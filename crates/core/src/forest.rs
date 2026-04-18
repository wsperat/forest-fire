//! Random forest implementation.
//!
//! The forest deliberately reuses the single-tree trainers instead of
//! maintaining a separate tree-building codepath. That keeps semantics aligned:
//! every constituent tree is still "just" a normal ForestFire tree trained on a
//! sampled table view with per-node feature subsampling.

use crate::bootstrap::BootstrapSampler;
use crate::ir::TrainingMetadata;
use crate::tree::shared::mix_seed;
use crate::{
    Criterion, FeaturePreprocessing, MaxFeatures, Model, Parallelism, PredictError, Task,
    TrainError, TreeType, capture_feature_preprocessing, training,
};
use forestfire_data::TableAccess;
use rayon::prelude::*;

/// Bagged ensemble of decision trees.
///
/// The forest stores full semantic [`Model`] trees rather than a bespoke forest
/// node format. That costs some memory, but it keeps IR conversion,
/// introspection, and optimized lowering consistent with the single-tree path.
#[derive(Debug, Clone)]
pub struct RandomForest {
    task: Task,
    criterion: Criterion,
    tree_type: TreeType,
    trees: Vec<Model>,
    compute_oob: bool,
    oob_score: Option<f64>,
    max_features: usize,
    seed: Option<u64>,
    num_features: usize,
    feature_preprocessing: Vec<FeaturePreprocessing>,
}

struct TrainedTree {
    model: Model,
    oob_rows: Vec<usize>,
}

struct SampledTable<'a> {
    base: &'a dyn TableAccess,
    row_indices: Vec<usize>,
}

struct NoCanaryTable<'a> {
    base: &'a dyn TableAccess,
}

impl RandomForest {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        task: Task,
        criterion: Criterion,
        tree_type: TreeType,
        trees: Vec<Model>,
        compute_oob: bool,
        oob_score: Option<f64>,
        max_features: usize,
        seed: Option<u64>,
        num_features: usize,
        feature_preprocessing: Vec<FeaturePreprocessing>,
    ) -> Self {
        Self {
            task,
            criterion,
            tree_type,
            trees,
            compute_oob,
            oob_score,
            max_features,
            seed,
            num_features,
            feature_preprocessing,
        }
    }

    pub(crate) fn train(
        train_set: &dyn TableAccess,
        config: training::RandomForestConfig,
        criterion: Criterion,
        parallelism: Parallelism,
    ) -> Result<Self, TrainError> {
        let n_trees = config.n_trees;
        if n_trees == 0 {
            return Err(TrainError::InvalidTreeCount(n_trees));
        }
        if matches!(config.max_features, MaxFeatures::Count(0)) {
            return Err(TrainError::InvalidMaxFeatures(0));
        }

        // Forests intentionally ignore canaries. For standalone trees they act as
        // a regularization/stopping heuristic, but in a forest they would make
        // bootstrap replicas overly conservative and reduce ensemble diversity.
        let train_set = NoCanaryTable::new(train_set);
        let sampler = BootstrapSampler::new(train_set.n_rows());
        let feature_preprocessing = capture_feature_preprocessing(&train_set);
        let max_features = config
            .max_features
            .resolve(config.task, train_set.binned_feature_count());
        let base_seed = config.seed.unwrap_or(0x0005_EEDF_0E57_u64);
        let tree_parallelism = Parallelism {
            thread_count: parallelism.thread_count,
        };
        let per_tree_parallelism = Parallelism::sequential();
        let train_tree = |tree_index: usize| -> Result<TrainedTree, TrainError> {
            let tree_seed = mix_seed(base_seed, tree_index as u64);
            let (sampled_rows, oob_rows) = sampler.sample_with_oob(tree_seed);
            // Sampling is implemented as a `TableAccess` view so the existing
            // tree trainers can stay oblivious to bootstrap mechanics.
            let sampled_table = SampledTable::new(&train_set, sampled_rows);
            let model = training::train_single_model_with_feature_subset(
                &sampled_table,
                training::SingleModelFeatureSubsetConfig {
                    base: training::SingleModelConfig {
                        task: config.task,
                        tree_type: config.tree_type,
                        criterion,
                        parallelism: per_tree_parallelism,
                        max_depth: config.max_depth,
                        min_samples_split: config.min_samples_split,
                        min_samples_leaf: config.min_samples_leaf,
                        missing_value_strategies: config.missing_value_strategies.clone(),
                        canary_filter: crate::CanaryFilter::default(),
                    },
                    max_features: Some(max_features),
                    random_seed: tree_seed,
                },
            )?;
            Ok(TrainedTree { model, oob_rows })
        };
        let trained_trees = if tree_parallelism.enabled() {
            (0..n_trees)
                .into_par_iter()
                .map(train_tree)
                .collect::<Result<Vec<_>, _>>()?
        } else {
            (0..n_trees)
                .map(train_tree)
                .collect::<Result<Vec<_>, _>>()?
        };
        let oob_score = if config.compute_oob {
            compute_oob_score(config.task, &trained_trees, &train_set)
        } else {
            None
        };
        let trees = trained_trees.into_iter().map(|tree| tree.model).collect();

        Ok(Self::new(
            config.task,
            criterion,
            config.tree_type,
            trees,
            config.compute_oob,
            oob_score,
            max_features,
            config.seed,
            train_set.n_features(),
            feature_preprocessing,
        ))
    }

    pub fn predict_table(&self, table: &dyn TableAccess) -> Vec<f64> {
        match self.task {
            Task::Regression => self.predict_regression_table(table),
            Task::Classification => self.predict_classification_table(table),
        }
    }

    pub fn predict_proba_table(
        &self,
        table: &dyn TableAccess,
    ) -> Result<Vec<Vec<f64>>, PredictError> {
        if self.task != Task::Classification {
            return Err(PredictError::ProbabilityPredictionRequiresClassification);
        }

        // Forest classification aggregates full class distributions rather than
        // voting on hard labels. This keeps `predict` and `predict_proba`
        // consistent and matches the optimized runtime lowering.
        let mut totals = self.trees[0].predict_proba_table(table)?;
        for tree in &self.trees[1..] {
            let probs = tree.predict_proba_table(table)?;
            for (row_totals, row_probs) in totals.iter_mut().zip(probs.iter()) {
                for (total, prob) in row_totals.iter_mut().zip(row_probs.iter()) {
                    *total += *prob;
                }
            }
        }

        let tree_count = self.trees.len() as f64;
        for row in &mut totals {
            for value in row {
                *value /= tree_count;
            }
        }

        Ok(totals)
    }

    pub fn task(&self) -> Task {
        self.task
    }

    pub fn criterion(&self) -> Criterion {
        self.criterion
    }

    pub fn tree_type(&self) -> TreeType {
        self.tree_type
    }

    pub fn trees(&self) -> &[Model] {
        &self.trees
    }

    pub fn num_features(&self) -> usize {
        self.num_features
    }

    pub fn feature_preprocessing(&self) -> &[FeaturePreprocessing] {
        &self.feature_preprocessing
    }

    pub fn training_metadata(&self) -> TrainingMetadata {
        let mut metadata = self.trees[0].training_metadata();
        metadata.algorithm = "rf".to_string();
        metadata.n_trees = Some(self.trees.len());
        metadata.max_features = Some(self.max_features);
        metadata.seed = self.seed;
        metadata.compute_oob = self.compute_oob;
        metadata.oob_score = self.oob_score;
        metadata.learning_rate = None;
        metadata.bootstrap = None;
        metadata.top_gradient_fraction = None;
        metadata.other_gradient_fraction = None;
        metadata
    }

    pub fn class_labels(&self) -> Option<Vec<f64>> {
        match self.task {
            Task::Classification => self.trees[0].class_labels(),
            Task::Regression => None,
        }
    }

    pub fn oob_score(&self) -> Option<f64> {
        self.oob_score
    }

    fn predict_regression_table(&self, table: &dyn TableAccess) -> Vec<f64> {
        let mut totals = self.trees[0].predict_table(table);
        for tree in &self.trees[1..] {
            let preds = tree.predict_table(table);
            for (total, pred) in totals.iter_mut().zip(preds.iter()) {
                *total += *pred;
            }
        }

        let tree_count = self.trees.len() as f64;
        for value in &mut totals {
            *value /= tree_count;
        }

        totals
    }

    fn predict_classification_table(&self, table: &dyn TableAccess) -> Vec<f64> {
        let probabilities = self
            .predict_proba_table(table)
            .expect("classification forest supports probabilities");
        let class_labels = self
            .class_labels()
            .expect("classification forest stores class labels");

        probabilities
            .into_iter()
            .map(|row| {
                let (best_index, _) = row
                    .iter()
                    .copied()
                    .enumerate()
                    .max_by(|(left_index, left), (right_index, right)| {
                        left.total_cmp(right)
                            .then_with(|| right_index.cmp(left_index))
                    })
                    .expect("classification probability row is non-empty");
                class_labels[best_index]
            })
            .collect()
    }
}

fn compute_oob_score(
    task: Task,
    trained_trees: &[TrainedTree],
    train_set: &dyn TableAccess,
) -> Option<f64> {
    match task {
        Task::Classification => compute_classification_oob_score(trained_trees, train_set),
        Task::Regression => compute_regression_oob_score(trained_trees, train_set),
    }
}

fn compute_classification_oob_score(
    trained_trees: &[TrainedTree],
    train_set: &dyn TableAccess,
) -> Option<f64> {
    let class_labels = trained_trees.first()?.model.class_labels()?;
    let mut totals = vec![vec![0.0; class_labels.len()]; train_set.n_rows()];
    let mut counts = vec![0usize; train_set.n_rows()];

    for tree in trained_trees {
        if tree.oob_rows.is_empty() {
            continue;
        }
        let oob_table = SampledTable::new(train_set, tree.oob_rows.clone());
        let probabilities = tree
            .model
            .predict_proba_table(&oob_table)
            .expect("classification tree supports predict_proba");
        for (&row_index, row_probs) in tree.oob_rows.iter().zip(probabilities.iter()) {
            for (total, prob) in totals[row_index].iter_mut().zip(row_probs.iter()) {
                *total += *prob;
            }
            counts[row_index] += 1;
        }
    }

    let mut correct = 0usize;
    let mut covered = 0usize;
    for row_index in 0..train_set.n_rows() {
        if counts[row_index] == 0 {
            continue;
        }
        covered += 1;
        let predicted = totals[row_index]
            .iter()
            .copied()
            .enumerate()
            .max_by(|(li, lv), (ri, rv)| lv.total_cmp(rv).then_with(|| ri.cmp(li)))
            .map(|(index, _)| class_labels[index])
            .expect("classification probability row is non-empty");
        if predicted
            .total_cmp(&train_set.target_value(row_index))
            .is_eq()
        {
            correct += 1;
        }
    }

    (covered > 0).then_some(correct as f64 / covered as f64)
}

fn compute_regression_oob_score(
    trained_trees: &[TrainedTree],
    train_set: &dyn TableAccess,
) -> Option<f64> {
    let mut totals = vec![0.0; train_set.n_rows()];
    let mut counts = vec![0usize; train_set.n_rows()];

    for tree in trained_trees {
        if tree.oob_rows.is_empty() {
            continue;
        }
        let oob_table = SampledTable::new(train_set, tree.oob_rows.clone());
        let predictions = tree.model.predict_table(&oob_table);
        for (&row_index, prediction) in tree.oob_rows.iter().zip(predictions.iter().copied()) {
            totals[row_index] += prediction;
            counts[row_index] += 1;
        }
    }

    let covered_rows: Vec<usize> = counts
        .iter()
        .enumerate()
        .filter_map(|(row_index, count)| (*count > 0).then_some(row_index))
        .collect();
    if covered_rows.is_empty() {
        return None;
    }

    let mean_target = covered_rows
        .iter()
        .map(|row_index| train_set.target_value(*row_index))
        .sum::<f64>()
        / covered_rows.len() as f64;
    let mut residual_sum = 0.0;
    let mut total_sum = 0.0;
    for row_index in covered_rows {
        let actual = train_set.target_value(row_index);
        let prediction = totals[row_index] / counts[row_index] as f64;
        residual_sum += (actual - prediction).powi(2);
        total_sum += (actual - mean_target).powi(2);
    }
    if total_sum == 0.0 {
        return None;
    }
    Some(1.0 - residual_sum / total_sum)
}

impl<'a> SampledTable<'a> {
    fn new(base: &'a dyn TableAccess, row_indices: Vec<usize>) -> Self {
        Self { base, row_indices }
    }

    fn resolve_row(&self, row_index: usize) -> usize {
        self.row_indices[row_index]
    }
}

impl<'a> NoCanaryTable<'a> {
    fn new(base: &'a dyn TableAccess) -> Self {
        Self { base }
    }
}

impl TableAccess for SampledTable<'_> {
    fn n_rows(&self) -> usize {
        self.row_indices.len()
    }

    fn n_features(&self) -> usize {
        self.base.n_features()
    }

    fn canaries(&self) -> usize {
        self.base.canaries()
    }

    fn numeric_bin_cap(&self) -> usize {
        self.base.numeric_bin_cap()
    }

    fn binned_feature_count(&self) -> usize {
        self.base.binned_feature_count()
    }

    fn feature_value(&self, feature_index: usize, row_index: usize) -> f64 {
        self.base
            .feature_value(feature_index, self.resolve_row(row_index))
    }

    fn is_missing(&self, feature_index: usize, row_index: usize) -> bool {
        self.base
            .is_missing(feature_index, self.resolve_row(row_index))
    }

    fn is_binary_feature(&self, index: usize) -> bool {
        self.base.is_binary_feature(index)
    }

    fn binned_value(&self, feature_index: usize, row_index: usize) -> u16 {
        self.base
            .binned_value(feature_index, self.resolve_row(row_index))
    }

    fn binned_boolean_value(&self, feature_index: usize, row_index: usize) -> Option<bool> {
        self.base
            .binned_boolean_value(feature_index, self.resolve_row(row_index))
    }

    fn binned_column_kind(&self, index: usize) -> forestfire_data::BinnedColumnKind {
        self.base.binned_column_kind(index)
    }

    fn is_binary_binned_feature(&self, index: usize) -> bool {
        self.base.is_binary_binned_feature(index)
    }

    fn target_value(&self, row_index: usize) -> f64 {
        self.base.target_value(self.resolve_row(row_index))
    }
}

impl TableAccess for NoCanaryTable<'_> {
    fn n_rows(&self) -> usize {
        self.base.n_rows()
    }

    fn n_features(&self) -> usize {
        self.base.n_features()
    }

    fn canaries(&self) -> usize {
        0
    }

    fn numeric_bin_cap(&self) -> usize {
        self.base.numeric_bin_cap()
    }

    fn binned_feature_count(&self) -> usize {
        self.base.binned_feature_count() - self.base.canaries()
    }

    fn feature_value(&self, feature_index: usize, row_index: usize) -> f64 {
        self.base.feature_value(feature_index, row_index)
    }

    fn is_missing(&self, feature_index: usize, row_index: usize) -> bool {
        self.base.is_missing(feature_index, row_index)
    }

    fn is_binary_feature(&self, index: usize) -> bool {
        self.base.is_binary_feature(index)
    }

    fn binned_value(&self, feature_index: usize, row_index: usize) -> u16 {
        self.base.binned_value(feature_index, row_index)
    }

    fn binned_boolean_value(&self, feature_index: usize, row_index: usize) -> Option<bool> {
        self.base.binned_boolean_value(feature_index, row_index)
    }

    fn binned_column_kind(&self, index: usize) -> forestfire_data::BinnedColumnKind {
        self.base.binned_column_kind(index)
    }

    fn is_binary_binned_feature(&self, index: usize) -> bool {
        self.base.is_binary_binned_feature(index)
    }

    fn target_value(&self, row_index: usize) -> f64 {
        self.base.target_value(row_index)
    }
}
