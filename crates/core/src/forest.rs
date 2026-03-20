use crate::bootstrap::BootstrapSampler;
use crate::ir::TrainingMetadata;
use crate::{
    Criterion, FeaturePreprocessing, MaxFeatures, Model, Parallelism, PredictError, Task,
    TrainConfig, TrainError, TreeType, capture_feature_preprocessing, training,
};
use forestfire_data::TableAccess;
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct RandomForest {
    task: Task,
    criterion: Criterion,
    tree_type: TreeType,
    trees: Vec<Model>,
    max_features: usize,
    seed: Option<u64>,
    num_features: usize,
    feature_preprocessing: Vec<FeaturePreprocessing>,
}

struct SampledTable<'a> {
    base: &'a dyn TableAccess,
    row_indices: Vec<usize>,
}

impl RandomForest {
    pub fn new(
        task: Task,
        criterion: Criterion,
        tree_type: TreeType,
        trees: Vec<Model>,
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
            max_features,
            seed,
            num_features,
            feature_preprocessing,
        }
    }

    pub(crate) fn train(
        train_set: &dyn TableAccess,
        config: TrainConfig,
        criterion: Criterion,
        parallelism: Parallelism,
    ) -> Result<Self, TrainError> {
        let n_trees = config.n_trees.unwrap_or(1000);
        if n_trees == 0 {
            return Err(TrainError::InvalidTreeCount(n_trees));
        }
        if matches!(config.max_features, MaxFeatures::Count(0)) {
            return Err(TrainError::InvalidMaxFeatures(0));
        }

        let sampler = BootstrapSampler::new(train_set.n_rows());
        let feature_preprocessing = capture_feature_preprocessing(train_set);
        let max_features = config
            .max_features
            .resolve(config.task, train_set.binned_feature_count());
        let base_seed = config.seed.unwrap_or(0x5EED_F0E5_7u64);
        let tree_parallelism = Parallelism {
            thread_count: parallelism.thread_count,
        };
        let per_tree_parallelism = Parallelism::sequential();
        let train_tree = |tree_index: usize| -> Result<Model, TrainError> {
            let tree_seed = mix_seed(base_seed, tree_index as u64);
            let sampled_rows = sampler.sample(tree_seed);
            let sampled_table = SampledTable::new(train_set, sampled_rows);
            training::train_single_model_with_feature_subset(
                &sampled_table,
                config.task,
                config.tree_type,
                criterion,
                per_tree_parallelism,
                config.min_samples_split.unwrap_or(2),
                config.min_samples_leaf.unwrap_or(1),
                Some(max_features),
                tree_seed,
            )
        };
        let trees = if tree_parallelism.enabled() {
            (0..n_trees)
                .into_par_iter()
                .map(train_tree)
                .collect::<Result<Vec<_>, _>>()?
        } else {
            (0..n_trees)
                .map(train_tree)
                .collect::<Result<Vec<_>, _>>()?
        };

        Ok(Self::new(
            config.task,
            criterion,
            config.tree_type,
            trees,
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
        metadata
    }

    pub fn class_labels(&self) -> Option<Vec<f64>> {
        match self.task {
            Task::Classification => self.trees[0].class_labels(),
            Task::Regression => None,
        }
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

fn mix_seed(base_seed: u64, value: u64) -> u64 {
    base_seed ^ value.wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(17)
}

impl<'a> SampledTable<'a> {
    fn new(base: &'a dyn TableAccess, row_indices: Vec<usize>) -> Self {
        Self { base, row_indices }
    }

    fn resolve_row(&self, row_index: usize) -> usize {
        self.row_indices[row_index]
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
