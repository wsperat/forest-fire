//! Gradient boosting implementation.
//!
//! The boosting path is intentionally "LightGBM-like" rather than a direct
//! clone. It uses second-order trees, shrinkage, and gradient-focused sampling,
//! but it keeps ForestFire's canary mechanism active so a stage can stop before
//! growing a tree whose best root split is indistinguishable from noise.

use crate::bootstrap::BootstrapSampler;
use crate::ir::TrainingMetadata;
use crate::tree::second_order::{
    SecondOrderRegressionTreeError, SecondOrderRegressionTreeOptions,
    train_cart_regressor_from_gradients_and_hessians_with_status,
    train_oblivious_regressor_from_gradients_and_hessians_with_status,
    train_randomized_regressor_from_gradients_and_hessians_with_status,
};
use crate::tree::shared::mix_seed;
use crate::{
    Criterion, FeaturePreprocessing, Model, Parallelism, PredictError, Task, TrainConfig, TreeType,
    capture_feature_preprocessing,
};
use forestfire_data::TableAccess;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

/// Stage-wise gradient-boosted tree ensemble.
///
/// The ensemble keeps explicit tree weights and a base score so the semantic
/// model can reconstruct raw margins exactly for both prediction and IR export.
#[derive(Debug, Clone)]
pub struct GradientBoostedTrees {
    task: Task,
    tree_type: TreeType,
    trees: Vec<Model>,
    tree_weights: Vec<f64>,
    base_score: f64,
    learning_rate: f64,
    bootstrap: bool,
    top_gradient_fraction: f64,
    other_gradient_fraction: f64,
    max_features: usize,
    seed: Option<u64>,
    num_features: usize,
    feature_preprocessing: Vec<FeaturePreprocessing>,
    class_labels: Option<Vec<f64>>,
    training_canaries: usize,
}

#[derive(Debug)]
pub enum BoostingError {
    InvalidTargetValue { row: usize, value: f64 },
    UnsupportedClassificationClassCount(usize),
    InvalidLearningRate(f64),
    InvalidTopGradientFraction(f64),
    InvalidOtherGradientFraction(f64),
    SecondOrderTree(SecondOrderRegressionTreeError),
}

impl std::fmt::Display for BoostingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BoostingError::InvalidTargetValue { row, value } => write!(
                f,
                "Boosting targets must be finite values. Found {} at row {}.",
                value, row
            ),
            BoostingError::UnsupportedClassificationClassCount(count) => write!(
                f,
                "Gradient boosting currently supports binary classification only. Found {} classes.",
                count
            ),
            BoostingError::InvalidLearningRate(value) => write!(
                f,
                "learning_rate must be finite and greater than 0. Found {}.",
                value
            ),
            BoostingError::InvalidTopGradientFraction(value) => write!(
                f,
                "top_gradient_fraction must be in the interval (0, 1]. Found {}.",
                value
            ),
            BoostingError::InvalidOtherGradientFraction(value) => write!(
                f,
                "other_gradient_fraction must be in the interval [0, 1), and top_gradient_fraction + other_gradient_fraction must be at most 1. Found {}.",
                value
            ),
            BoostingError::SecondOrderTree(err) => err.fmt(f),
        }
    }
}

impl std::error::Error for BoostingError {}

struct SampledTable<'a> {
    base: &'a dyn TableAccess,
    row_indices: Vec<usize>,
}

impl GradientBoostedTrees {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        task: Task,
        tree_type: TreeType,
        trees: Vec<Model>,
        tree_weights: Vec<f64>,
        base_score: f64,
        learning_rate: f64,
        bootstrap: bool,
        top_gradient_fraction: f64,
        other_gradient_fraction: f64,
        max_features: usize,
        seed: Option<u64>,
        num_features: usize,
        feature_preprocessing: Vec<FeaturePreprocessing>,
        class_labels: Option<Vec<f64>>,
        training_canaries: usize,
    ) -> Self {
        Self {
            task,
            tree_type,
            trees,
            tree_weights,
            base_score,
            learning_rate,
            bootstrap,
            top_gradient_fraction,
            other_gradient_fraction,
            max_features,
            seed,
            num_features,
            feature_preprocessing,
            class_labels,
            training_canaries,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn train(
        train_set: &dyn TableAccess,
        config: TrainConfig,
        parallelism: Parallelism,
    ) -> Result<Self, BoostingError> {
        let missing_value_strategies = config
            .missing_value_strategy
            .resolve_for_feature_count(train_set.binned_feature_count())
            .unwrap_or_else(|err| {
                panic!("unexpected training error while resolving missing strategy: {err}")
            });
        Self::train_with_missing_value_strategies(
            train_set,
            config,
            parallelism,
            missing_value_strategies,
        )
    }

    pub(crate) fn train_with_missing_value_strategies(
        train_set: &dyn TableAccess,
        config: TrainConfig,
        parallelism: Parallelism,
        missing_value_strategies: Vec<crate::MissingValueStrategy>,
    ) -> Result<Self, BoostingError> {
        let n_trees = config.n_trees.unwrap_or(100);
        let learning_rate = config.learning_rate.unwrap_or(0.1);
        let bootstrap = config.bootstrap;
        let top_gradient_fraction = config.top_gradient_fraction.unwrap_or(0.2);
        let other_gradient_fraction = config.other_gradient_fraction.unwrap_or(0.1);
        validate_boosting_parameters(
            train_set,
            learning_rate,
            top_gradient_fraction,
            other_gradient_fraction,
        )?;

        let max_features = config
            .max_features
            .resolve(config.task, train_set.n_features());
        let base_seed = config.seed.unwrap_or(0xB005_7EED_u64);
        let tree_options = crate::RegressionTreeOptions {
            max_depth: config.max_depth.unwrap_or(8),
            min_samples_split: config.min_samples_split.unwrap_or(2),
            min_samples_leaf: config.min_samples_leaf.unwrap_or(1),
            max_features: Some(max_features),
            random_seed: 0,
            missing_value_strategies,
            canary_filter: config.canary_filter,
        };
        let tree_options = SecondOrderRegressionTreeOptions {
            tree_options,
            l2_regularization: 1.0,
            min_sum_hessian_in_leaf: 1e-3,
            min_gain_to_split: 0.0,
        };
        let feature_preprocessing = capture_feature_preprocessing(train_set);
        let sampler = BootstrapSampler::new(train_set.n_rows());

        let (mut raw_predictions, class_labels, base_score) = match config.task {
            Task::Regression => {
                let targets = finite_targets(train_set)?;
                let base_score = targets.iter().sum::<f64>() / targets.len() as f64;
                (vec![base_score; train_set.n_rows()], None, base_score)
            }
            Task::Classification => {
                let (labels, encoded_targets) = binary_classification_targets(train_set)?;
                let positive_rate = (encoded_targets.iter().sum::<f64>()
                    / encoded_targets.len() as f64)
                    .clamp(1e-6, 1.0 - 1e-6);
                let base_score = (positive_rate / (1.0 - positive_rate)).ln();
                (
                    vec![base_score; train_set.n_rows()],
                    Some(labels),
                    base_score,
                )
            }
        };

        let mut trees = Vec::with_capacity(n_trees);
        let mut tree_weights = Vec::with_capacity(n_trees);
        let regression_targets = if config.task == Task::Regression {
            Some(finite_targets(train_set)?)
        } else {
            None
        };
        let classification_targets = if config.task == Task::Classification {
            Some(binary_classification_targets(train_set)?.1)
        } else {
            None
        };

        for tree_index in 0..n_trees {
            let stage_seed = mix_seed(base_seed, tree_index as u64);
            // Gradients/hessians are recomputed from the current ensemble margin
            // at every stage, which keeps the tree learner focused on residual
            // structure that earlier trees did not explain.
            let (gradients, hessians) = match config.task {
                Task::Regression => squared_error_gradients_and_hessians(
                    raw_predictions.as_slice(),
                    regression_targets
                        .as_ref()
                        .expect("regression targets exist for regression boosting"),
                ),
                Task::Classification => logistic_gradients_and_hessians(
                    raw_predictions.as_slice(),
                    classification_targets
                        .as_ref()
                        .expect("classification targets exist for classification boosting"),
                ),
            };

            let base_rows = if bootstrap {
                sampler.sample(stage_seed)
            } else {
                (0..train_set.n_rows()).collect()
            };
            // GOSS-style sampling keeps the largest gradients deterministically
            // and samples part of the remainder. This biases work toward the rows
            // where the current ensemble is most wrong.
            let sampled_rows = gradient_focus_sample(
                &base_rows,
                &gradients,
                &hessians,
                top_gradient_fraction,
                other_gradient_fraction,
                mix_seed(stage_seed, 0x6011_5A11),
            );
            let sampled_table = SampledTable::new(train_set, sampled_rows.row_indices);
            let mut stage_tree_options = tree_options.clone();
            stage_tree_options.tree_options.random_seed = stage_seed;
            let stage_result = match config.tree_type {
                TreeType::Cart => train_cart_regressor_from_gradients_and_hessians_with_status(
                    &sampled_table,
                    &sampled_rows.gradients,
                    &sampled_rows.hessians,
                    parallelism,
                    stage_tree_options,
                ),
                TreeType::Randomized => {
                    train_randomized_regressor_from_gradients_and_hessians_with_status(
                        &sampled_table,
                        &sampled_rows.gradients,
                        &sampled_rows.hessians,
                        parallelism,
                        stage_tree_options,
                    )
                }
                TreeType::Oblivious => {
                    train_oblivious_regressor_from_gradients_and_hessians_with_status(
                        &sampled_table,
                        &sampled_rows.gradients,
                        &sampled_rows.hessians,
                        parallelism,
                        stage_tree_options,
                    )
                }
                _ => unreachable!("boosting tree type validated by training dispatch"),
            }
            .map_err(BoostingError::SecondOrderTree)?;

            // A canary root win means the stage could not find a real feature
            // stronger than shuffled noise, so boosting stops early.
            if stage_result.root_canary_selected {
                break;
            }

            let stage_tree = stage_result.model;
            let stage_model = Model::DecisionTreeRegressor(stage_tree);
            let stage_predictions = stage_model.predict_table(train_set);
            for (raw_prediction, stage_prediction) in raw_predictions
                .iter_mut()
                .zip(stage_predictions.iter().copied())
            {
                // Trees are fit on raw margins; shrinkage is applied only when
                // updating the ensemble prediction.
                *raw_prediction += learning_rate * stage_prediction;
            }
            tree_weights.push(learning_rate);
            trees.push(stage_model);
        }

        Ok(Self::new(
            config.task,
            config.tree_type,
            trees,
            tree_weights,
            base_score,
            learning_rate,
            bootstrap,
            top_gradient_fraction,
            other_gradient_fraction,
            max_features,
            config.seed,
            train_set.n_features(),
            feature_preprocessing,
            class_labels,
            train_set.canaries(),
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
        Ok(self
            .raw_scores(table)
            .into_iter()
            .map(|score| {
                let positive = sigmoid(score);
                vec![1.0 - positive, positive]
            })
            .collect())
    }

    pub fn task(&self) -> Task {
        self.task
    }

    pub fn criterion(&self) -> Criterion {
        Criterion::SecondOrder
    }

    pub fn tree_type(&self) -> TreeType {
        self.tree_type
    }

    pub fn trees(&self) -> &[Model] {
        &self.trees
    }

    pub fn tree_weights(&self) -> &[f64] {
        &self.tree_weights
    }

    pub fn base_score(&self) -> f64 {
        self.base_score
    }

    pub fn num_features(&self) -> usize {
        self.num_features
    }

    pub fn feature_preprocessing(&self) -> &[FeaturePreprocessing] {
        &self.feature_preprocessing
    }

    pub fn class_labels(&self) -> Option<Vec<f64>> {
        self.class_labels.clone()
    }

    pub fn training_metadata(&self) -> TrainingMetadata {
        TrainingMetadata {
            algorithm: "gbm".to_string(),
            task: match self.task {
                Task::Regression => "regression".to_string(),
                Task::Classification => "classification".to_string(),
            },
            tree_type: match self.tree_type {
                TreeType::Cart => "cart".to_string(),
                TreeType::Randomized => "randomized".to_string(),
                TreeType::Oblivious => "oblivious".to_string(),
                _ => unreachable!("boosting only supports cart/randomized/oblivious"),
            },
            criterion: "second_order".to_string(),
            canaries: self.training_canaries,
            compute_oob: false,
            max_depth: self.trees.first().and_then(Model::max_depth),
            min_samples_split: self.trees.first().and_then(Model::min_samples_split),
            min_samples_leaf: self.trees.first().and_then(Model::min_samples_leaf),
            n_trees: Some(self.trees.len()),
            max_features: Some(self.max_features),
            seed: self.seed,
            oob_score: None,
            class_labels: self.class_labels.clone(),
            learning_rate: Some(self.learning_rate),
            bootstrap: Some(self.bootstrap),
            top_gradient_fraction: Some(self.top_gradient_fraction),
            other_gradient_fraction: Some(self.other_gradient_fraction),
        }
    }

    fn raw_scores(&self, table: &dyn TableAccess) -> Vec<f64> {
        let mut scores = vec![self.base_score; table.n_rows()];
        for (tree, weight) in self.trees.iter().zip(self.tree_weights.iter().copied()) {
            let predictions = tree.predict_table(table);
            for (score, prediction) in scores.iter_mut().zip(predictions.iter().copied()) {
                *score += weight * prediction;
            }
        }
        scores
    }

    fn predict_regression_table(&self, table: &dyn TableAccess) -> Vec<f64> {
        self.raw_scores(table)
    }

    fn predict_classification_table(&self, table: &dyn TableAccess) -> Vec<f64> {
        let class_labels = self
            .class_labels
            .as_ref()
            .expect("classification boosting stores class labels");
        self.raw_scores(table)
            .into_iter()
            .map(|score| {
                if sigmoid(score) >= 0.5 {
                    class_labels[1]
                } else {
                    class_labels[0]
                }
            })
            .collect()
    }
}

struct GradientFocusedSample {
    row_indices: Vec<usize>,
    gradients: Vec<f64>,
    hessians: Vec<f64>,
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

fn validate_boosting_parameters(
    train_set: &dyn TableAccess,
    learning_rate: f64,
    top_gradient_fraction: f64,
    other_gradient_fraction: f64,
) -> Result<(), BoostingError> {
    if train_set.n_rows() == 0 {
        return Err(BoostingError::InvalidLearningRate(learning_rate));
    }
    if !learning_rate.is_finite() || learning_rate <= 0.0 {
        return Err(BoostingError::InvalidLearningRate(learning_rate));
    }
    if !top_gradient_fraction.is_finite()
        || top_gradient_fraction <= 0.0
        || top_gradient_fraction > 1.0
    {
        return Err(BoostingError::InvalidTopGradientFraction(
            top_gradient_fraction,
        ));
    }
    if !other_gradient_fraction.is_finite()
        || !(0.0..1.0).contains(&other_gradient_fraction)
        || top_gradient_fraction + other_gradient_fraction > 1.0
    {
        return Err(BoostingError::InvalidOtherGradientFraction(
            other_gradient_fraction,
        ));
    }
    Ok(())
}

fn finite_targets(train_set: &dyn TableAccess) -> Result<Vec<f64>, BoostingError> {
    (0..train_set.n_rows())
        .map(|row_index| {
            let value = train_set.target_value(row_index);
            if value.is_finite() {
                Ok(value)
            } else {
                Err(BoostingError::InvalidTargetValue {
                    row: row_index,
                    value,
                })
            }
        })
        .collect()
}

fn binary_classification_targets(
    train_set: &dyn TableAccess,
) -> Result<(Vec<f64>, Vec<f64>), BoostingError> {
    let mut labels = finite_targets(train_set)?;
    labels.sort_by(|left, right| left.total_cmp(right));
    labels.dedup_by(|left, right| left.total_cmp(right).is_eq());
    if labels.len() != 2 {
        return Err(BoostingError::UnsupportedClassificationClassCount(
            labels.len(),
        ));
    }

    let negative = labels[0];
    let encoded = (0..train_set.n_rows())
        .map(|row_index| {
            if train_set
                .target_value(row_index)
                .total_cmp(&negative)
                .is_eq()
            {
                0.0
            } else {
                1.0
            }
        })
        .collect();
    Ok((labels, encoded))
}

fn squared_error_gradients_and_hessians(
    raw_predictions: &[f64],
    targets: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    (
        raw_predictions
            .iter()
            .zip(targets.iter())
            .map(|(prediction, target)| prediction - target)
            .collect(),
        vec![1.0; targets.len()],
    )
}

fn logistic_gradients_and_hessians(
    raw_predictions: &[f64],
    targets: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let mut gradients = Vec::with_capacity(targets.len());
    let mut hessians = Vec::with_capacity(targets.len());
    for (raw_prediction, target) in raw_predictions.iter().zip(targets.iter()) {
        let probability = sigmoid(*raw_prediction);
        gradients.push(probability - target);
        hessians.push((probability * (1.0 - probability)).max(1e-12));
    }
    (gradients, hessians)
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        let exp = (-value).exp();
        1.0 / (1.0 + exp)
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn gradient_focus_sample(
    base_rows: &[usize],
    gradients: &[f64],
    hessians: &[f64],
    top_gradient_fraction: f64,
    other_gradient_fraction: f64,
    seed: u64,
) -> GradientFocusedSample {
    let mut ranked = base_rows
        .iter()
        .copied()
        .map(|row_index| (row_index, gradients[row_index].abs()))
        .collect::<Vec<_>>();
    ranked.sort_by(|(left_row, left_abs), (right_row, right_abs)| {
        right_abs
            .total_cmp(left_abs)
            .then_with(|| left_row.cmp(right_row))
    });

    let top_count = ((ranked.len() as f64) * top_gradient_fraction)
        .ceil()
        .clamp(1.0, ranked.len() as f64) as usize;
    let mut row_indices = Vec::with_capacity(ranked.len());
    let mut sampled_gradients = Vec::with_capacity(ranked.len());
    let mut sampled_hessians = Vec::with_capacity(ranked.len());

    for (row_index, _) in ranked.iter().take(top_count) {
        row_indices.push(*row_index);
        sampled_gradients.push(gradients[*row_index]);
        sampled_hessians.push(hessians[*row_index]);
    }

    if top_count < ranked.len() && other_gradient_fraction > 0.0 {
        let remaining = ranked[top_count..]
            .iter()
            .map(|(row_index, _)| *row_index)
            .collect::<Vec<_>>();
        let other_count = ((remaining.len() as f64) * other_gradient_fraction)
            .ceil()
            .min(remaining.len() as f64) as usize;
        if other_count > 0 {
            let mut remaining = remaining;
            let mut rng = StdRng::seed_from_u64(seed);
            remaining.shuffle(&mut rng);
            let gradient_scale = (1.0 - top_gradient_fraction) / other_gradient_fraction;
            for row_index in remaining.into_iter().take(other_count) {
                row_indices.push(row_index);
                sampled_gradients.push(gradients[row_index] * gradient_scale);
                sampled_hessians.push(hessians[row_index] * gradient_scale);
            }
        }
    }

    GradientFocusedSample {
        row_indices,
        gradients: sampled_gradients,
        hessians: sampled_hessians,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CanaryFilter, MaxFeatures, TrainAlgorithm, TrainConfig};
    use forestfire_data::{BinnedColumnKind, TableAccess};
    use forestfire_data::{DenseTable, NumericBins};

    #[test]
    fn regression_boosting_fits_simple_signal() {
        let table = DenseTable::with_options(
            vec![
                vec![0.0],
                vec![0.0],
                vec![1.0],
                vec![1.0],
                vec![2.0],
                vec![2.0],
            ],
            vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
            0,
            NumericBins::fixed(8).unwrap(),
        )
        .unwrap();

        let model = GradientBoostedTrees::train(
            &table,
            TrainConfig {
                algorithm: TrainAlgorithm::Gbm,
                task: Task::Regression,
                tree_type: TreeType::Cart,
                criterion: Criterion::SecondOrder,
                n_trees: Some(20),
                learning_rate: Some(0.2),
                max_depth: Some(2),
                ..TrainConfig::default()
            },
            Parallelism::sequential(),
        )
        .unwrap();

        let predictions = model.predict_table(&table);
        assert!(predictions[0] < predictions[2]);
        assert!(predictions[2] < predictions[4]);
    }

    #[test]
    fn classification_boosting_produces_binary_probabilities() {
        let table = DenseTable::with_options(
            vec![vec![0.0], vec![0.1], vec![0.9], vec![1.0]],
            vec![0.0, 0.0, 1.0, 1.0],
            0,
            NumericBins::fixed(8).unwrap(),
        )
        .unwrap();

        let model = GradientBoostedTrees::train(
            &table,
            TrainConfig {
                algorithm: TrainAlgorithm::Gbm,
                task: Task::Classification,
                tree_type: TreeType::Cart,
                criterion: Criterion::SecondOrder,
                n_trees: Some(25),
                learning_rate: Some(0.2),
                max_depth: Some(2),
                ..TrainConfig::default()
            },
            Parallelism::sequential(),
        )
        .unwrap();

        let probabilities = model.predict_proba_table(&table).unwrap();
        assert_eq!(probabilities.len(), 4);
        assert!(probabilities[0][1] < 0.5);
        assert!(probabilities[3][1] > 0.5);
    }

    #[test]
    fn classification_boosting_rejects_multiclass_targets() {
        let table =
            DenseTable::new(vec![vec![0.0], vec![1.0], vec![2.0]], vec![0.0, 1.0, 2.0]).unwrap();

        let error = GradientBoostedTrees::train(
            &table,
            TrainConfig {
                algorithm: TrainAlgorithm::Gbm,
                task: Task::Classification,
                tree_type: TreeType::Cart,
                criterion: Criterion::SecondOrder,
                ..TrainConfig::default()
            },
            Parallelism::sequential(),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            BoostingError::UnsupportedClassificationClassCount(3)
        ));
    }

    struct RootCanaryTable;

    struct FilteredRootCanaryTable;

    impl TableAccess for RootCanaryTable {
        fn n_rows(&self) -> usize {
            4
        }

        fn n_features(&self) -> usize {
            1
        }

        fn canaries(&self) -> usize {
            1
        }

        fn numeric_bin_cap(&self) -> usize {
            2
        }

        fn binned_feature_count(&self) -> usize {
            2
        }

        fn feature_value(&self, _feature_index: usize, _row_index: usize) -> f64 {
            0.0
        }

        fn is_missing(&self, _feature_index: usize, _row_index: usize) -> bool {
            false
        }

        fn is_binary_feature(&self, _index: usize) -> bool {
            true
        }

        fn binned_value(&self, feature_index: usize, row_index: usize) -> u16 {
            match feature_index {
                0 => 0,
                1 => u16::from(row_index >= 2),
                _ => unreachable!(),
            }
        }

        fn binned_boolean_value(&self, feature_index: usize, row_index: usize) -> Option<bool> {
            Some(match feature_index {
                0 => false,
                1 => row_index >= 2,
                _ => unreachable!(),
            })
        }

        fn binned_column_kind(&self, index: usize) -> BinnedColumnKind {
            match index {
                0 => BinnedColumnKind::Real { source_index: 0 },
                1 => BinnedColumnKind::Canary {
                    source_index: 0,
                    copy_index: 0,
                },
                _ => unreachable!(),
            }
        }

        fn is_binary_binned_feature(&self, _index: usize) -> bool {
            true
        }

        fn target_value(&self, row_index: usize) -> f64 {
            [0.0, 0.0, 1.0, 1.0][row_index]
        }
    }

    impl TableAccess for FilteredRootCanaryTable {
        fn n_rows(&self) -> usize {
            4
        }

        fn n_features(&self) -> usize {
            1
        }

        fn canaries(&self) -> usize {
            1
        }

        fn numeric_bin_cap(&self) -> usize {
            2
        }

        fn binned_feature_count(&self) -> usize {
            2
        }

        fn feature_value(&self, _feature_index: usize, _row_index: usize) -> f64 {
            0.0
        }

        fn is_missing(&self, _feature_index: usize, _row_index: usize) -> bool {
            false
        }

        fn is_binary_feature(&self, _index: usize) -> bool {
            true
        }

        fn binned_value(&self, feature_index: usize, row_index: usize) -> u16 {
            u16::from(
                self.binned_boolean_value(feature_index, row_index)
                    .expect("all features are observed"),
            )
        }

        fn binned_boolean_value(&self, feature_index: usize, row_index: usize) -> Option<bool> {
            Some(match feature_index {
                0 => row_index == 3,
                1 => row_index >= 2,
                _ => unreachable!(),
            })
        }

        fn binned_column_kind(&self, index: usize) -> BinnedColumnKind {
            match index {
                0 => BinnedColumnKind::Real { source_index: 0 },
                1 => BinnedColumnKind::Canary {
                    source_index: 0,
                    copy_index: 0,
                },
                _ => unreachable!(),
            }
        }

        fn is_binary_binned_feature(&self, _index: usize) -> bool {
            true
        }

        fn target_value(&self, row_index: usize) -> f64 {
            [0.0, 0.0, 1.0, 1.0][row_index]
        }
    }

    #[test]
    fn boosting_stops_when_root_split_is_a_canary() {
        let table = RootCanaryTable;

        let model = GradientBoostedTrees::train(
            &table,
            TrainConfig {
                algorithm: TrainAlgorithm::Gbm,
                task: Task::Regression,
                tree_type: TreeType::Cart,
                criterion: Criterion::SecondOrder,
                n_trees: Some(10),
                max_features: MaxFeatures::All,
                learning_rate: Some(0.1),
                top_gradient_fraction: Some(1.0),
                other_gradient_fraction: Some(0.0),
                ..TrainConfig::default()
            },
            Parallelism::sequential(),
        )
        .unwrap();

        assert_eq!(model.trees().len(), 0);
        assert_eq!(model.training_metadata().n_trees, Some(0));
        assert!(
            model
                .predict_table(&table)
                .iter()
                .all(|value| value.is_finite())
        );
    }

    #[test]
    fn boosting_can_use_a_real_root_split_inside_top_n_canary_filter() {
        let table = FilteredRootCanaryTable;

        let model = GradientBoostedTrees::train(
            &table,
            TrainConfig {
                algorithm: TrainAlgorithm::Gbm,
                task: Task::Regression,
                tree_type: TreeType::Cart,
                criterion: Criterion::SecondOrder,
                n_trees: Some(10),
                max_features: MaxFeatures::All,
                learning_rate: Some(0.1),
                top_gradient_fraction: Some(1.0),
                other_gradient_fraction: Some(0.0),
                canary_filter: CanaryFilter::TopN(2),
                ..TrainConfig::default()
            },
            Parallelism::sequential(),
        )
        .unwrap();

        assert!(!model.trees().is_empty());
    }
}
