//! ForestFire core model, training, inference, and interchange layer.
//!
//! The crate is organized around a few stable abstractions:
//!
//! - [`forestfire_data::TableAccess`] is the common data boundary for both
//!   training and inference.
//! - [`TrainConfig`] is the normalized configuration surface shared by the Rust
//!   and Python APIs.
//! - [`Model`] is the semantic model view used for exact prediction,
//!   serialization, and introspection.
//! - [`OptimizedModel`] is a lowered runtime view used when prediction speed
//!   matters more than preserving the original tree layout.
//!
//! Keeping the semantic model and the runtime model separate is deliberate. It
//! makes export and introspection straightforward while still allowing the
//! optimized path to use layouts that are awkward to serialize directly.

use forestfire_data::{
    BinnedColumnKind, MAX_NUMERIC_BINS, NumericBins, TableAccess, numeric_bin_boundaries,
};
#[cfg(feature = "polars")]
use polars::prelude::{Column, DataFrame, DataType, IdxSize, LazyFrame};
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::Arc;
use wide::{u16x8, u32x8};

mod boosting;
mod bootstrap;
mod compiled_artifact;
mod forest;
mod inference_input;
mod introspection;
pub mod ir;
mod model_api;
mod optimized_runtime;
mod runtime_planning;
mod sampling;
mod training;
pub mod tree;

pub use boosting::BoostingError;
pub use boosting::GradientBoostedTrees;
pub use compiled_artifact::CompiledArtifactError;
pub use forest::RandomForest;
pub use introspection::IntrospectionError;
pub use introspection::PredictionHistogramEntry;
pub use introspection::PredictionValueStats;
pub use introspection::TreeStructureSummary;
pub use ir::IrError;
pub use ir::ModelPackageIr;
pub use model_api::OptimizedModel;
pub use tree::classifier::DecisionTreeAlgorithm;
pub use tree::classifier::DecisionTreeClassifier;
pub use tree::classifier::DecisionTreeError;
pub use tree::classifier::DecisionTreeOptions;
pub use tree::classifier::train_c45;
pub use tree::classifier::train_cart;
pub use tree::classifier::train_id3;
pub use tree::classifier::train_oblivious;
pub use tree::classifier::train_randomized;
pub use tree::regressor::DecisionTreeRegressor;
pub use tree::regressor::RegressionTreeAlgorithm;
pub use tree::regressor::RegressionTreeError;
pub use tree::regressor::RegressionTreeOptions;
pub use tree::regressor::train_cart_regressor;
pub use tree::regressor::train_oblivious_regressor;
pub use tree::regressor::train_randomized_regressor;
#[cfg(feature = "polars")]
const LAZYFRAME_PREDICT_BATCH_ROWS: usize = 10_000;
pub(crate) use inference_input::ColumnMajorBinnedMatrix;
pub(crate) use inference_input::CompactBinnedColumn;
pub(crate) use inference_input::InferenceTable;
pub(crate) use inference_input::ProjectedTableView;
#[cfg(feature = "polars")]
pub(crate) use inference_input::polars_named_columns;
pub(crate) use introspection::prediction_value_stats;
pub(crate) use introspection::tree_structure_summary;
pub(crate) use optimized_runtime::InferenceExecutor;
pub(crate) use optimized_runtime::OBLIVIOUS_SIMD_LANES;
pub(crate) use optimized_runtime::OptimizedBinaryClassifierNode;
pub(crate) use optimized_runtime::OptimizedBinaryRegressorNode;
pub(crate) use optimized_runtime::OptimizedClassifierNode;
pub(crate) use optimized_runtime::OptimizedRuntime;
pub(crate) use optimized_runtime::PARALLEL_INFERENCE_CHUNK_ROWS;
pub(crate) use optimized_runtime::STANDARD_BATCH_INFERENCE_CHUNK_ROWS;
pub(crate) use optimized_runtime::resolve_inference_thread_count;
pub(crate) use runtime_planning::build_feature_index_map;
pub(crate) use runtime_planning::build_feature_projection;
pub(crate) use runtime_planning::model_used_feature_indices;
pub(crate) use runtime_planning::ordered_ensemble_indices;
pub(crate) use runtime_planning::remap_feature_index;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainAlgorithm {
    /// Train a single decision tree.
    Dt,
    /// Train a bootstrap-aggregated random forest.
    Rf,
    /// Train a second-order gradient-boosted ensemble.
    Gbm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Criterion {
    /// Let training choose the appropriate criterion for the requested setup.
    Auto,
    /// Gini impurity for classification.
    Gini,
    /// Entropy / information gain for classification.
    Entropy,
    /// Mean-based regression criterion.
    Mean,
    /// Median-based regression criterion.
    Median,
    /// Internal second-order criterion used by gradient boosting.
    SecondOrder,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Task {
    /// Predict a continuous numeric value.
    Regression,
    /// Predict one label from a finite set.
    Classification,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreeType {
    /// Multiway information-gain tree.
    Id3,
    /// C4.5-style multiway tree.
    C45,
    /// Standard binary threshold tree.
    Cart,
    /// CART-style tree with randomized candidate selection.
    Randomized,
    /// Symmetric tree where every level shares the same split.
    Oblivious,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaxFeatures {
    /// Task-aware default: `sqrt` for classification, `third` for regression.
    Auto,
    /// Use all features at each split.
    All,
    /// Use `floor(sqrt(feature_count))` features.
    Sqrt,
    /// Use roughly one third of the features.
    Third,
    /// Use exactly this many features, capped to the available count.
    Count(usize),
}

impl MaxFeatures {
    pub fn resolve(self, task: Task, feature_count: usize) -> usize {
        match self {
            MaxFeatures::Auto => match task {
                Task::Classification => MaxFeatures::Sqrt.resolve(task, feature_count),
                Task::Regression => MaxFeatures::Third.resolve(task, feature_count),
            },
            MaxFeatures::All => feature_count.max(1),
            MaxFeatures::Sqrt => ((feature_count as f64).sqrt().floor() as usize).max(1),
            MaxFeatures::Third => (feature_count / 3).max(1),
            MaxFeatures::Count(count) => count.min(feature_count).max(1),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputFeatureKind {
    /// Numeric features are compared through their binned representation.
    Numeric,
    /// Binary features stay boolean all the way through the pipeline.
    Binary,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct NumericBinBoundary {
    /// Bin identifier in the preprocessed feature space.
    pub bin: u16,
    /// Largest raw floating-point value that still belongs to this bin.
    pub upper_bound: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FeaturePreprocessing {
    /// Numeric features are represented by explicit bin boundaries.
    Numeric {
        bin_boundaries: Vec<NumericBinBoundary>,
    },
    /// Binary features do not require numeric bin boundaries.
    Binary,
}

/// Unified training configuration shared by the Rust and Python entry points.
///
/// The crate keeps one normalized config type so the binding layer only has to
/// perform input validation and type conversion; all semantic decisions happen
/// from this one structure downward.
#[derive(Debug, Clone, Copy)]
pub struct TrainConfig {
    /// High-level training family.
    pub algorithm: TrainAlgorithm,
    /// Regression or classification.
    pub task: Task,
    /// Tree learner used by the selected algorithm family.
    pub tree_type: TreeType,
    /// Split criterion. [`Criterion::Auto`] is resolved by the trainer.
    pub criterion: Criterion,
    /// Maximum tree depth.
    pub max_depth: Option<usize>,
    /// Smallest node size that is still allowed to split.
    pub min_samples_split: Option<usize>,
    /// Minimum child size after a split.
    pub min_samples_leaf: Option<usize>,
    /// Optional cap on training-side rayon threads.
    pub physical_cores: Option<usize>,
    /// Number of trees for ensemble algorithms.
    pub n_trees: Option<usize>,
    /// Feature subsampling strategy.
    pub max_features: MaxFeatures,
    /// Seed used for reproducible sampling and randomized splits.
    pub seed: Option<u64>,
    /// Whether random forests should compute out-of-bag metrics.
    pub compute_oob: bool,
    /// Gradient boosting shrinkage factor.
    pub learning_rate: Option<f64>,
    /// Whether gradient boosting should bootstrap rows before gradient sampling.
    pub bootstrap: bool,
    /// Fraction of largest-gradient rows always kept by GOSS sampling.
    pub top_gradient_fraction: Option<f64>,
    /// Fraction of the remaining rows randomly retained by GOSS sampling.
    pub other_gradient_fraction: Option<f64>,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Regression,
            tree_type: TreeType::Cart,
            criterion: Criterion::Auto,
            max_depth: None,
            min_samples_split: None,
            min_samples_leaf: None,
            physical_cores: None,
            n_trees: None,
            max_features: MaxFeatures::Auto,
            seed: None,
            compute_oob: false,
            learning_rate: None,
            bootstrap: false,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
        }
    }
}

/// Top-level semantic model enum.
///
/// This type stays close to the learned structure rather than the fastest
/// possible runtime layout. That is what makes it suitable for introspection,
/// serialization, and exact behavior parity across bindings.
#[derive(Debug, Clone)]
pub enum Model {
    DecisionTreeClassifier(DecisionTreeClassifier),
    DecisionTreeRegressor(DecisionTreeRegressor),
    RandomForest(RandomForest),
    GradientBoostedTrees(GradientBoostedTrees),
}

#[derive(Debug)]
pub enum TrainError {
    DecisionTree(DecisionTreeError),
    RegressionTree(RegressionTreeError),
    Boosting(BoostingError),
    InvalidPhysicalCoreCount {
        requested: usize,
        available: usize,
    },
    ThreadPoolBuildFailed(String),
    UnsupportedConfiguration {
        task: Task,
        tree_type: TreeType,
        criterion: Criterion,
    },
    InvalidMaxDepth(usize),
    InvalidMinSamplesSplit(usize),
    InvalidMinSamplesLeaf(usize),
    InvalidTreeCount(usize),
    InvalidMaxFeatures(usize),
}

impl Display for TrainError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            TrainError::DecisionTree(err) => err.fmt(f),
            TrainError::RegressionTree(err) => err.fmt(f),
            TrainError::Boosting(err) => err.fmt(f),
            TrainError::InvalidPhysicalCoreCount {
                requested,
                available,
            } => write!(
                f,
                "Requested {} physical cores, but the available physical core count is {}.",
                requested, available
            ),
            TrainError::ThreadPoolBuildFailed(message) => {
                write!(f, "Failed to build training thread pool: {}.", message)
            }
            TrainError::UnsupportedConfiguration {
                task,
                tree_type,
                criterion,
            } => write!(
                f,
                "Unsupported training configuration: task={:?}, tree_type={:?}, criterion={:?}.",
                task, tree_type, criterion
            ),
            TrainError::InvalidMaxDepth(value) => {
                write!(f, "max_depth must be at least 1. Received {}.", value)
            }
            TrainError::InvalidMinSamplesSplit(value) => {
                write!(
                    f,
                    "min_samples_split must be at least 1. Received {}.",
                    value
                )
            }
            TrainError::InvalidMinSamplesLeaf(value) => {
                write!(
                    f,
                    "min_samples_leaf must be at least 1. Received {}.",
                    value
                )
            }
            TrainError::InvalidTreeCount(n_trees) => {
                write!(
                    f,
                    "Random forest requires at least one tree. Received {}.",
                    n_trees
                )
            }
            TrainError::InvalidMaxFeatures(count) => {
                write!(
                    f,
                    "max_features must be at least 1 when provided as an integer. Received {}.",
                    count
                )
            }
        }
    }
}

impl Error for TrainError {}

#[derive(Debug, Clone, PartialEq)]
pub enum PredictError {
    ProbabilityPredictionRequiresClassification,
    RaggedRows {
        row: usize,
        expected: usize,
        actual: usize,
    },
    FeatureCountMismatch {
        expected: usize,
        actual: usize,
    },
    ColumnLengthMismatch {
        feature: String,
        expected: usize,
        actual: usize,
    },
    MissingFeature(String),
    UnexpectedFeature(String),
    InvalidBinaryValue {
        feature_index: usize,
        row_index: usize,
        value: f64,
    },
    NullValue {
        feature: String,
        row_index: usize,
    },
    UnsupportedFeatureType {
        feature: String,
        dtype: String,
    },
    Polars(String),
}

impl Display for PredictError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            PredictError::ProbabilityPredictionRequiresClassification => write!(
                f,
                "predict_proba is only available for classification models."
            ),
            PredictError::RaggedRows {
                row,
                expected,
                actual,
            } => write!(
                f,
                "Ragged inference row at index {}: expected {} columns, found {}.",
                row, expected, actual
            ),
            PredictError::FeatureCountMismatch { expected, actual } => write!(
                f,
                "Inference input has {} features, but the model expects {}.",
                actual, expected
            ),
            PredictError::ColumnLengthMismatch {
                feature,
                expected,
                actual,
            } => write!(
                f,
                "Feature '{}' has {} values, expected {}.",
                feature, actual, expected
            ),
            PredictError::MissingFeature(feature) => {
                write!(f, "Missing required feature '{}'.", feature)
            }
            PredictError::UnexpectedFeature(feature) => {
                write!(f, "Unexpected feature '{}'.", feature)
            }
            PredictError::InvalidBinaryValue {
                feature_index,
                row_index,
                value,
            } => write!(
                f,
                "Feature {} at row {} must be binary for inference, found {}.",
                feature_index, row_index, value
            ),
            PredictError::NullValue { feature, row_index } => write!(
                f,
                "Feature '{}' contains a null value at row {}.",
                feature, row_index
            ),
            PredictError::UnsupportedFeatureType { feature, dtype } => write!(
                f,
                "Feature '{}' has unsupported dtype '{}'.",
                feature, dtype
            ),
            PredictError::Polars(message) => write!(f, "Polars inference failed: {}.", message),
        }
    }
}

impl Error for PredictError {}

#[cfg(feature = "polars")]
impl From<polars::error::PolarsError> for PredictError {
    fn from(value: polars::error::PolarsError) -> Self {
        PredictError::Polars(value.to_string())
    }
}

#[derive(Debug)]
pub enum OptimizeError {
    InvalidPhysicalCoreCount { requested: usize, available: usize },
    ThreadPoolBuildFailed(String),
    UnsupportedModelType(&'static str),
}

impl Display for OptimizeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            OptimizeError::InvalidPhysicalCoreCount {
                requested,
                available,
            } => write!(
                f,
                "Requested {} physical cores, but the available physical core count is {}.",
                requested, available
            ),
            OptimizeError::ThreadPoolBuildFailed(message) => {
                write!(f, "Failed to build inference thread pool: {}.", message)
            }
            OptimizeError::UnsupportedModelType(model_type) => {
                write!(
                    f,
                    "Optimized inference is not supported for model type '{}'.",
                    model_type
                )
            }
        }
    }
}

impl Error for OptimizeError {}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Parallelism {
    thread_count: usize,
}

impl Parallelism {
    pub(crate) fn sequential() -> Self {
        Self { thread_count: 1 }
    }

    #[cfg(test)]
    pub(crate) fn with_threads(thread_count: usize) -> Self {
        Self {
            thread_count: thread_count.max(1),
        }
    }

    pub(crate) fn enabled(self) -> bool {
        self.thread_count > 1
    }
}

pub(crate) fn capture_feature_preprocessing(table: &dyn TableAccess) -> Vec<FeaturePreprocessing> {
    (0..table.n_features())
        .map(|feature_index| {
            if table.is_binary_feature(feature_index) {
                FeaturePreprocessing::Binary
            } else {
                let values = (0..table.n_rows())
                    .map(|row_index| table.feature_value(feature_index, row_index))
                    .collect::<Vec<_>>();
                FeaturePreprocessing::Numeric {
                    bin_boundaries: numeric_bin_boundaries(
                        &values,
                        NumericBins::Fixed(table.numeric_bin_cap()),
                    )
                    .into_iter()
                    .map(|(bin, upper_bound)| NumericBinBoundary { bin, upper_bound })
                    .collect(),
                }
            }
        })
        .collect()
}

impl OptimizedRuntime {
    fn supports_batch_matrix(&self) -> bool {
        matches!(
            self,
            OptimizedRuntime::BinaryClassifier { .. }
                | OptimizedRuntime::BinaryRegressor { .. }
                | OptimizedRuntime::ObliviousClassifier { .. }
                | OptimizedRuntime::ObliviousRegressor { .. }
                | OptimizedRuntime::ForestClassifier { .. }
                | OptimizedRuntime::ForestRegressor { .. }
                | OptimizedRuntime::BoostedBinaryClassifier { .. }
                | OptimizedRuntime::BoostedRegressor { .. }
        )
    }

    fn should_use_batch_matrix(&self, n_rows: usize) -> bool {
        n_rows > 1 && self.supports_batch_matrix()
    }

    fn from_model(model: &Model, feature_index_map: &[usize]) -> Self {
        match model {
            Model::DecisionTreeClassifier(classifier) => {
                Self::from_classifier(classifier, feature_index_map)
            }
            Model::DecisionTreeRegressor(regressor) => {
                Self::from_regressor(regressor, feature_index_map)
            }
            Model::RandomForest(forest) => match forest.task() {
                Task::Classification => {
                    let tree_order = ordered_ensemble_indices(forest.trees());
                    Self::ForestClassifier {
                        trees: tree_order
                            .into_iter()
                            .map(|tree_index| {
                                Self::from_model(&forest.trees()[tree_index], feature_index_map)
                            })
                            .collect(),
                        class_labels: forest
                            .class_labels()
                            .expect("classification forest stores class labels"),
                    }
                }
                Task::Regression => {
                    let tree_order = ordered_ensemble_indices(forest.trees());
                    Self::ForestRegressor {
                        trees: tree_order
                            .into_iter()
                            .map(|tree_index| {
                                Self::from_model(&forest.trees()[tree_index], feature_index_map)
                            })
                            .collect(),
                    }
                }
            },
            Model::GradientBoostedTrees(model) => match model.task() {
                Task::Classification => {
                    let tree_order = ordered_ensemble_indices(model.trees());
                    Self::BoostedBinaryClassifier {
                        trees: tree_order
                            .iter()
                            .map(|tree_index| {
                                Self::from_model(&model.trees()[*tree_index], feature_index_map)
                            })
                            .collect(),
                        tree_weights: tree_order
                            .iter()
                            .map(|tree_index| model.tree_weights()[*tree_index])
                            .collect(),
                        base_score: model.base_score(),
                        class_labels: model
                            .class_labels()
                            .expect("classification boosting stores class labels"),
                    }
                }
                Task::Regression => {
                    let tree_order = ordered_ensemble_indices(model.trees());
                    Self::BoostedRegressor {
                        trees: tree_order
                            .iter()
                            .map(|tree_index| {
                                Self::from_model(&model.trees()[*tree_index], feature_index_map)
                            })
                            .collect(),
                        tree_weights: tree_order
                            .iter()
                            .map(|tree_index| model.tree_weights()[*tree_index])
                            .collect(),
                        base_score: model.base_score(),
                    }
                }
            },
        }
    }

    fn from_classifier(classifier: &DecisionTreeClassifier, feature_index_map: &[usize]) -> Self {
        match classifier.structure() {
            tree::classifier::TreeStructure::Standard { nodes, root } => {
                if classifier_nodes_are_binary_only(nodes) {
                    return Self::BinaryClassifier {
                        nodes: build_binary_classifier_layout(
                            nodes,
                            *root,
                            classifier.class_labels(),
                            feature_index_map,
                        ),
                        class_labels: classifier.class_labels().to_vec(),
                    };
                }

                let optimized_nodes = nodes
                    .iter()
                    .map(|node| match node {
                        tree::classifier::TreeNode::Leaf { class_counts, .. } => {
                            OptimizedClassifierNode::Leaf(normalized_probabilities_from_counts(
                                class_counts,
                            ))
                        }
                        tree::classifier::TreeNode::BinarySplit {
                            feature_index,
                            threshold_bin,
                            left_child,
                            right_child,
                            ..
                        } => OptimizedClassifierNode::Binary {
                            feature_index: remap_feature_index(*feature_index, feature_index_map),
                            threshold_bin: *threshold_bin,
                            children: [*left_child, *right_child],
                        },
                        tree::classifier::TreeNode::MultiwaySplit {
                            feature_index,
                            class_counts,
                            branches,
                            ..
                        } => {
                            let max_bin_index = branches
                                .iter()
                                .map(|(bin, _)| usize::from(*bin))
                                .max()
                                .unwrap_or(0);
                            let mut child_lookup = vec![usize::MAX; max_bin_index + 1];
                            for (bin, child_index) in branches {
                                child_lookup[usize::from(*bin)] = *child_index;
                            }
                            OptimizedClassifierNode::Multiway {
                                feature_index: remap_feature_index(
                                    *feature_index,
                                    feature_index_map,
                                ),
                                child_lookup,
                                max_bin_index,
                                fallback_probabilities: normalized_probabilities_from_counts(
                                    class_counts,
                                ),
                            }
                        }
                    })
                    .collect();

                Self::StandardClassifier {
                    nodes: optimized_nodes,
                    root: *root,
                    class_labels: classifier.class_labels().to_vec(),
                }
            }
            tree::classifier::TreeStructure::Oblivious {
                splits,
                leaf_class_counts,
                ..
            } => Self::ObliviousClassifier {
                feature_indices: splits
                    .iter()
                    .map(|split| remap_feature_index(split.feature_index, feature_index_map))
                    .collect(),
                threshold_bins: splits.iter().map(|split| split.threshold_bin).collect(),
                leaf_values: leaf_class_counts
                    .iter()
                    .map(|class_counts| normalized_probabilities_from_counts(class_counts))
                    .collect(),
                class_labels: classifier.class_labels().to_vec(),
            },
        }
    }

    fn from_regressor(regressor: &DecisionTreeRegressor, feature_index_map: &[usize]) -> Self {
        match regressor.structure() {
            tree::regressor::RegressionTreeStructure::Standard { nodes, root } => {
                Self::BinaryRegressor {
                    nodes: build_binary_regressor_layout(nodes, *root, feature_index_map),
                }
            }
            tree::regressor::RegressionTreeStructure::Oblivious {
                splits,
                leaf_values,
                ..
            } => Self::ObliviousRegressor {
                feature_indices: splits
                    .iter()
                    .map(|split| remap_feature_index(split.feature_index, feature_index_map))
                    .collect(),
                threshold_bins: splits.iter().map(|split| split.threshold_bin).collect(),
                leaf_values: leaf_values.clone(),
            },
        }
    }

    #[inline(always)]
    fn predict_table_row(&self, table: &dyn TableAccess, row_index: usize) -> f64 {
        match self {
            OptimizedRuntime::BinaryClassifier { .. }
            | OptimizedRuntime::StandardClassifier { .. }
            | OptimizedRuntime::ObliviousClassifier { .. }
            | OptimizedRuntime::ForestClassifier { .. }
            | OptimizedRuntime::BoostedBinaryClassifier { .. } => {
                let probabilities = self
                    .predict_proba_table_row(table, row_index)
                    .expect("classifier runtime supports probability prediction");
                class_label_from_probabilities(&probabilities, self.class_labels())
            }
            OptimizedRuntime::BinaryRegressor { nodes } => {
                predict_binary_regressor_row(nodes, |feature_index| {
                    table.binned_value(feature_index, row_index)
                })
            }
            OptimizedRuntime::ObliviousRegressor {
                feature_indices,
                threshold_bins,
                leaf_values,
            } => predict_oblivious_row(
                feature_indices,
                threshold_bins,
                leaf_values,
                |feature_index| table.binned_value(feature_index, row_index),
            ),
            OptimizedRuntime::ForestRegressor { trees } => {
                trees
                    .iter()
                    .map(|tree| tree.predict_table_row(table, row_index))
                    .sum::<f64>()
                    / trees.len() as f64
            }
            OptimizedRuntime::BoostedRegressor {
                trees,
                tree_weights,
                base_score,
            } => {
                *base_score
                    + trees
                        .iter()
                        .zip(tree_weights.iter().copied())
                        .map(|(tree, weight)| weight * tree.predict_table_row(table, row_index))
                        .sum::<f64>()
            }
        }
    }

    #[inline(always)]
    fn predict_proba_table_row(
        &self,
        table: &dyn TableAccess,
        row_index: usize,
    ) -> Result<Vec<f64>, PredictError> {
        match self {
            OptimizedRuntime::BinaryClassifier { nodes, .. } => Ok(
                predict_binary_classifier_probabilities_row(nodes, |feature_index| {
                    table.binned_value(feature_index, row_index)
                })
                .to_vec(),
            ),
            OptimizedRuntime::StandardClassifier { nodes, root, .. } => Ok(
                predict_standard_classifier_probabilities_row(nodes, *root, |feature_index| {
                    table.binned_value(feature_index, row_index)
                })
                .to_vec(),
            ),
            OptimizedRuntime::ObliviousClassifier {
                feature_indices,
                threshold_bins,
                leaf_values,
                ..
            } => Ok(predict_oblivious_probabilities_row(
                feature_indices,
                threshold_bins,
                leaf_values,
                |feature_index| table.binned_value(feature_index, row_index),
            )
            .to_vec()),
            OptimizedRuntime::ForestClassifier { trees, .. } => {
                let mut totals = trees[0].predict_proba_table_row(table, row_index)?;
                for tree in &trees[1..] {
                    let row = tree.predict_proba_table_row(table, row_index)?;
                    for (total, value) in totals.iter_mut().zip(row) {
                        *total += value;
                    }
                }
                let tree_count = trees.len() as f64;
                for value in &mut totals {
                    *value /= tree_count;
                }
                Ok(totals)
            }
            OptimizedRuntime::BoostedBinaryClassifier {
                trees,
                tree_weights,
                base_score,
                ..
            } => {
                let raw_score = *base_score
                    + trees
                        .iter()
                        .zip(tree_weights.iter().copied())
                        .map(|(tree, weight)| weight * tree.predict_table_row(table, row_index))
                        .sum::<f64>();
                let positive = sigmoid(raw_score);
                Ok(vec![1.0 - positive, positive])
            }
            OptimizedRuntime::BinaryRegressor { .. }
            | OptimizedRuntime::ObliviousRegressor { .. }
            | OptimizedRuntime::ForestRegressor { .. }
            | OptimizedRuntime::BoostedRegressor { .. } => {
                Err(PredictError::ProbabilityPredictionRequiresClassification)
            }
        }
    }

    fn predict_proba_table(
        &self,
        table: &dyn TableAccess,
        executor: &InferenceExecutor,
    ) -> Result<Vec<Vec<f64>>, PredictError> {
        match self {
            OptimizedRuntime::BinaryClassifier { .. }
            | OptimizedRuntime::StandardClassifier { .. }
            | OptimizedRuntime::ObliviousClassifier { .. }
            | OptimizedRuntime::ForestClassifier { .. }
            | OptimizedRuntime::BoostedBinaryClassifier { .. } => {
                if self.should_use_batch_matrix(table.n_rows()) {
                    let matrix = ColumnMajorBinnedMatrix::from_table_access(table);
                    self.predict_proba_column_major_matrix(&matrix, executor)
                } else {
                    (0..table.n_rows())
                        .map(|row_index| self.predict_proba_table_row(table, row_index))
                        .collect()
                }
            }
            OptimizedRuntime::BinaryRegressor { .. }
            | OptimizedRuntime::ObliviousRegressor { .. }
            | OptimizedRuntime::ForestRegressor { .. }
            | OptimizedRuntime::BoostedRegressor { .. } => {
                Err(PredictError::ProbabilityPredictionRequiresClassification)
            }
        }
    }

    fn predict_column_major_matrix(
        &self,
        matrix: &ColumnMajorBinnedMatrix,
        executor: &InferenceExecutor,
    ) -> Vec<f64> {
        match self {
            OptimizedRuntime::BinaryClassifier { .. }
            | OptimizedRuntime::StandardClassifier { .. }
            | OptimizedRuntime::ObliviousClassifier { .. }
            | OptimizedRuntime::ForestClassifier { .. }
            | OptimizedRuntime::BoostedBinaryClassifier { .. } => self
                .predict_proba_column_major_matrix(matrix, executor)
                .expect("classifier runtime supports probability prediction")
                .into_iter()
                .map(|row| class_label_from_probabilities(&row, self.class_labels()))
                .collect(),
            OptimizedRuntime::BinaryRegressor { nodes } => {
                predict_binary_regressor_column_major_matrix(nodes, matrix, executor)
            }
            OptimizedRuntime::ObliviousRegressor {
                feature_indices,
                threshold_bins,
                leaf_values,
            } => predict_oblivious_column_major_matrix(
                feature_indices,
                threshold_bins,
                leaf_values,
                matrix,
                executor,
            ),
            OptimizedRuntime::ForestRegressor { trees } => {
                let mut totals = trees[0].predict_column_major_matrix(matrix, executor);
                for tree in &trees[1..] {
                    let values = tree.predict_column_major_matrix(matrix, executor);
                    for (total, value) in totals.iter_mut().zip(values) {
                        *total += value;
                    }
                }
                let tree_count = trees.len() as f64;
                for total in &mut totals {
                    *total /= tree_count;
                }
                totals
            }
            OptimizedRuntime::BoostedRegressor {
                trees,
                tree_weights,
                base_score,
            } => {
                let mut totals = vec![*base_score; matrix.n_rows];
                for (tree, weight) in trees.iter().zip(tree_weights.iter().copied()) {
                    let values = tree.predict_column_major_matrix(matrix, executor);
                    for (total, value) in totals.iter_mut().zip(values) {
                        *total += weight * value;
                    }
                }
                totals
            }
        }
    }

    fn predict_proba_column_major_matrix(
        &self,
        matrix: &ColumnMajorBinnedMatrix,
        executor: &InferenceExecutor,
    ) -> Result<Vec<Vec<f64>>, PredictError> {
        match self {
            OptimizedRuntime::BinaryClassifier { nodes, .. } => {
                Ok(predict_binary_classifier_probabilities_column_major_matrix(
                    nodes, matrix, executor,
                ))
            }
            OptimizedRuntime::StandardClassifier { .. } => Ok((0..matrix.n_rows)
                .map(|row_index| {
                    self.predict_proba_binned_row_from_columns(matrix, row_index)
                        .expect("classifier runtime supports probability prediction")
                })
                .collect()),
            OptimizedRuntime::ObliviousClassifier {
                feature_indices,
                threshold_bins,
                leaf_values,
                ..
            } => Ok(predict_oblivious_probabilities_column_major_matrix(
                feature_indices,
                threshold_bins,
                leaf_values,
                matrix,
                executor,
            )),
            OptimizedRuntime::ForestClassifier { trees, .. } => {
                let mut totals = trees[0].predict_proba_column_major_matrix(matrix, executor)?;
                for tree in &trees[1..] {
                    let rows = tree.predict_proba_column_major_matrix(matrix, executor)?;
                    for (row_totals, row_values) in totals.iter_mut().zip(rows) {
                        for (total, value) in row_totals.iter_mut().zip(row_values) {
                            *total += value;
                        }
                    }
                }
                let tree_count = trees.len() as f64;
                for row in &mut totals {
                    for value in row {
                        *value /= tree_count;
                    }
                }
                Ok(totals)
            }
            OptimizedRuntime::BoostedBinaryClassifier {
                trees,
                tree_weights,
                base_score,
                ..
            } => {
                let mut raw_scores = vec![*base_score; matrix.n_rows];
                for (tree, weight) in trees.iter().zip(tree_weights.iter().copied()) {
                    let values = tree.predict_column_major_matrix(matrix, executor);
                    for (raw_score, value) in raw_scores.iter_mut().zip(values) {
                        *raw_score += weight * value;
                    }
                }
                Ok(raw_scores
                    .into_iter()
                    .map(|raw_score| {
                        let positive = sigmoid(raw_score);
                        vec![1.0 - positive, positive]
                    })
                    .collect())
            }
            OptimizedRuntime::BinaryRegressor { .. }
            | OptimizedRuntime::ObliviousRegressor { .. }
            | OptimizedRuntime::ForestRegressor { .. }
            | OptimizedRuntime::BoostedRegressor { .. } => {
                Err(PredictError::ProbabilityPredictionRequiresClassification)
            }
        }
    }

    fn class_labels(&self) -> &[f64] {
        match self {
            OptimizedRuntime::BinaryClassifier { class_labels, .. }
            | OptimizedRuntime::StandardClassifier { class_labels, .. }
            | OptimizedRuntime::ObliviousClassifier { class_labels, .. }
            | OptimizedRuntime::ForestClassifier { class_labels, .. }
            | OptimizedRuntime::BoostedBinaryClassifier { class_labels, .. } => class_labels,
            _ => &[],
        }
    }

    #[inline(always)]
    fn predict_binned_row_from_columns(
        &self,
        matrix: &ColumnMajorBinnedMatrix,
        row_index: usize,
    ) -> f64 {
        match self {
            OptimizedRuntime::BinaryRegressor { nodes } => {
                predict_binary_regressor_row(nodes, |feature_index| {
                    matrix.column(feature_index).value_at(row_index)
                })
            }
            OptimizedRuntime::ObliviousRegressor {
                feature_indices,
                threshold_bins,
                leaf_values,
            } => predict_oblivious_row(
                feature_indices,
                threshold_bins,
                leaf_values,
                |feature_index| matrix.column(feature_index).value_at(row_index),
            ),
            OptimizedRuntime::BoostedRegressor {
                trees,
                tree_weights,
                base_score,
            } => {
                *base_score
                    + trees
                        .iter()
                        .zip(tree_weights.iter().copied())
                        .map(|(tree, weight)| {
                            weight * tree.predict_binned_row_from_columns(matrix, row_index)
                        })
                        .sum::<f64>()
            }
            _ => self.predict_column_major_matrix(
                matrix,
                &InferenceExecutor::new(1).expect("inference executor"),
            )[row_index],
        }
    }

    #[inline(always)]
    fn predict_proba_binned_row_from_columns(
        &self,
        matrix: &ColumnMajorBinnedMatrix,
        row_index: usize,
    ) -> Result<Vec<f64>, PredictError> {
        match self {
            OptimizedRuntime::BinaryClassifier { nodes, .. } => Ok(
                predict_binary_classifier_probabilities_row(nodes, |feature_index| {
                    matrix.column(feature_index).value_at(row_index)
                })
                .to_vec(),
            ),
            OptimizedRuntime::StandardClassifier { nodes, root, .. } => Ok(
                predict_standard_classifier_probabilities_row(nodes, *root, |feature_index| {
                    matrix.column(feature_index).value_at(row_index)
                })
                .to_vec(),
            ),
            OptimizedRuntime::ObliviousClassifier {
                feature_indices,
                threshold_bins,
                leaf_values,
                ..
            } => Ok(predict_oblivious_probabilities_row(
                feature_indices,
                threshold_bins,
                leaf_values,
                |feature_index| matrix.column(feature_index).value_at(row_index),
            )
            .to_vec()),
            OptimizedRuntime::ForestClassifier { trees, .. } => {
                let mut totals =
                    trees[0].predict_proba_binned_row_from_columns(matrix, row_index)?;
                for tree in &trees[1..] {
                    let row = tree.predict_proba_binned_row_from_columns(matrix, row_index)?;
                    for (total, value) in totals.iter_mut().zip(row) {
                        *total += value;
                    }
                }
                let tree_count = trees.len() as f64;
                for value in &mut totals {
                    *value /= tree_count;
                }
                Ok(totals)
            }
            OptimizedRuntime::BoostedBinaryClassifier {
                trees,
                tree_weights,
                base_score,
                ..
            } => {
                let raw_score = *base_score
                    + trees
                        .iter()
                        .zip(tree_weights.iter().copied())
                        .map(|(tree, weight)| {
                            weight * tree.predict_binned_row_from_columns(matrix, row_index)
                        })
                        .sum::<f64>();
                let positive = sigmoid(raw_score);
                Ok(vec![1.0 - positive, positive])
            }
            OptimizedRuntime::BinaryRegressor { .. }
            | OptimizedRuntime::ObliviousRegressor { .. }
            | OptimizedRuntime::ForestRegressor { .. }
            | OptimizedRuntime::BoostedRegressor { .. } => {
                Err(PredictError::ProbabilityPredictionRequiresClassification)
            }
        }
    }
}

#[inline(always)]
fn predict_standard_classifier_probabilities_row<F>(
    nodes: &[OptimizedClassifierNode],
    root: usize,
    bin_at: F,
) -> &[f64]
where
    F: Fn(usize) -> u16,
{
    let mut node_index = root;
    loop {
        match &nodes[node_index] {
            OptimizedClassifierNode::Leaf(value) => return value,
            OptimizedClassifierNode::Binary {
                feature_index,
                threshold_bin,
                children,
            } => {
                let go_right = usize::from(bin_at(*feature_index) > *threshold_bin);
                node_index = children[go_right];
            }
            OptimizedClassifierNode::Multiway {
                feature_index,
                child_lookup,
                max_bin_index,
                fallback_probabilities,
            } => {
                let bin = usize::from(bin_at(*feature_index));
                if bin > *max_bin_index {
                    return fallback_probabilities;
                }
                let child_index = child_lookup[bin];
                if child_index == usize::MAX {
                    return fallback_probabilities;
                }
                node_index = child_index;
            }
        }
    }
}

#[inline(always)]
fn predict_binary_classifier_probabilities_row<F>(
    nodes: &[OptimizedBinaryClassifierNode],
    bin_at: F,
) -> &[f64]
where
    F: Fn(usize) -> u16,
{
    let mut node_index = 0usize;
    loop {
        match &nodes[node_index] {
            OptimizedBinaryClassifierNode::Leaf(value) => return value,
            OptimizedBinaryClassifierNode::Branch {
                feature_index,
                threshold_bin,
                jump_index,
                jump_if_greater,
            } => {
                let go_right = bin_at(*feature_index) > *threshold_bin;
                node_index = if go_right == *jump_if_greater {
                    *jump_index
                } else {
                    node_index + 1
                };
            }
        }
    }
}

#[inline(always)]
fn predict_binary_regressor_row<F>(nodes: &[OptimizedBinaryRegressorNode], bin_at: F) -> f64
where
    F: Fn(usize) -> u16,
{
    let mut node_index = 0usize;
    loop {
        match &nodes[node_index] {
            OptimizedBinaryRegressorNode::Leaf(value) => return *value,
            OptimizedBinaryRegressorNode::Branch {
                feature_index,
                threshold_bin,
                jump_index,
                jump_if_greater,
            } => {
                let go_right = bin_at(*feature_index) > *threshold_bin;
                node_index = if go_right == *jump_if_greater {
                    *jump_index
                } else {
                    node_index + 1
                };
            }
        }
    }
}

fn predict_binary_classifier_probabilities_column_major_matrix(
    nodes: &[OptimizedBinaryClassifierNode],
    matrix: &ColumnMajorBinnedMatrix,
    _executor: &InferenceExecutor,
) -> Vec<Vec<f64>> {
    (0..matrix.n_rows)
        .map(|row_index| {
            predict_binary_classifier_probabilities_row(nodes, |feature_index| {
                matrix.column(feature_index).value_at(row_index)
            })
            .to_vec()
        })
        .collect()
}

fn predict_binary_regressor_column_major_matrix(
    nodes: &[OptimizedBinaryRegressorNode],
    matrix: &ColumnMajorBinnedMatrix,
    executor: &InferenceExecutor,
) -> Vec<f64> {
    let mut outputs = vec![0.0; matrix.n_rows];
    executor.fill_chunks(
        &mut outputs,
        STANDARD_BATCH_INFERENCE_CHUNK_ROWS,
        |start_row, chunk| predict_binary_regressor_chunk(nodes, matrix, start_row, chunk),
    );
    outputs
}

fn predict_binary_regressor_chunk(
    nodes: &[OptimizedBinaryRegressorNode],
    matrix: &ColumnMajorBinnedMatrix,
    start_row: usize,
    output: &mut [f64],
) {
    let mut row_indices: Vec<usize> = (0..output.len()).collect();
    let mut stack = vec![(0usize, 0usize, output.len())];

    while let Some((node_index, start, end)) = stack.pop() {
        match &nodes[node_index] {
            OptimizedBinaryRegressorNode::Leaf(value) => {
                for position in start..end {
                    output[row_indices[position]] = *value;
                }
            }
            OptimizedBinaryRegressorNode::Branch {
                feature_index,
                threshold_bin,
                jump_index,
                jump_if_greater,
            } => {
                let fallthrough_index = node_index + 1;
                if *jump_index == fallthrough_index {
                    stack.push((fallthrough_index, start, end));
                    continue;
                }

                let column = matrix.column(*feature_index);
                let mut partition = start;
                let mut jump_start = end;
                match column {
                    CompactBinnedColumn::U8(values) if *threshold_bin <= u16::from(u8::MAX) => {
                        let threshold = *threshold_bin as u8;
                        while partition < jump_start {
                            let row_offset = row_indices[partition];
                            let go_right = values[start_row + row_offset] > threshold;
                            let goes_jump = go_right == *jump_if_greater;
                            if goes_jump {
                                jump_start -= 1;
                                row_indices.swap(partition, jump_start);
                            } else {
                                partition += 1;
                            }
                        }
                    }
                    _ => {
                        while partition < jump_start {
                            let row_offset = row_indices[partition];
                            let go_right = column.value_at(start_row + row_offset) > *threshold_bin;
                            let goes_jump = go_right == *jump_if_greater;
                            if goes_jump {
                                jump_start -= 1;
                                row_indices.swap(partition, jump_start);
                            } else {
                                partition += 1;
                            }
                        }
                    }
                }

                if jump_start < end {
                    stack.push((*jump_index, jump_start, end));
                }
                if start < jump_start {
                    stack.push((fallthrough_index, start, jump_start));
                }
            }
        }
    }
}

#[inline(always)]
fn predict_oblivious_row<F>(
    feature_indices: &[usize],
    threshold_bins: &[u16],
    leaf_values: &[f64],
    bin_at: F,
) -> f64
where
    F: Fn(usize) -> u16,
{
    let mut leaf_index = 0usize;
    for (&feature_index, &threshold_bin) in feature_indices.iter().zip(threshold_bins) {
        let go_right = usize::from(bin_at(feature_index) > threshold_bin);
        leaf_index = (leaf_index << 1) | go_right;
    }
    leaf_values[leaf_index]
}

#[inline(always)]
fn predict_oblivious_probabilities_row<'a, F>(
    feature_indices: &[usize],
    threshold_bins: &[u16],
    leaf_values: &'a [Vec<f64>],
    bin_at: F,
) -> &'a [f64]
where
    F: Fn(usize) -> u16,
{
    let mut leaf_index = 0usize;
    for (&feature_index, &threshold_bin) in feature_indices.iter().zip(threshold_bins) {
        let go_right = usize::from(bin_at(feature_index) > threshold_bin);
        leaf_index = (leaf_index << 1) | go_right;
    }
    leaf_values[leaf_index].as_slice()
}

fn normalized_probabilities_from_counts(class_counts: &[usize]) -> Vec<f64> {
    let total = class_counts.iter().sum::<usize>();
    if total == 0 {
        return vec![0.0; class_counts.len()];
    }

    class_counts
        .iter()
        .map(|count| *count as f64 / total as f64)
        .collect()
}

fn class_label_from_probabilities(probabilities: &[f64], class_labels: &[f64]) -> f64 {
    let best_index = probabilities
        .iter()
        .copied()
        .enumerate()
        .max_by(|(left_index, left), (right_index, right)| {
            left.total_cmp(right)
                .then_with(|| right_index.cmp(left_index))
        })
        .map(|(index, _)| index)
        .expect("classification probability row is non-empty");
    class_labels[best_index]
}

#[inline(always)]
fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        let exp = (-value).exp();
        1.0 / (1.0 + exp)
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn classifier_nodes_are_binary_only(nodes: &[tree::classifier::TreeNode]) -> bool {
    nodes.iter().all(|node| {
        matches!(
            node,
            tree::classifier::TreeNode::Leaf { .. }
                | tree::classifier::TreeNode::BinarySplit { .. }
        )
    })
}

fn classifier_node_sample_count(nodes: &[tree::classifier::TreeNode], node_index: usize) -> usize {
    match &nodes[node_index] {
        tree::classifier::TreeNode::Leaf { sample_count, .. }
        | tree::classifier::TreeNode::BinarySplit { sample_count, .. }
        | tree::classifier::TreeNode::MultiwaySplit { sample_count, .. } => *sample_count,
    }
}

fn build_binary_classifier_layout(
    nodes: &[tree::classifier::TreeNode],
    root: usize,
    _class_labels: &[f64],
    feature_index_map: &[usize],
) -> Vec<OptimizedBinaryClassifierNode> {
    let mut layout = Vec::with_capacity(nodes.len());
    append_binary_classifier_node(nodes, root, &mut layout, feature_index_map);
    layout
}

fn append_binary_classifier_node(
    nodes: &[tree::classifier::TreeNode],
    node_index: usize,
    layout: &mut Vec<OptimizedBinaryClassifierNode>,
    feature_index_map: &[usize],
) -> usize {
    let current_index = layout.len();
    layout.push(OptimizedBinaryClassifierNode::Leaf(Vec::new()));

    match &nodes[node_index] {
        tree::classifier::TreeNode::Leaf { class_counts, .. } => {
            layout[current_index] = OptimizedBinaryClassifierNode::Leaf(
                normalized_probabilities_from_counts(class_counts),
            );
        }
        tree::classifier::TreeNode::BinarySplit {
            feature_index,
            threshold_bin,
            left_child,
            right_child,
            ..
        } => {
            let (fallthrough_child, jump_child, jump_if_greater) = if left_child == right_child {
                (*left_child, *left_child, true)
            } else {
                let left_count = classifier_node_sample_count(nodes, *left_child);
                let right_count = classifier_node_sample_count(nodes, *right_child);
                if left_count >= right_count {
                    (*left_child, *right_child, true)
                } else {
                    (*right_child, *left_child, false)
                }
            };

            let fallthrough_index =
                append_binary_classifier_node(nodes, fallthrough_child, layout, feature_index_map);
            debug_assert_eq!(fallthrough_index, current_index + 1);
            let jump_index = if jump_child == fallthrough_child {
                fallthrough_index
            } else {
                append_binary_classifier_node(nodes, jump_child, layout, feature_index_map)
            };

            layout[current_index] = OptimizedBinaryClassifierNode::Branch {
                feature_index: remap_feature_index(*feature_index, feature_index_map),
                threshold_bin: *threshold_bin,
                jump_index,
                jump_if_greater,
            };
        }
        tree::classifier::TreeNode::MultiwaySplit { .. } => {
            unreachable!("multiway nodes are filtered out before binary layout construction");
        }
    }

    current_index
}

fn regressor_node_sample_count(
    nodes: &[tree::regressor::RegressionNode],
    node_index: usize,
) -> usize {
    match &nodes[node_index] {
        tree::regressor::RegressionNode::Leaf { sample_count, .. }
        | tree::regressor::RegressionNode::BinarySplit { sample_count, .. } => *sample_count,
    }
}

fn build_binary_regressor_layout(
    nodes: &[tree::regressor::RegressionNode],
    root: usize,
    feature_index_map: &[usize],
) -> Vec<OptimizedBinaryRegressorNode> {
    let mut layout = Vec::with_capacity(nodes.len());
    append_binary_regressor_node(nodes, root, &mut layout, feature_index_map);
    layout
}

fn append_binary_regressor_node(
    nodes: &[tree::regressor::RegressionNode],
    node_index: usize,
    layout: &mut Vec<OptimizedBinaryRegressorNode>,
    feature_index_map: &[usize],
) -> usize {
    let current_index = layout.len();
    layout.push(OptimizedBinaryRegressorNode::Leaf(0.0));

    match &nodes[node_index] {
        tree::regressor::RegressionNode::Leaf { value, .. } => {
            layout[current_index] = OptimizedBinaryRegressorNode::Leaf(*value);
        }
        tree::regressor::RegressionNode::BinarySplit {
            feature_index,
            threshold_bin,
            left_child,
            right_child,
            ..
        } => {
            let (fallthrough_child, jump_child, jump_if_greater) = if left_child == right_child {
                (*left_child, *left_child, true)
            } else {
                let left_count = regressor_node_sample_count(nodes, *left_child);
                let right_count = regressor_node_sample_count(nodes, *right_child);
                if left_count >= right_count {
                    (*left_child, *right_child, true)
                } else {
                    (*right_child, *left_child, false)
                }
            };

            let fallthrough_index =
                append_binary_regressor_node(nodes, fallthrough_child, layout, feature_index_map);
            debug_assert_eq!(fallthrough_index, current_index + 1);
            let jump_index = if jump_child == fallthrough_child {
                fallthrough_index
            } else {
                append_binary_regressor_node(nodes, jump_child, layout, feature_index_map)
            };

            layout[current_index] = OptimizedBinaryRegressorNode::Branch {
                feature_index: remap_feature_index(*feature_index, feature_index_map),
                threshold_bin: *threshold_bin,
                jump_index,
                jump_if_greater,
            };
        }
    }

    current_index
}

fn predict_oblivious_column_major_matrix(
    feature_indices: &[usize],
    threshold_bins: &[u16],
    leaf_values: &[f64],
    matrix: &ColumnMajorBinnedMatrix,
    executor: &InferenceExecutor,
) -> Vec<f64> {
    let mut outputs = vec![0.0; matrix.n_rows];
    executor.fill_chunks(
        &mut outputs,
        PARALLEL_INFERENCE_CHUNK_ROWS,
        |start_row, chunk| {
            predict_oblivious_chunk(
                feature_indices,
                threshold_bins,
                leaf_values,
                matrix,
                start_row,
                chunk,
            )
        },
    );
    outputs
}

fn predict_oblivious_probabilities_column_major_matrix(
    feature_indices: &[usize],
    threshold_bins: &[u16],
    leaf_values: &[Vec<f64>],
    matrix: &ColumnMajorBinnedMatrix,
    _executor: &InferenceExecutor,
) -> Vec<Vec<f64>> {
    (0..matrix.n_rows)
        .map(|row_index| {
            predict_oblivious_probabilities_row(
                feature_indices,
                threshold_bins,
                leaf_values,
                |feature_index| matrix.column(feature_index).value_at(row_index),
            )
            .to_vec()
        })
        .collect()
}

fn predict_oblivious_chunk(
    feature_indices: &[usize],
    threshold_bins: &[u16],
    leaf_values: &[f64],
    matrix: &ColumnMajorBinnedMatrix,
    start_row: usize,
    output: &mut [f64],
) {
    let processed = simd_predict_oblivious_chunk(
        feature_indices,
        threshold_bins,
        leaf_values,
        matrix,
        start_row,
        output,
    );

    for (offset, out) in output.iter_mut().enumerate().skip(processed) {
        let row_index = start_row + offset;
        *out = predict_oblivious_row(
            feature_indices,
            threshold_bins,
            leaf_values,
            |feature_index| matrix.column(feature_index).value_at(row_index),
        );
    }
}

fn simd_predict_oblivious_chunk(
    feature_indices: &[usize],
    threshold_bins: &[u16],
    leaf_values: &[f64],
    matrix: &ColumnMajorBinnedMatrix,
    start_row: usize,
    output: &mut [f64],
) -> usize {
    let mut processed = 0usize;
    let ones = u32x8::splat(1);

    while processed + OBLIVIOUS_SIMD_LANES <= output.len() {
        let base_row = start_row + processed;
        let mut leaf_indices = u32x8::splat(0);

        for (&feature_index, &threshold_bin) in feature_indices.iter().zip(threshold_bins) {
            let column = matrix.column(feature_index);
            let bins = if let Some(lanes) = column.slice_u8(base_row, OBLIVIOUS_SIMD_LANES) {
                let lanes: [u8; OBLIVIOUS_SIMD_LANES] = lanes
                    .try_into()
                    .expect("lane width matches the fixed SIMD width");
                u32x8::new([
                    u32::from(lanes[0]),
                    u32::from(lanes[1]),
                    u32::from(lanes[2]),
                    u32::from(lanes[3]),
                    u32::from(lanes[4]),
                    u32::from(lanes[5]),
                    u32::from(lanes[6]),
                    u32::from(lanes[7]),
                ])
            } else {
                let lanes: [u16; OBLIVIOUS_SIMD_LANES] = column
                    .slice_u16(base_row, OBLIVIOUS_SIMD_LANES)
                    .expect("column is u16 when not u8")
                    .try_into()
                    .expect("lane width matches the fixed SIMD width");
                u32x8::from(u16x8::new(lanes))
            };
            let threshold = u32x8::splat(u32::from(threshold_bin));
            let bit = bins.cmp_gt(threshold) & ones;
            leaf_indices = (leaf_indices << 1) | bit;
        }

        let lane_indices = leaf_indices.to_array();
        for lane in 0..OBLIVIOUS_SIMD_LANES {
            output[processed + lane] =
                leaf_values[usize::try_from(lane_indices[lane]).expect("leaf index fits usize")];
        }
        processed += OBLIVIOUS_SIMD_LANES;
    }

    processed
}

pub fn train(train_set: &dyn TableAccess, config: TrainConfig) -> Result<Model, TrainError> {
    training::train(train_set, config)
}

#[cfg(test)]
mod tests;
