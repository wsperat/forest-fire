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
mod forest;
pub mod ir;
mod sampling;
mod training;
pub mod tree;

pub use boosting::BoostingError;
pub use boosting::GradientBoostedTrees;
pub use forest::RandomForest;
pub use ir::IrError;
pub use ir::ModelPackageIr;
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

const PARALLEL_INFERENCE_ROW_THRESHOLD: usize = 256;
const PARALLEL_INFERENCE_CHUNK_ROWS: usize = 256;
const STANDARD_BATCH_INFERENCE_CHUNK_ROWS: usize = 4096;
const OBLIVIOUS_SIMD_LANES: usize = 8;
#[cfg(feature = "polars")]
const LAZYFRAME_PREDICT_BATCH_ROWS: usize = 10_000;
const COMPILED_ARTIFACT_MAGIC: [u8; 4] = *b"FFCA";
const COMPILED_ARTIFACT_VERSION: u16 = 1;
const COMPILED_ARTIFACT_BACKEND_CPU: u16 = 1;
const COMPILED_ARTIFACT_HEADER_LEN: usize = 8;

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeStructureSummary {
    /// Logical representation used by the tree.
    pub representation: String,
    /// Total node count including leaves.
    pub node_count: usize,
    /// Count of decision nodes.
    pub internal_node_count: usize,
    /// Count of leaves.
    pub leaf_count: usize,
    /// Maximum realized depth.
    pub actual_depth: usize,
    /// Shortest root-to-leaf path.
    pub shortest_path: usize,
    /// Longest root-to-leaf path.
    pub longest_path: usize,
    /// Mean root-to-leaf path length.
    pub average_path: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionValueStats {
    /// Number of leaf predictions included in the summary.
    pub count: usize,
    /// Number of distinct prediction values across leaves.
    pub unique_count: usize,
    /// Minimum prediction value.
    pub min: f64,
    /// Maximum prediction value.
    pub max: f64,
    /// Mean prediction value.
    pub mean: f64,
    /// Standard deviation of prediction values.
    pub std_dev: f64,
    /// Exact-value histogram over leaves.
    pub histogram: Vec<PredictionHistogramEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionHistogramEntry {
    /// Exact leaf prediction value.
    pub prediction: f64,
    /// Number of leaves with this value.
    pub count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntrospectionError {
    TreeIndexOutOfBounds { requested: usize, available: usize },
    NodeIndexOutOfBounds { requested: usize, available: usize },
    LevelIndexOutOfBounds { requested: usize, available: usize },
    LeafIndexOutOfBounds { requested: usize, available: usize },
    NotANodeTree,
    NotAnObliviousTree,
}

impl Display for IntrospectionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            IntrospectionError::TreeIndexOutOfBounds {
                requested,
                available,
            } => write!(
                f,
                "Tree index {} is out of bounds for model with {} trees.",
                requested, available
            ),
            IntrospectionError::NodeIndexOutOfBounds {
                requested,
                available,
            } => write!(
                f,
                "Node index {} is out of bounds for tree with {} nodes.",
                requested, available
            ),
            IntrospectionError::LevelIndexOutOfBounds {
                requested,
                available,
            } => write!(
                f,
                "Level index {} is out of bounds for tree with {} levels.",
                requested, available
            ),
            IntrospectionError::LeafIndexOutOfBounds {
                requested,
                available,
            } => write!(
                f,
                "Leaf index {} is out of bounds for tree with {} leaves.",
                requested, available
            ),
            IntrospectionError::NotANodeTree => write!(
                f,
                "This tree uses oblivious-level representation; inspect levels or leaves instead."
            ),
            IntrospectionError::NotAnObliviousTree => write!(
                f,
                "This tree uses node-tree representation; inspect nodes instead."
            ),
        }
    }
}

impl Error for IntrospectionError {}

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

#[derive(Debug)]
pub enum CompiledArtifactError {
    ArtifactTooShort { actual: usize, minimum: usize },
    InvalidMagic([u8; 4]),
    UnsupportedVersion(u16),
    UnsupportedBackend(u16),
    Encode(String),
    Decode(String),
    InvalidSemanticModel(IrError),
    InvalidRuntime(OptimizeError),
}

impl Display for CompiledArtifactError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            CompiledArtifactError::ArtifactTooShort { actual, minimum } => write!(
                f,
                "Compiled artifact is too short: expected at least {} bytes, found {}.",
                minimum, actual
            ),
            CompiledArtifactError::InvalidMagic(magic) => {
                write!(f, "Compiled artifact has invalid magic bytes: {:?}.", magic)
            }
            CompiledArtifactError::UnsupportedVersion(version) => {
                write!(f, "Unsupported compiled artifact version: {}.", version)
            }
            CompiledArtifactError::UnsupportedBackend(backend) => {
                write!(f, "Unsupported compiled artifact backend: {}.", backend)
            }
            CompiledArtifactError::Encode(message) => {
                write!(f, "Failed to encode compiled artifact: {}.", message)
            }
            CompiledArtifactError::Decode(message) => {
                write!(f, "Failed to decode compiled artifact: {}.", message)
            }
            CompiledArtifactError::InvalidSemanticModel(err) => err.fmt(f),
            CompiledArtifactError::InvalidRuntime(err) => err.fmt(f),
        }
    }
}

impl Error for CompiledArtifactError {}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Parallelism {
    thread_count: usize,
}

impl Parallelism {
    pub(crate) fn sequential() -> Self {
        Self { thread_count: 1 }
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

#[derive(Debug, Clone)]
enum InferenceFeatureColumn {
    Numeric(Vec<f64>),
    Binary(Vec<bool>),
}

#[derive(Debug, Clone)]
enum InferenceBinnedColumn {
    Numeric(Vec<u16>),
    Binary(Vec<bool>),
}

#[derive(Debug, Clone)]
enum CompactBinnedColumn {
    U8(Vec<u8>),
    U16(Vec<u16>),
}

impl CompactBinnedColumn {
    #[inline(always)]
    fn value_at(&self, row_index: usize) -> u16 {
        match self {
            CompactBinnedColumn::U8(values) => u16::from(values[row_index]),
            CompactBinnedColumn::U16(values) => values[row_index],
        }
    }

    #[inline(always)]
    fn slice_u8(&self, start: usize, len: usize) -> Option<&[u8]> {
        match self {
            CompactBinnedColumn::U8(values) => Some(&values[start..start + len]),
            CompactBinnedColumn::U16(_) => None,
        }
    }

    #[inline(always)]
    fn slice_u16(&self, start: usize, len: usize) -> Option<&[u16]> {
        match self {
            CompactBinnedColumn::U8(_) => None,
            CompactBinnedColumn::U16(values) => Some(&values[start..start + len]),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct InferenceTable {
    feature_columns: Vec<InferenceFeatureColumn>,
    binned_feature_columns: Vec<InferenceBinnedColumn>,
    n_rows: usize,
}

impl InferenceTable {
    pub(crate) fn from_rows(
        rows: Vec<Vec<f64>>,
        preprocessing: &[FeaturePreprocessing],
    ) -> Result<Self, PredictError> {
        let expected = preprocessing.len();
        if let Some((row_index, actual)) = rows
            .iter()
            .enumerate()
            .find_map(|(row_index, row)| (row.len() != expected).then_some((row_index, row.len())))
        {
            return Err(PredictError::RaggedRows {
                row: row_index,
                expected,
                actual,
            });
        }

        let columns = (0..expected)
            .map(|feature_index| {
                rows.iter()
                    .map(|row| row[feature_index])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        Self::from_columns(columns, preprocessing)
    }

    pub(crate) fn from_named_columns(
        columns: BTreeMap<String, Vec<f64>>,
        preprocessing: &[FeaturePreprocessing],
    ) -> Result<Self, PredictError> {
        let expected = preprocessing.len();
        if columns.len() != expected {
            for feature_index in 0..expected {
                let name = format!("f{}", feature_index);
                if !columns.contains_key(&name) {
                    return Err(PredictError::MissingFeature(name));
                }
            }
            if let Some(unexpected) = columns.keys().find(|name| {
                name.strip_prefix('f')
                    .and_then(|idx| idx.parse::<usize>().ok())
                    .is_none_or(|idx| idx >= expected)
            }) {
                return Err(PredictError::UnexpectedFeature(unexpected.clone()));
            }
        }

        let n_rows = columns.values().next().map_or(0, Vec::len);
        let ordered = (0..expected)
            .map(|feature_index| {
                let feature_name = format!("f{}", feature_index);
                let values = columns
                    .get(&feature_name)
                    .ok_or_else(|| PredictError::MissingFeature(feature_name.clone()))?;
                if values.len() != n_rows {
                    return Err(PredictError::ColumnLengthMismatch {
                        feature: feature_name,
                        expected: n_rows,
                        actual: values.len(),
                    });
                }
                Ok(values.clone())
            })
            .collect::<Result<Vec<_>, _>>()?;

        Self::from_columns(ordered, preprocessing)
    }

    pub(crate) fn from_sparse_binary_columns(
        n_rows: usize,
        n_features: usize,
        columns: Vec<Vec<usize>>,
        preprocessing: &[FeaturePreprocessing],
    ) -> Result<Self, PredictError> {
        if n_features != preprocessing.len() {
            return Err(PredictError::FeatureCountMismatch {
                expected: preprocessing.len(),
                actual: n_features,
            });
        }

        let mut dense_columns = Vec::with_capacity(n_features);
        for (feature_index, row_indices) in columns.into_iter().enumerate() {
            match preprocessing.get(feature_index) {
                Some(FeaturePreprocessing::Binary) => {
                    let mut values = vec![false; n_rows];
                    for row_index in row_indices {
                        if row_index >= n_rows {
                            return Err(PredictError::ColumnLengthMismatch {
                                feature: format!("f{}", feature_index),
                                expected: n_rows,
                                actual: row_index + 1,
                            });
                        }
                        values[row_index] = true;
                    }
                    dense_columns.push(values.into_iter().map(f64::from).collect());
                }
                Some(FeaturePreprocessing::Numeric { .. }) => {
                    return Err(PredictError::InvalidBinaryValue {
                        feature_index,
                        row_index: 0,
                        value: 1.0,
                    });
                }
                None => unreachable!("validated feature count"),
            }
        }

        Self::from_columns(dense_columns, preprocessing)
    }

    fn from_columns(
        columns: Vec<Vec<f64>>,
        preprocessing: &[FeaturePreprocessing],
    ) -> Result<Self, PredictError> {
        if columns.len() != preprocessing.len() {
            return Err(PredictError::FeatureCountMismatch {
                expected: preprocessing.len(),
                actual: columns.len(),
            });
        }

        let n_rows = columns.first().map_or(0, Vec::len);
        let mut feature_columns = Vec::with_capacity(columns.len());
        let mut binned_feature_columns = Vec::with_capacity(columns.len());

        for (feature_index, (column, feature_preprocessing)) in
            columns.into_iter().zip(preprocessing.iter()).enumerate()
        {
            if column.len() != n_rows {
                return Err(PredictError::ColumnLengthMismatch {
                    feature: format!("f{}", feature_index),
                    expected: n_rows,
                    actual: column.len(),
                });
            }
            match feature_preprocessing {
                FeaturePreprocessing::Binary => {
                    let values = column
                        .into_iter()
                        .enumerate()
                        .map(|(row_index, value)| match value {
                            v if v.total_cmp(&0.0).is_eq() => Ok(false),
                            v if v.total_cmp(&1.0).is_eq() => Ok(true),
                            v => Err(PredictError::InvalidBinaryValue {
                                feature_index,
                                row_index,
                                value: v,
                            }),
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    feature_columns.push(InferenceFeatureColumn::Binary(values.clone()));
                    binned_feature_columns.push(InferenceBinnedColumn::Binary(values));
                }
                FeaturePreprocessing::Numeric { bin_boundaries } => {
                    let bins = column
                        .iter()
                        .map(|value| infer_numeric_bin(*value, bin_boundaries))
                        .collect();
                    feature_columns.push(InferenceFeatureColumn::Numeric(column));
                    binned_feature_columns.push(InferenceBinnedColumn::Numeric(bins));
                }
            }
        }

        Ok(Self {
            feature_columns,
            binned_feature_columns,
            n_rows,
        })
    }

    pub(crate) fn to_column_major_binned_matrix(&self) -> ColumnMajorBinnedMatrix {
        let n_features = self.feature_columns.len();
        let columns = (0..n_features)
            .map(
                |feature_index| match &self.binned_feature_columns[feature_index] {
                    InferenceBinnedColumn::Numeric(values) => compact_binned_column(values),
                    InferenceBinnedColumn::Binary(values) => CompactBinnedColumn::U8(
                        values.iter().map(|value| u8::from(*value)).collect(),
                    ),
                },
            )
            .collect();

        ColumnMajorBinnedMatrix {
            n_rows: self.n_rows,
            columns,
        }
    }
}

#[derive(Debug, Clone)]
struct ColumnMajorBinnedMatrix {
    n_rows: usize,
    columns: Vec<CompactBinnedColumn>,
}

impl ColumnMajorBinnedMatrix {
    fn from_table_access(table: &dyn TableAccess) -> Self {
        let columns = (0..table.n_features())
            .map(|feature_index| {
                if table.is_binary_binned_feature(feature_index) {
                    CompactBinnedColumn::U8(
                        (0..table.n_rows())
                            .map(|row_index| {
                                u8::from(
                                    table
                                        .binned_boolean_value(feature_index, row_index)
                                        .unwrap_or(false),
                                )
                            })
                            .collect(),
                    )
                } else {
                    compact_binned_column(
                        &(0..table.n_rows())
                            .map(|row_index| table.binned_value(feature_index, row_index))
                            .collect::<Vec<_>>(),
                    )
                }
            })
            .collect();

        Self {
            n_rows: table.n_rows(),
            columns,
        }
    }

    #[inline(always)]
    fn column(&self, feature_index: usize) -> &CompactBinnedColumn {
        &self.columns[feature_index]
    }
}

fn infer_numeric_bin(value: f64, boundaries: &[NumericBinBoundary]) -> u16 {
    boundaries
        .iter()
        .find(|boundary| value <= boundary.upper_bound)
        .map_or_else(
            || boundaries.last().map_or(0, |boundary| boundary.bin),
            |boundary| boundary.bin,
        )
}

fn compact_binned_column(values: &[u16]) -> CompactBinnedColumn {
    if values.iter().all(|value| *value <= u16::from(u8::MAX)) {
        CompactBinnedColumn::U8(values.iter().map(|value| *value as u8).collect())
    } else {
        CompactBinnedColumn::U16(values.to_vec())
    }
}

impl TableAccess for InferenceTable {
    fn n_rows(&self) -> usize {
        self.n_rows
    }

    fn n_features(&self) -> usize {
        self.feature_columns.len()
    }

    fn canaries(&self) -> usize {
        0
    }

    fn numeric_bin_cap(&self) -> usize {
        MAX_NUMERIC_BINS
    }

    fn binned_feature_count(&self) -> usize {
        self.binned_feature_columns.len()
    }

    fn feature_value(&self, feature_index: usize, row_index: usize) -> f64 {
        match &self.feature_columns[feature_index] {
            InferenceFeatureColumn::Numeric(values) => values[row_index],
            InferenceFeatureColumn::Binary(values) => f64::from(u8::from(values[row_index])),
        }
    }

    fn is_binary_feature(&self, index: usize) -> bool {
        matches!(
            self.feature_columns[index],
            InferenceFeatureColumn::Binary(_)
        )
    }

    fn binned_value(&self, feature_index: usize, row_index: usize) -> u16 {
        match &self.binned_feature_columns[feature_index] {
            InferenceBinnedColumn::Numeric(values) => values[row_index],
            InferenceBinnedColumn::Binary(values) => u16::from(values[row_index]),
        }
    }

    fn binned_boolean_value(&self, feature_index: usize, row_index: usize) -> Option<bool> {
        match &self.binned_feature_columns[feature_index] {
            InferenceBinnedColumn::Numeric(_) => None,
            InferenceBinnedColumn::Binary(values) => Some(values[row_index]),
        }
    }

    fn binned_column_kind(&self, index: usize) -> BinnedColumnKind {
        BinnedColumnKind::Real {
            source_index: index,
        }
    }

    fn is_binary_binned_feature(&self, index: usize) -> bool {
        matches!(
            self.binned_feature_columns[index],
            InferenceBinnedColumn::Binary(_)
        )
    }

    fn target_value(&self, _row_index: usize) -> f64 {
        0.0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum OptimizedRuntime {
    BinaryClassifier {
        nodes: Vec<OptimizedBinaryClassifierNode>,
        class_labels: Vec<f64>,
    },
    StandardClassifier {
        nodes: Vec<OptimizedClassifierNode>,
        root: usize,
        class_labels: Vec<f64>,
    },
    ObliviousClassifier {
        feature_indices: Vec<usize>,
        threshold_bins: Vec<u16>,
        leaf_values: Vec<Vec<f64>>,
        class_labels: Vec<f64>,
    },
    BinaryRegressor {
        nodes: Vec<OptimizedBinaryRegressorNode>,
    },
    ObliviousRegressor {
        feature_indices: Vec<usize>,
        threshold_bins: Vec<u16>,
        leaf_values: Vec<f64>,
    },
    ForestClassifier {
        trees: Vec<OptimizedRuntime>,
        class_labels: Vec<f64>,
    },
    ForestRegressor {
        trees: Vec<OptimizedRuntime>,
    },
    BoostedBinaryClassifier {
        trees: Vec<OptimizedRuntime>,
        tree_weights: Vec<f64>,
        base_score: f64,
        class_labels: Vec<f64>,
    },
    BoostedRegressor {
        trees: Vec<OptimizedRuntime>,
        tree_weights: Vec<f64>,
        base_score: f64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum OptimizedClassifierNode {
    Leaf(Vec<f64>),
    Binary {
        feature_index: usize,
        threshold_bin: u16,
        children: [usize; 2],
    },
    Multiway {
        feature_index: usize,
        child_lookup: Vec<usize>,
        max_bin_index: usize,
        fallback_probabilities: Vec<f64>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum OptimizedBinaryClassifierNode {
    Leaf(Vec<f64>),
    Branch {
        feature_index: usize,
        threshold_bin: u16,
        jump_index: usize,
        jump_if_greater: bool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum OptimizedBinaryRegressorNode {
    Leaf(f64),
    Branch {
        feature_index: usize,
        threshold_bin: u16,
        jump_index: usize,
        jump_if_greater: bool,
    },
}

#[derive(Debug, Clone)]
struct InferenceExecutor {
    thread_count: usize,
    pool: Option<Arc<rayon::ThreadPool>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CompiledArtifactPayload {
    semantic_ir: ModelPackageIr,
    runtime: OptimizedRuntime,
}

impl InferenceExecutor {
    fn new(thread_count: usize) -> Result<Self, OptimizeError> {
        let pool = if thread_count > 1 {
            Some(Arc::new(
                ThreadPoolBuilder::new()
                    .num_threads(thread_count)
                    .build()
                    .map_err(|err| OptimizeError::ThreadPoolBuildFailed(err.to_string()))?,
            ))
        } else {
            None
        };

        Ok(Self { thread_count, pool })
    }

    fn predict_rows<F>(&self, n_rows: usize, predict_row: F) -> Vec<f64>
    where
        F: Fn(usize) -> f64 + Sync + Send,
    {
        if self.thread_count == 1 || n_rows < PARALLEL_INFERENCE_ROW_THRESHOLD {
            return (0..n_rows).map(predict_row).collect();
        }

        self.pool
            .as_ref()
            .expect("thread pool exists when parallel inference is enabled")
            .install(|| (0..n_rows).into_par_iter().map(predict_row).collect())
    }

    fn fill_chunks<F>(&self, outputs: &mut [f64], chunk_rows: usize, fill_chunk: F)
    where
        F: Fn(usize, &mut [f64]) + Sync + Send,
    {
        if self.thread_count == 1 || outputs.len() < PARALLEL_INFERENCE_ROW_THRESHOLD {
            for (chunk_index, chunk) in outputs.chunks_mut(chunk_rows).enumerate() {
                fill_chunk(chunk_index * chunk_rows, chunk);
            }
            return;
        }

        self.pool
            .as_ref()
            .expect("thread pool exists when parallel inference is enabled")
            .install(|| {
                outputs
                    .par_chunks_mut(chunk_rows)
                    .enumerate()
                    .for_each(|(chunk_index, chunk)| fill_chunk(chunk_index * chunk_rows, chunk));
            });
    }
}

/// Runtime-lowered model used for faster inference.
///
/// The optimized model keeps a copy of the source [`Model`] so it can preserve
/// serialization and introspection behavior even after the runtime has been
/// lowered into lookup-table-friendly structures.
#[derive(Debug, Clone)]
pub struct OptimizedModel {
    source_model: Model,
    runtime: OptimizedRuntime,
    executor: InferenceExecutor,
}

impl OptimizedModel {
    fn new(source_model: Model, physical_cores: Option<usize>) -> Result<Self, OptimizeError> {
        let thread_count = resolve_inference_thread_count(physical_cores)?;
        let runtime = OptimizedRuntime::from_model(&source_model);
        let executor = InferenceExecutor::new(thread_count)?;

        Ok(Self {
            source_model,
            runtime,
            executor,
        })
    }

    /// Predict directly from a preprocessed table.
    pub fn predict_table(&self, table: &dyn TableAccess) -> Vec<f64> {
        if self.runtime.should_use_batch_matrix(table.n_rows()) {
            let matrix = ColumnMajorBinnedMatrix::from_table_access(table);
            return self.predict_column_major_binned_matrix(&matrix);
        }

        self.executor.predict_rows(table.n_rows(), |row_index| {
            self.runtime.predict_table_row(table, row_index)
        })
    }

    /// Predict from raw row-major inputs.
    pub fn predict_rows(&self, rows: Vec<Vec<f64>>) -> Result<Vec<f64>, PredictError> {
        let table = InferenceTable::from_rows(rows, self.source_model.feature_preprocessing())?;
        if self.runtime.should_use_batch_matrix(table.n_rows()) {
            let matrix = table.to_column_major_binned_matrix();
            Ok(self.predict_column_major_binned_matrix(&matrix))
        } else {
            Ok(self.predict_table(&table))
        }
    }

    /// Predict from named columns keyed by feature name.
    pub fn predict_named_columns(
        &self,
        columns: BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<f64>, PredictError> {
        let table =
            InferenceTable::from_named_columns(columns, self.source_model.feature_preprocessing())?;
        if self.runtime.should_use_batch_matrix(table.n_rows()) {
            let matrix = table.to_column_major_binned_matrix();
            Ok(self.predict_column_major_binned_matrix(&matrix))
        } else {
            Ok(self.predict_table(&table))
        }
    }

    /// Return class probabilities for classification models.
    pub fn predict_proba_table(
        &self,
        table: &dyn TableAccess,
    ) -> Result<Vec<Vec<f64>>, PredictError> {
        self.runtime.predict_proba_table(table, &self.executor)
    }

    /// Return class probabilities from raw row-major inputs.
    pub fn predict_proba_rows(&self, rows: Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>, PredictError> {
        let table = InferenceTable::from_rows(rows, self.source_model.feature_preprocessing())?;
        self.predict_proba_table(&table)
    }

    /// Return class probabilities from named columns keyed by feature name.
    pub fn predict_proba_named_columns(
        &self,
        columns: BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<Vec<f64>>, PredictError> {
        let table =
            InferenceTable::from_named_columns(columns, self.source_model.feature_preprocessing())?;
        self.predict_proba_table(&table)
    }

    /// Return class probabilities from sparse binary column storage.
    pub fn predict_proba_sparse_binary_columns(
        &self,
        n_rows: usize,
        n_features: usize,
        columns: Vec<Vec<usize>>,
    ) -> Result<Vec<Vec<f64>>, PredictError> {
        let table = InferenceTable::from_sparse_binary_columns(
            n_rows,
            n_features,
            columns,
            self.source_model.feature_preprocessing(),
        )?;
        self.predict_proba_table(&table)
    }

    /// Predict from sparse binary column storage.
    pub fn predict_sparse_binary_columns(
        &self,
        n_rows: usize,
        n_features: usize,
        columns: Vec<Vec<usize>>,
    ) -> Result<Vec<f64>, PredictError> {
        let table = InferenceTable::from_sparse_binary_columns(
            n_rows,
            n_features,
            columns,
            self.source_model.feature_preprocessing(),
        )?;
        if self.runtime.should_use_batch_matrix(table.n_rows()) {
            let matrix = table.to_column_major_binned_matrix();
            Ok(self.predict_column_major_binned_matrix(&matrix))
        } else {
            Ok(self.predict_table(&table))
        }
    }

    #[cfg(feature = "polars")]
    pub fn predict_polars_dataframe(&self, df: &DataFrame) -> Result<Vec<f64>, PredictError> {
        let columns = polars_named_columns(df)?;
        self.predict_named_columns(columns)
    }

    #[cfg(feature = "polars")]
    pub fn predict_polars_lazyframe(&self, lf: &LazyFrame) -> Result<Vec<f64>, PredictError> {
        let mut predictions = Vec::new();
        let mut offset = 0i64;
        loop {
            let batch = lf
                .clone()
                .slice(offset, LAZYFRAME_PREDICT_BATCH_ROWS as IdxSize)
                .collect()?;
            let height = batch.height();
            if height == 0 {
                break;
            }
            predictions.extend(self.predict_polars_dataframe(&batch)?);
            if height < LAZYFRAME_PREDICT_BATCH_ROWS {
                break;
            }
            offset += height as i64;
        }
        Ok(predictions)
    }

    /// The semantic algorithm remains the one from the source model.
    pub fn algorithm(&self) -> TrainAlgorithm {
        self.source_model.algorithm()
    }

    pub fn task(&self) -> Task {
        self.source_model.task()
    }

    pub fn criterion(&self) -> Criterion {
        self.source_model.criterion()
    }

    pub fn tree_type(&self) -> TreeType {
        self.source_model.tree_type()
    }

    pub fn mean_value(&self) -> Option<f64> {
        self.source_model.mean_value()
    }

    pub fn canaries(&self) -> usize {
        self.source_model.canaries()
    }

    pub fn max_depth(&self) -> Option<usize> {
        self.source_model.max_depth()
    }

    pub fn min_samples_split(&self) -> Option<usize> {
        self.source_model.min_samples_split()
    }

    pub fn min_samples_leaf(&self) -> Option<usize> {
        self.source_model.min_samples_leaf()
    }

    pub fn n_trees(&self) -> Option<usize> {
        self.source_model.n_trees()
    }

    pub fn max_features(&self) -> Option<usize> {
        self.source_model.max_features()
    }

    pub fn seed(&self) -> Option<u64> {
        self.source_model.seed()
    }

    pub fn compute_oob(&self) -> bool {
        self.source_model.compute_oob()
    }

    pub fn oob_score(&self) -> Option<f64> {
        self.source_model.oob_score()
    }

    pub fn learning_rate(&self) -> Option<f64> {
        self.source_model.learning_rate()
    }

    pub fn bootstrap(&self) -> bool {
        self.source_model.bootstrap()
    }

    pub fn top_gradient_fraction(&self) -> Option<f64> {
        self.source_model.top_gradient_fraction()
    }

    pub fn other_gradient_fraction(&self) -> Option<f64> {
        self.source_model.other_gradient_fraction()
    }

    pub fn tree_count(&self) -> usize {
        self.source_model.tree_count()
    }

    /// Introspection is delegated to the source model so lowering never changes
    /// the observable semantic tree structure.
    pub fn tree_structure(
        &self,
        tree_index: usize,
    ) -> Result<TreeStructureSummary, IntrospectionError> {
        self.source_model.tree_structure(tree_index)
    }

    pub fn tree_prediction_stats(
        &self,
        tree_index: usize,
    ) -> Result<PredictionValueStats, IntrospectionError> {
        self.source_model.tree_prediction_stats(tree_index)
    }

    pub fn tree_node(
        &self,
        tree_index: usize,
        node_index: usize,
    ) -> Result<ir::NodeTreeNode, IntrospectionError> {
        self.source_model.tree_node(tree_index, node_index)
    }

    pub fn tree_level(
        &self,
        tree_index: usize,
        level_index: usize,
    ) -> Result<ir::ObliviousLevel, IntrospectionError> {
        self.source_model.tree_level(tree_index, level_index)
    }

    pub fn tree_leaf(
        &self,
        tree_index: usize,
        leaf_index: usize,
    ) -> Result<ir::IndexedLeaf, IntrospectionError> {
        self.source_model.tree_leaf(tree_index, leaf_index)
    }

    pub fn to_ir(&self) -> ModelPackageIr {
        self.source_model.to_ir()
    }

    pub fn to_ir_json(&self) -> Result<String, serde_json::Error> {
        self.source_model.to_ir_json()
    }

    pub fn to_ir_json_pretty(&self) -> Result<String, serde_json::Error> {
        self.source_model.to_ir_json_pretty()
    }

    pub fn serialize(&self) -> Result<String, serde_json::Error> {
        self.source_model.serialize()
    }

    pub fn serialize_pretty(&self) -> Result<String, serde_json::Error> {
        self.source_model.serialize_pretty()
    }

    pub fn serialize_compiled(&self) -> Result<Vec<u8>, CompiledArtifactError> {
        let payload = CompiledArtifactPayload {
            semantic_ir: self.source_model.to_ir(),
            runtime: self.runtime.clone(),
        };
        let mut payload_bytes = Vec::new();
        ciborium::into_writer(&payload, &mut payload_bytes)
            .map_err(|err| CompiledArtifactError::Encode(err.to_string()))?;
        let mut bytes = Vec::with_capacity(COMPILED_ARTIFACT_HEADER_LEN + payload_bytes.len());
        bytes.extend_from_slice(&COMPILED_ARTIFACT_MAGIC);
        bytes.extend_from_slice(&COMPILED_ARTIFACT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&COMPILED_ARTIFACT_BACKEND_CPU.to_le_bytes());
        bytes.extend_from_slice(&payload_bytes);
        Ok(bytes)
    }

    pub fn deserialize_compiled(
        serialized: &[u8],
        physical_cores: Option<usize>,
    ) -> Result<Self, CompiledArtifactError> {
        if serialized.len() < COMPILED_ARTIFACT_HEADER_LEN {
            return Err(CompiledArtifactError::ArtifactTooShort {
                actual: serialized.len(),
                minimum: COMPILED_ARTIFACT_HEADER_LEN,
            });
        }

        let magic = [serialized[0], serialized[1], serialized[2], serialized[3]];
        if magic != COMPILED_ARTIFACT_MAGIC {
            return Err(CompiledArtifactError::InvalidMagic(magic));
        }

        let version = u16::from_le_bytes([serialized[4], serialized[5]]);
        if version != COMPILED_ARTIFACT_VERSION {
            return Err(CompiledArtifactError::UnsupportedVersion(version));
        }

        let backend = u16::from_le_bytes([serialized[6], serialized[7]]);
        if backend != COMPILED_ARTIFACT_BACKEND_CPU {
            return Err(CompiledArtifactError::UnsupportedBackend(backend));
        }

        let payload: CompiledArtifactPayload = ciborium::from_reader(std::io::Cursor::new(
            &serialized[COMPILED_ARTIFACT_HEADER_LEN..],
        ))
        .map_err(|err| CompiledArtifactError::Decode(err.to_string()))?;
        let source_model = ir::model_from_ir(payload.semantic_ir)
            .map_err(CompiledArtifactError::InvalidSemanticModel)?;
        let thread_count = resolve_inference_thread_count(physical_cores)
            .map_err(CompiledArtifactError::InvalidRuntime)?;
        let executor =
            InferenceExecutor::new(thread_count).map_err(CompiledArtifactError::InvalidRuntime)?;

        Ok(Self {
            source_model,
            runtime: payload.runtime,
            executor,
        })
    }

    fn predict_column_major_binned_matrix(&self, matrix: &ColumnMajorBinnedMatrix) -> Vec<f64> {
        self.runtime
            .predict_column_major_matrix(matrix, &self.executor)
    }
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

    fn from_model(model: &Model) -> Self {
        match model {
            Model::DecisionTreeClassifier(classifier) => Self::from_classifier(classifier),
            Model::DecisionTreeRegressor(regressor) => Self::from_regressor(regressor),
            Model::RandomForest(forest) => match forest.task() {
                Task::Classification => Self::ForestClassifier {
                    trees: forest.trees().iter().map(Self::from_model).collect(),
                    class_labels: forest
                        .class_labels()
                        .expect("classification forest stores class labels"),
                },
                Task::Regression => Self::ForestRegressor {
                    trees: forest.trees().iter().map(Self::from_model).collect(),
                },
            },
            Model::GradientBoostedTrees(model) => match model.task() {
                Task::Classification => Self::BoostedBinaryClassifier {
                    trees: model.trees().iter().map(Self::from_model).collect(),
                    tree_weights: model.tree_weights().to_vec(),
                    base_score: model.base_score(),
                    class_labels: model
                        .class_labels()
                        .expect("classification boosting stores class labels"),
                },
                Task::Regression => Self::BoostedRegressor {
                    trees: model.trees().iter().map(Self::from_model).collect(),
                    tree_weights: model.tree_weights().to_vec(),
                    base_score: model.base_score(),
                },
            },
        }
    }

    fn from_classifier(classifier: &DecisionTreeClassifier) -> Self {
        match classifier.structure() {
            tree::classifier::TreeStructure::Standard { nodes, root } => {
                if classifier_nodes_are_binary_only(nodes) {
                    return Self::BinaryClassifier {
                        nodes: build_binary_classifier_layout(
                            nodes,
                            *root,
                            classifier.class_labels(),
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
                            feature_index: *feature_index,
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
                                feature_index: *feature_index,
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
                feature_indices: splits.iter().map(|split| split.feature_index).collect(),
                threshold_bins: splits.iter().map(|split| split.threshold_bin).collect(),
                leaf_values: leaf_class_counts
                    .iter()
                    .map(|class_counts| normalized_probabilities_from_counts(class_counts))
                    .collect(),
                class_labels: classifier.class_labels().to_vec(),
            },
        }
    }

    fn from_regressor(regressor: &DecisionTreeRegressor) -> Self {
        match regressor.structure() {
            tree::regressor::RegressionTreeStructure::Standard { nodes, root } => {
                Self::BinaryRegressor {
                    nodes: build_binary_regressor_layout(nodes, *root),
                }
            }
            tree::regressor::RegressionTreeStructure::Oblivious {
                splits,
                leaf_values,
                ..
            } => Self::ObliviousRegressor {
                feature_indices: splits.iter().map(|split| split.feature_index).collect(),
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
) -> Vec<OptimizedBinaryClassifierNode> {
    let mut layout = Vec::with_capacity(nodes.len());
    append_binary_classifier_node(nodes, root, &mut layout);
    layout
}

fn append_binary_classifier_node(
    nodes: &[tree::classifier::TreeNode],
    node_index: usize,
    layout: &mut Vec<OptimizedBinaryClassifierNode>,
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

            let fallthrough_index = append_binary_classifier_node(nodes, fallthrough_child, layout);
            debug_assert_eq!(fallthrough_index, current_index + 1);
            let jump_index = if jump_child == fallthrough_child {
                fallthrough_index
            } else {
                append_binary_classifier_node(nodes, jump_child, layout)
            };

            layout[current_index] = OptimizedBinaryClassifierNode::Branch {
                feature_index: *feature_index,
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
) -> Vec<OptimizedBinaryRegressorNode> {
    let mut layout = Vec::with_capacity(nodes.len());
    append_binary_regressor_node(nodes, root, &mut layout);
    layout
}

fn append_binary_regressor_node(
    nodes: &[tree::regressor::RegressionNode],
    node_index: usize,
    layout: &mut Vec<OptimizedBinaryRegressorNode>,
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

            let fallthrough_index = append_binary_regressor_node(nodes, fallthrough_child, layout);
            debug_assert_eq!(fallthrough_index, current_index + 1);
            let jump_index = if jump_child == fallthrough_child {
                fallthrough_index
            } else {
                append_binary_regressor_node(nodes, jump_child, layout)
            };

            layout[current_index] = OptimizedBinaryRegressorNode::Branch {
                feature_index: *feature_index,
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

fn resolve_inference_thread_count(physical_cores: Option<usize>) -> Result<usize, OptimizeError> {
    let available = num_cpus::get_physical().max(1);
    let requested = physical_cores.unwrap_or(available);

    if requested == 0 {
        return Err(OptimizeError::InvalidPhysicalCoreCount {
            requested,
            available,
        });
    }

    Ok(requested.min(available))
}

impl Model {
    /// Predict directly from a preprocessed table.
    pub fn predict_table(&self, table: &dyn TableAccess) -> Vec<f64> {
        match self {
            Model::DecisionTreeClassifier(model) => model.predict_table(table),
            Model::DecisionTreeRegressor(model) => model.predict_table(table),
            Model::RandomForest(model) => model.predict_table(table),
            Model::GradientBoostedTrees(model) => model.predict_table(table),
        }
    }

    /// Predict from raw row-major input.
    pub fn predict_rows(&self, rows: Vec<Vec<f64>>) -> Result<Vec<f64>, PredictError> {
        let table = InferenceTable::from_rows(rows, self.feature_preprocessing())?;
        Ok(self.predict_table(&table))
    }

    /// Return class probabilities for classification models.
    pub fn predict_proba_table(
        &self,
        table: &dyn TableAccess,
    ) -> Result<Vec<Vec<f64>>, PredictError> {
        match self {
            Model::DecisionTreeClassifier(model) => Ok(model.predict_proba_table(table)),
            Model::RandomForest(model) => model.predict_proba_table(table),
            Model::GradientBoostedTrees(model) => model.predict_proba_table(table),
            Model::DecisionTreeRegressor(_) => {
                Err(PredictError::ProbabilityPredictionRequiresClassification)
            }
        }
    }

    /// Return class probabilities from raw row-major input.
    pub fn predict_proba_rows(&self, rows: Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>, PredictError> {
        let table = InferenceTable::from_rows(rows, self.feature_preprocessing())?;
        self.predict_proba_table(&table)
    }

    /// Predict from named columns keyed by feature name.
    pub fn predict_named_columns(
        &self,
        columns: BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<f64>, PredictError> {
        let table = InferenceTable::from_named_columns(columns, self.feature_preprocessing())?;
        Ok(self.predict_table(&table))
    }

    /// Return class probabilities from named columns keyed by feature name.
    pub fn predict_proba_named_columns(
        &self,
        columns: BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<Vec<f64>>, PredictError> {
        let table = InferenceTable::from_named_columns(columns, self.feature_preprocessing())?;
        self.predict_proba_table(&table)
    }

    /// Predict from sparse binary column storage.
    pub fn predict_sparse_binary_columns(
        &self,
        n_rows: usize,
        n_features: usize,
        columns: Vec<Vec<usize>>,
    ) -> Result<Vec<f64>, PredictError> {
        let table = InferenceTable::from_sparse_binary_columns(
            n_rows,
            n_features,
            columns,
            self.feature_preprocessing(),
        )?;
        Ok(self.predict_table(&table))
    }

    /// Return class probabilities from sparse binary column storage.
    pub fn predict_proba_sparse_binary_columns(
        &self,
        n_rows: usize,
        n_features: usize,
        columns: Vec<Vec<usize>>,
    ) -> Result<Vec<Vec<f64>>, PredictError> {
        let table = InferenceTable::from_sparse_binary_columns(
            n_rows,
            n_features,
            columns,
            self.feature_preprocessing(),
        )?;
        self.predict_proba_table(&table)
    }

    #[cfg(feature = "polars")]
    pub fn predict_polars_dataframe(&self, df: &DataFrame) -> Result<Vec<f64>, PredictError> {
        let columns = polars_named_columns(df)?;
        self.predict_named_columns(columns)
    }

    #[cfg(feature = "polars")]
    pub fn predict_polars_lazyframe(&self, lf: &LazyFrame) -> Result<Vec<f64>, PredictError> {
        let mut predictions = Vec::new();
        let mut offset = 0i64;
        loop {
            let batch = lf
                .clone()
                .slice(offset, LAZYFRAME_PREDICT_BATCH_ROWS as IdxSize)
                .collect()?;
            let height = batch.height();
            if height == 0 {
                break;
            }
            predictions.extend(self.predict_polars_dataframe(&batch)?);
            if height < LAZYFRAME_PREDICT_BATCH_ROWS {
                break;
            }
            offset += height as i64;
        }
        Ok(predictions)
    }

    /// Report the semantic algorithm family used to train the model.
    pub fn algorithm(&self) -> TrainAlgorithm {
        match self {
            Model::DecisionTreeClassifier(_) | Model::DecisionTreeRegressor(_) => {
                TrainAlgorithm::Dt
            }
            Model::RandomForest(_) => TrainAlgorithm::Rf,
            Model::GradientBoostedTrees(_) => TrainAlgorithm::Gbm,
        }
    }

    pub fn task(&self) -> Task {
        match self {
            Model::DecisionTreeRegressor(_) => Task::Regression,
            Model::DecisionTreeClassifier(_) => Task::Classification,
            Model::RandomForest(model) => model.task(),
            Model::GradientBoostedTrees(model) => model.task(),
        }
    }

    pub fn criterion(&self) -> Criterion {
        match self {
            Model::DecisionTreeClassifier(model) => model.criterion(),
            Model::DecisionTreeRegressor(model) => model.criterion(),
            Model::RandomForest(model) => model.criterion(),
            Model::GradientBoostedTrees(model) => model.criterion(),
        }
    }

    pub fn tree_type(&self) -> TreeType {
        match self {
            Model::DecisionTreeClassifier(model) => match model.algorithm() {
                DecisionTreeAlgorithm::Id3 => TreeType::Id3,
                DecisionTreeAlgorithm::C45 => TreeType::C45,
                DecisionTreeAlgorithm::Cart => TreeType::Cart,
                DecisionTreeAlgorithm::Randomized => TreeType::Randomized,
                DecisionTreeAlgorithm::Oblivious => TreeType::Oblivious,
            },
            Model::DecisionTreeRegressor(model) => match model.algorithm() {
                RegressionTreeAlgorithm::Cart => TreeType::Cart,
                RegressionTreeAlgorithm::Randomized => TreeType::Randomized,
                RegressionTreeAlgorithm::Oblivious => TreeType::Oblivious,
            },
            Model::RandomForest(model) => model.tree_type(),
            Model::GradientBoostedTrees(model) => model.tree_type(),
        }
    }

    pub fn mean_value(&self) -> Option<f64> {
        match self {
            Model::DecisionTreeClassifier(_)
            | Model::DecisionTreeRegressor(_)
            | Model::RandomForest(_)
            | Model::GradientBoostedTrees(_) => None,
        }
    }

    pub fn canaries(&self) -> usize {
        self.training_metadata().canaries
    }

    pub fn max_depth(&self) -> Option<usize> {
        self.training_metadata().max_depth
    }

    pub fn min_samples_split(&self) -> Option<usize> {
        self.training_metadata().min_samples_split
    }

    pub fn min_samples_leaf(&self) -> Option<usize> {
        self.training_metadata().min_samples_leaf
    }

    pub fn n_trees(&self) -> Option<usize> {
        self.training_metadata().n_trees
    }

    pub fn max_features(&self) -> Option<usize> {
        self.training_metadata().max_features
    }

    pub fn seed(&self) -> Option<u64> {
        self.training_metadata().seed
    }

    pub fn compute_oob(&self) -> bool {
        self.training_metadata().compute_oob
    }

    pub fn oob_score(&self) -> Option<f64> {
        self.training_metadata().oob_score
    }

    pub fn learning_rate(&self) -> Option<f64> {
        self.training_metadata().learning_rate
    }

    pub fn bootstrap(&self) -> bool {
        self.training_metadata().bootstrap.unwrap_or(false)
    }

    pub fn top_gradient_fraction(&self) -> Option<f64> {
        self.training_metadata().top_gradient_fraction
    }

    pub fn other_gradient_fraction(&self) -> Option<f64> {
        self.training_metadata().other_gradient_fraction
    }

    /// Count trees after normalizing both single-tree and ensemble models to
    /// the shared IR introspection view.
    pub fn tree_count(&self) -> usize {
        self.to_ir().model.trees.len()
    }

    /// Summarize the structure of one tree inside the model.
    pub fn tree_structure(
        &self,
        tree_index: usize,
    ) -> Result<TreeStructureSummary, IntrospectionError> {
        tree_structure_summary(self.tree_definition(tree_index)?)
    }

    /// Summarize the values stored in one tree's leaves.
    pub fn tree_prediction_stats(
        &self,
        tree_index: usize,
    ) -> Result<PredictionValueStats, IntrospectionError> {
        prediction_value_stats(self.tree_definition(tree_index)?)
    }

    /// Inspect a node-tree node by index.
    pub fn tree_node(
        &self,
        tree_index: usize,
        node_index: usize,
    ) -> Result<ir::NodeTreeNode, IntrospectionError> {
        match self.tree_definition(tree_index)? {
            ir::TreeDefinition::NodeTree { nodes, .. } => {
                let available = nodes.len();
                nodes
                    .into_iter()
                    .nth(node_index)
                    .ok_or(IntrospectionError::NodeIndexOutOfBounds {
                        requested: node_index,
                        available,
                    })
            }
            ir::TreeDefinition::ObliviousLevels { .. } => Err(IntrospectionError::NotANodeTree),
        }
    }

    /// Inspect an oblivious-tree level by index.
    pub fn tree_level(
        &self,
        tree_index: usize,
        level_index: usize,
    ) -> Result<ir::ObliviousLevel, IntrospectionError> {
        match self.tree_definition(tree_index)? {
            ir::TreeDefinition::ObliviousLevels { levels, .. } => {
                let available = levels.len();
                levels.into_iter().nth(level_index).ok_or(
                    IntrospectionError::LevelIndexOutOfBounds {
                        requested: level_index,
                        available,
                    },
                )
            }
            ir::TreeDefinition::NodeTree { .. } => Err(IntrospectionError::NotAnObliviousTree),
        }
    }

    /// Inspect a leaf by index regardless of the underlying tree representation.
    pub fn tree_leaf(
        &self,
        tree_index: usize,
        leaf_index: usize,
    ) -> Result<ir::IndexedLeaf, IntrospectionError> {
        match self.tree_definition(tree_index)? {
            ir::TreeDefinition::ObliviousLevels { leaves, .. } => {
                let available = leaves.len();
                leaves
                    .into_iter()
                    .nth(leaf_index)
                    .ok_or(IntrospectionError::LeafIndexOutOfBounds {
                        requested: leaf_index,
                        available,
                    })
            }
            ir::TreeDefinition::NodeTree { nodes, .. } => {
                let leaves = nodes
                    .into_iter()
                    .filter_map(|node| match node {
                        ir::NodeTreeNode::Leaf {
                            node_id,
                            leaf,
                            stats,
                            ..
                        } => Some(ir::IndexedLeaf {
                            leaf_index: node_id,
                            leaf,
                            stats: ir::NodeStats {
                                sample_count: stats.sample_count,
                                impurity: stats.impurity,
                                gain: stats.gain,
                                class_counts: stats.class_counts,
                                variance: stats.variance,
                            },
                        }),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                let available = leaves.len();
                leaves
                    .into_iter()
                    .nth(leaf_index)
                    .ok_or(IntrospectionError::LeafIndexOutOfBounds {
                        requested: leaf_index,
                        available,
                    })
            }
        }
    }

    /// Convert the model to the stable IR used for serialization and
    /// binding-independent introspection.
    pub fn to_ir(&self) -> ModelPackageIr {
        ir::model_to_ir(self)
    }

    pub fn to_ir_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(&self.to_ir())
    }

    pub fn to_ir_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.to_ir())
    }

    pub fn serialize(&self) -> Result<String, serde_json::Error> {
        self.to_ir_json()
    }

    pub fn serialize_pretty(&self) -> Result<String, serde_json::Error> {
        self.to_ir_json_pretty()
    }

    /// Lower the model into a runtime-oriented representation while preserving
    /// the original semantic model for serialization and inspection.
    pub fn optimize_inference(
        &self,
        physical_cores: Option<usize>,
    ) -> Result<OptimizedModel, OptimizeError> {
        OptimizedModel::new(self.clone(), physical_cores)
    }

    pub fn json_schema() -> schemars::schema::RootSchema {
        ModelPackageIr::json_schema()
    }

    pub fn json_schema_json() -> Result<String, IrError> {
        ModelPackageIr::json_schema_json()
    }

    pub fn json_schema_json_pretty() -> Result<String, IrError> {
        ModelPackageIr::json_schema_json_pretty()
    }

    pub fn deserialize(serialized: &str) -> Result<Self, IrError> {
        let ir: ModelPackageIr =
            serde_json::from_str(serialized).map_err(|err| IrError::Json(err.to_string()))?;
        ir::model_from_ir(ir)
    }

    pub(crate) fn num_features(&self) -> usize {
        match self {
            Model::DecisionTreeClassifier(model) => model.num_features(),
            Model::DecisionTreeRegressor(model) => model.num_features(),
            Model::RandomForest(model) => model.num_features(),
            Model::GradientBoostedTrees(model) => model.num_features(),
        }
    }

    pub(crate) fn feature_preprocessing(&self) -> &[FeaturePreprocessing] {
        match self {
            Model::DecisionTreeClassifier(model) => model.feature_preprocessing(),
            Model::DecisionTreeRegressor(model) => model.feature_preprocessing(),
            Model::RandomForest(model) => model.feature_preprocessing(),
            Model::GradientBoostedTrees(model) => model.feature_preprocessing(),
        }
    }

    pub(crate) fn class_labels(&self) -> Option<Vec<f64>> {
        match self {
            Model::DecisionTreeClassifier(model) => Some(model.class_labels().to_vec()),
            Model::RandomForest(model) => model.class_labels(),
            Model::GradientBoostedTrees(model) => model.class_labels(),
            Model::DecisionTreeRegressor(_) => None,
        }
    }

    pub(crate) fn training_metadata(&self) -> ir::TrainingMetadata {
        match self {
            Model::DecisionTreeClassifier(model) => model.training_metadata(),
            Model::DecisionTreeRegressor(model) => model.training_metadata(),
            Model::RandomForest(model) => model.training_metadata(),
            Model::GradientBoostedTrees(model) => model.training_metadata(),
        }
    }

    fn tree_definition(&self, tree_index: usize) -> Result<ir::TreeDefinition, IntrospectionError> {
        let trees = self.to_ir().model.trees;
        let available = trees.len();
        trees
            .into_iter()
            .nth(tree_index)
            .ok_or(IntrospectionError::TreeIndexOutOfBounds {
                requested: tree_index,
                available,
            })
    }
}

fn tree_structure_summary(
    tree: ir::TreeDefinition,
) -> Result<TreeStructureSummary, IntrospectionError> {
    match tree {
        ir::TreeDefinition::NodeTree {
            root_node_id,
            nodes,
            ..
        } => {
            let node_map = nodes
                .iter()
                .cloned()
                .map(|node| match &node {
                    ir::NodeTreeNode::Leaf { node_id, .. }
                    | ir::NodeTreeNode::BinaryBranch { node_id, .. }
                    | ir::NodeTreeNode::MultiwayBranch { node_id, .. } => (*node_id, node),
                })
                .collect::<BTreeMap<_, _>>();
            let mut leaf_depths = Vec::new();
            collect_leaf_depths(&node_map, root_node_id, &mut leaf_depths)?;
            let internal_node_count = nodes
                .iter()
                .filter(|node| !matches!(node, ir::NodeTreeNode::Leaf { .. }))
                .count();
            let leaf_count = leaf_depths.len();
            let shortest_path = *leaf_depths.iter().min().unwrap_or(&0);
            let longest_path = *leaf_depths.iter().max().unwrap_or(&0);
            let average_path = if leaf_depths.is_empty() {
                0.0
            } else {
                leaf_depths.iter().sum::<usize>() as f64 / leaf_depths.len() as f64
            };
            Ok(TreeStructureSummary {
                representation: "node_tree".to_string(),
                node_count: internal_node_count + leaf_count,
                internal_node_count,
                leaf_count,
                actual_depth: longest_path,
                shortest_path,
                longest_path,
                average_path,
            })
        }
        ir::TreeDefinition::ObliviousLevels { depth, leaves, .. } => Ok(TreeStructureSummary {
            representation: "oblivious_levels".to_string(),
            node_count: ((1usize << depth) - 1) + leaves.len(),
            internal_node_count: (1usize << depth) - 1,
            leaf_count: leaves.len(),
            actual_depth: depth,
            shortest_path: depth,
            longest_path: depth,
            average_path: depth as f64,
        }),
    }
}

fn collect_leaf_depths(
    nodes: &BTreeMap<usize, ir::NodeTreeNode>,
    node_id: usize,
    output: &mut Vec<usize>,
) -> Result<(), IntrospectionError> {
    match nodes
        .get(&node_id)
        .ok_or(IntrospectionError::NodeIndexOutOfBounds {
            requested: node_id,
            available: nodes.len(),
        })? {
        ir::NodeTreeNode::Leaf { depth, .. } => output.push(*depth),
        ir::NodeTreeNode::BinaryBranch {
            depth: _, children, ..
        } => {
            collect_leaf_depths(nodes, children.left, output)?;
            collect_leaf_depths(nodes, children.right, output)?;
        }
        ir::NodeTreeNode::MultiwayBranch {
            depth,
            branches,
            unmatched_leaf: _,
            ..
        } => {
            output.push(depth + 1);
            for branch in branches {
                collect_leaf_depths(nodes, branch.child, output)?;
            }
        }
    }
    Ok(())
}

fn prediction_value_stats(
    tree: ir::TreeDefinition,
) -> Result<PredictionValueStats, IntrospectionError> {
    let predictions = match tree {
        ir::TreeDefinition::NodeTree { nodes, .. } => nodes
            .into_iter()
            .flat_map(|node| match node {
                ir::NodeTreeNode::Leaf { leaf, .. } => vec![leaf_payload_value(&leaf)],
                ir::NodeTreeNode::MultiwayBranch { unmatched_leaf, .. } => {
                    vec![leaf_payload_value(&unmatched_leaf)]
                }
                ir::NodeTreeNode::BinaryBranch { .. } => Vec::new(),
            })
            .collect::<Vec<_>>(),
        ir::TreeDefinition::ObliviousLevels { leaves, .. } => leaves
            .into_iter()
            .map(|leaf| leaf_payload_value(&leaf.leaf))
            .collect::<Vec<_>>(),
    };

    let count = predictions.len();
    let min = predictions
        .iter()
        .copied()
        .min_by(f64::total_cmp)
        .unwrap_or(0.0);
    let max = predictions
        .iter()
        .copied()
        .max_by(f64::total_cmp)
        .unwrap_or(0.0);
    let mean = if count == 0 {
        0.0
    } else {
        predictions.iter().sum::<f64>() / count as f64
    };
    let std_dev = if count == 0 {
        0.0
    } else {
        let variance = predictions
            .iter()
            .map(|value| (*value - mean).powi(2))
            .sum::<f64>()
            / count as f64;
        variance.sqrt()
    };
    let mut histogram = BTreeMap::<String, usize>::new();
    for prediction in &predictions {
        *histogram.entry(prediction.to_string()).or_insert(0) += 1;
    }
    let histogram = histogram
        .into_iter()
        .map(|(prediction, count)| PredictionHistogramEntry {
            prediction: prediction
                .parse::<f64>()
                .expect("histogram keys are numeric"),
            count,
        })
        .collect::<Vec<_>>();

    Ok(PredictionValueStats {
        count,
        unique_count: histogram.len(),
        min,
        max,
        mean,
        std_dev,
        histogram,
    })
}

fn leaf_payload_value(leaf: &ir::LeafPayload) -> f64 {
    match leaf {
        ir::LeafPayload::RegressionValue { value } => *value,
        ir::LeafPayload::ClassIndex { class_value, .. } => *class_value,
    }
}

#[cfg(feature = "polars")]
fn polars_named_columns(df: &DataFrame) -> Result<BTreeMap<String, Vec<f64>>, PredictError> {
    df.get_columns()
        .iter()
        .map(|column| {
            let name = column.name().to_string();
            Ok((name, polars_column_values(column)?))
        })
        .collect()
}

#[cfg(feature = "polars")]
fn polars_column_values(column: &Column) -> Result<Vec<f64>, PredictError> {
    let name = column.name().to_string();
    let series = column.as_materialized_series();
    match series.dtype() {
        DataType::Boolean => series
            .bool()?
            .into_iter()
            .enumerate()
            .map(|(row_index, value)| {
                value
                    .map(|value| f64::from(u8::from(value)))
                    .ok_or_else(|| PredictError::NullValue {
                        feature: name.clone(),
                        row_index,
                    })
            })
            .collect(),
        DataType::Float64 => series
            .f64()?
            .into_iter()
            .enumerate()
            .map(|(row_index, value)| {
                value.ok_or_else(|| PredictError::NullValue {
                    feature: name.clone(),
                    row_index,
                })
            })
            .collect(),
        DataType::Float32
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64 => {
            let casted = series.cast(&DataType::Float64)?;
            casted
                .f64()?
                .into_iter()
                .enumerate()
                .map(|(row_index, value)| {
                    value.ok_or_else(|| PredictError::NullValue {
                        feature: name.clone(),
                        row_index,
                    })
                })
                .collect()
        }
        dtype => Err(PredictError::UnsupportedFeatureType {
            feature: name,
            dtype: dtype.to_string(),
        }),
    }
}

#[cfg(test)]
mod tests;
