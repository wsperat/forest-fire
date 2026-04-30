use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::boosting::BoostingError;
use crate::config::{Criterion, SplitStrategy, Task, TrainAlgorithm, TreeType};
use crate::tree::classifier::DecisionTreeError;
use crate::tree::regressor::RegressionTreeError;

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
    UnsupportedSplitStrategy {
        algorithm: TrainAlgorithm,
        task: Task,
        tree_type: TreeType,
        split_strategy: SplitStrategy,
    },
    InvalidMaxDepth(usize),
    InvalidMinSamplesSplit(usize),
    InvalidMinSamplesLeaf(usize),
    InvalidLookaheadDepth(usize),
    InvalidLookaheadTopK(usize),
    InvalidLookaheadWeight(f64),
    InvalidBeamWidth(usize),
    InvalidTreeCount(usize),
    InvalidMaxFeatures(usize),
    InvalidCanaryFilterTopN(usize),
    InvalidCanaryFilterTopFraction(f64),
    InvalidMissingValueStrategyFeature {
        feature_index: usize,
        feature_count: usize,
    },
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
            TrainError::UnsupportedSplitStrategy {
                algorithm,
                task,
                tree_type,
                split_strategy,
            } => write!(
                f,
                "Unsupported split strategy: algorithm={:?}, task={:?}, tree_type={:?}, split_strategy={:?}.",
                algorithm, task, tree_type, split_strategy
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
            TrainError::InvalidLookaheadDepth(value) => {
                write!(f, "lookahead_depth must be at least 1. Received {}.", value)
            }
            TrainError::InvalidLookaheadTopK(value) => {
                write!(f, "lookahead_top_k must be at least 1. Received {}.", value)
            }
            TrainError::InvalidLookaheadWeight(value) => {
                write!(
                    f,
                    "lookahead_weight must be finite and non-negative. Received {}.",
                    value
                )
            }
            TrainError::InvalidBeamWidth(value) => {
                write!(f, "beam_width must be at least 1. Received {}.", value)
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
            TrainError::InvalidCanaryFilterTopN(count) => {
                write!(
                    f,
                    "canary_filter top-n must be at least 1. Received {}.",
                    count
                )
            }
            TrainError::InvalidCanaryFilterTopFraction(fraction) => {
                write!(
                    f,
                    "canary_filter top fraction must be finite and in (0, 1]. Received {}.",
                    fraction
                )
            }
            TrainError::InvalidMissingValueStrategyFeature {
                feature_index,
                feature_count,
            } => write!(
                f,
                "missing_value_strategy references feature {}, but the training table only has {} features.",
                feature_index, feature_count
            ),
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
