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
    numeric_missing_bin,
};
#[cfg(feature = "polars")]
use polars::prelude::{Column, DataFrame, DataType, IdxSize, LazyFrame};
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

mod boosting;
mod bootstrap;
pub mod categorical;
mod compiled_artifact;
mod config;
mod errors;
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
pub use categorical::CategoricalConfig;
pub use categorical::CategoricalError;
pub use categorical::CategoricalModel;
pub use categorical::CategoricalOptimizedModel;
pub use categorical::CategoricalStrategy;
pub use categorical::CategoricalValue;
pub use compiled_artifact::CompiledArtifactError;
pub(crate) use config::Parallelism;
pub use config::{
    BuilderStrategy, CanaryFilter, Criterion, FeaturePreprocessing, InputFeatureKind, MaxFeatures,
    MissingValueStrategy, MissingValueStrategyConfig, NumericBinBoundary, SplitStrategy, Task,
    TrainAlgorithm, TrainConfig, TreeType,
};
pub use errors::{OptimizeError, PredictError, TrainError};
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
pub(crate) use optimized_runtime::OptimizedRuntime;
pub(crate) use optimized_runtime::resolve_inference_thread_count;
pub(crate) use runtime_planning::build_feature_index_map;
pub(crate) use runtime_planning::build_feature_projection;
pub(crate) use runtime_planning::model_used_feature_indices;
pub(crate) use runtime_planning::ordered_ensemble_indices;
pub(crate) use runtime_planning::remap_feature_index;

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

impl CanaryFilter {
    pub(crate) fn validate(self) -> Result<(), TrainError> {
        match self {
            Self::TopN(0) => Err(TrainError::InvalidCanaryFilterTopN(0)),
            Self::TopN(_) => Ok(()),
            Self::TopFraction(fraction)
                if !fraction.is_finite() || fraction <= 0.0 || fraction > 1.0 =>
            {
                Err(TrainError::InvalidCanaryFilterTopFraction(fraction))
            }
            Self::TopFraction(_) => Ok(()),
        }
    }
}

impl MissingValueStrategyConfig {
    pub fn resolve_for_feature_count(
        &self,
        feature_count: usize,
    ) -> Result<Vec<MissingValueStrategy>, TrainError> {
        match self {
            MissingValueStrategyConfig::Global(strategy) => Ok(vec![*strategy; feature_count]),
            MissingValueStrategyConfig::PerFeature(strategies) => {
                let mut resolved = vec![MissingValueStrategy::Heuristic; feature_count];
                for (&feature_index, &strategy) in strategies {
                    if feature_index >= feature_count {
                        return Err(TrainError::InvalidMissingValueStrategyFeature {
                            feature_index,
                            feature_count,
                        });
                    }
                    resolved[feature_index] = strategy;
                }
                Ok(resolved)
            }
        }
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
                    missing_bin: numeric_missing_bin(NumericBins::Fixed(table.numeric_bin_cap())),
                }
            }
        })
        .collect()
}

pub fn train(train_set: &dyn TableAccess, config: TrainConfig) -> Result<Model, TrainError> {
    training::train(train_set, config)
}

#[cfg(test)]
mod tests;
