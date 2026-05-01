use forestfire_data::NumericBins;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

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
pub enum SplitStrategy {
    /// Use ordinary one-feature threshold or boolean splits.
    AxisAligned,
    /// Use experimental sparse linear-combination splits where supported.
    Oblique,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuilderStrategy {
    /// Rank splits by immediate local gain only.
    Greedy,
    /// Rank splits by a finite lookahead horizon.
    Lookahead,
    /// Rank splits by a width-limited continuation search.
    Beam,
    /// Rank splits by the exact best subtree objective within a shallow horizon.
    Optimal,
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CanaryFilter {
    /// Require the chosen real feature to appear within the top `n` scored splits.
    TopN(usize),
    /// Require the chosen real feature to appear within the top fraction of scored splits.
    TopFraction(f64),
}

impl Default for CanaryFilter {
    fn default() -> Self {
        Self::TopN(1)
    }
}

impl CanaryFilter {
    pub(crate) fn selection_size(self, candidate_count: usize) -> usize {
        if candidate_count == 0 {
            return 0;
        }

        match self {
            Self::TopN(count) => count.clamp(1, candidate_count),
            Self::TopFraction(fraction) => {
                ((fraction * candidate_count as f64).ceil() as usize).clamp(1, candidate_count)
            }
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
        missing_bin: u16,
    },
    /// Binary features do not require numeric bin boundaries.
    Binary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingValueStrategy {
    Heuristic,
    Optimal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MissingValueStrategyConfig {
    Global(MissingValueStrategy),
    PerFeature(BTreeMap<usize, MissingValueStrategy>),
}

impl MissingValueStrategyConfig {
    pub fn heuristic() -> Self {
        Self::Global(MissingValueStrategy::Heuristic)
    }

    pub fn optimal() -> Self {
        Self::Global(MissingValueStrategy::Optimal)
    }
}

/// Unified training configuration shared by the Rust and Python entry points.
///
/// The crate keeps one normalized config type so the binding layer only has to
/// perform input validation and type conversion; all semantic decisions happen
/// from this one structure downward.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// High-level training family.
    pub algorithm: TrainAlgorithm,
    /// Regression or classification.
    pub task: Task,
    /// Tree learner used by the selected algorithm family.
    pub tree_type: TreeType,
    /// Split family used by binary tree learners.
    pub split_strategy: SplitStrategy,
    /// Tree-construction strategy used to rank candidate splits.
    pub builder: BuilderStrategy,
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
    /// Window used when skipping canaries during split selection.
    pub canary_filter: CanaryFilter,
    /// Optional retry window for the first boosting stage when the default
    /// canary filter blocks the root.
    ///
    /// `Some(CanaryFilter::TopN(1))` preserves the strict default behavior.
    pub boosting_first_stage_retry_filter: Option<CanaryFilter>,
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
    /// Strategy used to evaluate missing-value routing during split search.
    pub missing_value_strategy: MissingValueStrategyConfig,
    /// Optional numeric histogram bin configuration for training-time split search.
    ///
    /// `None` preserves the incoming table's existing numeric bins. `Some(...)`
    /// rebuilds the numeric training view at the requested resolution before
    /// fitting, while leaving the caller's source table unchanged.
    pub histogram_bins: Option<NumericBins>,
    /// Number of tree levels used when ranking split candidates.
    ///
    /// A value of `1` preserves the current greedy behavior. Larger values
    /// evaluate candidate splits by also considering recursively chosen
    /// descendants up to the requested horizon.
    pub lookahead_depth: usize,
    /// Number of highest immediate-gain candidates to rescore with lookahead.
    pub lookahead_top_k: usize,
    /// Weight applied to future split value when lookahead rescoring is enabled.
    pub lookahead_weight: f64,
    /// Number of continuation candidates kept alive at each lookahead step for beam search.
    pub beam_width: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Regression,
            tree_type: TreeType::Cart,
            split_strategy: SplitStrategy::AxisAligned,
            builder: BuilderStrategy::Greedy,
            criterion: Criterion::Auto,
            max_depth: None,
            min_samples_split: None,
            min_samples_leaf: None,
            physical_cores: None,
            n_trees: None,
            max_features: MaxFeatures::Auto,
            seed: None,
            canary_filter: CanaryFilter::default(),
            boosting_first_stage_retry_filter: Some(CanaryFilter::TopN(1)),
            compute_oob: false,
            learning_rate: None,
            bootstrap: false,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
            missing_value_strategy: MissingValueStrategyConfig::heuristic(),
            histogram_bins: None,
            lookahead_depth: 1,
            lookahead_top_k: 8,
            lookahead_weight: 1.0,
            beam_width: 4,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Parallelism {
    pub(crate) thread_count: usize,
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
