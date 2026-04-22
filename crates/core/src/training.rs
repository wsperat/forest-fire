//! Training dispatch layer.
//!
//! This module is intentionally thin: it resolves defaults and validation once,
//! then forwards into the specialized tree/forest/boosting trainers. Keeping the
//! dispatch layer small makes it easier to reason about which behavior belongs
//! to "shared configuration policy" versus "algorithm-specific learning logic".

use crate::{
    Criterion, GradientBoostedTrees, Model, Parallelism, RandomForest, SplitStrategy, Task,
    TrainAlgorithm, TrainConfig, TrainError, TreeType, tree,
};
use forestfire_data::{
    BinnedColumnKind, NumericBins, TableAccess, numeric_bin_boundaries, numeric_missing_bin,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rayon::ThreadPoolBuilder;

enum TrainingTableRef<'a> {
    Borrowed(&'a dyn TableAccess),
    Rebinned(HistogramBinnedTable<'a>),
}

impl TableAccess for TrainingTableRef<'_> {
    fn n_rows(&self) -> usize {
        match self {
            Self::Borrowed(table) => table.n_rows(),
            Self::Rebinned(table) => table.n_rows(),
        }
    }

    fn n_features(&self) -> usize {
        match self {
            Self::Borrowed(table) => table.n_features(),
            Self::Rebinned(table) => table.n_features(),
        }
    }

    fn canaries(&self) -> usize {
        match self {
            Self::Borrowed(table) => table.canaries(),
            Self::Rebinned(table) => table.canaries(),
        }
    }

    fn numeric_bin_cap(&self) -> usize {
        match self {
            Self::Borrowed(table) => table.numeric_bin_cap(),
            Self::Rebinned(table) => table.numeric_bin_cap(),
        }
    }

    fn binned_feature_count(&self) -> usize {
        match self {
            Self::Borrowed(table) => table.binned_feature_count(),
            Self::Rebinned(table) => table.binned_feature_count(),
        }
    }

    fn feature_value(&self, feature_index: usize, row_index: usize) -> f64 {
        match self {
            Self::Borrowed(table) => table.feature_value(feature_index, row_index),
            Self::Rebinned(table) => table.feature_value(feature_index, row_index),
        }
    }

    fn is_missing(&self, feature_index: usize, row_index: usize) -> bool {
        match self {
            Self::Borrowed(table) => table.is_missing(feature_index, row_index),
            Self::Rebinned(table) => table.is_missing(feature_index, row_index),
        }
    }

    fn is_binary_feature(&self, index: usize) -> bool {
        match self {
            Self::Borrowed(table) => table.is_binary_feature(index),
            Self::Rebinned(table) => table.is_binary_feature(index),
        }
    }

    fn binned_value(&self, feature_index: usize, row_index: usize) -> u16 {
        match self {
            Self::Borrowed(table) => table.binned_value(feature_index, row_index),
            Self::Rebinned(table) => table.binned_value(feature_index, row_index),
        }
    }

    fn binned_boolean_value(&self, feature_index: usize, row_index: usize) -> Option<bool> {
        match self {
            Self::Borrowed(table) => table.binned_boolean_value(feature_index, row_index),
            Self::Rebinned(table) => table.binned_boolean_value(feature_index, row_index),
        }
    }

    fn binned_column_kind(&self, index: usize) -> BinnedColumnKind {
        match self {
            Self::Borrowed(table) => table.binned_column_kind(index),
            Self::Rebinned(table) => table.binned_column_kind(index),
        }
    }

    fn is_binary_binned_feature(&self, index: usize) -> bool {
        match self {
            Self::Borrowed(table) => table.is_binary_binned_feature(index),
            Self::Rebinned(table) => table.is_binary_binned_feature(index),
        }
    }

    fn target_value(&self, row_index: usize) -> f64 {
        match self {
            Self::Borrowed(table) => table.target_value(row_index),
            Self::Rebinned(table) => table.target_value(row_index),
        }
    }
}

struct HistogramBinnedTable<'a> {
    base: &'a dyn TableAccess,
    numeric_bins: NumericBins,
    rebinned_numeric_columns: Vec<Option<Vec<u16>>>,
}

impl<'a> HistogramBinnedTable<'a> {
    fn new(base: &'a dyn TableAccess, numeric_bins: NumericBins) -> Self {
        let mut rebinned_numeric_columns = vec![None; base.binned_feature_count()];

        for (feature_index, rebinned_column) in rebinned_numeric_columns
            .iter_mut()
            .enumerate()
            .take(base.n_features())
        {
            if base.is_binary_feature(feature_index) {
                continue;
            }
            let values = (0..base.n_rows())
                .map(|row_index| base.feature_value(feature_index, row_index))
                .collect::<Vec<_>>();
            *rebinned_column = Some(rebin_numeric_column(&values, numeric_bins));
        }

        for feature_index in base.n_features()..base.binned_feature_count() {
            let BinnedColumnKind::Canary {
                source_index,
                copy_index,
            } = base.binned_column_kind(feature_index)
            else {
                continue;
            };
            let Some(source_column) = rebinned_numeric_columns[source_index].as_ref() else {
                continue;
            };
            let mut shuffled = source_column.clone();
            shuffle_values(&mut shuffled, copy_index, source_index);
            rebinned_numeric_columns[feature_index] = Some(shuffled);
        }

        Self {
            base,
            numeric_bins,
            rebinned_numeric_columns,
        }
    }
}

impl TableAccess for HistogramBinnedTable<'_> {
    fn n_rows(&self) -> usize {
        self.base.n_rows()
    }

    fn n_features(&self) -> usize {
        self.base.n_features()
    }

    fn canaries(&self) -> usize {
        self.base.canaries()
    }

    fn numeric_bin_cap(&self) -> usize {
        self.numeric_bins.cap()
    }

    fn binned_feature_count(&self) -> usize {
        self.base.binned_feature_count()
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
        self.rebinned_numeric_columns[feature_index]
            .as_ref()
            .map_or_else(
                || self.base.binned_value(feature_index, row_index),
                |column| column[row_index],
            )
    }

    fn binned_boolean_value(&self, feature_index: usize, row_index: usize) -> Option<bool> {
        self.base.binned_boolean_value(feature_index, row_index)
    }

    fn binned_column_kind(&self, index: usize) -> BinnedColumnKind {
        self.base.binned_column_kind(index)
    }

    fn is_binary_binned_feature(&self, index: usize) -> bool {
        self.base.is_binary_binned_feature(index)
    }

    fn target_value(&self, row_index: usize) -> f64 {
        self.base.target_value(row_index)
    }
}

fn rebin_numeric_column(values: &[f64], numeric_bins: NumericBins) -> Vec<u16> {
    let missing_bin = numeric_missing_bin(numeric_bins);
    let boundaries = numeric_bin_boundaries(values, numeric_bins);
    values
        .iter()
        .map(|value| infer_numeric_bin(*value, &boundaries, missing_bin))
        .collect()
}

fn infer_numeric_bin(value: f64, boundaries: &[(u16, f64)], missing_bin: u16) -> u16 {
    if value.is_nan() {
        return missing_bin;
    }
    boundaries
        .iter()
        .find(|(_, upper_bound)| value <= *upper_bound)
        .map_or_else(
            || boundaries.last().map_or(0, |(bin, _)| *bin),
            |(bin, _)| *bin,
        )
}

fn shuffle_values<T>(values: &mut [T], copy_index: usize, source_index: usize) {
    let seed = 0xA11CE5EED_u64
        ^ ((copy_index as u64) << 32)
        ^ (source_index as u64)
        ^ ((values.len() as u64) << 16);
    let mut rng = StdRng::seed_from_u64(seed);
    values.shuffle(&mut rng);
}

pub fn train(train_set: &dyn TableAccess, config: TrainConfig) -> Result<Model, TrainError> {
    let train_table = config.histogram_bins.map_or_else(
        || TrainingTableRef::Borrowed(train_set),
        |numeric_bins| {
            TrainingTableRef::Rebinned(HistogramBinnedTable::new(train_set, numeric_bins))
        },
    );

    // Criterion resolution happens once here so the downstream trainers can
    // operate on explicit semantics instead of carrying `Auto` branches.
    let criterion = resolve_criterion(
        config.algorithm,
        config.task,
        config.tree_type,
        config.criterion,
    )?;
    validate_split_strategy(
        config.algorithm,
        config.task,
        config.tree_type,
        config.split_strategy,
    )?;
    let missing_value_strategies = config
        .missing_value_strategy
        .resolve_for_feature_count(train_table.binned_feature_count())?;
    let parallelism = resolve_parallelism(config.physical_cores)?;
    let max_depth = config.max_depth.unwrap_or(8);
    if max_depth == 0 {
        return Err(TrainError::InvalidMaxDepth(max_depth));
    }
    let min_samples_split = config.min_samples_split.unwrap_or(2);
    if min_samples_split == 0 {
        return Err(TrainError::InvalidMinSamplesSplit(min_samples_split));
    }
    let min_samples_leaf = config.min_samples_leaf.unwrap_or(1);
    if min_samples_leaf == 0 {
        return Err(TrainError::InvalidMinSamplesLeaf(min_samples_leaf));
    }
    config.canary_filter.validate()?;

    // Parallelism is installed around the whole training call so nested trainers
    // can use rayon consistently without rebuilding pools at every split.
    run_with_parallelism(parallelism, || match config.algorithm {
        TrainAlgorithm::Dt => train_single_model(
            &train_table,
            SingleModelConfig {
                task: config.task,
                tree_type: config.tree_type,
                split_strategy: config.split_strategy,
                criterion,
                parallelism,
                max_depth,
                min_samples_split,
                min_samples_leaf,
                missing_value_strategies: missing_value_strategies.clone(),
                canary_filter: config.canary_filter,
            },
        ),
        TrainAlgorithm::Rf => train_random_forest(
            &train_table,
            RandomForestConfig {
                task: config.task,
                tree_type: config.tree_type,
                split_strategy: config.split_strategy,
                criterion,
                parallelism,
                n_trees: config.n_trees.unwrap_or(1000),
                max_depth,
                min_samples_split,
                min_samples_leaf,
                missing_value_strategies: missing_value_strategies.clone(),
                max_features: config.max_features,
                seed: config.seed,
                compute_oob: config.compute_oob,
            },
        ),
        TrainAlgorithm::Gbm => train_gradient_boosting(
            &train_table,
            TrainConfig {
                criterion,
                ..config
            },
            parallelism,
            missing_value_strategies,
        ),
    })
}

pub(crate) struct SingleModelConfig {
    pub(crate) task: Task,
    pub(crate) tree_type: TreeType,
    pub(crate) split_strategy: SplitStrategy,
    pub(crate) criterion: Criterion,
    pub(crate) parallelism: Parallelism,
    pub(crate) max_depth: usize,
    pub(crate) min_samples_split: usize,
    pub(crate) min_samples_leaf: usize,
    pub(crate) missing_value_strategies: Vec<crate::MissingValueStrategy>,
    pub(crate) canary_filter: crate::CanaryFilter,
}

/// Internal single-tree config with optional per-node feature subsampling.
///
/// The public API exposes `max_features` at the algorithm level, but random
/// forests and boosting need to reuse the same tree trainers with additional
/// feature-subset control.
pub(crate) struct SingleModelFeatureSubsetConfig {
    pub(crate) base: SingleModelConfig,
    pub(crate) max_features: Option<usize>,
    pub(crate) random_seed: u64,
}

#[derive(Clone)]
pub(crate) struct RandomForestConfig {
    pub(crate) task: Task,
    pub(crate) tree_type: TreeType,
    pub(crate) split_strategy: SplitStrategy,
    pub(crate) criterion: Criterion,
    pub(crate) parallelism: Parallelism,
    pub(crate) n_trees: usize,
    pub(crate) max_depth: usize,
    pub(crate) min_samples_split: usize,
    pub(crate) min_samples_leaf: usize,
    pub(crate) missing_value_strategies: Vec<crate::MissingValueStrategy>,
    pub(crate) max_features: crate::MaxFeatures,
    pub(crate) seed: Option<u64>,
    pub(crate) compute_oob: bool,
}

pub(crate) fn train_single_model(
    train_set: &dyn TableAccess,
    config: SingleModelConfig,
) -> Result<Model, TrainError> {
    train_single_model_with_feature_subset(
        train_set,
        SingleModelFeatureSubsetConfig {
            base: config,
            max_features: None,
            random_seed: 0,
        },
    )
}

pub(crate) fn train_single_model_with_feature_subset(
    train_set: &dyn TableAccess,
    config: SingleModelFeatureSubsetConfig,
) -> Result<Model, TrainError> {
    let SingleModelFeatureSubsetConfig {
        base:
            SingleModelConfig {
                task,
                tree_type,
                split_strategy,
                criterion,
                parallelism,
                max_depth,
                min_samples_split,
                min_samples_leaf,
                missing_value_strategies,
                canary_filter,
            },
        max_features,
        random_seed,
    } = config;
    let classifier_options = tree::classifier::DecisionTreeOptions {
        max_depth,
        min_samples_split,
        min_samples_leaf,
        max_features,
        random_seed,
        missing_value_strategies: missing_value_strategies.clone(),
        canary_filter,
        split_strategy,
    };
    let regressor_options = tree::regressor::RegressionTreeOptions {
        max_depth,
        min_samples_split,
        min_samples_leaf,
        max_features,
        random_seed,
        missing_value_strategies,
        canary_filter,
        split_strategy,
    };

    match (task, tree_type, criterion) {
        (Task::Classification, TreeType::Id3, Criterion::Gini)
        | (Task::Classification, TreeType::Id3, Criterion::Entropy) => {
            tree::classifier::train_id3_with_criterion_parallelism_and_options(
                train_set,
                criterion,
                parallelism,
                classifier_options,
            )
            .map(Model::DecisionTreeClassifier)
            .map_err(TrainError::DecisionTree)
        }
        (Task::Classification, TreeType::C45, Criterion::Gini)
        | (Task::Classification, TreeType::C45, Criterion::Entropy) => {
            tree::classifier::train_c45_with_criterion_parallelism_and_options(
                train_set,
                criterion,
                parallelism,
                classifier_options,
            )
            .map(Model::DecisionTreeClassifier)
            .map_err(TrainError::DecisionTree)
        }
        (Task::Classification, TreeType::Cart, Criterion::Gini)
        | (Task::Classification, TreeType::Cart, Criterion::Entropy) => {
            tree::classifier::train_cart_with_criterion_parallelism_and_options(
                train_set,
                criterion,
                parallelism,
                classifier_options,
            )
            .map(Model::DecisionTreeClassifier)
            .map_err(TrainError::DecisionTree)
        }
        (Task::Classification, TreeType::Randomized, Criterion::Gini)
        | (Task::Classification, TreeType::Randomized, Criterion::Entropy) => {
            tree::classifier::train_randomized_with_criterion_parallelism_and_options(
                train_set,
                criterion,
                parallelism,
                classifier_options,
            )
            .map(Model::DecisionTreeClassifier)
            .map_err(TrainError::DecisionTree)
        }
        (Task::Classification, TreeType::Oblivious, Criterion::Gini)
        | (Task::Classification, TreeType::Oblivious, Criterion::Entropy) => {
            tree::classifier::train_oblivious_with_criterion_parallelism_and_options(
                train_set,
                criterion,
                parallelism,
                classifier_options,
            )
            .map(Model::DecisionTreeClassifier)
            .map_err(TrainError::DecisionTree)
        }
        (Task::Regression, TreeType::Cart, Criterion::Mean)
        | (Task::Regression, TreeType::Cart, Criterion::Median) => {
            tree::regressor::train_cart_regressor_with_criterion_parallelism_and_options(
                train_set,
                criterion,
                parallelism,
                regressor_options,
            )
            .map(Model::DecisionTreeRegressor)
            .map_err(TrainError::RegressionTree)
        }
        (Task::Regression, TreeType::Randomized, Criterion::Mean)
        | (Task::Regression, TreeType::Randomized, Criterion::Median) => {
            tree::regressor::train_randomized_regressor_with_criterion_parallelism_and_options(
                train_set,
                criterion,
                parallelism,
                regressor_options,
            )
            .map(Model::DecisionTreeRegressor)
            .map_err(TrainError::RegressionTree)
        }
        (Task::Regression, TreeType::Oblivious, Criterion::Mean)
        | (Task::Regression, TreeType::Oblivious, Criterion::Median) => {
            tree::regressor::train_oblivious_regressor_with_criterion_parallelism_and_options(
                train_set,
                criterion,
                parallelism,
                regressor_options,
            )
            .map(Model::DecisionTreeRegressor)
            .map_err(TrainError::RegressionTree)
        }
        (task, tree_type, criterion) => Err(TrainError::UnsupportedConfiguration {
            task,
            tree_type,
            criterion,
        }),
    }
}

fn train_random_forest(
    train_set: &dyn TableAccess,
    config: RandomForestConfig,
) -> Result<Model, TrainError> {
    RandomForest::train(
        train_set,
        config.clone(),
        config.criterion,
        config.parallelism,
    )
    .map(Model::RandomForest)
}

fn train_gradient_boosting(
    train_set: &dyn TableAccess,
    config: TrainConfig,
    parallelism: Parallelism,
    missing_value_strategies: Vec<crate::MissingValueStrategy>,
) -> Result<Model, TrainError> {
    GradientBoostedTrees::train_with_missing_value_strategies(
        train_set,
        config,
        parallelism,
        missing_value_strategies,
    )
    .map(Model::GradientBoostedTrees)
    .map_err(TrainError::Boosting)
}

fn resolve_criterion(
    algorithm: TrainAlgorithm,
    task: Task,
    tree_type: TreeType,
    criterion: Criterion,
) -> Result<Criterion, TrainError> {
    let resolved = match (algorithm, task, tree_type, criterion) {
        (
            TrainAlgorithm::Gbm,
            Task::Regression | Task::Classification,
            TreeType::Cart | TreeType::Randomized | TreeType::Oblivious,
            Criterion::Auto,
        ) => Criterion::SecondOrder,
        (
            TrainAlgorithm::Dt | TrainAlgorithm::Rf,
            Task::Regression,
            TreeType::Cart | TreeType::Randomized | TreeType::Oblivious,
            Criterion::Auto,
        ) => Criterion::Mean,
        (
            TrainAlgorithm::Dt | TrainAlgorithm::Rf,
            Task::Regression,
            TreeType::Cart | TreeType::Randomized | TreeType::Oblivious,
            Criterion::Mean | Criterion::Median,
        ) => criterion,
        (
            TrainAlgorithm::Dt | TrainAlgorithm::Rf,
            Task::Classification,
            TreeType::Id3 | TreeType::C45,
            Criterion::Auto,
        ) => Criterion::Entropy,
        (
            TrainAlgorithm::Dt | TrainAlgorithm::Rf,
            Task::Classification,
            TreeType::Id3 | TreeType::C45,
            Criterion::Gini | Criterion::Entropy,
        ) => criterion,
        (
            TrainAlgorithm::Dt | TrainAlgorithm::Rf,
            Task::Classification,
            TreeType::Cart | TreeType::Randomized | TreeType::Oblivious,
            Criterion::Auto,
        ) => Criterion::Gini,
        (
            TrainAlgorithm::Dt | TrainAlgorithm::Rf,
            Task::Classification,
            TreeType::Cart | TreeType::Randomized | TreeType::Oblivious,
            Criterion::Gini | Criterion::Entropy,
        ) => criterion,
        (_, task, tree_type, criterion) => {
            return Err(TrainError::UnsupportedConfiguration {
                task,
                tree_type,
                criterion,
            });
        }
    };

    Ok(resolved)
}

fn resolve_parallelism(physical_cores: Option<usize>) -> Result<Parallelism, TrainError> {
    let available = num_cpus::get_physical().max(1);
    let requested = physical_cores.unwrap_or(available);

    if requested == 0 {
        return Err(TrainError::InvalidPhysicalCoreCount {
            requested,
            available,
        });
    }

    Ok(Parallelism {
        thread_count: requested.min(available),
    })
}

fn validate_split_strategy(
    algorithm: TrainAlgorithm,
    task: Task,
    tree_type: TreeType,
    split_strategy: SplitStrategy,
) -> Result<(), TrainError> {
    if matches!(split_strategy, SplitStrategy::AxisAligned) {
        return Ok(());
    }

    if matches!(algorithm, TrainAlgorithm::Dt | TrainAlgorithm::Rf)
        && matches!(tree_type, TreeType::Cart | TreeType::Randomized)
        && matches!(task, Task::Regression | Task::Classification)
    {
        return Ok(());
    }

    Err(TrainError::UnsupportedSplitStrategy {
        algorithm,
        task,
        tree_type,
        split_strategy,
    })
}

fn run_with_parallelism<T, F>(parallelism: Parallelism, train_fn: F) -> Result<T, TrainError>
where
    T: Send,
    F: FnOnce() -> Result<T, TrainError> + Send,
{
    if !parallelism.enabled() {
        return train_fn();
    }

    ThreadPoolBuilder::new()
        .num_threads(parallelism.thread_count)
        .build()
        .map_err(|err| TrainError::ThreadPoolBuildFailed(err.to_string()))?
        .install(train_fn)
}
