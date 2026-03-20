use crate::{
    Criterion, GradientBoostedTrees, Model, Parallelism, RandomForest, Task, TrainAlgorithm,
    TrainConfig, TrainError, TreeType, tree,
};
use forestfire_data::TableAccess;
use rayon::ThreadPoolBuilder;

pub fn train(train_set: &dyn TableAccess, config: TrainConfig) -> Result<Model, TrainError> {
    let criterion = resolve_criterion(
        config.algorithm,
        config.task,
        config.tree_type,
        config.criterion,
    )?;
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

    run_with_parallelism(parallelism, || match config.algorithm {
        TrainAlgorithm::Dt => train_single_model(
            train_set,
            config.task,
            config.tree_type,
            criterion,
            parallelism,
            max_depth,
            min_samples_split,
            min_samples_leaf,
        ),
        TrainAlgorithm::Rf => train_random_forest(
            train_set,
            config.task,
            config.tree_type,
            criterion,
            parallelism,
            config.n_trees.unwrap_or(1000),
            max_depth,
            min_samples_split,
            min_samples_leaf,
            config.max_features,
            config.seed,
            config.compute_oob,
        ),
        TrainAlgorithm::Gbm => train_gradient_boosting(
            train_set,
            TrainConfig {
                criterion,
                ..config
            },
            parallelism,
        ),
    })
}

pub(crate) fn train_single_model(
    train_set: &dyn TableAccess,
    task: Task,
    tree_type: TreeType,
    criterion: Criterion,
    parallelism: Parallelism,
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
) -> Result<Model, TrainError> {
    train_single_model_with_feature_subset(
        train_set,
        task,
        tree_type,
        criterion,
        parallelism,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        None,
        0,
    )
}

pub(crate) fn train_single_model_with_feature_subset(
    train_set: &dyn TableAccess,
    task: Task,
    tree_type: TreeType,
    criterion: Criterion,
    parallelism: Parallelism,
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_features: Option<usize>,
    random_seed: u64,
) -> Result<Model, TrainError> {
    let classifier_options = tree::classifier::DecisionTreeOptions {
        max_depth,
        min_samples_split,
        min_samples_leaf,
        max_features,
        random_seed,
        ..tree::classifier::DecisionTreeOptions::default()
    };
    let regressor_options = tree::regressor::RegressionTreeOptions {
        max_depth,
        min_samples_split,
        min_samples_leaf,
        max_features,
        random_seed,
        ..tree::regressor::RegressionTreeOptions::default()
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
    task: Task,
    tree_type: TreeType,
    criterion: Criterion,
    parallelism: Parallelism,
    n_trees: usize,
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_features: crate::MaxFeatures,
    seed: Option<u64>,
    compute_oob: bool,
) -> Result<Model, TrainError> {
    RandomForest::train(
        train_set,
        TrainConfig {
            algorithm: TrainAlgorithm::Rf,
            task,
            tree_type,
            criterion,
            max_depth: Some(max_depth),
            min_samples_split: Some(min_samples_split),
            min_samples_leaf: Some(min_samples_leaf),
            physical_cores: None,
            n_trees: Some(n_trees),
            max_features,
            seed,
            compute_oob,
            learning_rate: None,
            bootstrap: false,
            top_gradient_fraction: None,
            other_gradient_fraction: None,
        },
        criterion,
        parallelism,
    )
    .map(Model::RandomForest)
}

fn train_gradient_boosting(
    train_set: &dyn TableAccess,
    config: TrainConfig,
    parallelism: Parallelism,
) -> Result<Model, TrainError> {
    GradientBoostedTrees::train(train_set, config, parallelism)
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
