use crate::{
    Criterion, Model, Parallelism, Task, TrainAlgorithm, TrainConfig, TrainError, TreeType, tree,
};
use forestfire_data::TableAccess;
use rayon::ThreadPoolBuilder;

pub fn train(train_set: &dyn TableAccess, config: TrainConfig) -> Result<Model, TrainError> {
    let criterion = resolve_criterion(config.task, config.tree_type, config.criterion)?;
    let parallelism = resolve_parallelism(config.physical_cores)?;

    run_with_parallelism(parallelism, || {
        match (config.algorithm, config.task, config.tree_type, criterion) {
            (TrainAlgorithm::Dt, Task::Regression, TreeType::TargetMean, Criterion::Mean)
            | (TrainAlgorithm::Dt, Task::Regression, TreeType::TargetMean, Criterion::Median) => {
                tree::mean_tree::train_target_mean_with_criterion(train_set, criterion)
                    .map(Model::TargetMean)
                    .map_err(TrainError::Mean)
            }
            (TrainAlgorithm::Dt, Task::Classification, TreeType::Id3, Criterion::Gini)
            | (TrainAlgorithm::Dt, Task::Classification, TreeType::Id3, Criterion::Entropy) => {
                tree::classifier::train_id3_with_criterion_and_parallelism(
                    train_set,
                    criterion,
                    parallelism,
                )
                .map(Model::DecisionTreeClassifier)
                .map_err(TrainError::DecisionTree)
            }
            (TrainAlgorithm::Dt, Task::Classification, TreeType::C45, Criterion::Gini)
            | (TrainAlgorithm::Dt, Task::Classification, TreeType::C45, Criterion::Entropy) => {
                tree::classifier::train_c45_with_criterion_and_parallelism(
                    train_set,
                    criterion,
                    parallelism,
                )
                .map(Model::DecisionTreeClassifier)
                .map_err(TrainError::DecisionTree)
            }
            (TrainAlgorithm::Dt, Task::Classification, TreeType::Cart, Criterion::Gini)
            | (TrainAlgorithm::Dt, Task::Classification, TreeType::Cart, Criterion::Entropy) => {
                tree::classifier::train_cart_with_criterion_and_parallelism(
                    train_set,
                    criterion,
                    parallelism,
                )
                .map(Model::DecisionTreeClassifier)
                .map_err(TrainError::DecisionTree)
            }
            (TrainAlgorithm::Dt, Task::Classification, TreeType::Randomized, Criterion::Gini)
            | (
                TrainAlgorithm::Dt,
                Task::Classification,
                TreeType::Randomized,
                Criterion::Entropy,
            ) => tree::classifier::train_randomized_with_criterion_and_parallelism(
                train_set,
                criterion,
                parallelism,
            )
            .map(Model::DecisionTreeClassifier)
            .map_err(TrainError::DecisionTree),
            (TrainAlgorithm::Dt, Task::Classification, TreeType::Oblivious, Criterion::Gini)
            | (TrainAlgorithm::Dt, Task::Classification, TreeType::Oblivious, Criterion::Entropy) => {
                tree::classifier::train_oblivious_with_criterion_and_parallelism(
                    train_set,
                    criterion,
                    parallelism,
                )
                .map(Model::DecisionTreeClassifier)
                .map_err(TrainError::DecisionTree)
            }
            (TrainAlgorithm::Dt, Task::Regression, TreeType::Cart, Criterion::Mean)
            | (TrainAlgorithm::Dt, Task::Regression, TreeType::Cart, Criterion::Median) => {
                tree::regressor::train_cart_regressor_with_criterion_and_parallelism(
                    train_set,
                    criterion,
                    parallelism,
                )
                .map(Model::DecisionTreeRegressor)
                .map_err(TrainError::RegressionTree)
            }
            (TrainAlgorithm::Dt, Task::Regression, TreeType::Randomized, Criterion::Mean)
            | (TrainAlgorithm::Dt, Task::Regression, TreeType::Randomized, Criterion::Median) => {
                tree::regressor::train_randomized_regressor_with_criterion_and_parallelism(
                    train_set,
                    criterion,
                    parallelism,
                )
                .map(Model::DecisionTreeRegressor)
                .map_err(TrainError::RegressionTree)
            }
            (TrainAlgorithm::Dt, Task::Regression, TreeType::Oblivious, Criterion::Mean)
            | (TrainAlgorithm::Dt, Task::Regression, TreeType::Oblivious, Criterion::Median) => {
                tree::regressor::train_oblivious_regressor_with_criterion_and_parallelism(
                    train_set,
                    criterion,
                    parallelism,
                )
                .map(Model::DecisionTreeRegressor)
                .map_err(TrainError::RegressionTree)
            }
            (TrainAlgorithm::Dt, task, tree_type, criterion) => {
                Err(TrainError::UnsupportedConfiguration {
                    task,
                    tree_type,
                    criterion,
                })
            }
        }
    })
}

fn resolve_criterion(
    task: Task,
    tree_type: TreeType,
    criterion: Criterion,
) -> Result<Criterion, TrainError> {
    let resolved = match (task, tree_type, criterion) {
        (Task::Regression, TreeType::TargetMean, Criterion::Auto) => Criterion::Mean,
        (Task::Regression, TreeType::TargetMean, Criterion::Mean | Criterion::Median) => criterion,
        (
            Task::Regression,
            TreeType::Cart | TreeType::Randomized | TreeType::Oblivious,
            Criterion::Auto,
        ) => Criterion::Mean,
        (
            Task::Regression,
            TreeType::Cart | TreeType::Randomized | TreeType::Oblivious,
            Criterion::Mean | Criterion::Median,
        ) => criterion,
        (Task::Classification, TreeType::Id3 | TreeType::C45, Criterion::Auto) => {
            Criterion::Entropy
        }
        (
            Task::Classification,
            TreeType::Id3 | TreeType::C45,
            Criterion::Gini | Criterion::Entropy,
        ) => criterion,
        (
            Task::Classification,
            TreeType::Cart | TreeType::Randomized | TreeType::Oblivious,
            Criterion::Auto,
        ) => Criterion::Gini,
        (
            Task::Classification,
            TreeType::Cart | TreeType::Randomized | TreeType::Oblivious,
            Criterion::Gini | Criterion::Entropy,
        ) => criterion,
        (task, tree_type, criterion) => {
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
