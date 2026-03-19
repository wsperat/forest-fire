use forestfire_data::DenseTable;
use std::error::Error;
use std::fmt::{Display, Formatter};

pub mod tree;

pub use tree::classifier::DecisionTreeAlgorithm;
pub use tree::classifier::DecisionTreeClassifier;
pub use tree::classifier::DecisionTreeError;
pub use tree::classifier::DecisionTreeOptions;
pub use tree::classifier::train_c45;
pub use tree::classifier::train_cart;
pub use tree::classifier::train_id3;
pub use tree::classifier::train_oblivious;
pub use tree::mean_tree::ModelError;
pub use tree::mean_tree::TargetMeanTree;
pub use tree::mean_tree::train_target_mean;
pub use tree::regressor::DecisionTreeRegressor;
pub use tree::regressor::RegressionTreeAlgorithm;
pub use tree::regressor::RegressionTreeError;
pub use tree::regressor::RegressionTreeOptions;
pub use tree::regressor::train_cart_regressor;
pub use tree::regressor::train_oblivious_regressor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainAlgorithm {
    Dt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Task {
    Regression,
    Classification,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreeType {
    TargetMean,
    Id3,
    C45,
    Cart,
    Oblivious,
}

#[derive(Debug, Clone, Copy)]
pub struct TrainConfig {
    pub algorithm: TrainAlgorithm,
    pub task: Task,
    pub tree_type: TreeType,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Regression,
            tree_type: TreeType::TargetMean,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Model {
    TargetMean(TargetMeanTree),
    DecisionTreeClassifier(DecisionTreeClassifier),
    DecisionTreeRegressor(DecisionTreeRegressor),
}

#[derive(Debug)]
pub enum TrainError {
    Mean(ModelError),
    DecisionTree(DecisionTreeError),
    RegressionTree(RegressionTreeError),
    UnsupportedConfiguration { task: Task, tree_type: TreeType },
}

impl Display for TrainError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            TrainError::Mean(err) => err.fmt(f),
            TrainError::DecisionTree(err) => err.fmt(f),
            TrainError::RegressionTree(err) => err.fmt(f),
            TrainError::UnsupportedConfiguration { task, tree_type } => write!(
                f,
                "Unsupported training configuration: task={:?}, tree_type={:?}.",
                task, tree_type
            ),
        }
    }
}

impl Error for TrainError {}

pub fn train(train_set: &DenseTable, config: TrainConfig) -> Result<Model, TrainError> {
    match (config.algorithm, config.task, config.tree_type) {
        (TrainAlgorithm::Dt, Task::Regression, TreeType::TargetMean) => {
            train_target_mean(train_set)
                .map(Model::TargetMean)
                .map_err(TrainError::Mean)
        }
        (TrainAlgorithm::Dt, Task::Classification, TreeType::Id3) => train_id3(train_set)
            .map(Model::DecisionTreeClassifier)
            .map_err(TrainError::DecisionTree),
        (TrainAlgorithm::Dt, Task::Classification, TreeType::C45) => train_c45(train_set)
            .map(Model::DecisionTreeClassifier)
            .map_err(TrainError::DecisionTree),
        (TrainAlgorithm::Dt, Task::Classification, TreeType::Cart) => train_cart(train_set)
            .map(Model::DecisionTreeClassifier)
            .map_err(TrainError::DecisionTree),
        (TrainAlgorithm::Dt, Task::Classification, TreeType::Oblivious) => {
            train_oblivious(train_set)
                .map(Model::DecisionTreeClassifier)
                .map_err(TrainError::DecisionTree)
        }
        (TrainAlgorithm::Dt, Task::Regression, TreeType::Cart) => train_cart_regressor(train_set)
            .map(Model::DecisionTreeRegressor)
            .map_err(TrainError::RegressionTree),
        (TrainAlgorithm::Dt, Task::Regression, TreeType::Oblivious) => {
            train_oblivious_regressor(train_set)
                .map(Model::DecisionTreeRegressor)
                .map_err(TrainError::RegressionTree)
        }
        (TrainAlgorithm::Dt, task, tree_type) => {
            Err(TrainError::UnsupportedConfiguration { task, tree_type })
        }
    }
}

impl Model {
    pub fn predict_table(&self, table: &DenseTable) -> Vec<f64> {
        match self {
            Model::TargetMean(model) => model.predict_table(table),
            Model::DecisionTreeClassifier(model) => model.predict_table(table),
            Model::DecisionTreeRegressor(model) => model.predict_table(table),
        }
    }

    pub fn algorithm(&self) -> TrainAlgorithm {
        match self {
            Model::TargetMean(_)
            | Model::DecisionTreeClassifier(_)
            | Model::DecisionTreeRegressor(_) => TrainAlgorithm::Dt,
        }
    }

    pub fn task(&self) -> Task {
        match self {
            Model::TargetMean(_) | Model::DecisionTreeRegressor(_) => Task::Regression,
            Model::DecisionTreeClassifier(_) => Task::Classification,
        }
    }

    pub fn tree_type(&self) -> TreeType {
        match self {
            Model::TargetMean(_) => TreeType::TargetMean,
            Model::DecisionTreeClassifier(model) => match model.algorithm() {
                DecisionTreeAlgorithm::Id3 => TreeType::Id3,
                DecisionTreeAlgorithm::C45 => TreeType::C45,
                DecisionTreeAlgorithm::Cart => TreeType::Cart,
                DecisionTreeAlgorithm::Oblivious => TreeType::Oblivious,
            },
            Model::DecisionTreeRegressor(model) => match model.algorithm() {
                RegressionTreeAlgorithm::Cart => TreeType::Cart,
                RegressionTreeAlgorithm::Oblivious => TreeType::Oblivious,
            },
        }
    }

    pub fn mean_value(&self) -> Option<f64> {
        match self {
            Model::TargetMean(model) => Some(model.mean),
            Model::DecisionTreeClassifier(_) | Model::DecisionTreeRegressor(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unified_train_dispatches_regression_cart() {
        let table = DenseTable::new(
            vec![
                vec![0.0],
                vec![1.0],
                vec![2.0],
                vec![3.0],
                vec![4.0],
                vec![5.0],
            ],
            vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0],
        )
        .unwrap();

        let model = train(
            &table,
            TrainConfig {
                algorithm: TrainAlgorithm::Dt,
                task: Task::Regression,
                tree_type: TreeType::Cart,
            },
        )
        .unwrap();

        assert!(matches!(model, Model::DecisionTreeRegressor(_)));
        assert_eq!(model.task(), Task::Regression);
        assert_eq!(model.tree_type(), TreeType::Cart);
    }

    #[test]
    fn unified_train_rejects_unsupported_task_tree_pair() {
        let table = DenseTable::new(vec![vec![0.0], vec![1.0]], vec![0.0, 1.0]).unwrap();

        let err = train(
            &table,
            TrainConfig {
                algorithm: TrainAlgorithm::Dt,
                task: Task::Regression,
                tree_type: TreeType::Id3,
            },
        )
        .unwrap_err();

        assert!(matches!(
            err,
            TrainError::UnsupportedConfiguration {
                task: Task::Regression,
                tree_type: TreeType::Id3,
            }
        ));
    }
}
