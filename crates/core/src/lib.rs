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
pub use tree::mean_tree::ModelError;
pub use tree::mean_tree::TargetMeanTree;
pub use tree::mean_tree::train_target_mean;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainAlgorithm {
    Dt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreeType {
    TargetMean,
    Id3,
    C45,
    Cart,
}

#[derive(Debug, Clone, Copy)]
pub struct TrainConfig {
    pub algorithm: TrainAlgorithm,
    pub tree_type: TreeType,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            algorithm: TrainAlgorithm::Dt,
            tree_type: TreeType::TargetMean,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Model {
    TargetMean(TargetMeanTree),
    DecisionTree(DecisionTreeClassifier),
}

#[derive(Debug)]
pub enum TrainError {
    Mean(ModelError),
    DecisionTree(DecisionTreeError),
}

impl Display for TrainError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            TrainError::Mean(err) => err.fmt(f),
            TrainError::DecisionTree(err) => err.fmt(f),
        }
    }
}

impl Error for TrainError {}

pub fn train(train_set: &DenseTable, config: TrainConfig) -> Result<Model, TrainError> {
    match (config.algorithm, config.tree_type) {
        (TrainAlgorithm::Dt, TreeType::TargetMean) => train_target_mean(train_set)
            .map(Model::TargetMean)
            .map_err(TrainError::Mean),
        (TrainAlgorithm::Dt, TreeType::Id3) => train_id3(train_set)
            .map(Model::DecisionTree)
            .map_err(TrainError::DecisionTree),
        (TrainAlgorithm::Dt, TreeType::C45) => train_c45(train_set)
            .map(Model::DecisionTree)
            .map_err(TrainError::DecisionTree),
        (TrainAlgorithm::Dt, TreeType::Cart) => train_cart(train_set)
            .map(Model::DecisionTree)
            .map_err(TrainError::DecisionTree),
    }
}

impl Model {
    pub fn predict_table(&self, table: &DenseTable) -> Vec<f64> {
        match self {
            Model::TargetMean(model) => model.predict_table(table),
            Model::DecisionTree(model) => model.predict_table(table),
        }
    }

    pub fn algorithm(&self) -> TrainAlgorithm {
        match self {
            Model::TargetMean(_) | Model::DecisionTree(_) => TrainAlgorithm::Dt,
        }
    }

    pub fn tree_type(&self) -> TreeType {
        match self {
            Model::TargetMean(_) => TreeType::TargetMean,
            Model::DecisionTree(model) => match model.algorithm() {
                DecisionTreeAlgorithm::Id3 => TreeType::Id3,
                DecisionTreeAlgorithm::C45 => TreeType::C45,
                DecisionTreeAlgorithm::Cart => TreeType::Cart,
            },
        }
    }

    pub fn mean_value(&self) -> Option<f64> {
        match self {
            Model::TargetMean(model) => Some(model.mean),
            Model::DecisionTree(_) => None,
        }
    }
}
