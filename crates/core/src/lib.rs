use forestfire_data::{TableAccess, numeric_bin_boundaries};
use rayon::ThreadPoolBuilder;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt::{Display, Formatter};

pub mod ir;
pub mod tree;

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
pub enum Criterion {
    Auto,
    Gini,
    Entropy,
    Mean,
    Median,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputFeatureKind {
    Numeric,
    Binary,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NumericBinBoundary {
    pub bin: u16,
    pub upper_bound: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FeaturePreprocessing {
    Numeric {
        bin_boundaries: Vec<NumericBinBoundary>,
    },
    Binary,
}

#[derive(Debug, Clone, Copy)]
pub struct TrainConfig {
    pub algorithm: TrainAlgorithm,
    pub task: Task,
    pub tree_type: TreeType,
    pub criterion: Criterion,
    pub physical_cores: Option<usize>,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Regression,
            tree_type: TreeType::TargetMean,
            criterion: Criterion::Auto,
            physical_cores: None,
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
}

impl Display for TrainError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            TrainError::Mean(err) => err.fmt(f),
            TrainError::DecisionTree(err) => err.fmt(f),
            TrainError::RegressionTree(err) => err.fmt(f),
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
        }
    }
}

impl Error for TrainError {}

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
                    bin_boundaries: numeric_bin_boundaries(&values)
                        .into_iter()
                        .map(|(bin, upper_bound)| NumericBinBoundary { bin, upper_bound })
                        .collect(),
                }
            }
        })
        .collect()
}

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
        (Task::Regression, TreeType::Cart | TreeType::Oblivious, Criterion::Auto) => {
            Criterion::Mean
        }
        (
            Task::Regression,
            TreeType::Cart | TreeType::Oblivious,
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
        (Task::Classification, TreeType::Cart | TreeType::Oblivious, Criterion::Auto) => {
            Criterion::Gini
        }
        (
            Task::Classification,
            TreeType::Cart | TreeType::Oblivious,
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

impl Model {
    pub fn predict_table(&self, table: &dyn TableAccess) -> Vec<f64> {
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

    pub fn criterion(&self) -> Criterion {
        match self {
            Model::TargetMean(model) => model.criterion(),
            Model::DecisionTreeClassifier(model) => model.criterion(),
            Model::DecisionTreeRegressor(model) => model.criterion(),
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

    pub fn deserialize(serialized: &str) -> Result<Self, IrError> {
        let ir: ModelPackageIr =
            serde_json::from_str(serialized).map_err(|err| IrError::Json(err.to_string()))?;
        ir::model_from_ir(ir)
    }

    pub(crate) fn num_features(&self) -> usize {
        match self {
            Model::TargetMean(model) => model.num_features(),
            Model::DecisionTreeClassifier(model) => model.num_features(),
            Model::DecisionTreeRegressor(model) => model.num_features(),
        }
    }

    pub(crate) fn feature_preprocessing(&self) -> &[FeaturePreprocessing] {
        match self {
            Model::TargetMean(model) => model.feature_preprocessing(),
            Model::DecisionTreeClassifier(model) => model.feature_preprocessing(),
            Model::DecisionTreeRegressor(model) => model.feature_preprocessing(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forestfire_data::DenseTable;

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
                criterion: Criterion::Mean,
                physical_cores: Some(1),
            },
        )
        .unwrap();

        assert!(matches!(model, Model::DecisionTreeRegressor(_)));
        assert_eq!(model.task(), Task::Regression);
        assert_eq!(model.tree_type(), TreeType::Cart);
        assert_eq!(model.criterion(), Criterion::Mean);
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
                criterion: Criterion::Mean,
                physical_cores: Some(1),
            },
        )
        .unwrap_err();

        assert!(matches!(
            err,
            TrainError::UnsupportedConfiguration {
                task: Task::Regression,
                tree_type: TreeType::Id3,
                criterion: Criterion::Mean,
            }
        ));
    }

    #[test]
    fn unified_train_resolves_auto_criterion_across_supported_matrix() {
        let classification_table = DenseTable::with_canaries(
            vec![
                vec![0.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
            ],
            vec![0.0, 0.0, 0.0, 1.0],
            0,
        )
        .unwrap();
        let regression_table = DenseTable::with_canaries(
            vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]],
            vec![1.0, 3.0, 5.0, 7.0],
            0,
        )
        .unwrap();

        for (table, config, expected_criterion) in [
            (
                &regression_table,
                TrainConfig {
                    algorithm: TrainAlgorithm::Dt,
                    task: Task::Regression,
                    tree_type: TreeType::TargetMean,
                    criterion: Criterion::Auto,
                    physical_cores: Some(1),
                },
                Criterion::Mean,
            ),
            (
                &regression_table,
                TrainConfig {
                    algorithm: TrainAlgorithm::Dt,
                    task: Task::Regression,
                    tree_type: TreeType::Cart,
                    criterion: Criterion::Auto,
                    physical_cores: Some(1),
                },
                Criterion::Mean,
            ),
            (
                &regression_table,
                TrainConfig {
                    algorithm: TrainAlgorithm::Dt,
                    task: Task::Regression,
                    tree_type: TreeType::Oblivious,
                    criterion: Criterion::Auto,
                    physical_cores: Some(1),
                },
                Criterion::Mean,
            ),
            (
                &classification_table,
                TrainConfig {
                    algorithm: TrainAlgorithm::Dt,
                    task: Task::Classification,
                    tree_type: TreeType::Id3,
                    criterion: Criterion::Auto,
                    physical_cores: Some(1),
                },
                Criterion::Entropy,
            ),
            (
                &classification_table,
                TrainConfig {
                    algorithm: TrainAlgorithm::Dt,
                    task: Task::Classification,
                    tree_type: TreeType::C45,
                    criterion: Criterion::Auto,
                    physical_cores: Some(1),
                },
                Criterion::Entropy,
            ),
            (
                &classification_table,
                TrainConfig {
                    algorithm: TrainAlgorithm::Dt,
                    task: Task::Classification,
                    tree_type: TreeType::Cart,
                    criterion: Criterion::Auto,
                    physical_cores: Some(1),
                },
                Criterion::Gini,
            ),
            (
                &classification_table,
                TrainConfig {
                    algorithm: TrainAlgorithm::Dt,
                    task: Task::Classification,
                    tree_type: TreeType::Oblivious,
                    criterion: Criterion::Auto,
                    physical_cores: Some(1),
                },
                Criterion::Gini,
            ),
        ] {
            let model = train(table, config).unwrap();
            assert_eq!(model.criterion(), expected_criterion);
        }
    }

    #[test]
    fn unified_train_parallel_matches_single_core_across_supported_tree_types() {
        let classification_table = DenseTable::with_canaries(
            vec![
                vec![0.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
                vec![0.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
            ],
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            0,
        )
        .unwrap();
        let regression_table = DenseTable::with_canaries(
            vec![
                vec![0.0],
                vec![1.0],
                vec![2.0],
                vec![3.0],
                vec![4.0],
                vec![5.0],
            ],
            vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0],
            0,
        )
        .unwrap();

        for config in [
            TrainConfig {
                algorithm: TrainAlgorithm::Dt,
                task: Task::Regression,
                tree_type: TreeType::TargetMean,
                criterion: Criterion::Mean,
                physical_cores: Some(1),
            },
            TrainConfig {
                algorithm: TrainAlgorithm::Dt,
                task: Task::Regression,
                tree_type: TreeType::Cart,
                criterion: Criterion::Mean,
                physical_cores: Some(1),
            },
            TrainConfig {
                algorithm: TrainAlgorithm::Dt,
                task: Task::Regression,
                tree_type: TreeType::Oblivious,
                criterion: Criterion::Mean,
                physical_cores: Some(1),
            },
        ] {
            let single_core = train(&regression_table, config).unwrap();
            let parallel = train(
                &regression_table,
                TrainConfig {
                    physical_cores: Some(2),
                    ..config
                },
            )
            .unwrap();

            assert_eq!(
                single_core.predict_table(&regression_table),
                parallel.predict_table(&regression_table)
            );
        }

        for config in [
            TrainConfig {
                algorithm: TrainAlgorithm::Dt,
                task: Task::Classification,
                tree_type: TreeType::Id3,
                criterion: Criterion::Entropy,
                physical_cores: Some(1),
            },
            TrainConfig {
                algorithm: TrainAlgorithm::Dt,
                task: Task::Classification,
                tree_type: TreeType::C45,
                criterion: Criterion::Entropy,
                physical_cores: Some(1),
            },
            TrainConfig {
                algorithm: TrainAlgorithm::Dt,
                task: Task::Classification,
                tree_type: TreeType::Cart,
                criterion: Criterion::Gini,
                physical_cores: Some(1),
            },
            TrainConfig {
                algorithm: TrainAlgorithm::Dt,
                task: Task::Classification,
                tree_type: TreeType::Oblivious,
                criterion: Criterion::Gini,
                physical_cores: Some(1),
            },
        ] {
            let single_core = train(&classification_table, config).unwrap();
            let parallel = train(
                &classification_table,
                TrainConfig {
                    physical_cores: Some(2),
                    ..config
                },
            )
            .unwrap();

            assert_eq!(
                single_core.predict_table(&classification_table),
                parallel.predict_table(&classification_table)
            );
        }
    }

    #[test]
    fn unified_train_rejects_zero_physical_cores() {
        let table = DenseTable::new(vec![vec![0.0], vec![1.0]], vec![0.0, 1.0]).unwrap();

        let err = train(
            &table,
            TrainConfig {
                physical_cores: Some(0),
                ..TrainConfig::default()
            },
        )
        .unwrap_err();

        assert!(matches!(
            err,
            TrainError::InvalidPhysicalCoreCount { requested: 0, .. }
        ));
    }

    #[test]
    fn unified_train_caps_physical_cores_to_available_hardware() {
        let table = DenseTable::with_canaries(
            vec![
                vec![0.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
            ],
            vec![0.0, 0.0, 0.0, 1.0],
            0,
        )
        .unwrap();

        let single_core = train(
            &table,
            TrainConfig {
                algorithm: TrainAlgorithm::Dt,
                task: Task::Classification,
                tree_type: TreeType::Cart,
                criterion: Criterion::Gini,
                physical_cores: Some(1),
            },
        )
        .unwrap();
        let overprovisioned = train(
            &table,
            TrainConfig {
                algorithm: TrainAlgorithm::Dt,
                task: Task::Classification,
                tree_type: TreeType::Cart,
                criterion: Criterion::Gini,
                physical_cores: Some(usize::MAX),
            },
        )
        .unwrap();

        assert_eq!(
            single_core.predict_table(&table),
            overprovisioned.predict_table(&table)
        );
    }

    #[test]
    fn ir_exports_target_mean_with_training_binning() {
        let table = DenseTable::with_canaries(
            vec![
                vec![0.0, 0.0],
                vec![1.0, 10.0],
                vec![0.0, 20.0],
                vec![1.0, 30.0],
            ],
            vec![1.0, 3.0, 5.0, 7.0],
            2,
        )
        .unwrap();

        let model = train(
            &table,
            TrainConfig {
                algorithm: TrainAlgorithm::Dt,
                task: Task::Regression,
                tree_type: TreeType::TargetMean,
                criterion: Criterion::Mean,
                physical_cores: Some(1),
            },
        )
        .unwrap();

        let ir = model.to_ir();

        assert_eq!(ir.ir_version, "1.0.0");
        assert_eq!(ir.model.algorithm, "dt");
        assert_eq!(ir.model.tree_type, "target_mean");
        assert_eq!(ir.input_schema.feature_count, 2);
        assert_eq!(ir.training_metadata.canaries, 2);
        assert!(matches!(
            &ir.preprocessing.numeric_binning.features[0],
            ir::FeatureBinning::Binary { feature_index: 0 }
        ));
        assert!(matches!(
            &ir.preprocessing.numeric_binning.features[1],
            ir::FeatureBinning::Numeric {
                feature_index: 1,
                ..
            }
        ));
        let ir::TreeDefinition::NodeTree { nodes, .. } = &ir.model.trees[0] else {
            panic!("target mean should export as a node tree");
        };
        let ir::NodeTreeNode::Leaf { leaf, .. } = &nodes[0] else {
            panic!("target mean tree should contain a single leaf");
        };
        assert!(
            matches!(leaf, ir::LeafPayload::RegressionValue { value } if (*value - 4.0).abs() < 1e-12)
        );
    }

    #[test]
    fn ir_exports_classifier_with_multiway_postprocessing() {
        let table = DenseTable::with_canaries(
            vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]],
            vec![2.0, 4.0, 6.0, 8.0],
            0,
        )
        .unwrap();

        let model = train(
            &table,
            TrainConfig {
                algorithm: TrainAlgorithm::Dt,
                task: Task::Classification,
                tree_type: TreeType::Id3,
                criterion: Criterion::Entropy,
                physical_cores: Some(1),
            },
        )
        .unwrap();

        let ir = model.to_ir();

        assert_eq!(ir.model.representation, "node_tree");
        assert_eq!(ir.output_schema.class_order, Some(vec![2.0, 4.0, 6.0, 8.0]));
        assert!(matches!(
            &ir.postprocessing.steps[0],
            ir::PostprocessingStep::MapClassIndexToLabel { labels }
                if labels == &vec![2.0, 4.0, 6.0, 8.0]
        ));
        let ir::TreeDefinition::NodeTree { nodes, .. } = &ir.model.trees[0] else {
            panic!("id3 should export as a node tree");
        };
        assert!(
            nodes
                .iter()
                .any(|node| matches!(node, ir::NodeTreeNode::MultiwayBranch { .. }))
        );
    }

    #[test]
    fn ir_exports_oblivious_regressor_with_msb_leaf_indexing() {
        let table = DenseTable::with_canaries(
            vec![
                vec![0.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
            ],
            vec![0.0, 1.0, 1.0, 2.0],
            0,
        )
        .unwrap();

        let model = train(
            &table,
            TrainConfig {
                algorithm: TrainAlgorithm::Dt,
                task: Task::Regression,
                tree_type: TreeType::Oblivious,
                criterion: Criterion::Mean,
                physical_cores: Some(1),
            },
        )
        .unwrap();

        let ir = model.to_ir();

        let ir::TreeDefinition::ObliviousLevels {
            depth,
            leaf_indexing,
            leaves,
            ..
        } = &ir.model.trees[0]
        else {
            panic!("oblivious regressor should export as oblivious_levels");
        };

        assert_eq!(*depth, 2);
        assert_eq!(leaf_indexing.bit_order, "msb_first");
        assert_eq!(leaves.len(), 4);

        let json = model.to_ir_json().unwrap();
        let parsed: ModelPackageIr = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.model.tree_type, "oblivious");
    }

    #[test]
    fn serialized_model_round_trips_through_deserialize() {
        let table = DenseTable::with_canaries(
            vec![
                vec![0.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
            ],
            vec![0.0, 0.0, 0.0, 1.0],
            2,
        )
        .unwrap();

        let model = train(
            &table,
            TrainConfig {
                algorithm: TrainAlgorithm::Dt,
                task: Task::Classification,
                tree_type: TreeType::Cart,
                criterion: Criterion::Gini,
                physical_cores: Some(1),
            },
        )
        .unwrap();

        let serialized = model.serialize().unwrap();
        let restored = Model::deserialize(&serialized).unwrap();

        assert_eq!(model.algorithm(), restored.algorithm());
        assert_eq!(model.task(), restored.task());
        assert_eq!(model.tree_type(), restored.tree_type());
        assert_eq!(model.criterion(), restored.criterion());
        assert_eq!(model.predict_table(&table), restored.predict_table(&table));
    }
}
