use forestfire_data::{BinnedColumnKind, TableAccess, numeric_bin_boundaries};
#[cfg(feature = "polars")]
use polars::prelude::{Column, DataFrame, DataType, LazyFrame};
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

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

const OPTIMIZED_MULTIWAY_LOOKUP_SIZE: usize = 512;
const PARALLEL_INFERENCE_ROW_THRESHOLD: usize = 256;

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

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
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

#[derive(Debug, Clone, PartialEq)]
pub enum PredictError {
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
        }
    }
}

impl Error for OptimizeError {}

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

    pub(crate) fn to_binned_matrix(&self) -> BinnedInferenceMatrix {
        let n_features = self.feature_columns.len();
        let mut bins = vec![0u16; self.n_rows * n_features];

        for row_index in 0..self.n_rows {
            let row_offset = row_index * n_features;
            for feature_index in 0..n_features {
                bins[row_offset + feature_index] = self.binned_value(feature_index, row_index);
            }
        }

        BinnedInferenceMatrix {
            n_rows: self.n_rows,
            n_features,
            bins,
        }
    }
}

#[derive(Debug, Clone)]
struct BinnedInferenceMatrix {
    n_rows: usize,
    n_features: usize,
    bins: Vec<u16>,
}

impl BinnedInferenceMatrix {
    #[inline(always)]
    fn row(&self, row_index: usize) -> &[u16] {
        let start = row_index * self.n_features;
        &self.bins[start..start + self.n_features]
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

#[derive(Debug, Clone)]
enum OptimizedRuntime {
    TargetMean {
        value: f64,
    },
    StandardClassifier {
        nodes: Vec<OptimizedClassifierNode>,
        root: usize,
    },
    ObliviousClassifier {
        feature_indices: Vec<usize>,
        threshold_bins: Vec<u16>,
        leaf_values: Vec<f64>,
    },
    StandardRegressor {
        nodes: Vec<OptimizedRegressorNode>,
        root: usize,
    },
    ObliviousRegressor {
        feature_indices: Vec<usize>,
        threshold_bins: Vec<u16>,
        leaf_values: Vec<f64>,
    },
}

#[derive(Debug, Clone)]
enum OptimizedClassifierNode {
    Leaf(f64),
    Binary {
        feature_index: usize,
        threshold_bin: u16,
        children: [usize; 2],
    },
    Multiway {
        feature_index: usize,
        child_lookup: Vec<usize>,
        fallback_value: f64,
    },
}

#[derive(Debug, Clone)]
enum OptimizedRegressorNode {
    Leaf(f64),
    Binary {
        feature_index: usize,
        threshold_bin: u16,
        children: [usize; 2],
    },
}

#[derive(Debug, Clone)]
struct InferenceExecutor {
    thread_count: usize,
    pool: Option<Arc<rayon::ThreadPool>>,
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
}

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

    pub fn predict_table(&self, table: &dyn TableAccess) -> Vec<f64> {
        self.executor.predict_rows(table.n_rows(), |row_index| {
            self.runtime.predict_table_row(table, row_index)
        })
    }

    pub fn predict_rows(&self, rows: Vec<Vec<f64>>) -> Result<Vec<f64>, PredictError> {
        let table = InferenceTable::from_rows(rows, self.source_model.feature_preprocessing())?;
        let matrix = table.to_binned_matrix();
        Ok(self.predict_binned_matrix(&matrix))
    }

    pub fn predict_named_columns(
        &self,
        columns: BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<f64>, PredictError> {
        let table =
            InferenceTable::from_named_columns(columns, self.source_model.feature_preprocessing())?;
        let matrix = table.to_binned_matrix();
        Ok(self.predict_binned_matrix(&matrix))
    }

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
        let matrix = table.to_binned_matrix();
        Ok(self.predict_binned_matrix(&matrix))
    }

    #[cfg(feature = "polars")]
    pub fn predict_polars_dataframe(&self, df: &DataFrame) -> Result<Vec<f64>, PredictError> {
        let columns = polars_named_columns(df)?;
        self.predict_named_columns(columns)
    }

    #[cfg(feature = "polars")]
    pub fn predict_polars_lazyframe(&self, lf: &LazyFrame) -> Result<Vec<f64>, PredictError> {
        let df = lf.clone().collect()?;
        self.predict_polars_dataframe(&df)
    }

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

    fn predict_binned_matrix(&self, matrix: &BinnedInferenceMatrix) -> Vec<f64> {
        self.executor.predict_rows(matrix.n_rows, |row_index| {
            self.runtime.predict_binned_row(matrix.row(row_index))
        })
    }
}

impl OptimizedRuntime {
    fn from_model(model: &Model) -> Self {
        match model {
            Model::TargetMean(target_mean) => Self::TargetMean {
                value: target_mean.mean,
            },
            Model::DecisionTreeClassifier(classifier) => Self::from_classifier(classifier),
            Model::DecisionTreeRegressor(regressor) => Self::from_regressor(regressor),
        }
    }

    fn from_classifier(classifier: &DecisionTreeClassifier) -> Self {
        match classifier.structure() {
            tree::classifier::TreeStructure::Standard { nodes, root } => {
                let optimized_nodes = nodes
                    .iter()
                    .map(|node| match node {
                        tree::classifier::TreeNode::Leaf { class_index, .. } => {
                            OptimizedClassifierNode::Leaf(classifier.class_labels()[*class_index])
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
                            fallback_class_index,
                            branches,
                            ..
                        } => {
                            let mut child_lookup = vec![usize::MAX; OPTIMIZED_MULTIWAY_LOOKUP_SIZE];
                            for (bin, child_index) in branches {
                                let idx = usize::from(*bin);
                                if idx < OPTIMIZED_MULTIWAY_LOOKUP_SIZE {
                                    child_lookup[idx] = *child_index;
                                }
                            }
                            OptimizedClassifierNode::Multiway {
                                feature_index: *feature_index,
                                child_lookup,
                                fallback_value: classifier.class_labels()[*fallback_class_index],
                            }
                        }
                    })
                    .collect();

                Self::StandardClassifier {
                    nodes: optimized_nodes,
                    root: *root,
                }
            }
            tree::classifier::TreeStructure::Oblivious {
                splits,
                leaf_class_indices,
                ..
            } => Self::ObliviousClassifier {
                feature_indices: splits.iter().map(|split| split.feature_index).collect(),
                threshold_bins: splits.iter().map(|split| split.threshold_bin).collect(),
                leaf_values: leaf_class_indices
                    .iter()
                    .map(|class_index| classifier.class_labels()[*class_index])
                    .collect(),
            },
        }
    }

    fn from_regressor(regressor: &DecisionTreeRegressor) -> Self {
        match regressor.structure() {
            tree::regressor::RegressionTreeStructure::Standard { nodes, root } => {
                let optimized_nodes = nodes
                    .iter()
                    .map(|node| match node {
                        tree::regressor::RegressionNode::Leaf { value, .. } => {
                            OptimizedRegressorNode::Leaf(*value)
                        }
                        tree::regressor::RegressionNode::BinarySplit {
                            feature_index,
                            threshold_bin,
                            left_child,
                            right_child,
                            ..
                        } => OptimizedRegressorNode::Binary {
                            feature_index: *feature_index,
                            threshold_bin: *threshold_bin,
                            children: [*left_child, *right_child],
                        },
                    })
                    .collect();

                Self::StandardRegressor {
                    nodes: optimized_nodes,
                    root: *root,
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
            OptimizedRuntime::TargetMean { value } => *value,
            OptimizedRuntime::StandardClassifier { nodes, root } => {
                predict_standard_classifier_row(nodes, *root, |feature_index| {
                    table.binned_value(feature_index, row_index)
                })
            }
            OptimizedRuntime::ObliviousClassifier {
                feature_indices,
                threshold_bins,
                leaf_values,
            } => predict_oblivious_row(
                feature_indices,
                threshold_bins,
                leaf_values,
                |feature_index| table.binned_value(feature_index, row_index),
            ),
            OptimizedRuntime::StandardRegressor { nodes, root } => {
                predict_standard_regressor_row(nodes, *root, |feature_index| {
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
        }
    }

    #[inline(always)]
    fn predict_binned_row(&self, row_bins: &[u16]) -> f64 {
        match self {
            OptimizedRuntime::TargetMean { value } => *value,
            OptimizedRuntime::StandardClassifier { nodes, root } => {
                predict_standard_classifier_row(nodes, *root, |feature_index| {
                    row_bins[feature_index]
                })
            }
            OptimizedRuntime::ObliviousClassifier {
                feature_indices,
                threshold_bins,
                leaf_values,
            } => predict_oblivious_row(
                feature_indices,
                threshold_bins,
                leaf_values,
                |feature_index| row_bins[feature_index],
            ),
            OptimizedRuntime::StandardRegressor { nodes, root } => {
                predict_standard_regressor_row(nodes, *root, |feature_index| {
                    row_bins[feature_index]
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
                |feature_index| row_bins[feature_index],
            ),
        }
    }
}

#[inline(always)]
fn predict_standard_classifier_row<F>(
    nodes: &[OptimizedClassifierNode],
    root: usize,
    bin_at: F,
) -> f64
where
    F: Fn(usize) -> u16,
{
    let mut node_index = root;
    loop {
        match &nodes[node_index] {
            OptimizedClassifierNode::Leaf(value) => return *value,
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
                fallback_value,
            } => {
                let child_index = child_lookup[usize::from(bin_at(*feature_index))];
                if child_index == usize::MAX {
                    return *fallback_value;
                }
                node_index = child_index;
            }
        }
    }
}

#[inline(always)]
fn predict_standard_regressor_row<F>(
    nodes: &[OptimizedRegressorNode],
    root: usize,
    bin_at: F,
) -> f64
where
    F: Fn(usize) -> u16,
{
    let mut node_index = root;
    loop {
        match &nodes[node_index] {
            OptimizedRegressorNode::Leaf(value) => return *value,
            OptimizedRegressorNode::Binary {
                feature_index,
                threshold_bin,
                children,
            } => {
                let go_right = usize::from(bin_at(*feature_index) > *threshold_bin);
                node_index = children[go_right];
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

    pub fn predict_rows(&self, rows: Vec<Vec<f64>>) -> Result<Vec<f64>, PredictError> {
        let table = InferenceTable::from_rows(rows, self.feature_preprocessing())?;
        Ok(self.predict_table(&table))
    }

    pub fn predict_named_columns(
        &self,
        columns: BTreeMap<String, Vec<f64>>,
    ) -> Result<Vec<f64>, PredictError> {
        let table = InferenceTable::from_named_columns(columns, self.feature_preprocessing())?;
        Ok(self.predict_table(&table))
    }

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

    #[cfg(feature = "polars")]
    pub fn predict_polars_dataframe(&self, df: &DataFrame) -> Result<Vec<f64>, PredictError> {
        let columns = polars_named_columns(df)?;
        self.predict_named_columns(columns)
    }

    #[cfg(feature = "polars")]
    pub fn predict_polars_lazyframe(&self, lf: &LazyFrame) -> Result<Vec<f64>, PredictError> {
        let df = lf.clone().collect()?;
        self.predict_polars_dataframe(&df)
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
mod tests {
    use super::*;
    use forestfire_data::DenseTable;
    #[cfg(feature = "polars")]
    use polars::prelude::{DataFrame, IntoLazy, NamedFrom, Series};
    use std::collections::BTreeMap;

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

    #[test]
    fn optimized_model_matches_base_model_and_ir_for_standard_classifier() {
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
        let optimized = model.optimize_inference(Some(1)).unwrap();

        assert_eq!(model.to_ir_json().unwrap(), optimized.to_ir_json().unwrap());
        assert_eq!(model.serialize().unwrap(), optimized.serialize().unwrap());
        assert_eq!(model.predict_table(&table), optimized.predict_table(&table));
        assert_eq!(
            model
                .predict_rows(vec![vec![0.0, 1.0], vec![1.0, 1.0]])
                .unwrap(),
            optimized
                .predict_rows(vec![vec![0.0, 1.0], vec![1.0, 1.0]])
                .unwrap()
        );
    }

    #[test]
    fn optimized_model_matches_base_model_and_ir_for_oblivious_regressor() {
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
        let optimized = model.optimize_inference(Some(2)).unwrap();

        assert_eq!(model.to_ir_json().unwrap(), optimized.to_ir_json().unwrap());
        assert_eq!(model.predict_table(&table), optimized.predict_table(&table));
        assert_eq!(
            model
                .predict_named_columns(BTreeMap::from([
                    ("f0".to_string(), vec![0.0, 1.0]),
                    ("f1".to_string(), vec![1.0, 1.0]),
                ]))
                .unwrap(),
            optimized
                .predict_named_columns(BTreeMap::from([
                    ("f0".to_string(), vec![0.0, 1.0]),
                    ("f1".to_string(), vec![1.0, 1.0]),
                ]))
                .unwrap()
        );
    }

    #[test]
    fn optimized_model_rejects_zero_physical_cores() {
        let table = DenseTable::with_canaries(vec![vec![0.0]], vec![1.0], 0).unwrap();
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

        let err = model.optimize_inference(Some(0)).unwrap_err();

        assert!(matches!(
            err,
            OptimizeError::InvalidPhysicalCoreCount { requested: 0, .. }
        ));
    }

    #[test]
    fn model_predicts_from_raw_rows_without_building_a_training_table() {
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

        let preds = model
            .predict_rows(vec![
                vec![0.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
            ])
            .unwrap();

        assert_eq!(preds, vec![0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn model_predicts_from_named_columns() {
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

        let preds = model
            .predict_named_columns(BTreeMap::from([
                ("f0".to_string(), vec![0.0, 0.0, 1.0, 1.0]),
                ("f1".to_string(), vec![0.0, 1.0, 0.0, 1.0]),
            ]))
            .unwrap();

        assert_eq!(preds, vec![0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn model_rejects_missing_named_feature() {
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

        let err = model
            .predict_named_columns(BTreeMap::from([("f0".to_string(), vec![0.0, 1.0])]))
            .unwrap_err();

        assert!(matches!(err, PredictError::MissingFeature(feature) if feature == "f1"));
    }

    #[test]
    fn model_rejects_unexpected_named_feature() {
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

        let err = model
            .predict_named_columns(BTreeMap::from([
                ("f0".to_string(), vec![0.0, 1.0]),
                ("f1".to_string(), vec![0.0, 1.0]),
                ("f2".to_string(), vec![0.0, 1.0]),
            ]))
            .unwrap_err();

        assert!(matches!(err, PredictError::UnexpectedFeature(feature) if feature == "f2"));
    }

    #[test]
    fn model_rejects_invalid_binary_value_during_inference() {
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

        let err = model.predict_rows(vec![vec![0.5, 1.0]]).unwrap_err();

        assert!(matches!(
            err,
            PredictError::InvalidBinaryValue {
                feature_index: 0,
                row_index: 0,
                ..
            }
        ));
    }

    #[test]
    fn model_predicts_from_sparse_binary_columns() {
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

        let preds = model
            .predict_sparse_binary_columns(4, 2, vec![vec![2, 3], vec![1, 3]])
            .unwrap();

        assert_eq!(preds, vec![0.0, 0.0, 0.0, 1.0]);
    }

    #[cfg(feature = "polars")]
    #[test]
    fn model_predicts_from_polars_dataframe() {
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
        let df = DataFrame::new(vec![
            Series::new("f0".into(), &[0.0, 0.0, 1.0, 1.0]).into(),
            Series::new("f1".into(), &[0.0, 1.0, 0.0, 1.0]).into(),
        ])
        .unwrap();

        let preds = model.predict_polars_dataframe(&df).unwrap();

        assert_eq!(preds, vec![0.0, 0.0, 0.0, 1.0]);
    }

    #[cfg(feature = "polars")]
    #[test]
    fn model_predicts_from_polars_lazyframe() {
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
        let df = DataFrame::new(vec![
            Series::new("f0".into(), &[0.0, 0.0, 1.0, 1.0]).into(),
            Series::new("f1".into(), &[0.0, 1.0, 0.0, 1.0]).into(),
        ])
        .unwrap();

        let preds = model.predict_polars_lazyframe(&df.lazy()).unwrap();

        assert_eq!(preds, vec![0.0, 0.0, 0.0, 1.0]);
    }

    #[cfg(feature = "polars")]
    #[test]
    fn model_rejects_polars_nulls() {
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
        let df = DataFrame::new(vec![
            Series::new("f0".into(), &[Some(0.0), None]).into(),
            Series::new("f1".into(), &[Some(0.0), Some(1.0)]).into(),
        ])
        .unwrap();

        let err = model.predict_polars_dataframe(&df).unwrap_err();

        assert!(
            matches!(err, PredictError::NullValue { feature, row_index } if feature == "f0" && row_index == 1)
        );
    }

    #[test]
    fn ir_serializes_node_stats_for_standard_and_oblivious_trees() {
        let classifier_table = DenseTable::with_canaries(
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
        let classifier = train(
            &classifier_table,
            TrainConfig {
                algorithm: TrainAlgorithm::Dt,
                task: Task::Classification,
                tree_type: TreeType::Cart,
                criterion: Criterion::Gini,
                physical_cores: Some(1),
            },
        )
        .unwrap()
        .to_ir();

        let ir::TreeDefinition::NodeTree { nodes, .. } = &classifier.model.trees[0] else {
            panic!("classifier should export as node_tree");
        };
        assert!(nodes.iter().all(|node| match node {
            ir::NodeTreeNode::Leaf { stats, .. } => stats.sample_count > 0,
            ir::NodeTreeNode::BinaryBranch { stats, .. }
            | ir::NodeTreeNode::MultiwayBranch { stats, .. } => {
                stats.sample_count > 0 && stats.impurity.is_some() && stats.gain.is_some()
            }
        }));

        let regressor_table = DenseTable::with_canaries(
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
        let regressor = train(
            &regressor_table,
            TrainConfig {
                algorithm: TrainAlgorithm::Dt,
                task: Task::Regression,
                tree_type: TreeType::Oblivious,
                criterion: Criterion::Mean,
                physical_cores: Some(1),
            },
        )
        .unwrap()
        .to_ir();

        let ir::TreeDefinition::ObliviousLevels { levels, leaves, .. } = &regressor.model.trees[0]
        else {
            panic!("regressor should export as oblivious_levels");
        };
        assert!(levels.iter().all(|level| {
            level.stats.sample_count > 0
                && level.stats.impurity.is_some()
                && level.stats.gain.is_some()
        }));
        assert!(leaves.iter().all(|leaf| leaf.stats.sample_count > 0));
    }

    #[test]
    fn generated_json_schema_matches_checked_in_schema() {
        let generated = Model::json_schema_json_pretty().unwrap();
        let checked_in = include_str!("../schema/forestfire-ir.schema.json");
        assert_eq!(generated.trim_end(), checked_in.trim_end());
    }
}
