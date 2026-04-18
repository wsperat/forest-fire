//! Python bindings for ForestFire.
//!
//! The binding layer does three jobs:
//!
//! - convert Python-native inputs into the normalized Rust table/config surface
//! - preserve Python ergonomics such as `task="auto"`, string class labels, and
//!   single-row prediction shortcuts
//! - keep the heavy lifting in Rust while releasing the GIL around long-running
//!   train/predict/optimize calls

use forestfire_core::{
    CanaryFilter, CategoricalConfig, CategoricalModel, CategoricalOptimizedModel,
    CategoricalStrategy, CategoricalValue, Criterion, IntrospectionError, MaxFeatures,
    MissingValueStrategy, MissingValueStrategyConfig, Model, OptimizedModel as CoreOptimizedModel,
    Task, TrainAlgorithm, TrainConfig, TreeType, categorical, train as train_model,
};
use forestfire_data::{MAX_NUMERIC_BINS, NumericBins, Table, TableAccess, TableKind};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::types::{PyBytes, PyDict, PyType};
use pyo3::{Bound, prelude::*};
use std::collections::BTreeMap;

const LAZYFRAME_PREDICT_BATCH_ROWS: usize = 10_000;

#[pyclass(name = "Model")]
struct PyModel {
    inner: Model,
    string_class_labels: Option<Vec<String>>,
}

#[pyclass(name = "OptimizedModel")]
struct PyOptimizedModel {
    inner: CoreOptimizedModel,
    string_class_labels: Option<Vec<String>>,
}

#[pyclass(name = "CategoricalModel")]
struct PyCategoricalModel {
    inner: CategoricalModel,
    string_class_labels: Option<Vec<String>>,
}

#[pyclass(name = "CategoricalOptimizedModel")]
struct PyCategoricalOptimizedModel {
    inner: CategoricalOptimizedModel,
    string_class_labels: Option<Vec<String>>,
}

#[pyclass(name = "Table")]
#[derive(Clone)]
struct PyTable {
    inner: Table,
}

#[derive(serde::Serialize)]
struct TreeDataFrameRow {
    tree_index: usize,
    representation: String,
    node_type: String,
    node_index: String,
    depth: usize,
    parent_index: Option<String>,
    left_child: Option<String>,
    right_child: Option<String>,
    branch_bins: Option<Vec<u16>>,
    branch_children: Option<Vec<String>>,
    split_feature: Option<usize>,
    split_feature_name: Option<String>,
    split_type: Option<String>,
    threshold_bin: Option<u16>,
    threshold_upper_bound: Option<f64>,
    operator: Option<String>,
    leaf_value: Option<f64>,
    leaf_class_index: Option<usize>,
    leaf_label: Option<String>,
    sample_count: usize,
    impurity: Option<f64>,
    gain: Option<f64>,
    variance: Option<f64>,
    class_counts: Option<Vec<usize>>,
}

fn table_kind_name(kind: TableKind) -> &'static str {
    match kind {
        TableKind::Dense => "dense",
        TableKind::Sparse => "sparse",
    }
}

enum TrainingTargets {
    Numeric(Vec<f64>),
    StringClasses {
        encoded: Vec<f64>,
        labels: Vec<String>,
    },
}

fn build_training_table(
    x: &Bound<PyAny>,
    y: Option<&Bound<PyAny>>,
    canaries: usize,
    bins: NumericBins,
) -> PyResult<(Table, Option<Vec<String>>)> {
    if let Ok(table) = x.extract::<PyRef<'_, PyTable>>() {
        if y.is_some() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "y must be omitted when x is already a Table.",
            ));
        }
        if bins != NumericBins::Auto {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "bins must be omitted when x is already a Table.",
            ));
        }
        return Ok((table.inner.clone(), None));
    }

    let y = y.ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "y is required unless x is already a Table.",
        )
    })?;
    if is_scipy_sparse_matrix(x)? {
        return build_sparse_training_table(x, y, canaries, bins);
    }
    let x_rows = extract_matrix(x)?;
    let (y_values, string_class_labels) = match extract_training_targets(y)? {
        TrainingTargets::Numeric(values) => (values, None),
        TrainingTargets::StringClasses { encoded, labels } => (encoded, Some(labels)),
    };

    Table::with_options(x_rows, y_values, canaries, bins)
        .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))
        .map(|table| (table, string_class_labels))
}

fn build_feature_table(x: &Bound<PyAny>, bins: NumericBins) -> PyResult<Table> {
    if let Ok(table) = x.extract::<PyRef<'_, PyTable>>() {
        if bins != NumericBins::Auto {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "bins must be omitted when x is already a Table.",
            ));
        }
        return Ok(table.inner.clone());
    }

    if is_scipy_sparse_matrix(x)? {
        return build_sparse_feature_table(x, bins);
    }

    let x_rows = extract_matrix(x)?;
    let y_values = vec![0.0; x_rows.len()];

    Table::with_options(x_rows, y_values, 0, bins)
        .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))
}

enum InferenceInput {
    Rows(Vec<Vec<f64>>),
    NamedColumns(BTreeMap<String, Vec<f64>>),
    SparseBinaryColumns {
        n_rows: usize,
        n_features: usize,
        columns: Vec<Vec<usize>>,
    },
    TrainingTable(Table),
}

fn build_inference_input(x: &Bound<PyAny>) -> PyResult<InferenceInput> {
    if let Ok(table) = x.extract::<PyRef<'_, PyTable>>() {
        return Ok(InferenceInput::TrainingTable(table.inner.clone()));
    }

    if is_scipy_sparse_matrix(x)? {
        let (n_rows, n_features, columns) = extract_sparse_binary_columns(x)?;
        return Ok(InferenceInput::SparseBinaryColumns {
            n_rows,
            n_features,
            columns,
        });
    }

    // Inference is intentionally permissive: a single row like `[1, 2, 3]`
    // should not require the caller to wrap it as `[[1, 2, 3]]`.
    if let Ok(rows) = extract_rows_or_single_row(x) {
        return Ok(InferenceInput::Rows(rows));
    }

    if let Ok(columns) = extract_named_columns(x) {
        return Ok(InferenceInput::NamedColumns(columns));
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "Input could not be interpreted as rows or named columns.",
    ))
}

fn is_scipy_sparse_matrix(x: &Bound<PyAny>) -> PyResult<bool> {
    Ok(x.hasattr("getnnz")? && x.hasattr("nonzero")?)
}

fn is_polars_lazyframe(x: &Bound<PyAny>) -> PyResult<bool> {
    if !(x.hasattr("collect")? && x.hasattr("slice")?) {
        return Ok(false);
    }

    let class = x.getattr("__class__")?;
    let name: String = class.getattr("__name__")?.extract()?;
    let module: String = class.getattr("__module__")?.extract()?;
    Ok(name == "LazyFrame" && module.starts_with("polars"))
}

fn row_count(x: &Bound<PyAny>) -> PyResult<usize> {
    if x.hasattr("height")? {
        return x.getattr("height")?.extract();
    }

    if x.hasattr("shape")? {
        let shape: Vec<usize> = x.getattr("shape")?.extract()?;
        return Ok(shape.first().copied().unwrap_or(0));
    }

    x.len()
}

fn predict_lazyframe_in_batches<F>(x: &Bound<PyAny>, predict_batch: F) -> PyResult<Vec<f64>>
where
    F: Fn(InferenceInput) -> PyResult<Vec<f64>>,
{
    let mut predictions = Vec::new();
    let mut offset = 0usize;
    loop {
        let sliced = x.call_method1("slice", (offset, LAZYFRAME_PREDICT_BATCH_ROWS))?;
        let batch = sliced.call_method0("collect")?;
        let height = row_count(&batch)?;
        if height == 0 {
            break;
        }
        // LazyFrames are collected in bounded batches so prediction can scale to
        // larger-than-memory logical plans without forcing one giant materialize.
        predictions.extend(predict_batch(build_inference_input(&batch)?)?);
        if height < LAZYFRAME_PREDICT_BATCH_ROWS {
            break;
        }
        offset += height;
    }
    Ok(predictions)
}

fn predict_proba_lazyframe_in_batches<F>(
    x: &Bound<PyAny>,
    predict_batch: F,
) -> PyResult<Vec<Vec<f64>>>
where
    F: Fn(InferenceInput) -> PyResult<Vec<Vec<f64>>>,
{
    let mut predictions = Vec::new();
    let mut offset = 0usize;
    loop {
        let sliced = x.call_method1("slice", (offset, LAZYFRAME_PREDICT_BATCH_ROWS))?;
        let batch = sliced.call_method0("collect")?;
        let height = row_count(&batch)?;
        if height == 0 {
            break;
        }
        predictions.extend(predict_batch(build_inference_input(&batch)?)?);
        if height < LAZYFRAME_PREDICT_BATCH_ROWS {
            break;
        }
        offset += height;
    }
    Ok(predictions)
}

fn predict_input_with_model_result(
    model: &Model,
    input: InferenceInput,
) -> Result<Vec<f64>, String> {
    match input {
        InferenceInput::Rows(rows) => model.predict_rows(rows),
        InferenceInput::NamedColumns(columns) => model.predict_named_columns(columns),
        InferenceInput::SparseBinaryColumns {
            n_rows,
            n_features,
            columns,
        } => model.predict_sparse_binary_columns(n_rows, n_features, columns),
        InferenceInput::TrainingTable(table) => Ok(model.predict_table(&table)),
    }
    .map_err(|err| err.to_string())
}

fn predict_proba_input_with_model_result(
    model: &Model,
    input: InferenceInput,
) -> Result<Vec<Vec<f64>>, String> {
    match input {
        InferenceInput::Rows(rows) => model.predict_proba_rows(rows),
        InferenceInput::NamedColumns(columns) => model.predict_proba_named_columns(columns),
        InferenceInput::SparseBinaryColumns {
            n_rows,
            n_features,
            columns,
        } => model.predict_proba_sparse_binary_columns(n_rows, n_features, columns),
        InferenceInput::TrainingTable(table) => model.predict_proba_table(&table),
    }
    .map_err(|err| err.to_string())
}

fn predict_input_with_optimized_model_result(
    model: &CoreOptimizedModel,
    input: InferenceInput,
) -> Result<Vec<f64>, String> {
    match input {
        InferenceInput::Rows(rows) => model.predict_rows(rows),
        InferenceInput::NamedColumns(columns) => model.predict_named_columns(columns),
        InferenceInput::SparseBinaryColumns {
            n_rows,
            n_features,
            columns,
        } => model.predict_sparse_binary_columns(n_rows, n_features, columns),
        InferenceInput::TrainingTable(table) => Ok(model.predict_table(&table)),
    }
    .map_err(|err| err.to_string())
}

fn predict_proba_input_with_optimized_model_result(
    model: &CoreOptimizedModel,
    input: InferenceInput,
) -> Result<Vec<Vec<f64>>, String> {
    match input {
        InferenceInput::Rows(rows) => model.predict_proba_rows(rows),
        InferenceInput::NamedColumns(columns) => model.predict_proba_named_columns(columns),
        InferenceInput::SparseBinaryColumns {
            n_rows,
            n_features,
            columns,
        } => model.predict_proba_sparse_binary_columns(n_rows, n_features, columns),
        InferenceInput::TrainingTable(table) => model.predict_proba_table(&table),
    }
    .map_err(|err| err.to_string())
}

fn train_model_detached(py: Python<'_>, table: Table, config: TrainConfig) -> PyResult<Model> {
    py.detach(move || train_model(&table, config).map_err(|err| err.to_string()))
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
}

fn optimize_model_detached(
    py: Python<'_>,
    model: &Model,
    physical_cores: Option<usize>,
    missing_features: Option<Vec<usize>>,
) -> PyResult<CoreOptimizedModel> {
    py.detach(|| {
        model
            .optimize_inference_with_missing_features(physical_cores, missing_features)
            .map_err(|err| err.to_string())
    })
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
}

fn predict_input_with_model_detached(
    py: Python<'_>,
    model: &Model,
    input: InferenceInput,
) -> PyResult<Vec<f64>> {
    py.detach(|| predict_input_with_model_result(model, input))
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
}

fn predict_proba_input_with_model_detached(
    py: Python<'_>,
    model: &Model,
    input: InferenceInput,
) -> PyResult<Vec<Vec<f64>>> {
    py.detach(|| predict_proba_input_with_model_result(model, input))
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
}

fn predict_input_with_optimized_model_detached(
    py: Python<'_>,
    model: &CoreOptimizedModel,
    input: InferenceInput,
) -> PyResult<Vec<f64>> {
    py.detach(|| predict_input_with_optimized_model_result(model, input))
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
}

fn predict_proba_input_with_optimized_model_detached(
    py: Python<'_>,
    model: &CoreOptimizedModel,
    input: InferenceInput,
) -> PyResult<Vec<Vec<f64>>> {
    py.detach(|| predict_proba_input_with_optimized_model_result(model, input))
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
}

fn decoded_predictions<'py>(
    py: Python<'py>,
    preds: Vec<f64>,
    string_class_labels: Option<&[String]>,
) -> PyResult<Bound<'py, PyAny>> {
    if let Some(labels) = string_class_labels {
        let decoded = preds
            .into_iter()
            .map(|value| {
                let index = value as usize;
                labels.get(index).cloned().ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Prediction referenced an unknown string class label.",
                    )
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let numpy = py.import("numpy")?;
        return numpy.getattr("array")?.call1((decoded,));
    }

    Ok(PyArray1::from_vec(py, preds).into_any())
}

fn to_python_json_value<'py, T: serde::Serialize>(
    py: Python<'py>,
    value: &T,
) -> PyResult<Bound<'py, PyAny>> {
    let json = py.import("json")?;
    let serialized = serde_json::to_string(value)
        .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;
    json.getattr("loads")?.call1((serialized,))
}

fn build_dataframe<'py, T: serde::Serialize>(
    py: Python<'py>,
    value: &T,
) -> PyResult<Bound<'py, PyAny>> {
    let rows = to_python_json_value(py, value)?;
    match py.import("polars") {
        Ok(polars) => polars.getattr("DataFrame")?.call1((rows.clone(),)),
        Err(_) => {
            let pyarrow = py.import("pyarrow")?;
            pyarrow
                .getattr("Table")?
                .getattr("from_pylist")?
                .call1((rows,))
        }
    }
}

fn string_label_for_leaf(
    labels: Option<&[String]>,
    leaf_class_index: Option<usize>,
) -> Option<String> {
    labels.and_then(|labels| leaf_class_index.and_then(|index| labels.get(index).cloned()))
}

type SplitDetails = (
    Option<usize>,
    Option<String>,
    Option<String>,
    Option<u16>,
    Option<f64>,
    Option<String>,
);

fn split_details(split: &serde_json::Value) -> SplitDetails {
    (
        split
            .get("feature_index")
            .and_then(|value| value.as_u64())
            .map(|value| value as usize),
        split
            .get("feature_name")
            .and_then(|value| value.as_str())
            .map(ToOwned::to_owned),
        split
            .get("split_type")
            .and_then(|value| value.as_str())
            .map(ToOwned::to_owned),
        split
            .get("threshold_bin")
            .and_then(|value| value.as_u64())
            .map(|value| value as u16),
        split
            .get("threshold_upper_bound")
            .and_then(|value| value.as_f64()),
        split
            .get("operator")
            .and_then(|value| value.as_str())
            .map(ToOwned::to_owned),
    )
}

fn leaf_payload_details(
    leaf: &serde_json::Value,
    string_class_labels: Option<&[String]>,
) -> (Option<f64>, Option<usize>, Option<String>) {
    match leaf.get("prediction_kind").and_then(|value| value.as_str()) {
        Some("regression_value") => (
            leaf.get("value").and_then(|value| value.as_f64()),
            None,
            None,
        ),
        Some("class_index") => {
            let class_index = leaf
                .get("class_index")
                .and_then(|value| value.as_u64())
                .map(|value| value as usize);
            (
                leaf.get("class_value").and_then(|value| value.as_f64()),
                class_index,
                string_label_for_leaf(string_class_labels, class_index),
            )
        }
        _ => (None, None, None),
    }
}

fn tree_rows_from_model(
    model: &Model,
    string_class_labels: Option<&[String]>,
    tree_index: usize,
) -> PyResult<Vec<TreeDataFrameRow>> {
    let summary = model
        .tree_structure(tree_index)
        .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;
    let mut rows = Vec::new();

    match summary.representation.as_str() {
        "node_tree" => {
            let mut nodes = Vec::new();
            let mut node_index = 0usize;
            loop {
                match model.tree_node(tree_index, node_index) {
                    Ok(node) => {
                        nodes.push(node);
                        node_index += 1;
                    }
                    Err(IntrospectionError::NodeIndexOutOfBounds { .. }) => break,
                    Err(err) => {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            err.to_string(),
                        ));
                    }
                }
            }

            let mut parents = BTreeMap::<String, String>::new();
            for node in &nodes {
                match node {
                    forestfire_core::ir::NodeTreeNode::BinaryBranch {
                        node_id, children, ..
                    } => {
                        let parent = node_id.to_string();
                        parents.insert(children.left.to_string(), parent.clone());
                        parents.insert(children.right.to_string(), parent);
                    }
                    forestfire_core::ir::NodeTreeNode::MultiwayBranch {
                        node_id, branches, ..
                    } => {
                        let parent = node_id.to_string();
                        for branch in branches {
                            parents.insert(branch.child.to_string(), parent.clone());
                        }
                    }
                    forestfire_core::ir::NodeTreeNode::Leaf { .. } => {}
                }
            }

            for node in nodes {
                match node {
                    forestfire_core::ir::NodeTreeNode::Leaf {
                        node_id,
                        depth,
                        leaf,
                        stats,
                    } => {
                        let (leaf_value, leaf_class_index, leaf_label) = leaf_payload_details(
                            &serde_json::to_value(&leaf).unwrap(),
                            string_class_labels,
                        );
                        rows.push(TreeDataFrameRow {
                            tree_index,
                            representation: "node_tree".to_string(),
                            node_type: "leaf".to_string(),
                            node_index: node_id.to_string(),
                            depth,
                            parent_index: parents.get(&node_id.to_string()).cloned(),
                            left_child: None,
                            right_child: None,
                            branch_bins: None,
                            branch_children: None,
                            split_feature: None,
                            split_feature_name: None,
                            split_type: None,
                            threshold_bin: None,
                            threshold_upper_bound: None,
                            operator: None,
                            leaf_value,
                            leaf_class_index,
                            leaf_label,
                            sample_count: stats.sample_count,
                            impurity: stats.impurity,
                            gain: stats.gain,
                            variance: stats.variance,
                            class_counts: stats.class_counts,
                        });
                    }
                    forestfire_core::ir::NodeTreeNode::BinaryBranch {
                        node_id,
                        depth,
                        split,
                        children,
                        stats,
                    } => {
                        let split = serde_json::to_value(&split).unwrap();
                        let (
                            split_feature,
                            split_feature_name,
                            split_type,
                            threshold_bin,
                            threshold_upper_bound,
                            operator,
                        ) = split_details(&split);
                        rows.push(TreeDataFrameRow {
                            tree_index,
                            representation: "node_tree".to_string(),
                            node_type: "split".to_string(),
                            node_index: node_id.to_string(),
                            depth,
                            parent_index: parents.get(&node_id.to_string()).cloned(),
                            left_child: Some(children.left.to_string()),
                            right_child: Some(children.right.to_string()),
                            branch_bins: None,
                            branch_children: None,
                            split_feature,
                            split_feature_name,
                            split_type,
                            threshold_bin,
                            threshold_upper_bound,
                            operator,
                            leaf_value: None,
                            leaf_class_index: None,
                            leaf_label: None,
                            sample_count: stats.sample_count,
                            impurity: stats.impurity,
                            gain: stats.gain,
                            variance: stats.variance,
                            class_counts: stats.class_counts,
                        });
                    }
                    forestfire_core::ir::NodeTreeNode::MultiwayBranch {
                        node_id,
                        depth,
                        split,
                        branches,
                        unmatched_leaf,
                        stats,
                    } => {
                        let split = serde_json::to_value(&split).unwrap();
                        let (
                            split_feature,
                            split_feature_name,
                            split_type,
                            threshold_bin,
                            threshold_upper_bound,
                            operator,
                        ) = split_details(&split);
                        rows.push(TreeDataFrameRow {
                            tree_index,
                            representation: "node_tree".to_string(),
                            node_type: "split".to_string(),
                            node_index: node_id.to_string(),
                            depth,
                            parent_index: parents.get(&node_id.to_string()).cloned(),
                            left_child: None,
                            right_child: None,
                            branch_bins: Some(branches.iter().map(|branch| branch.bin).collect()),
                            branch_children: Some(
                                branches
                                    .iter()
                                    .map(|branch| branch.child.to_string())
                                    .collect(),
                            ),
                            split_feature,
                            split_feature_name,
                            split_type,
                            threshold_bin,
                            threshold_upper_bound,
                            operator,
                            leaf_value: None,
                            leaf_class_index: None,
                            leaf_label: None,
                            sample_count: stats.sample_count,
                            impurity: stats.impurity,
                            gain: stats.gain,
                            variance: stats.variance,
                            class_counts: stats.class_counts.clone(),
                        });

                        let (leaf_value, leaf_class_index, leaf_label) = leaf_payload_details(
                            &serde_json::to_value(&unmatched_leaf).unwrap(),
                            string_class_labels,
                        );
                        rows.push(TreeDataFrameRow {
                            tree_index,
                            representation: "node_tree".to_string(),
                            node_type: "unmatched_leaf".to_string(),
                            node_index: format!("{node_id}.missing"),
                            depth: depth + 1,
                            parent_index: Some(node_id.to_string()),
                            left_child: None,
                            right_child: None,
                            branch_bins: None,
                            branch_children: None,
                            split_feature: None,
                            split_feature_name: None,
                            split_type: None,
                            threshold_bin: None,
                            threshold_upper_bound: None,
                            operator: None,
                            leaf_value,
                            leaf_class_index,
                            leaf_label,
                            sample_count: stats.sample_count,
                            impurity: None,
                            gain: None,
                            variance: None,
                            class_counts: stats.class_counts,
                        });
                    }
                }
            }
        }
        "oblivious_levels" => {
            for level_index in 0..summary.actual_depth {
                let level = model.tree_level(tree_index, level_index).map_err(|err| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string())
                })?;
                let split = serde_json::to_value(&level.split).unwrap();
                let (
                    split_feature,
                    split_feature_name,
                    split_type,
                    threshold_bin,
                    threshold_upper_bound,
                    operator,
                ) = split_details(&split);
                rows.push(TreeDataFrameRow {
                    tree_index,
                    representation: "oblivious_levels".to_string(),
                    node_type: "level".to_string(),
                    node_index: format!("level_{level_index}"),
                    depth: level.level,
                    parent_index: (level.level > 0).then(|| format!("level_{}", level.level - 1)),
                    left_child: None,
                    right_child: None,
                    branch_bins: None,
                    branch_children: None,
                    split_feature,
                    split_feature_name,
                    split_type,
                    threshold_bin,
                    threshold_upper_bound,
                    operator,
                    leaf_value: None,
                    leaf_class_index: None,
                    leaf_label: None,
                    sample_count: level.stats.sample_count,
                    impurity: level.stats.impurity,
                    gain: level.stats.gain,
                    variance: level.stats.variance,
                    class_counts: level.stats.class_counts,
                });
            }

            for leaf_index in 0..summary.leaf_count {
                let leaf = model.tree_leaf(tree_index, leaf_index).map_err(|err| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string())
                })?;
                let (leaf_value, leaf_class_index, leaf_label) = leaf_payload_details(
                    &serde_json::to_value(&leaf.leaf).unwrap(),
                    string_class_labels,
                );
                rows.push(TreeDataFrameRow {
                    tree_index,
                    representation: "oblivious_levels".to_string(),
                    node_type: "leaf".to_string(),
                    node_index: format!("leaf_{leaf_index}"),
                    depth: summary.actual_depth,
                    parent_index: (summary.actual_depth > 0)
                        .then(|| format!("level_{}", summary.actual_depth - 1)),
                    left_child: None,
                    right_child: None,
                    branch_bins: None,
                    branch_children: None,
                    split_feature: None,
                    split_feature_name: None,
                    split_type: None,
                    threshold_bin: None,
                    threshold_upper_bound: None,
                    operator: None,
                    leaf_value,
                    leaf_class_index,
                    leaf_label,
                    sample_count: leaf.stats.sample_count,
                    impurity: leaf.stats.impurity,
                    gain: leaf.stats.gain,
                    variance: leaf.stats.variance,
                    class_counts: leaf.stats.class_counts,
                });
            }
        }
        _ => {}
    }

    Ok(rows)
}

fn dataframe_from_model<'py>(
    py: Python<'py>,
    model: &Model,
    string_class_labels: Option<&[String]>,
    tree_index: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let mut rows = Vec::new();
    let indices = if let Some(tree_index) = tree_index {
        vec![tree_index]
    } else {
        (0..model.tree_count()).collect()
    };
    for tree_index in indices {
        rows.extend(tree_rows_from_model(
            model,
            string_class_labels,
            tree_index,
        )?);
    }
    build_dataframe(py, &rows)
}

fn tree_rows_from_optimized_model(
    model: &CoreOptimizedModel,
    string_class_labels: Option<&[String]>,
    tree_index: usize,
) -> PyResult<Vec<TreeDataFrameRow>> {
    let summary = model
        .tree_structure(tree_index)
        .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;
    let mut rows = Vec::new();

    match summary.representation.as_str() {
        "node_tree" => {
            let mut nodes = Vec::new();
            let mut node_index = 0usize;
            loop {
                match model.tree_node(tree_index, node_index) {
                    Ok(node) => {
                        nodes.push(node);
                        node_index += 1;
                    }
                    Err(IntrospectionError::NodeIndexOutOfBounds { .. }) => break,
                    Err(err) => {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            err.to_string(),
                        ));
                    }
                }
            }

            let mut parents = BTreeMap::<String, String>::new();
            for node in &nodes {
                match node {
                    forestfire_core::ir::NodeTreeNode::BinaryBranch {
                        node_id, children, ..
                    } => {
                        let parent = node_id.to_string();
                        parents.insert(children.left.to_string(), parent.clone());
                        parents.insert(children.right.to_string(), parent);
                    }
                    forestfire_core::ir::NodeTreeNode::MultiwayBranch {
                        node_id, branches, ..
                    } => {
                        let parent = node_id.to_string();
                        for branch in branches {
                            parents.insert(branch.child.to_string(), parent.clone());
                        }
                    }
                    forestfire_core::ir::NodeTreeNode::Leaf { .. } => {}
                }
            }

            for node in nodes {
                match node {
                    forestfire_core::ir::NodeTreeNode::Leaf {
                        node_id,
                        depth,
                        leaf,
                        stats,
                    } => {
                        let (leaf_value, leaf_class_index, leaf_label) = leaf_payload_details(
                            &serde_json::to_value(&leaf).unwrap(),
                            string_class_labels,
                        );
                        rows.push(TreeDataFrameRow {
                            tree_index,
                            representation: "node_tree".to_string(),
                            node_type: "leaf".to_string(),
                            node_index: node_id.to_string(),
                            depth,
                            parent_index: parents.get(&node_id.to_string()).cloned(),
                            left_child: None,
                            right_child: None,
                            branch_bins: None,
                            branch_children: None,
                            split_feature: None,
                            split_feature_name: None,
                            split_type: None,
                            threshold_bin: None,
                            threshold_upper_bound: None,
                            operator: None,
                            leaf_value,
                            leaf_class_index,
                            leaf_label,
                            sample_count: stats.sample_count,
                            impurity: stats.impurity,
                            gain: stats.gain,
                            variance: stats.variance,
                            class_counts: stats.class_counts,
                        });
                    }
                    forestfire_core::ir::NodeTreeNode::BinaryBranch {
                        node_id,
                        depth,
                        split,
                        children,
                        stats,
                    } => {
                        let split = serde_json::to_value(&split).unwrap();
                        let (
                            split_feature,
                            split_feature_name,
                            split_type,
                            threshold_bin,
                            threshold_upper_bound,
                            operator,
                        ) = split_details(&split);
                        rows.push(TreeDataFrameRow {
                            tree_index,
                            representation: "node_tree".to_string(),
                            node_type: "split".to_string(),
                            node_index: node_id.to_string(),
                            depth,
                            parent_index: parents.get(&node_id.to_string()).cloned(),
                            left_child: Some(children.left.to_string()),
                            right_child: Some(children.right.to_string()),
                            branch_bins: None,
                            branch_children: None,
                            split_feature,
                            split_feature_name,
                            split_type,
                            threshold_bin,
                            threshold_upper_bound,
                            operator,
                            leaf_value: None,
                            leaf_class_index: None,
                            leaf_label: None,
                            sample_count: stats.sample_count,
                            impurity: stats.impurity,
                            gain: stats.gain,
                            variance: stats.variance,
                            class_counts: stats.class_counts,
                        });
                    }
                    forestfire_core::ir::NodeTreeNode::MultiwayBranch {
                        node_id,
                        depth,
                        split,
                        branches,
                        unmatched_leaf,
                        stats,
                    } => {
                        let split = serde_json::to_value(&split).unwrap();
                        let (
                            split_feature,
                            split_feature_name,
                            split_type,
                            threshold_bin,
                            threshold_upper_bound,
                            operator,
                        ) = split_details(&split);
                        rows.push(TreeDataFrameRow {
                            tree_index,
                            representation: "node_tree".to_string(),
                            node_type: "split".to_string(),
                            node_index: node_id.to_string(),
                            depth,
                            parent_index: parents.get(&node_id.to_string()).cloned(),
                            left_child: None,
                            right_child: None,
                            branch_bins: Some(branches.iter().map(|branch| branch.bin).collect()),
                            branch_children: Some(
                                branches
                                    .iter()
                                    .map(|branch| branch.child.to_string())
                                    .collect(),
                            ),
                            split_feature,
                            split_feature_name,
                            split_type,
                            threshold_bin,
                            threshold_upper_bound,
                            operator,
                            leaf_value: None,
                            leaf_class_index: None,
                            leaf_label: None,
                            sample_count: stats.sample_count,
                            impurity: stats.impurity,
                            gain: stats.gain,
                            variance: stats.variance,
                            class_counts: stats.class_counts.clone(),
                        });

                        let (leaf_value, leaf_class_index, leaf_label) = leaf_payload_details(
                            &serde_json::to_value(&unmatched_leaf).unwrap(),
                            string_class_labels,
                        );
                        rows.push(TreeDataFrameRow {
                            tree_index,
                            representation: "node_tree".to_string(),
                            node_type: "unmatched_leaf".to_string(),
                            node_index: format!("{node_id}.missing"),
                            depth: depth + 1,
                            parent_index: Some(node_id.to_string()),
                            left_child: None,
                            right_child: None,
                            branch_bins: None,
                            branch_children: None,
                            split_feature: None,
                            split_feature_name: None,
                            split_type: None,
                            threshold_bin: None,
                            threshold_upper_bound: None,
                            operator: None,
                            leaf_value,
                            leaf_class_index,
                            leaf_label,
                            sample_count: stats.sample_count,
                            impurity: None,
                            gain: None,
                            variance: None,
                            class_counts: stats.class_counts,
                        });
                    }
                }
            }
        }
        "oblivious_levels" => {
            for level_index in 0..summary.actual_depth {
                let level = model.tree_level(tree_index, level_index).map_err(|err| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string())
                })?;
                let split = serde_json::to_value(&level.split).unwrap();
                let (
                    split_feature,
                    split_feature_name,
                    split_type,
                    threshold_bin,
                    threshold_upper_bound,
                    operator,
                ) = split_details(&split);
                rows.push(TreeDataFrameRow {
                    tree_index,
                    representation: "oblivious_levels".to_string(),
                    node_type: "level".to_string(),
                    node_index: format!("level_{level_index}"),
                    depth: level.level,
                    parent_index: (level.level > 0).then(|| format!("level_{}", level.level - 1)),
                    left_child: None,
                    right_child: None,
                    branch_bins: None,
                    branch_children: None,
                    split_feature,
                    split_feature_name,
                    split_type,
                    threshold_bin,
                    threshold_upper_bound,
                    operator,
                    leaf_value: None,
                    leaf_class_index: None,
                    leaf_label: None,
                    sample_count: level.stats.sample_count,
                    impurity: level.stats.impurity,
                    gain: level.stats.gain,
                    variance: level.stats.variance,
                    class_counts: level.stats.class_counts,
                });
            }

            for leaf_index in 0..summary.leaf_count {
                let leaf = model.tree_leaf(tree_index, leaf_index).map_err(|err| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string())
                })?;
                let (leaf_value, leaf_class_index, leaf_label) = leaf_payload_details(
                    &serde_json::to_value(&leaf.leaf).unwrap(),
                    string_class_labels,
                );
                rows.push(TreeDataFrameRow {
                    tree_index,
                    representation: "oblivious_levels".to_string(),
                    node_type: "leaf".to_string(),
                    node_index: format!("leaf_{leaf_index}"),
                    depth: summary.actual_depth,
                    parent_index: (summary.actual_depth > 0)
                        .then(|| format!("level_{}", summary.actual_depth - 1)),
                    left_child: None,
                    right_child: None,
                    branch_bins: None,
                    branch_children: None,
                    split_feature: None,
                    split_feature_name: None,
                    split_type: None,
                    threshold_bin: None,
                    threshold_upper_bound: None,
                    operator: None,
                    leaf_value,
                    leaf_class_index,
                    leaf_label,
                    sample_count: leaf.stats.sample_count,
                    impurity: leaf.stats.impurity,
                    gain: leaf.stats.gain,
                    variance: leaf.stats.variance,
                    class_counts: leaf.stats.class_counts,
                });
            }
        }
        _ => {}
    }

    Ok(rows)
}

fn dataframe_from_optimized_model<'py>(
    py: Python<'py>,
    model: &CoreOptimizedModel,
    string_class_labels: Option<&[String]>,
    tree_index: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let mut rows = Vec::new();
    let indices = if let Some(tree_index) = tree_index {
        vec![tree_index]
    } else {
        (0..model.tree_count()).collect()
    };
    for tree_index in indices {
        rows.extend(tree_rows_from_optimized_model(
            model,
            string_class_labels,
            tree_index,
        )?);
    }
    build_dataframe(py, &rows)
}

fn build_sparse_training_table(
    x: &Bound<PyAny>,
    y: &Bound<PyAny>,
    canaries: usize,
    bins: NumericBins,
) -> PyResult<(Table, Option<Vec<String>>)> {
    let (n_rows, n_features, columns) = extract_sparse_binary_columns(x)?;
    let (y_values, string_class_labels) = match extract_training_targets(y)? {
        TrainingTargets::Numeric(values) => (values, None),
        TrainingTargets::StringClasses { encoded, labels } => (encoded, Some(labels)),
    };
    let table = forestfire_data::SparseTable::from_sparse_binary_columns_with_options(
        n_rows, n_features, columns, y_values, canaries, bins,
    )
    .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;
    Ok((Table::Sparse(table), string_class_labels))
}

fn build_sparse_feature_table(x: &Bound<PyAny>, bins: NumericBins) -> PyResult<Table> {
    let (n_rows, n_features, columns) = extract_sparse_binary_columns(x)?;
    let table = forestfire_data::SparseTable::from_sparse_binary_columns_with_options(
        n_rows,
        n_features,
        columns,
        vec![0.0; n_rows],
        0,
        bins,
    )
    .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;
    Ok(Table::Sparse(table))
}

fn extract_sparse_binary_columns(x: &Bound<PyAny>) -> PyResult<(usize, usize, Vec<Vec<usize>>)> {
    let shape_any = x.getattr("shape")?;
    let shape: Vec<usize> = shape_any.extract()?;
    if shape.len() != 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Sparse inputs must be two-dimensional.",
        ));
    }
    let n_rows = shape[0];
    let n_features = shape[1];

    let data = x.getattr("data")?;
    let values = extract_vector(&data)?;
    if values
        .iter()
        .any(|value| value.total_cmp(&1.0) != std::cmp::Ordering::Equal)
    {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "SparseTable requires binary sparse inputs with stored values equal to 1.",
        ));
    }

    let nonzero = x.call_method0("nonzero")?;
    let (row_indices_any, col_indices_any): (Bound<'_, PyAny>, Bound<'_, PyAny>) =
        nonzero.extract()?;
    let row_indices: Vec<usize> = extract_index_vector(&row_indices_any)?;
    let col_indices: Vec<usize> = extract_index_vector(&col_indices_any)?;

    if row_indices.len() != col_indices.len() || row_indices.len() != values.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Sparse input metadata is inconsistent.",
        ));
    }

    let mut columns = vec![Vec::new(); n_features];
    for (row_idx, col_idx) in row_indices.into_iter().zip(col_indices) {
        if row_idx >= n_rows || col_idx >= n_features {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Sparse input indices are out of bounds.",
            ));
        }
        columns[col_idx].push(row_idx);
    }

    Ok((n_rows, n_features, columns))
}

fn extract_index_vector(values: &Bound<PyAny>) -> PyResult<Vec<usize>> {
    if let Ok(array) = values.extract::<PyReadonlyArray1<'_, i64>>() {
        return array
            .as_array()
            .iter()
            .map(|value| {
                usize::try_from(*value).map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Sparse indices must be non-negative.",
                    )
                })
            })
            .collect();
    }

    if let Ok(array) = values.extract::<PyReadonlyArray1<'_, i32>>() {
        return array
            .as_array()
            .iter()
            .map(|value| {
                usize::try_from(*value).map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Sparse indices must be non-negative.",
                    )
                })
            })
            .collect();
    }

    values
        .try_iter()?
        .map(|value| {
            value.and_then(|value| {
                let value = value.extract::<i64>()?;
                usize::try_from(value).map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Sparse indices must be non-negative.",
                    )
                })
            })
        })
        .collect()
}

fn extract_matrix(x: &Bound<PyAny>) -> PyResult<Vec<Vec<f64>>> {
    if x.hasattr("collect")? {
        let collected = x.call_method0("collect")?;
        return extract_matrix(&collected);
    }

    if let Ok(array) = x.extract::<PyReadonlyArray2<'_, f64>>() {
        let view = array.as_array();
        return Ok(view.rows().into_iter().map(|row| row.to_vec()).collect());
    }

    if let Ok(array) = x.extract::<PyReadonlyArray2<'_, bool>>() {
        let view = array.as_array();
        return Ok(view
            .rows()
            .into_iter()
            .map(|row| {
                row.iter()
                    .map(|value| f64::from(u8::from(*value)))
                    .collect()
            })
            .collect());
    }

    if let Ok(array) = x.extract::<PyReadonlyArray2<'_, i64>>() {
        let view = array.as_array();
        return Ok(view
            .rows()
            .into_iter()
            .map(|row| row.iter().map(|value| *value as f64).collect())
            .collect());
    }

    if let Ok(array) = x.extract::<PyReadonlyArray2<'_, i32>>() {
        let view = array.as_array();
        return Ok(view
            .rows()
            .into_iter()
            .map(|row| row.iter().map(|value| f64::from(*value)).collect())
            .collect());
    }

    if x.hasattr("toarray")? {
        let as_array = x.call_method0("toarray")?;
        return extract_matrix(&as_array);
    }

    if x.hasattr("todense")? {
        let as_dense = x.call_method0("todense")?;
        return extract_matrix(&as_dense);
    }

    if x.hasattr("__array__")? {
        let as_array = x.call_method0("__array__")?;
        if as_array.is(x) {
            return extract_matrix_from_rows(x);
        }
        return extract_matrix(&as_array);
    }

    if x.hasattr("to_numpy")? {
        let as_numpy = x.call_method0("to_numpy")?;
        return extract_matrix(&as_numpy);
    }

    if x.hasattr("to_pydict")? {
        let columns = x.call_method0("to_pydict")?;
        return extract_matrix_from_columns(&columns);
    }

    if x.hasattr("to_pylist")? {
        let rows = x.call_method0("to_pylist")?;
        return extract_matrix_from_rows(&rows);
    }

    extract_matrix_from_rows(x)
}

fn extract_rows_or_single_row(x: &Bound<PyAny>) -> PyResult<Vec<Vec<f64>>> {
    match extract_matrix(x) {
        Ok(rows) => Ok(rows),
        Err(matrix_error) => match extract_vector(x) {
            Ok(row) => Ok(vec![row]),
            Err(_) => Err(matrix_error),
        },
    }
}

fn extract_named_columns(x: &Bound<PyAny>) -> PyResult<BTreeMap<String, Vec<f64>>> {
    if x.hasattr("collect")? {
        let collected = x.call_method0("collect")?;
        return extract_named_columns(&collected);
    }

    if let Ok(columns) = x.cast::<PyDict>() {
        return extract_named_columns_from_dict(columns);
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "Input is not a named-column mapping.",
    ))
}

fn extract_named_columns_from_dict(
    columns: &Bound<'_, PyDict>,
) -> PyResult<BTreeMap<String, Vec<f64>>> {
    let scalar_row = columns
        .iter()
        .next()
        .is_some_and(|(_, value)| value.try_iter().is_err());
    if scalar_row {
        return columns
            .iter()
            .map(|(name, value)| Ok((name.extract::<String>()?, vec![extract_scalar(&value)?])))
            .collect();
    }

    columns
        .iter()
        .map(|(name, values)| Ok((name.extract::<String>()?, extract_vector(&values)?)))
        .collect()
}

fn extract_matrix_from_columns(columns: &Bound<PyAny>) -> PyResult<Vec<Vec<f64>>> {
    let columns = columns.cast::<PyDict>()?;
    let column_values: Vec<Vec<f64>> = columns
        .iter()
        .map(|(_name, values)| extract_vector(&values))
        .collect::<PyResult<_>>()?;

    let n_rows = column_values.first().map_or(0, Vec::len);
    if column_values.iter().any(|column| column.len() != n_rows) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Column-oriented input must contain equally sized columns.",
        ));
    }

    Ok((0..n_rows)
        .map(|row_idx| {
            column_values
                .iter()
                .map(|column| column[row_idx])
                .collect::<Vec<_>>()
        })
        .collect())
}

fn extract_matrix_from_rows(rows: &Bound<PyAny>) -> PyResult<Vec<Vec<f64>>> {
    rows.try_iter()?
        .map(|row| {
            let row = row?;
            row.try_iter()?
                .map(|value| value.and_then(|value| extract_scalar(&value)))
                .collect::<PyResult<Vec<_>>>()
        })
        .collect()
}

fn extract_categorical_scalar(value: &Bound<PyAny>) -> PyResult<CategoricalValue> {
    if value.is_none() {
        return Ok(CategoricalValue::Missing);
    }
    if let Ok(value) = value.extract::<String>() {
        return Ok(CategoricalValue::String(value));
    }
    if let Ok(value) = value.extract::<bool>() {
        return Ok(CategoricalValue::Numeric(f64::from(u8::from(value))));
    }
    if let Ok(value) = value.extract::<f64>() {
        if value.is_nan() {
            return Ok(CategoricalValue::Missing);
        }
        return Ok(CategoricalValue::Numeric(value));
    }
    if let Ok(value) = value.extract::<i64>() {
        return Ok(CategoricalValue::Numeric(value as f64));
    }
    if let Ok(value) = value.extract::<i32>() {
        return Ok(CategoricalValue::Numeric(f64::from(value)));
    }
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "Categorical inputs must contain strings, booleans, numbers, or missing values.",
    ))
}

fn extract_categorical_matrix_from_rows(
    rows: &Bound<PyAny>,
) -> PyResult<Vec<Vec<CategoricalValue>>> {
    rows.try_iter()?
        .map(|row| {
            let row = row?;
            row.try_iter()?
                .map(|value| value.and_then(|value| extract_categorical_scalar(&value)))
                .collect::<PyResult<Vec<_>>>()
        })
        .collect()
}

fn extract_categorical_matrix(x: &Bound<PyAny>) -> PyResult<Vec<Vec<CategoricalValue>>> {
    if x.hasattr("collect")? {
        let collected = x.call_method0("collect")?;
        return extract_categorical_matrix(&collected);
    }

    if x.hasattr("to_pylist")? {
        let rows = x.call_method0("to_pylist")?;
        return extract_categorical_matrix_from_rows(&rows);
    }

    if x.hasattr("to_dict")? {
        if let Ok(columns) = x.call_method("to_dict", ("list",), None) {
            let dict = columns.cast::<PyDict>()?;
            let column_names = dict
                .iter()
                .map(|(name, _)| name.extract::<String>())
                .collect::<PyResult<Vec<_>>>()?;
            let values = dict
                .iter()
                .map(|(_, values)| {
                    values
                        .try_iter()?
                        .map(|value| value.and_then(|value| extract_categorical_scalar(&value)))
                        .collect::<PyResult<Vec<_>>>()
                })
                .collect::<PyResult<Vec<_>>>()?;
            let n_rows = values.first().map_or(0, Vec::len);
            let _ = column_names;
            return Ok((0..n_rows)
                .map(|row_idx| {
                    values
                        .iter()
                        .map(|column| column[row_idx].clone())
                        .collect()
                })
                .collect());
        }
    }

    if x.hasattr("__array__")? {
        let as_array = x.call_method0("__array__")?;
        if !as_array.is(x) {
            return extract_categorical_matrix(&as_array);
        }
    }

    extract_categorical_matrix_from_rows(x)
}

fn extract_vector(y: &Bound<PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(array) = y.extract::<PyReadonlyArray1<'_, f64>>() {
        return Ok(array.as_array().to_vec());
    }

    if let Ok(array) = y.extract::<PyReadonlyArray2<'_, f64>>() {
        let shape = array.shape();
        if shape[0] == 1 || shape[1] == 1 {
            return Ok(array.as_array().iter().copied().collect());
        }
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Target arrays must be one-dimensional or have a single row/column.",
        ));
    }

    if let Ok(array) = y.extract::<PyReadonlyArray1<'_, bool>>() {
        return Ok(array
            .as_array()
            .iter()
            .map(|value| f64::from(u8::from(*value)))
            .collect());
    }

    if let Ok(array) = y.extract::<PyReadonlyArray2<'_, bool>>() {
        let shape = array.shape();
        if shape[0] == 1 || shape[1] == 1 {
            return Ok(array
                .as_array()
                .iter()
                .map(|value| f64::from(u8::from(*value)))
                .collect());
        }
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Target arrays must be one-dimensional or have a single row/column.",
        ));
    }

    if let Ok(array) = y.extract::<PyReadonlyArray1<'_, i64>>() {
        return Ok(array.as_array().iter().map(|value| *value as f64).collect());
    }

    if let Ok(array) = y.extract::<PyReadonlyArray2<'_, i64>>() {
        let shape = array.shape();
        if shape[0] == 1 || shape[1] == 1 {
            return Ok(array.as_array().iter().map(|value| *value as f64).collect());
        }
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Target arrays must be one-dimensional or have a single row/column.",
        ));
    }

    if let Ok(array) = y.extract::<PyReadonlyArray1<'_, i32>>() {
        return Ok(array
            .as_array()
            .iter()
            .map(|value| f64::from(*value))
            .collect());
    }

    if let Ok(array) = y.extract::<PyReadonlyArray2<'_, i32>>() {
        let shape = array.shape();
        if shape[0] == 1 || shape[1] == 1 {
            return Ok(array
                .as_array()
                .iter()
                .map(|value| f64::from(*value))
                .collect());
        }
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Target arrays must be one-dimensional or have a single row/column.",
        ));
    }

    if y.hasattr("toarray")? {
        let as_array = y.call_method0("toarray")?;
        return extract_vector(&as_array);
    }

    if y.hasattr("todense")? {
        let as_dense = y.call_method0("todense")?;
        return extract_vector(&as_dense);
    }

    if y.hasattr("__array__")? {
        let as_array = y.call_method0("__array__")?;
        if as_array.is(y) {
            return y
                .try_iter()?
                .map(|value| value.and_then(|value| extract_scalar(&value)))
                .collect();
        }
        return extract_vector(&as_array);
    }

    if y.hasattr("to_numpy")? {
        let as_numpy = y.call_method0("to_numpy")?;
        return extract_vector(&as_numpy);
    }

    if y.hasattr("to_pylist")? {
        let as_list = y.call_method0("to_pylist")?;
        return extract_vector(&as_list);
    }

    y.try_iter()?
        .map(|value| value.and_then(|value| extract_scalar(&value)))
        .collect()
}

fn extract_training_targets(y: &Bound<PyAny>) -> PyResult<TrainingTargets> {
    if y.hasattr("to_pylist")? {
        let as_list = y.call_method0("to_pylist")?;
        if let Ok(strings) = as_list.extract::<Vec<String>>() {
            return encode_string_labels(strings);
        }
        if let Ok(strings) = as_list.extract::<Vec<Vec<String>>>() {
            let is_vector = strings.len() == 1 || strings.iter().all(|row| row.len() == 1);
            if is_vector {
                return encode_string_labels(
                    strings
                        .into_iter()
                        .flat_map(|row| row.into_iter())
                        .collect::<Vec<_>>(),
                );
            }
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Target arrays must be one-dimensional or have a single row/column.",
            ));
        }
    }

    if y.hasattr("tolist")? {
        let as_list = y.call_method0("tolist")?;
        if let Ok(strings) = as_list.extract::<Vec<String>>() {
            return encode_string_labels(strings);
        }
        if let Ok(strings) = as_list.extract::<Vec<Vec<String>>>() {
            let is_vector = strings.len() == 1 || strings.iter().all(|row| row.len() == 1);
            if is_vector {
                return encode_string_labels(
                    strings
                        .into_iter()
                        .flat_map(|row| row.into_iter())
                        .collect::<Vec<_>>(),
                );
            }
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Target arrays must be one-dimensional or have a single row/column.",
            ));
        }
    }

    Ok(TrainingTargets::Numeric(extract_vector(y)?))
}

fn encode_string_labels(labels: Vec<String>) -> PyResult<TrainingTargets> {
    let mut vocabulary = Vec::<String>::new();
    let mut encoded = Vec::with_capacity(labels.len());

    for label in labels {
        let index = if let Some(index) = vocabulary.iter().position(|candidate| candidate == &label)
        {
            index
        } else {
            vocabulary.push(label.clone());
            vocabulary.len() - 1
        };
        encoded.push(index as f64);
    }

    Ok(TrainingTargets::StringClasses {
        encoded,
        labels: vocabulary,
    })
}

fn extract_scalar(value: &Bound<PyAny>) -> PyResult<f64> {
    if value.is_none() {
        return Ok(f64::NAN);
    }

    if let Ok(value) = value.extract::<bool>() {
        return Ok(f64::from(u8::from(value)));
    }

    value.extract::<f64>()
}

fn parse_algorithm(algorithm: &str) -> PyResult<TrainAlgorithm> {
    match algorithm {
        "dt" => Ok(TrainAlgorithm::Dt),
        "rf" => Ok(TrainAlgorithm::Rf),
        "gbm" => Ok(TrainAlgorithm::Gbm),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unsupported algorithm '{}'. Expected one of: dt, rf, gbm",
            algorithm
        ))),
    }
}

fn parse_tree_type(tree_type: &str) -> PyResult<TreeType> {
    match tree_type {
        "id3" => Ok(TreeType::Id3),
        "c45" => Ok(TreeType::C45),
        "cart" => Ok(TreeType::Cart),
        "randomized" => Ok(TreeType::Randomized),
        "oblivious" => Ok(TreeType::Oblivious),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unsupported tree_type '{}'. Expected one of: id3, c45, cart, randomized, oblivious",
            tree_type
        ))),
    }
}

fn resolve_tree_type(requested_tree_type: &str, task_was_auto: bool) -> PyResult<TreeType> {
    if task_was_auto && requested_tree_type == "cart" {
        return Ok(TreeType::Cart);
    }

    parse_tree_type(requested_tree_type)
}

fn parse_task(task: &str) -> PyResult<Task> {
    match task {
        "regression" => Ok(Task::Regression),
        "classification" => Ok(Task::Classification),
        "auto" => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Internal error: task='auto' must be resolved before parse_task.",
        )),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unsupported task '{}'. Expected one of: regression, classification, auto",
            task
        ))),
    }
}

fn resolve_task(x: &Bound<PyAny>, y: Option<&Bound<PyAny>>, task: &str) -> PyResult<Task> {
    if task != "auto" {
        return parse_task(task);
    }

    if let Some(y) = y {
        if target_is_string_like(y)? || target_is_integer_like(y)? {
            return Ok(Task::Classification);
        }
        return Ok(Task::Regression);
    }

    if let Ok(table) = x.extract::<PyRef<'_, PyTable>>() {
        let all_integral = (0..table.inner.n_rows()).all(|row_idx| {
            let value = table.inner.target_value(row_idx);
            value.is_finite() && value.fract() == 0.0
        });
        if all_integral {
            return Ok(Task::Classification);
        }
        return Ok(Task::Regression);
    }

    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
        "task='auto' requires y unless x is already a Table.",
    ))
}

fn target_is_integer_like(y: &Bound<PyAny>) -> PyResult<bool> {
    if y.extract::<PyReadonlyArray1<'_, i64>>().is_ok()
        || y.extract::<PyReadonlyArray2<'_, i64>>().is_ok()
        || y.extract::<PyReadonlyArray1<'_, i32>>().is_ok()
        || y.extract::<PyReadonlyArray2<'_, i32>>().is_ok()
        || y.extract::<PyReadonlyArray1<'_, bool>>().is_ok()
        || y.extract::<PyReadonlyArray2<'_, bool>>().is_ok()
    {
        return Ok(true);
    }

    if y.hasattr("to_pylist")? {
        let as_list = y.call_method0("to_pylist")?;
        if let Ok(values) = as_list.extract::<Vec<i64>>() {
            return Ok(!values.is_empty() || as_list.len().unwrap_or(0) == 0);
        }
    }

    if y.hasattr("tolist")? {
        let as_list = y.call_method0("tolist")?;
        if as_list.extract::<Vec<i64>>().is_ok() || as_list.extract::<Vec<Vec<i64>>>().is_ok() {
            return Ok(true);
        }
    }

    Ok(false)
}

fn target_is_string_like(y: &Bound<PyAny>) -> PyResult<bool> {
    if y.hasattr("to_pylist")? {
        let as_list = y.call_method0("to_pylist")?;
        if as_list.extract::<Vec<String>>().is_ok() || as_list.extract::<Vec<Vec<String>>>().is_ok()
        {
            return Ok(true);
        }
    }

    if y.hasattr("tolist")? {
        let as_list = y.call_method0("tolist")?;
        if as_list.extract::<Vec<String>>().is_ok() || as_list.extract::<Vec<Vec<String>>>().is_ok()
        {
            return Ok(true);
        }
    }

    Ok(false)
}

fn parse_criterion(criterion: &str) -> PyResult<Criterion> {
    match criterion {
        "auto" => Ok(Criterion::Auto),
        "gini" => Ok(Criterion::Gini),
        "entropy" => Ok(Criterion::Entropy),
        "mean" => Ok(Criterion::Mean),
        "median" => Ok(Criterion::Median),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unsupported criterion '{}'. Expected one of: auto, gini, entropy, mean, median",
            criterion
        ))),
    }
}

fn parse_missing_value_strategy_value(value: &str) -> PyResult<MissingValueStrategy> {
    match value {
        "heuristic" => Ok(MissingValueStrategy::Heuristic),
        "optimal" => Ok(MissingValueStrategy::Optimal),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unsupported missing_value_strategy '{}'. Expected 'heuristic' or 'optimal'.",
            value
        ))),
    }
}

fn parse_missing_value_strategy_key(key: &str) -> PyResult<usize> {
    if let Some(index) = key.strip_prefix('f') {
        return index.parse::<usize>().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!(
                    "Invalid missing_value_strategy feature key '{}'. Expected names like 'f0' or 'col_1'.",
                    key
                ),
            )
        });
    }
    if let Some(index) = key.strip_prefix("col_") {
        let parsed = index.parse::<usize>().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!(
                    "Invalid missing_value_strategy feature key '{}'. Expected names like 'f0' or 'col_1'.",
                    key
                ),
            )
        })?;
        return Ok(parsed.saturating_sub(1));
    }
    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
        "Invalid missing_value_strategy feature key '{}'. Expected names like 'f0' or 'col_1'.",
        key
    )))
}

fn parse_missing_value_strategy(
    missing_value_strategy: Option<&Bound<'_, PyAny>>,
) -> PyResult<MissingValueStrategyConfig> {
    let Some(missing_value_strategy) = missing_value_strategy else {
        return Ok(MissingValueStrategyConfig::heuristic());
    };
    if let Ok(strategy) = missing_value_strategy.extract::<String>() {
        return Ok(MissingValueStrategyConfig::Global(
            parse_missing_value_strategy_value(&strategy)?,
        ));
    }
    if let Ok(strategy_map) = missing_value_strategy.cast::<PyDict>() {
        let mut resolved = BTreeMap::new();
        for (key, value) in strategy_map.iter() {
            let key = key.extract::<String>()?;
            let value = value.extract::<String>()?;
            resolved.insert(
                parse_missing_value_strategy_key(&key)?,
                parse_missing_value_strategy_value(&value)?,
            );
        }
        return Ok(MissingValueStrategyConfig::PerFeature(resolved));
    }
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "missing_value_strategy must be 'heuristic', 'optimal', or a dict like {'col_1': 'optimal'}.",
    ))
}

fn parse_max_features(value: Option<&Bound<PyAny>>) -> PyResult<MaxFeatures> {
    let Some(value) = value else {
        return Ok(MaxFeatures::Auto);
    };

    if value.is_none() {
        return Ok(MaxFeatures::Auto);
    }

    if let Ok(text) = value.extract::<String>() {
        return match text.as_str() {
            "auto" => Ok(MaxFeatures::Auto),
            "all" => Ok(MaxFeatures::All),
            "sqrt" => Ok(MaxFeatures::Sqrt),
            "third" => Ok(MaxFeatures::Third),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unsupported max_features '{}'. Expected one of: auto, all, sqrt, third, or a positive integer",
                text
            ))),
        };
    }

    if let Ok(count) = value.extract::<usize>() {
        return Ok(MaxFeatures::Count(count));
    }

    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
        "Unsupported max_features value. Expected one of: auto, all, sqrt, third, or a positive integer",
    ))
}

fn parse_optional_positive_usize(value: Option<usize>, name: &str) -> PyResult<Option<usize>> {
    match value {
        Some(0) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{} must be at least 1.",
            name
        ))),
        Some(value) => Ok(Some(value)),
        None => Ok(None),
    }
}

fn parse_optional_positive_f64(value: Option<f64>, name: &str) -> PyResult<Option<f64>> {
    match value {
        Some(value) if !value.is_finite() || value <= 0.0 => {
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "{} must be a finite value greater than 0.",
                name
            )))
        }
        Some(value) => Ok(Some(value)),
        None => Ok(None),
    }
}

fn parse_optional_fraction(value: Option<f64>, name: &str) -> PyResult<Option<f64>> {
    match value {
        Some(value) if !value.is_finite() || !(0.0..=1.0).contains(&value) => {
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "{} must be a finite value between 0 and 1.",
                name
            )))
        }
        Some(value) => Ok(Some(value)),
        None => Ok(None),
    }
}

fn parse_canary_filter(value: Option<&Bound<PyAny>>) -> PyResult<CanaryFilter> {
    let Some(value) = value else {
        return Ok(CanaryFilter::default());
    };

    if value.is_none() {
        return Ok(CanaryFilter::default());
    }

    if let Ok(count) = value.extract::<usize>() {
        if count == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "filter must be a positive integer when provided as an int.",
            ));
        }
        return Ok(CanaryFilter::TopN(count));
    }

    if let Ok(alpha) = value.extract::<f64>() {
        if !alpha.is_finite() || !(0.0..1.0).contains(&alpha) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "filter must be a finite float in [0, 1) when provided as a float.",
            ));
        }
        return Ok(CanaryFilter::TopFraction(1.0 - alpha));
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "filter must be a positive integer, a float in [0, 1), or None.",
    ))
}

fn parse_categorical_strategy(value: Option<&str>) -> PyResult<Option<CategoricalStrategy>> {
    match value {
        None => Ok(None),
        Some("dummy") | Some("one_hot") => Ok(Some(CategoricalStrategy::Dummy)),
        Some("target") => Ok(Some(CategoricalStrategy::Target)),
        Some("fisher") => Ok(Some(CategoricalStrategy::Fisher)),
        Some(value) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unsupported categorical_strategy '{}'. Expected one of: dummy, target, fisher.",
            value
        ))),
    }
}

fn parse_categorical_features(value: Option<&Bound<PyAny>>) -> PyResult<Option<Vec<usize>>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    if let Ok(text) = value.extract::<String>() {
        if text == "all" {
            return Ok(Some(Vec::new()));
        }
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "categorical_features string form currently only supports 'all'.",
        ));
    }
    if let Ok(features) = value.extract::<Vec<usize>>() {
        return Ok(Some(features));
    }
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "categorical_features must be None, 'all', or a list of integer feature indices.",
    ))
}

fn parse_bins(bins: Option<&Bound<PyAny>>) -> PyResult<NumericBins> {
    let Some(bins) = bins else {
        return Ok(NumericBins::Auto);
    };

    if let Ok(value) = bins.extract::<usize>() {
        return NumericBins::fixed(value)
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()));
    }

    if let Ok(value) = bins.extract::<String>() {
        if value == "auto" {
            return Ok(NumericBins::Auto);
        }
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unsupported bins value '{}'. Expected 'auto' or an integer between 1 and {}.",
            value, MAX_NUMERIC_BINS
        )));
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
        "Unsupported bins value. Expected 'auto' or an integer between 1 and {}.",
        MAX_NUMERIC_BINS
    )))
}

fn algorithm_name(algorithm: TrainAlgorithm) -> &'static str {
    match algorithm {
        TrainAlgorithm::Dt => "dt",
        TrainAlgorithm::Rf => "rf",
        TrainAlgorithm::Gbm => "gbm",
    }
}

fn task_name(task: Task) -> &'static str {
    match task {
        Task::Regression => "regression",
        Task::Classification => "classification",
    }
}

fn criterion_name(criterion: Criterion) -> &'static str {
    match criterion {
        Criterion::Auto => "auto",
        Criterion::Gini => "gini",
        Criterion::Entropy => "entropy",
        Criterion::Mean => "mean",
        Criterion::Median => "median",
        Criterion::SecondOrder => "second_order",
    }
}

fn tree_type_name(tree_type: TreeType) -> &'static str {
    match tree_type {
        TreeType::Id3 => "id3",
        TreeType::C45 => "c45",
        TreeType::Cart => "cart",
        TreeType::Randomized => "randomized",
        TreeType::Oblivious => "oblivious",
    }
}

#[pyfunction]
#[pyo3(signature = (x, y=None, algorithm="dt", task="auto", tree_type="cart", criterion="auto", canaries=2, bins=None, histogram_bins=None, physical_cores=None, max_depth=None, min_samples_split=None, min_samples_leaf=None, n_trees=None, max_features=None, seed=None, compute_oob=false, learning_rate=None, bootstrap=false, top_gradient_fraction=None, other_gradient_fraction=None, missing_value_strategy=None, filter=None, categorical_strategy=None, categorical_features=None, target_smoothing=20.0))]
#[allow(clippy::too_many_arguments)]
fn train(
    py: Python<'_>,
    x: &Bound<PyAny>,
    y: Option<&Bound<PyAny>>,
    algorithm: &str,
    task: &str,
    tree_type: &str,
    criterion: &str,
    canaries: usize,
    bins: Option<&Bound<PyAny>>,
    histogram_bins: Option<&Bound<PyAny>>,
    physical_cores: Option<usize>,
    max_depth: Option<usize>,
    min_samples_split: Option<usize>,
    min_samples_leaf: Option<usize>,
    n_trees: Option<usize>,
    max_features: Option<&Bound<PyAny>>,
    seed: Option<u64>,
    compute_oob: bool,
    learning_rate: Option<f64>,
    bootstrap: bool,
    top_gradient_fraction: Option<f64>,
    other_gradient_fraction: Option<f64>,
    missing_value_strategy: Option<&Bound<PyAny>>,
    filter: Option<&Bound<PyAny>>,
    categorical_strategy: Option<&str>,
    categorical_features: Option<&Bound<PyAny>>,
    target_smoothing: f64,
) -> PyResult<Py<PyAny>> {
    let task_was_auto = task == "auto";
    let resolved_task = resolve_task(x, y, task)?;
    let parsed_strategy = parse_categorical_strategy(categorical_strategy)?;
    let parsed_bins = parse_bins(bins)?;
    let config = TrainConfig {
        algorithm: parse_algorithm(algorithm)?,
        task: resolved_task,
        tree_type: resolve_tree_type(tree_type, task_was_auto)?,
        criterion: parse_criterion(criterion)?,
        max_depth: parse_optional_positive_usize(max_depth, "max_depth")?,
        min_samples_split: parse_optional_positive_usize(min_samples_split, "min_samples_split")?,
        min_samples_leaf: parse_optional_positive_usize(min_samples_leaf, "min_samples_leaf")?,
        physical_cores,
        n_trees,
        max_features: parse_max_features(max_features)?,
        seed,
        canary_filter: parse_canary_filter(filter)?,
        compute_oob,
        learning_rate: parse_optional_positive_f64(learning_rate, "learning_rate")?,
        bootstrap,
        top_gradient_fraction: parse_optional_fraction(
            top_gradient_fraction,
            "top_gradient_fraction",
        )?,
        other_gradient_fraction: parse_optional_fraction(
            other_gradient_fraction,
            "other_gradient_fraction",
        )?,
        missing_value_strategy: parse_missing_value_strategy(missing_value_strategy)?,
        histogram_bins: histogram_bins
            .map(|value| parse_bins(Some(value)))
            .transpose()?,
    };
    if let Some(parsed_strategy) = parsed_strategy {
        let y = y.ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "y is required unless x is already a Table.",
            )
        })?;
        let rows = extract_categorical_matrix(x)?;
        let (targets, string_class_labels) = match extract_training_targets(y)? {
            TrainingTargets::Numeric(values) => (values, None),
            TrainingTargets::StringClasses { encoded, labels } => (encoded, Some(labels)),
        };
        let parsed_features = parse_categorical_features(categorical_features)?;
        let parsed_features = parsed_features.map(|features| {
            if features.is_empty() {
                (0..rows.first().map_or(0, |row| row.len())).collect()
            } else {
                features
            }
        });
        let inner = py
            .detach(move || {
                categorical::train(
                    rows,
                    targets,
                    None,
                    canaries,
                    parsed_bins,
                    config,
                    CategoricalConfig {
                        strategy: parsed_strategy,
                        categorical_features: parsed_features,
                        target_smoothing,
                    },
                )
                .map_err(|err| err.to_string())
            })
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        let model = PyCategoricalModel {
            inner,
            string_class_labels,
        };
        return Py::new(py, model).map(|obj| obj.into_any());
    }

    let (table, string_class_labels) = build_training_table(x, y, canaries, parsed_bins)?;
    let model = train_model_detached(py, table, config)?;
    Py::new(
        py,
        PyModel {
            inner: model,
            string_class_labels,
        },
    )
    .map(|obj| obj.into_any())
}

#[pymethods]
impl PyModel {
    #[classmethod]
    fn deserialize(_cls: &Bound<PyType>, serialized: &str) -> PyResult<Self> {
        let inner = Model::deserialize(serialized)
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;
        Ok(Self {
            inner,
            string_class_labels: None,
        })
    }

    fn predict<'py>(&self, py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let preds = if is_polars_lazyframe(x)? {
            predict_lazyframe_in_batches(x, |input| {
                predict_input_with_model_detached(py, &self.inner, input)
            })?
        } else {
            predict_input_with_model_detached(py, &self.inner, build_inference_input(x)?)?
        };
        decoded_predictions(py, preds, self.string_class_labels.as_deref())
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let preds = if is_polars_lazyframe(x)? {
            predict_proba_lazyframe_in_batches(x, |input| {
                predict_proba_input_with_model_detached(py, &self.inner, input)
            })?
        } else {
            predict_proba_input_with_model_detached(py, &self.inner, build_inference_input(x)?)?
        };
        PyArray2::from_vec2(py, &preds)
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))
    }

    #[getter]
    fn tree_count(&self) -> usize {
        self.inner.tree_count()
    }

    #[getter]
    fn used_feature_count(&self) -> usize {
        self.inner.used_feature_count()
    }

    #[getter]
    fn used_feature_indices(&self) -> Vec<usize> {
        self.inner.used_feature_indices()
    }

    #[pyo3(signature = (tree_index=0))]
    fn tree_structure<'py>(
        &self,
        py: Python<'py>,
        tree_index: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let summary = self
            .inner
            .tree_structure(tree_index)
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;
        to_python_json_value(py, &summary)
    }

    #[pyo3(signature = (tree_index=0))]
    fn tree_prediction_stats<'py>(
        &self,
        py: Python<'py>,
        tree_index: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let stats = self
            .inner
            .tree_prediction_stats(tree_index)
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;
        to_python_json_value(py, &stats)
    }

    #[pyo3(signature = (node_index, tree_index=0))]
    fn tree_node<'py>(
        &self,
        py: Python<'py>,
        node_index: usize,
        tree_index: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let node = self
            .inner
            .tree_node(tree_index, node_index)
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;
        to_python_json_value(py, &node)
    }

    #[pyo3(signature = (level_index, tree_index=0))]
    fn tree_level<'py>(
        &self,
        py: Python<'py>,
        level_index: usize,
        tree_index: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let level = self
            .inner
            .tree_level(tree_index, level_index)
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;
        to_python_json_value(py, &level)
    }

    #[pyo3(signature = (leaf_index, tree_index=0))]
    fn tree_leaf<'py>(
        &self,
        py: Python<'py>,
        leaf_index: usize,
        tree_index: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let leaf = self
            .inner
            .tree_leaf(tree_index, leaf_index)
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;
        to_python_json_value(py, &leaf)
    }

    #[pyo3(signature = (tree_index=None))]
    fn to_dataframe<'py>(
        &self,
        py: Python<'py>,
        tree_index: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        dataframe_from_model(
            py,
            &self.inner,
            self.string_class_labels.as_deref(),
            tree_index,
        )
    }

    #[pyo3(signature = (physical_cores=None, missing_features=None))]
    fn optimize_inference(
        &self,
        py: Python<'_>,
        physical_cores: Option<usize>,
        missing_features: Option<Vec<usize>>,
    ) -> PyResult<PyOptimizedModel> {
        let inner = optimize_model_detached(py, &self.inner, physical_cores, missing_features)?;
        Ok(PyOptimizedModel {
            inner,
            string_class_labels: self.string_class_labels.clone(),
        })
    }

    #[getter]
    fn algorithm(&self) -> &'static str {
        algorithm_name(self.inner.algorithm())
    }

    #[getter]
    fn task(&self) -> &'static str {
        task_name(self.inner.task())
    }

    #[getter]
    fn criterion(&self) -> &'static str {
        criterion_name(self.inner.criterion())
    }

    #[getter]
    fn tree_type(&self) -> &'static str {
        tree_type_name(self.inner.tree_type())
    }

    #[getter]
    fn canaries(&self) -> usize {
        self.inner.canaries()
    }

    #[getter]
    fn max_depth(&self) -> Option<usize> {
        self.inner.max_depth()
    }

    #[getter]
    fn min_samples_split(&self) -> Option<usize> {
        self.inner.min_samples_split()
    }

    #[getter]
    fn min_samples_leaf(&self) -> Option<usize> {
        self.inner.min_samples_leaf()
    }

    #[getter]
    fn n_trees(&self) -> Option<usize> {
        self.inner.n_trees()
    }

    #[getter]
    fn max_features(&self) -> Option<usize> {
        self.inner.max_features()
    }

    #[getter]
    fn seed(&self) -> Option<u64> {
        self.inner.seed()
    }

    #[getter]
    fn compute_oob(&self) -> bool {
        self.inner.compute_oob()
    }

    #[getter]
    fn oob_score(&self) -> Option<f64> {
        self.inner.oob_score()
    }

    #[getter]
    fn learning_rate(&self) -> Option<f64> {
        self.inner.learning_rate()
    }

    #[getter]
    fn bootstrap(&self) -> bool {
        self.inner.bootstrap()
    }

    #[getter]
    fn top_gradient_fraction(&self) -> Option<f64> {
        self.inner.top_gradient_fraction()
    }

    #[getter]
    fn other_gradient_fraction(&self) -> Option<f64> {
        self.inner.other_gradient_fraction()
    }

    #[pyo3(signature = (pretty=false))]
    fn to_ir_json(&self, pretty: bool) -> PyResult<String> {
        if self.string_class_labels.is_some() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "IR export is not supported for models trained with string class labels.",
            ));
        }
        if pretty {
            self.inner.to_ir_json_pretty()
        } else {
            self.inner.to_ir_json()
        }
        .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))
    }

    #[pyo3(signature = (pretty=false))]
    fn serialize(&self, pretty: bool) -> PyResult<String> {
        if self.string_class_labels.is_some() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Serialization is not supported for models trained with string class labels.",
            ));
        }
        if pretty {
            self.inner.serialize_pretty()
        } else {
            self.inner.serialize()
        }
        .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))
    }
}

#[pymethods]
impl PyCategoricalModel {
    fn predict<'py>(&self, py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let rows = extract_categorical_matrix(x)?;
        let preds = py
            .detach(|| self.inner.predict_rows(rows).map_err(|err| err.to_string()))
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        decoded_predictions(py, preds, self.string_class_labels.as_deref())
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let rows = extract_categorical_matrix(x)?;
        let preds = py
            .detach(|| {
                self.inner
                    .predict_proba_rows(rows)
                    .map_err(|err| err.to_string())
            })
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        PyArray2::from_vec2(py, &preds)
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))
    }

    #[pyo3(signature = (physical_cores=None, missing_features=None))]
    fn optimize_inference(
        &self,
        _py: Python<'_>,
        physical_cores: Option<usize>,
        missing_features: Option<Vec<usize>>,
    ) -> PyResult<PyCategoricalOptimizedModel> {
        let inner = self
            .inner
            .optimize_inference(physical_cores, missing_features.as_deref())
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;
        Ok(PyCategoricalOptimizedModel {
            inner,
            string_class_labels: self.string_class_labels.clone(),
        })
    }

    #[getter]
    fn algorithm(&self) -> &'static str {
        algorithm_name(self.inner.inner().algorithm())
    }

    #[getter]
    fn task(&self) -> &'static str {
        task_name(self.inner.inner().task())
    }

    #[getter]
    fn criterion(&self) -> &'static str {
        criterion_name(self.inner.inner().criterion())
    }

    #[getter]
    fn tree_type(&self) -> &'static str {
        tree_type_name(self.inner.inner().tree_type())
    }

    #[getter]
    fn n_trees(&self) -> Option<usize> {
        self.inner.inner().n_trees()
    }
}

#[pymethods]
impl PyOptimizedModel {
    #[classmethod]
    #[pyo3(signature = (serialized, physical_cores=None))]
    fn deserialize_compiled(
        _cls: &Bound<PyType>,
        serialized: &[u8],
        physical_cores: Option<usize>,
    ) -> PyResult<Self> {
        let inner = CoreOptimizedModel::deserialize_compiled(serialized, physical_cores)
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;
        Ok(Self {
            inner,
            string_class_labels: None,
        })
    }

    fn predict<'py>(&self, py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let preds = if is_polars_lazyframe(x)? {
            predict_lazyframe_in_batches(x, |input| {
                predict_input_with_optimized_model_detached(py, &self.inner, input)
            })?
        } else {
            predict_input_with_optimized_model_detached(py, &self.inner, build_inference_input(x)?)?
        };
        decoded_predictions(py, preds, self.string_class_labels.as_deref())
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let preds = if is_polars_lazyframe(x)? {
            predict_proba_lazyframe_in_batches(x, |input| {
                predict_proba_input_with_optimized_model_detached(py, &self.inner, input)
            })?
        } else {
            predict_proba_input_with_optimized_model_detached(
                py,
                &self.inner,
                build_inference_input(x)?,
            )?
        };
        PyArray2::from_vec2(py, &preds)
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))
    }

    #[getter]
    fn tree_count(&self) -> usize {
        self.inner.tree_count()
    }

    #[getter]
    fn used_feature_count(&self) -> usize {
        self.inner.used_feature_count()
    }

    #[getter]
    fn used_feature_indices(&self) -> Vec<usize> {
        self.inner.used_feature_indices()
    }

    #[pyo3(signature = (tree_index=0))]
    fn tree_structure<'py>(
        &self,
        py: Python<'py>,
        tree_index: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let summary = self
            .inner
            .tree_structure(tree_index)
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;
        to_python_json_value(py, &summary)
    }

    #[pyo3(signature = (tree_index=0))]
    fn tree_prediction_stats<'py>(
        &self,
        py: Python<'py>,
        tree_index: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let stats = self
            .inner
            .tree_prediction_stats(tree_index)
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;
        to_python_json_value(py, &stats)
    }

    #[pyo3(signature = (node_index, tree_index=0))]
    fn tree_node<'py>(
        &self,
        py: Python<'py>,
        node_index: usize,
        tree_index: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let node = self
            .inner
            .tree_node(tree_index, node_index)
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;
        to_python_json_value(py, &node)
    }

    #[pyo3(signature = (level_index, tree_index=0))]
    fn tree_level<'py>(
        &self,
        py: Python<'py>,
        level_index: usize,
        tree_index: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let level = self
            .inner
            .tree_level(tree_index, level_index)
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;
        to_python_json_value(py, &level)
    }

    #[pyo3(signature = (leaf_index, tree_index=0))]
    fn tree_leaf<'py>(
        &self,
        py: Python<'py>,
        leaf_index: usize,
        tree_index: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let leaf = self
            .inner
            .tree_leaf(tree_index, leaf_index)
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;
        to_python_json_value(py, &leaf)
    }

    #[pyo3(signature = (tree_index=None))]
    fn to_dataframe<'py>(
        &self,
        py: Python<'py>,
        tree_index: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        dataframe_from_optimized_model(
            py,
            &self.inner,
            self.string_class_labels.as_deref(),
            tree_index,
        )
    }

    #[getter]
    fn algorithm(&self) -> &'static str {
        algorithm_name(self.inner.algorithm())
    }

    #[getter]
    fn task(&self) -> &'static str {
        task_name(self.inner.task())
    }

    #[getter]
    fn criterion(&self) -> &'static str {
        criterion_name(self.inner.criterion())
    }

    #[getter]
    fn tree_type(&self) -> &'static str {
        tree_type_name(self.inner.tree_type())
    }

    #[getter]
    fn canaries(&self) -> usize {
        self.inner.canaries()
    }

    #[getter]
    fn max_depth(&self) -> Option<usize> {
        self.inner.max_depth()
    }

    #[getter]
    fn min_samples_split(&self) -> Option<usize> {
        self.inner.min_samples_split()
    }

    #[getter]
    fn min_samples_leaf(&self) -> Option<usize> {
        self.inner.min_samples_leaf()
    }

    #[getter]
    fn n_trees(&self) -> Option<usize> {
        self.inner.n_trees()
    }

    #[getter]
    fn max_features(&self) -> Option<usize> {
        self.inner.max_features()
    }

    #[getter]
    fn seed(&self) -> Option<u64> {
        self.inner.seed()
    }

    #[getter]
    fn compute_oob(&self) -> bool {
        self.inner.compute_oob()
    }

    #[getter]
    fn oob_score(&self) -> Option<f64> {
        self.inner.oob_score()
    }

    #[getter]
    fn learning_rate(&self) -> Option<f64> {
        self.inner.learning_rate()
    }

    #[getter]
    fn bootstrap(&self) -> bool {
        self.inner.bootstrap()
    }

    #[getter]
    fn top_gradient_fraction(&self) -> Option<f64> {
        self.inner.top_gradient_fraction()
    }

    #[getter]
    fn other_gradient_fraction(&self) -> Option<f64> {
        self.inner.other_gradient_fraction()
    }

    #[pyo3(signature = (pretty=false))]
    fn to_ir_json(&self, pretty: bool) -> PyResult<String> {
        if self.string_class_labels.is_some() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "IR export is not supported for models trained with string class labels.",
            ));
        }
        if pretty {
            self.inner.to_ir_json_pretty()
        } else {
            self.inner.to_ir_json()
        }
        .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))
    }

    #[pyo3(signature = (pretty=false))]
    fn serialize(&self, pretty: bool) -> PyResult<String> {
        if self.string_class_labels.is_some() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Serialization is not supported for models trained with string class labels.",
            ));
        }
        if pretty {
            self.inner.serialize_pretty()
        } else {
            self.inner.serialize()
        }
        .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))
    }

    fn serialize_compiled<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        if self.string_class_labels.is_some() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Compiled serialization is not supported for models trained with string class labels.",
            ));
        }
        let bytes = self
            .inner
            .serialize_compiled()
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;
        Ok(PyBytes::new(py, &bytes))
    }
}

#[pymethods]
impl PyCategoricalOptimizedModel {
    fn predict<'py>(&self, py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let rows = extract_categorical_matrix(x)?;
        let preds = py
            .detach(|| self.inner.predict_rows(rows).map_err(|err| err.to_string()))
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        decoded_predictions(py, preds, self.string_class_labels.as_deref())
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let rows = extract_categorical_matrix(x)?;
        let preds = py
            .detach(|| {
                self.inner
                    .predict_proba_rows(rows)
                    .map_err(|err| err.to_string())
            })
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        PyArray2::from_vec2(py, &preds)
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))
    }
}

#[pymethods]
impl PyTable {
    #[new]
    #[pyo3(signature = (x, y=None, canaries=2, bins=None))]
    fn new(
        x: &Bound<PyAny>,
        y: Option<&Bound<PyAny>>,
        canaries: usize,
        bins: Option<&Bound<PyAny>>,
    ) -> PyResult<Self> {
        let bins = parse_bins(bins)?;
        let inner = if let Some(y) = y {
            let (table, string_class_labels) = build_training_table(x, Some(y), canaries, bins)?;
            if string_class_labels.is_some() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Table does not support string targets; pass raw inputs to train(...) instead.",
                ));
            }
            table
        } else {
            build_feature_table(x, bins)?
        };

        Ok(Self { inner })
    }

    #[getter]
    fn kind(&self) -> &'static str {
        table_kind_name(self.inner.kind())
    }

    #[getter]
    fn n_rows(&self) -> usize {
        self.inner.n_rows()
    }

    #[getter]
    fn n_features(&self) -> usize {
        self.inner.n_features()
    }

    #[getter]
    fn canaries(&self) -> usize {
        self.inner.canaries()
    }
}

#[pymodule]
fn _core(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyModel>()?;
    m.add_class::<PyOptimizedModel>()?;
    m.add_class::<PyCategoricalModel>()?;
    m.add_class::<PyCategoricalOptimizedModel>()?;
    m.add_class::<PyTable>()?;
    m.add_function(wrap_pyfunction!(train, m)?)?;
    m.add(
        "__all__",
        vec![
            "Model",
            "OptimizedModel",
            "CategoricalModel",
            "CategoricalOptimizedModel",
            "Table",
            "train",
        ],
    )?;
    Ok(())
}
