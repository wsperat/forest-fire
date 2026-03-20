use forestfire_core::{
    Criterion, Model, Task, TrainAlgorithm, TrainConfig, TreeType, train as train_model,
};
use forestfire_data::{Table, TableAccess, TableKind};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::types::{PyDict, PyType};
use pyo3::{Bound, prelude::*};
use std::collections::BTreeMap;

#[pyclass(name = "Model")]
struct PyModel {
    inner: Model,
}

#[pyclass(name = "Table")]
#[derive(Clone)]
struct PyTable {
    inner: Table,
}

fn table_kind_name(kind: TableKind) -> &'static str {
    match kind {
        TableKind::Dense => "dense",
        TableKind::Sparse => "sparse",
    }
}

fn build_training_table(
    x: &Bound<PyAny>,
    y: Option<&Bound<PyAny>>,
    canaries: usize,
) -> PyResult<Table> {
    if let Ok(table) = x.extract::<PyRef<'_, PyTable>>() {
        if y.is_some() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "y must be omitted when x is already a Table.",
            ));
        }
        return Ok(table.inner.clone());
    }

    let y = y.ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "y is required unless x is already a Table.",
        )
    })?;
    if is_scipy_sparse_matrix(x)? {
        return build_sparse_training_table(x, y, canaries);
    }
    let x_rows = extract_matrix(x)?;
    let y_values = extract_vector(y)?;

    Table::with_canaries(x_rows, y_values, canaries)
        .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))
}

fn build_feature_table(x: &Bound<PyAny>) -> PyResult<Table> {
    if let Ok(table) = x.extract::<PyRef<'_, PyTable>>() {
        return Ok(table.inner.clone());
    }

    if is_scipy_sparse_matrix(x)? {
        return build_sparse_feature_table(x);
    }

    let x_rows = extract_matrix(x)?;
    let y_values = vec![0.0; x_rows.len()];

    Table::with_canaries(x_rows, y_values, 0)
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

    if let Ok(columns) = extract_named_columns(x) {
        return Ok(InferenceInput::NamedColumns(columns));
    }

    Ok(InferenceInput::Rows(extract_matrix(x)?))
}

fn is_scipy_sparse_matrix(x: &Bound<PyAny>) -> PyResult<bool> {
    Ok(x.hasattr("getnnz")? && x.hasattr("nonzero")?)
}

fn build_sparse_training_table(
    x: &Bound<PyAny>,
    y: &Bound<PyAny>,
    canaries: usize,
) -> PyResult<Table> {
    let (n_rows, n_features, columns) = extract_sparse_binary_columns(x)?;
    let y_values = extract_vector(y)?;
    let table = forestfire_data::SparseTable::from_sparse_binary_columns(
        n_rows, n_features, columns, y_values, canaries,
    )
    .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;
    Ok(Table::Sparse(table))
}

fn build_sparse_feature_table(x: &Bound<PyAny>) -> PyResult<Table> {
    let (n_rows, n_features, columns) = extract_sparse_binary_columns(x)?;
    let table = forestfire_data::SparseTable::from_sparse_binary_columns(
        n_rows,
        n_features,
        columns,
        vec![0.0; n_rows],
        0,
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
    for (row_idx, col_idx) in row_indices.into_iter().zip(col_indices.into_iter()) {
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

fn extract_named_columns(x: &Bound<PyAny>) -> PyResult<BTreeMap<String, Vec<f64>>> {
    if x.hasattr("collect")? {
        let collected = x.call_method0("collect")?;
        return extract_named_columns(&collected);
    }

    if x.hasattr("to_pydict")? {
        let columns = x.call_method0("to_pydict")?;
        return extract_named_columns_from_dict(columns.cast::<PyDict>()?);
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

fn extract_scalar(value: &Bound<PyAny>) -> PyResult<f64> {
    if let Ok(value) = value.extract::<bool>() {
        return Ok(f64::from(u8::from(value)));
    }

    value.extract::<f64>()
}

fn parse_algorithm(algorithm: &str) -> PyResult<TrainAlgorithm> {
    match algorithm {
        "dt" => Ok(TrainAlgorithm::Dt),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unsupported algorithm '{}'. Expected one of: dt",
            algorithm
        ))),
    }
}

fn parse_tree_type(tree_type: &str) -> PyResult<TreeType> {
    match tree_type {
        "target_mean" => Ok(TreeType::TargetMean),
        "id3" => Ok(TreeType::Id3),
        "c45" => Ok(TreeType::C45),
        "cart" => Ok(TreeType::Cart),
        "oblivious" => Ok(TreeType::Oblivious),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unsupported tree_type '{}'. Expected one of: target_mean, id3, c45, cart, oblivious",
            tree_type
        ))),
    }
}

fn parse_task(task: &str) -> PyResult<Task> {
    match task {
        "regression" => Ok(Task::Regression),
        "classification" => Ok(Task::Classification),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unsupported task '{}'. Expected one of: regression, classification",
            task
        ))),
    }
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

fn algorithm_name(algorithm: TrainAlgorithm) -> &'static str {
    match algorithm {
        TrainAlgorithm::Dt => "dt",
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
    }
}

fn tree_type_name(tree_type: TreeType) -> &'static str {
    match tree_type {
        TreeType::TargetMean => "target_mean",
        TreeType::Id3 => "id3",
        TreeType::C45 => "c45",
        TreeType::Cart => "cart",
        TreeType::Oblivious => "oblivious",
    }
}

#[pyfunction]
#[pyo3(signature = (x, y=None, algorithm="dt", task="regression", tree_type="target_mean", criterion="auto", canaries=2, physical_cores=None))]
#[allow(clippy::too_many_arguments)]
fn train(
    x: &Bound<PyAny>,
    y: Option<&Bound<PyAny>>,
    algorithm: &str,
    task: &str,
    tree_type: &str,
    criterion: &str,
    canaries: usize,
    physical_cores: Option<usize>,
) -> PyResult<PyModel> {
    let table = build_training_table(x, y, canaries)?;
    let config = TrainConfig {
        algorithm: parse_algorithm(algorithm)?,
        task: parse_task(task)?,
        tree_type: parse_tree_type(tree_type)?,
        criterion: parse_criterion(criterion)?,
        physical_cores,
    };
    let model = train_model(&table, config)
        .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;

    Ok(PyModel { inner: model })
}

#[pymethods]
impl PyModel {
    #[classmethod]
    fn deserialize(_cls: &Bound<PyType>, serialized: &str) -> PyResult<Self> {
        let inner = Model::deserialize(serialized)
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;
        Ok(Self { inner })
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let preds = match build_inference_input(x)? {
            InferenceInput::Rows(rows) => self.inner.predict_rows(rows),
            InferenceInput::NamedColumns(columns) => self.inner.predict_named_columns(columns),
            InferenceInput::SparseBinaryColumns {
                n_rows,
                n_features,
                columns,
            } => self
                .inner
                .predict_sparse_binary_columns(n_rows, n_features, columns),
            InferenceInput::TrainingTable(table) => Ok(self.inner.predict_table(&table)),
        }
        .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;
        Ok(PyArray1::from_vec(py, preds))
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
    fn mean_(&self) -> Option<f64> {
        self.inner.mean_value()
    }

    #[pyo3(signature = (pretty=false))]
    fn to_ir_json(&self, pretty: bool) -> PyResult<String> {
        if pretty {
            self.inner.to_ir_json_pretty()
        } else {
            self.inner.to_ir_json()
        }
        .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))
    }

    #[pyo3(signature = (pretty=false))]
    fn serialize(&self, pretty: bool) -> PyResult<String> {
        if pretty {
            self.inner.serialize_pretty()
        } else {
            self.inner.serialize()
        }
        .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))
    }
}

#[pymethods]
impl PyTable {
    #[new]
    #[pyo3(signature = (x, y=None, canaries=2))]
    fn new(x: &Bound<PyAny>, y: Option<&Bound<PyAny>>, canaries: usize) -> PyResult<Self> {
        let inner = if let Some(y) = y {
            build_training_table(x, Some(y), canaries)?
        } else {
            build_feature_table(x)?
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
fn forestfire(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyModel>()?;
    m.add_class::<PyTable>()?;
    m.add_function(wrap_pyfunction!(train, m)?)?;
    m.add("__all__", vec!["Model", "Table", "train"])?;
    Ok(())
}
