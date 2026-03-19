use forestfire_core::{
    Criterion, Model, Task, TrainAlgorithm, TrainConfig, TreeType, train as train_model,
};
use forestfire_data::DenseTable;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::{Bound, prelude::*};

#[pyclass(name = "Model")]
struct PyModel {
    inner: Model,
}

fn build_table(
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    canaries: usize,
) -> PyResult<DenseTable> {
    let x_view = x.as_array();
    let y_view = y.as_array();

    let x_vec: Vec<Vec<f64>> = x_view.rows().into_iter().map(|row| row.to_vec()).collect();
    let y_vec: Vec<f64> = y_view.to_vec();

    DenseTable::with_canaries(x_vec, y_vec, canaries)
        .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))
}

fn build_feature_table(x: PyReadonlyArray2<f64>) -> PyResult<DenseTable> {
    let x_view = x.as_array();
    let x_vec: Vec<Vec<f64>> = x_view.rows().into_iter().map(|row| row.to_vec()).collect();
    let y_vec = vec![0.0; x.shape()[0]];

    DenseTable::with_canaries(x_vec, y_vec, 0)
        .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))
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
#[pyo3(signature = (x, y, algorithm="dt", task="regression", tree_type="target_mean", criterion="auto", canaries=2))]
fn train(
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    algorithm: &str,
    task: &str,
    tree_type: &str,
    criterion: &str,
    canaries: usize,
) -> PyResult<PyModel> {
    let table = build_table(x, y, canaries)?;
    let config = TrainConfig {
        algorithm: parse_algorithm(algorithm)?,
        task: parse_task(task)?,
        tree_type: parse_tree_type(tree_type)?,
        criterion: parse_criterion(criterion)?,
    };
    let model = train_model(&table, config)
        .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;

    Ok(PyModel { inner: model })
}

#[pymethods]
impl PyModel {
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let table = build_feature_table(x)?;
        let preds = self.inner.predict_table(&table);
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
}

#[pymodule]
fn forestfire(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyModel>()?;
    m.add_function(wrap_pyfunction!(train, m)?)?;
    m.add("__all__", vec!["Model", "train"])?;
    Ok(())
}
