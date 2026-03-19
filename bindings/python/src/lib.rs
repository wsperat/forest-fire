use forestfire_core::{TargetMeanTree, train as train_model};
use forestfire_data::DenseTable;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::{Bound, prelude::*};

#[pyclass(name = "TargetMeanTree")]
struct PyTargetMeanTree {
    inner: TargetMeanTree,
}

fn build_table(x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<DenseTable> {
    let x_view = x.as_array();
    let y_view = y.as_array();

    let x_vec: Vec<Vec<f64>> = x_view.rows().into_iter().map(|row| row.to_vec()).collect();
    let y_vec: Vec<f64> = y_view.to_vec();

    DenseTable::new(x_vec, y_vec)
        .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))
}

#[pyfunction]
fn train(x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<PyTargetMeanTree> {
    let table = build_table(x, y)?;
    let model = train_model(&table)
        .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;

    Ok(PyTargetMeanTree { inner: model })
}

#[pymethods]
impl PyTargetMeanTree {
    /// predict(X) -> np.ndarray[n_samples]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let n_rows = x.shape()[0];
        let preds = self.inner.predict_many(n_rows);
        Ok(PyArray1::from_vec(py, preds))
    }

    /// mean_ -> float
    #[getter]
    fn mean_(&self) -> f64 {
        self.inner.mean
    }
}

#[pymodule]
fn forestfire(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyTargetMeanTree>()?;
    m.add_function(wrap_pyfunction!(train, m)?)?;
    m.add("__all__", vec!["TargetMeanTree", "train"])?;
    Ok(())
}
