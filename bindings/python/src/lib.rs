use forestfire_core::TargetMeanTree;
use forestfire_data::DenseDataset;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::{Bound, prelude::*, types::PyType};

#[pyclass(name = "TargetMeanTree")]
struct PyTargetMeanTree {
    inner: TargetMeanTree,
}

#[pymethods]
impl PyTargetMeanTree {
    /// classmethod: fit(X, y) -> TargetMeanTree
    #[classmethod]
    fn fit<'py>(
        _cls: &Bound<'py, PyType>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<f64>,
    ) -> PyResult<Self> {
        let x_view = x.as_array();
        let y_view = y.as_array();

        // Convert to Vec<Vec<f64>> / Vec<f64>
        let x_vec: Vec<Vec<f64>> = x_view.rows().into_iter().map(|r| r.to_vec()).collect();
        let y_vec: Vec<f64> = y_view.to_vec();

        let ds = DenseDataset::new(x_vec, y_vec)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

        let model = TargetMeanTree::train(&ds)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        Ok(Self { inner: model })
    }

    /// predict(X) -> np.ndarray[n_samples]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let n = x.shape()[0]; // or: x.as_array().shape()[0]
        let preds = self.inner.predict_many(n);
        Ok(PyArray1::from_vec(py, preds))
    }

    /// mean_ -> float (scikit-learn style)
    #[getter]
    fn mean_(&self) -> f64 {
        self.inner.mean
    }
}

#[pymodule]
fn forestfire(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyTargetMeanTree>()?;
    Ok(())
}
