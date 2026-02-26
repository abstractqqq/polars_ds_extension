#![allow(non_snake_case)]
/// Linear Regression Interop with Python
use crate::linear::{
    lr::{
        lr_solvers::{ElasticNet, LR},
        LinearModel,
    },
    online_lr::lr_online_solvers::OnlineLR,
    LinalgErrors,
};
use super::numpy_faer::{PyArrRef, PyArr, PyFaerRef, PyFaerMat};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

impl From<LinalgErrors> for PyErr {
    fn from(value: LinalgErrors) -> Self {
        PyValueError::new_err(value.to_string())
    }
}

#[pyclass(subclass)]
pub struct PyLR {
    lr: LR<f64>,
}

#[pymethods]
impl PyLR {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(
        solver = "qr",
        lambda_ = 0.,
        add_bias = false,
    ))]
    pub fn new(solver: &str, lambda_: f64, add_bias: bool) -> Self {
        PyLR {
            lr: LR::new(solver, lambda_, add_bias),
        }
    }

    pub fn is_fit(&self) -> bool {
        self.lr.is_fit()
    }                   

    pub fn fit<'py>(&mut self, X: PyFaerRef, y: PyFaerRef) -> PyResult<()> {
        match self.lr.fit(X.0, y.0) {
            Ok(_) => Ok(()),
            Err(e) => Err(e.into()),
        }
    }

    pub fn set_coeffs_and_bias<'py>(
        &mut self,
        coeffs: PyArrRef,
        bias: f64,
    ) -> PyResult<()> {
        Ok(self.lr.set_coeffs_and_bias(coeffs.0, bias))
    }

    pub fn predict<'py>(
        &self,
        py: Python<'py>,
        X: PyFaerRef
    ) -> PyResult<Bound<'py, PyFaerMat>> {
        match self.lr.predict(X.0) {
            Ok(result) => Bound::new(py, PyFaerMat(result))
            , Err(e) => Err(e.into()),
        }
    }

    #[getter]
    pub fn coeffs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArr>> {
        match self.lr.coeffs_as_vec() {
            Ok(v) => Bound::new(py, PyArr(v))
            , Err(e) => Err(e.into()),
        }
    }

    #[getter]
    pub fn bias(&self) -> f64 {
        self.lr.bias()
    }

    #[getter]
    pub fn lambda_(&self) -> f64 {
        self.lr.lambda
    }
}

#[pyclass(subclass)]
pub struct PyElasticNet {
    lr: ElasticNet<f64>,
}

#[pymethods]
impl PyElasticNet {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(
        l1_reg,
        l2_reg,
        add_bias = false,
        tol = 1e-5,
        max_iter = 2000,
    ))]
    pub fn new(l1_reg: f64, l2_reg: f64, add_bias: bool, tol: f64, max_iter: usize) -> Self {
        PyElasticNet {
            lr: ElasticNet::new(l1_reg, l2_reg, add_bias, tol, max_iter),
        }
    }

    pub fn set_coeffs_and_bias<'py>(
        &mut self,
        coeffs: PyArrRef,
        bias: f64,
    ) -> PyResult<()> {
        Ok(self.lr.set_coeffs_and_bias(coeffs.0, bias))
    }

    pub fn is_fit(&self) -> bool {
        self.lr.is_fit()
    }

    pub fn fit<'py>(&mut self, X: PyFaerRef, y: PyFaerRef) -> PyResult<()> {
        match self.lr.fit(X.0, y.0) {
            Ok(_) => Ok(()),
            Err(e) => Err(e.into()),
        }
    }

    pub fn predict<'py>(
        &self,
        py: Python<'py>,
        X: PyFaerRef,
    ) -> PyResult<Bound<'py, PyFaerMat>> {
        match self.lr.predict(X.0) {
            Ok(result) => Bound::new(py, PyFaerMat(result))
            , Err(e) => Err(e.into()),
        }
    }

    pub fn add_bias(&self) -> bool {
        self.lr.add_bias()
    }

    #[getter]
    pub fn coeffs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArr>> {
        match self.lr.coeffs_as_vec() {
            Ok(v) => Bound::new(py, PyArr(v))
            , Err(e) => Err(e.into()),
        }
    }

    #[getter]
    pub fn bias(&self) -> f64 {
        self.lr.bias()
    }

    #[getter]
    pub fn regularizers(&self) -> (f64, f64) {
        self.lr.regularizers()
    }
}

#[pyclass(subclass)]
pub struct PyOnlineLR {
    lr: OnlineLR<f64>,
}

#[pymethods]
impl PyOnlineLR {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(lambda_=0., add_bias=false))]
    pub fn new(lambda_: f64, add_bias: bool) -> Self {
        PyOnlineLR {
            lr: OnlineLR::new(lambda_, add_bias),
        }
    }

    pub fn is_fit(&self) -> bool {
        self.lr.is_fit()
    }

    pub fn fit<'py>(&mut self, X: PyFaerRef, y: PyFaerRef) -> PyResult<()> {
        match self.lr.fit(X.0, y.0) {
            Ok(_) => Ok(()),
            Err(e) => Err(e.into()),
        }
    }

    pub fn update<'py>(&mut self, X: PyFaerRef, y: PyFaerRef, c: f64) -> PyResult<()> {
        Ok(self.lr.update(X.0, y.0, c))
    }

    pub fn set_coeffs_bias_inverse<'py>(
        &mut self,
        coeffs: PyArrRef,
        bias: f64,
        inv: PyFaerRef,
    ) -> PyResult<()> {
        match self.lr.set_coeffs_bias_inverse(coeffs.0, bias, inv.0) {
            Ok(_) => Ok(()),
            Err(e) => Err(e.into()),
        }
    }

    pub fn predict<'py>(
        &self,
        py: Python<'py>,
        X: PyFaerRef,
    ) -> PyResult<Bound<'py, PyFaerMat>> {
        match self.lr.predict(X.0) {
            Ok(result) => Bound::new(py, PyFaerMat(result))
            , Err(e) => Err(e.into()),
        }
    }

    #[getter]
    pub fn coeffs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArr>> {
        match self.lr.coeffs_as_vec() {
            Ok(v) => Bound::new(py, PyArr(v))
            , Err(e) => Err(e.into()),
        }
    }

    #[getter]
    pub fn inv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyFaerMat>> {
        match self.lr.get_inv() {
            Ok(matrix) => Bound::new(py, PyFaerMat(matrix.to_owned()))
            , Err(e) => Err(e.into()),
        }
    }

    #[getter]
    pub fn bias(&self) -> f64 {
        self.lr.bias()
    }

    #[getter]
    pub fn lambda_(&self) -> f64 {
        self.lr.lambda
    }
}
