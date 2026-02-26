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
use super::{
    numpy_faer::{
        numpy_mat_to_faer
        , numpy_1d_to_slice
        , PyFaerMat
        , PyVec
    }
};
// use crate::utils::interop::{IntoFaer, IntoNdarray};
// use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
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

    pub fn fit<'py>(&mut self, X: Bound<'py, PyAny>, y: Bound<'py, PyAny>) -> PyResult<()> {
        // return Err(pyo3::exceptions::PyValueError::new_err(
        //         "Error here!"
        //     )
        // );
        let x = numpy_mat_to_faer(X)?;
        let y = numpy_mat_to_faer(y)?;
        match self.lr.fit(x, y) {
            Ok(_) => Ok(()),
            Err(e) => Err(e.into()),
        }
    }

    pub fn set_coeffs_and_bias<'py>(
        &mut self,
        coeffs: Bound<'py, PyAny>,
        bias: f64,
    ) -> PyResult<()> {
        match numpy_1d_to_slice(coeffs) {
            Ok(s) => Ok(self.lr.set_coeffs_and_bias(s, bias)),
            Err(e) => Err(e.into()),
        }
    }

    pub fn predict<'py>(
        &self,
        py: Python<'py>,
        X: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyFaerMat>> {
        let x = numpy_mat_to_faer(X)?;
        match self.lr.predict(x) {
            Ok(result) => {
                let out = Bound::new(py, PyFaerMat{mat: result})?;
                Ok(out)
            }
            Err(e) => Err(e.into()),
        }
    }

    #[getter]
    pub fn coeffs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyVec>> {
        match self.lr.coeffs_as_vec() {
            Ok(v) => {
                let out = Bound::new(py, PyVec{data: v})?;
                Ok(out)
            },
            Err(e) => Err(e.into()),
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
        coeffs: Bound<'py, PyAny>,
        bias: f64,
    ) -> PyResult<()> {
        match numpy_1d_to_slice(coeffs) {
            Ok(s) => Ok(self.lr.set_coeffs_and_bias(s, bias)),
            Err(e) => Err(e.into()),
        }
    }

    pub fn is_fit(&self) -> bool {
        self.lr.is_fit()
    }

    pub fn fit<'py>(&mut self, X: Bound<'py, PyAny>, y: Bound<'py, PyAny>) -> PyResult<()> {
        let x = numpy_mat_to_faer(X)?;
        let y = numpy_mat_to_faer(y)?;
        match self.lr.fit(x, y) {
            Ok(_) => Ok(()),
            Err(e) => Err(e.into()),
        }
    }

    pub fn predict<'py>(
        &self,
        py: Python<'py>,
        X: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyFaerMat>> {
        let x = numpy_mat_to_faer(X)?;
        match self.lr.predict(x) {
            Ok(result) => {
                let out = Bound::new(py, PyFaerMat{mat: result})?;
                Ok(out)
            }
            Err(e) => Err(e.into()),
        }
    }

    pub fn add_bias(&self) -> bool {
        self.lr.add_bias()
    }

    #[getter]
    pub fn coeffs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyVec>> {
        match self.lr.coeffs_as_vec() {
            Ok(v) => {
                let out = Bound::new(py, PyVec{data: v})?;
                Ok(out)
            },
            Err(e) => Err(e.into()),
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

    pub fn fit<'py>(&mut self, X: Bound<'py, PyAny>, y: Bound<'py, PyAny>) -> PyResult<()> {
        let x = numpy_mat_to_faer(X)?;
        let y = numpy_mat_to_faer(y)?;
        match self.lr.fit(x, y) {
            Ok(_) => Ok(()),
            Err(e) => Err(e.into()),
        }
    }

    pub fn update<'py>(&mut self, X: Bound<'py, PyAny>, y: Bound<'py, PyAny>, c: f64) -> PyResult<()> {
        let x = numpy_mat_to_faer(X)?;
        let y = numpy_mat_to_faer(y)?;
        self.lr.update(x, y, c);
        Ok(())
    }

    // pub fn set_coeffs_and_bias<'py>(
    //     &mut self,
    //     coeffs: Bound<'py, PyAny>,
    //     bias: f64,
    // ) -> PyResult<()> {
    //     match numpy_1d_to_slice(coeffs) {
    //         Ok(s) => Ok(self.lr.set_coeffs_and_bias(s, bias)),
    //         Err(e) => Err(e.into()),
    //     }
    // }

    pub fn set_coeffs_bias_inverse<'py>(
        &mut self,
        coeffs: Bound<'py, PyAny>,
        bias: f64,
        inv: Bound<'py, PyAny>,
    ) -> PyResult<()> {
        let inverse = numpy_mat_to_faer(inv)?;
        match numpy_1d_to_slice(coeffs) {
            Ok(s) => match self.lr.set_coeffs_bias_inverse(s, bias, inverse) {
                Ok(_) => Ok(()),
                Err(e) => Err(e.into()),
            },
            Err(e) => Err(e.into()),
        }
    }

    pub fn predict<'py>(
        &self,
        py: Python<'py>,
        X: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyFaerMat>> {
        let x = numpy_mat_to_faer(X)?;
        match self.lr.predict(x) {
            Ok(result) => {
                let out = Bound::new(py, PyFaerMat{mat: result})?;
                Ok(out)
            }
            Err(e) => Err(e.into()),
        }
    }

    #[getter]
    pub fn coeffs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyVec>> {
        match self.lr.coeffs_as_vec() {
            Ok(v) => {
                let out = Bound::new(py, PyVec{data: v})?;
                Ok(out)
            },
            Err(e) => Err(e.into()),
        }
    }

    #[getter]
    pub fn inv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyFaerMat>> {
        match self.lr.get_inv() {
            Ok(matrix) => {
                let mat = PyFaerMat {mat: matrix.to_owned()};
                let out = Bound::new(py, mat)?;
                Ok(out)
            }
            Err(e) => Err(e.into()),
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
