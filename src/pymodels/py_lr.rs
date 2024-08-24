#![allow(non_snake_case)]
/// Linear Regression Interop with Python
use crate::linalg::{LinalgErrors, lstsq::LR};
use faer_ext::IntoFaer;
use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};

impl From<LinalgErrors> for PyErr {
    fn from(value: LinalgErrors) -> Self {
        let msg = match value {
            LinalgErrors::DimensionMismatch => "Dimension mismatch.".to_string(),
            LinalgErrors::NotContiguousArray => "Input array is not contiguous.".to_string(),
            LinalgErrors::Other(s) => s,
        };
        PyValueError::new_err(msg)
    }
}

#[pyclass(subclass)]
pub struct PyLR {
    lr: LR
}

#[pymethods]
impl PyLR {

    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(
        solver = "qr",
        lambda_ = 0.,
        fit_bias = false,
    ))]
    pub fn new(solver:&str, lambda_:f64, fit_bias:bool) -> Self {
        PyLR {
            lr: LR::new(solver, lambda_, fit_bias)
        }
    }

    pub fn is_fit(&self) -> bool {
        self.lr.is_fit()
    }

    pub fn fit(
        &mut self,
        X: PyReadonlyArray2<f64>,
        y: PyReadonlyArray2<f64>,
    ) -> PyResult<()> {
        let x = X.as_array().into_faer();
        let y = y.as_array().into_faer();
        match self.lr.fit(x, y) {
            Ok(_) => Ok(()),
            Err(e) => Err(e.into()),
        }
    }

    pub fn set_coeffs_and_bias(
        &mut self,
        coeffs: PyReadonlyArray1<f64>,
        bias: f64
    ) {
        self.lr.set_coeffs_and_bias(coeffs.as_slice().unwrap(), bias)
    }

    pub fn predict<'py>(
        &self,
        py: Python<'py>,
        X: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {

        let x = X.as_array().into_faer();
        match self.lr.predict(x) {
            Ok(result) => {
                // result should be n by 1, where n = x.nrows()
                let res = result.col_as_slice(0);
                let v = res.to_vec();
                Ok(v.into_pyarray_bound(py))
            },
            Err(e) => Err(e.into()),
        }
    }

    #[getter]
    pub fn coeffs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        match self.lr.coeffs_as_vec() {
            Ok(v) => Ok(v.into_pyarray_bound(py)),
            Err(e) => Err(e.into()),
        }
    }

    #[getter]
    pub fn bias(&self) -> f64 {
        self.lr.bias
    }

    #[getter]
    pub fn lambda_(&self) -> f64 {
        self.lr.lambda
    }

}

// #[pyclass(subclass)]
// pub struct PyOnlineLR {
//     lr: LR
// }

// #[pymethods]
// impl PyOnlineLR {

//     #[new]
//     #[allow(clippy::too_many_arguments)]
//     #[pyo3(signature=(
//         solver = "qr",
//         lambda_ = 0.,
//         fit_bias = false,
//     ))]
//     pub fn new(solver:&str, lambda_:f64, fit_bias:bool) -> Self {
//         PyLR {
//             lr: LR::new(solver, lambda_, fit_bias)
//         }
//     }

//     pub fn is_fit(&self) -> bool {
//         self.lr.is_fit()
//     }

//     pub fn fit(
//         &mut self,
//         X: PyReadonlyArray2<f64>,
//         y: PyReadonlyArray2<f64>,
//     ) -> PyResult<()> {
//         let x = X.as_array().into_faer();
//         let y = y.as_array().into_faer();
//         match self.lr.fit(x, y) {
//             Ok(_) => Ok(()),
//             Err(e) => Err(e.into()),
//         }
//     }

//     pub fn set_coeffs_and_bias(
//         &mut self,
//         coeffs: PyReadonlyArray1<f64>,
//         bias: f64
//     ) {
//         self.lr.set_coeffs_and_bias(coeffs.as_slice().unwrap(), bias)
//     }

//     pub fn predict<'py>(
//         &self,
//         py: Python<'py>,
//         X: PyReadonlyArray2<f64>,
//     ) -> PyResult<Bound<'py, PyArray1<f64>>> {

//         let x = X.as_array().into_faer();
//         match self.lr.predict(x) {
//             Ok(result) => {
//                 // result should be n by 1, where n = x.nrows()
//                 let res = result.col_as_slice(0);
//                 let v = res.to_vec();
//                 Ok(v.into_pyarray_bound(py))
//             },
//             Err(e) => Err(e.into()),
//         }
//     }

//     #[getter]
//     pub fn coeffs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
//         match self.lr.coeffs_as_vec() {
//             Ok(v) => Ok(v.into_pyarray_bound(py)),
//             Err(e) => Err(e.into()),
//         }
//     }

//     #[getter]
//     pub fn bias(&self) -> f64 {
//         self.lr.bias
//     }

//     #[getter]
//     pub fn lambda_(&self) -> f64 {
//         self.lr.lambda
//     }

// }


