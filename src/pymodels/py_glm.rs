#![allow(non_snake_case)]
/// Linear Regression Interop with Python
use crate::linear::{
    glm::glm_solvers::{GLMFamily, GLMParams, GLM},
    GeneralizedLinearModel, LinalgErrors, LinearModel,
};
use faer_ext::IntoFaer;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

#[pyclass(subclass)]
pub struct PyGLM {
    glm: GLM<f64>,
}

#[pymethods]
impl PyGLM {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(
        add_bias = false,
        family = "normal",
        solver = "irls",
        max_iter = 100,
        tol = 1e-8
    ))]
    pub fn new(add_bias: bool, family: &str, solver: &str, max_iter: usize, tol:f64) -> Self {
        let glm_family: GLMFamily = family.into();
        let glm = GLM::new(
            solver,
            0f64,
            add_bias,
            glm_family.link_function(),
            glm_family.variance_function(),
            GLMParams::new(max_iter, tol)        
        );
        PyGLM { glm: glm }
    }

    pub fn describe(&self) -> String {
        self.glm.to_string()
    }

    pub fn is_fit(&self) -> bool {
        self.glm.is_fit()
    }

    pub fn fit(&mut self, X: PyReadonlyArray2<f64>, y: PyReadonlyArray2<f64>) -> PyResult<()> {
        let x = X.into_faer();
        let y = y.into_faer();
        match self.glm.fit(x, y) {
            Ok(_) => Ok(()),
            Err(e) => Err(e.into()),
        }
    }

    pub fn set_coeffs_and_bias(
        &mut self,
        coeffs: PyReadonlyArray1<f64>,
        bias: f64,
    ) -> PyResult<()> {
        if coeffs.len().unwrap_or(0) <= 0 {
            return Err(LinalgErrors::Other("Input coefficients array is empty.".into()).into());
        }
        match coeffs.as_slice() {
            Ok(s) => Ok(self.glm.set_coeffs_and_bias(s, bias)),
            Err(_) => {
                // Copy if not contiguous
                let vec = coeffs.as_array().iter().copied().collect::<Vec<_>>();
                Ok(self.glm.set_coeffs_and_bias(&vec, bias))
            }
        }
    }

    pub fn linear_predict<'py>(
        &self,
        py: Python<'py>,
        X: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x = X.into_faer();
        match self.glm.predict(x) {
            Ok(result) => {
                // result should be n by 1, where n = x.nrows()
                let res = result.col_as_slice(0);
                Ok(PyArray1::from_slice(py, res))
            }
            Err(e) => Err(e.into()),
        }
    }

    pub fn predict<'py>(
        &self,
        py: Python<'py>,
        X: PyReadonlyArray2<f64>,
        linear: bool,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x = X.into_faer();
        let prediction = if linear {
            self.glm.predict(x)
        } else {
            self.glm.glm_predict(x)
        };
        match prediction {
            Ok(result) => {
                // result should be n by 1, where n = x.nrows()
                let res = result.col_as_slice(0);
                Ok(PyArray1::from_slice(py, res))
            }
            Err(e) => Err(e.into()),
        }
    }

    #[getter]
    pub fn coeffs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        match self.glm.coeffs_as_vec() {
            Ok(v) => Ok(v.into_pyarray(py)),
            Err(e) => Err(e.into()),
        }
    }

    #[getter]
    pub fn bias(&self) -> f64 {
        self.glm.bias()
    }

    #[getter]
    pub fn lambda_(&self) -> f64 {
        self.glm.lambda
    }
}
