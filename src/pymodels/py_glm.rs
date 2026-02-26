#![allow(non_snake_case)]
/// Linear Regression Interop with Python
use crate::linear::{
    glm::{
        glm_solvers::{GLMFamily, GLMParams, GLM},
        GeneralizedLinearModel,
    },
    lr::LinearModel,
};
use super::numpy_faer::{PyArrRef, PyArr, PyFaerRef, PyFaerMat};
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
        tol = 1e-5
    ))]
    pub fn new(add_bias: bool, family: &str, solver: &str, max_iter: usize, tol: f64) -> Self {
        let glm_family: GLMFamily = family.into();
        let glm = GLM::new(
            solver,
            0f64,
            add_bias,
            glm_family.link_function(),
            glm_family.variance_function(),
            GLMParams::new(max_iter, tol),
        );
        PyGLM { glm: glm }
    }

    pub fn is_fit(&self) -> bool {
        self.glm.is_fit()
    }

    pub fn fit(&mut self, X: PyFaerRef, y: PyFaerRef) -> PyResult<()> {
        match self.glm.fit(X.0, y.0) {
            Ok(_) => Ok(()),
            Err(e) => Err(e.into()),
        }
    }

    pub fn set_coeffs_and_bias(
        &mut self,
        coeffs: PyArrRef,
        bias: f64,
    ) -> PyResult<()> {
        Ok(self.glm.set_coeffs_and_bias(coeffs.0, bias))
    }

    pub fn linear_predict<'py>(
        &self,
        py: Python<'py>,
        X: PyFaerRef,
    ) -> PyResult<Bound<'py, PyFaerMat>> {
        match self.glm.predict(X.0) {
            Ok(result) => Bound::new(py, PyFaerMat(result))
            , Err(e) => Err(e.into()),
        }
    }

    pub fn predict<'py>(
        &self,
        py: Python<'py>,
        X: PyFaerRef,
        linear: bool,
    ) -> PyResult<Bound<'py, PyFaerMat>> {
        let prediction = if linear {
            self.glm.predict(X.0)
        } else {
            self.glm.glm_predict(X.0)
        };
        match prediction {
            Ok(result) => Bound::new(py, PyFaerMat(result))
            , Err(e) => Err(e.into()),
        }
    }

    #[getter]
    pub fn coeffs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArr>> {
        match self.glm.coeffs_as_vec() {
            Ok(v) => Bound::new(py, PyArr(v)),
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
