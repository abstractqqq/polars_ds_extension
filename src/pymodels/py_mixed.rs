#![allow(non_snake_case)]
use super::numpy_faer::{PyArr, PyArrRef, PyFaerRef};
use crate::linear::mixed::fit_reml;
use pyo3::prelude::*;

#[pyclass(subclass)]
pub struct PyMixedModel {
    coeffs: Vec<f64>,
    std_errors: Vec<f64>,
    dfs: Vec<f64>,
    gamma: f64,
    resid_variance: f64,
    is_fit: bool,
}

#[pymethods]
impl PyMixedModel {
    #[new]
    pub fn new() -> Self {
        PyMixedModel {
            coeffs: Vec::new(),
            std_errors: Vec::new(),
            dfs: Vec::new(),
            gamma: 0.0,
            resid_variance: 0.0,
            is_fit: false,
        }
    }

    pub fn is_fit(&self) -> bool {
        self.is_fit
    }

    /// Fits a random-intercept model via REML.
    ///
    /// `group_codes` is a dense `0..n_groups` grouping per row (as f64, rounded to usize).
    /// `between_idx` are the column indices of `X` constant within every group level,
    /// used for containment degrees of freedom.
    #[pyo3(signature=(X, y, group_codes, n_groups, between_idx, max_iter=200, tol=1e-10))]
    #[allow(clippy::too_many_arguments)]
    pub fn fit(
        &mut self,
        X: PyFaerRef,
        y: PyArrRef,
        group_codes: PyArrRef,
        n_groups: usize,
        between_idx: Vec<usize>,
        max_iter: usize,
        tol: f64,
    ) -> PyResult<()> {
        let codes: Vec<usize> = group_codes.0.iter().map(|v| *v as usize).collect();
        let result = fit_reml(X.0, y.0, &codes, n_groups, &between_idx, max_iter, tol)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;

        self.coeffs = result.coeffs;
        self.std_errors = result.std_errors;
        self.dfs = result.dfs;
        self.gamma = result.gamma;
        self.resid_variance = result.resid_variance;
        self.is_fit = true;
        Ok(())
    }

    #[getter]
    pub fn coeffs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArr>> {
        Bound::new(py, PyArr(self.coeffs.clone()))
    }

    #[getter]
    pub fn std_errors<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArr>> {
        Bound::new(py, PyArr(self.std_errors.clone()))
    }

    #[getter]
    pub fn dfs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArr>> {
        Bound::new(py, PyArr(self.dfs.clone()))
    }

    #[getter]
    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    #[getter]
    pub fn resid_variance(&self) -> f64 {
        self.resid_variance
    }
}
