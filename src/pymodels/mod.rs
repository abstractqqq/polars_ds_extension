pub mod py_kdt;
pub mod py_lr;
pub mod py_glm;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use crate::linear::LinalgErrors;

impl From<LinalgErrors> for PyErr {
    fn from(value: LinalgErrors) -> Self {
        PyValueError::new_err(value.to_string())
    }
}