pub mod py_glm;
pub mod py_kdt;
pub mod py_lr;

use crate::linear::LinalgErrors;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

impl From<LinalgErrors> for PyErr {
    fn from(value: LinalgErrors) -> Self {
        PyValueError::new_err(value.to_string())
    }
}
