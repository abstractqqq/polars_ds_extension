use crate::stats_utils::beta::student_t_sf;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Two-sided p-value for a Student's t statistic with `df` degrees of freedom,
/// via the regularized incomplete beta function.
#[pyfunction]
pub fn student_t_two_sided_pvalue(t: f64, df: f64) -> PyResult<f64> {
    let sf = student_t_sf(t.abs(), df).map_err(PyValueError::new_err)?;
    Ok((2.0 * sf).clamp(0.0, 1.0))
}
