#![feature(float_gamma)]

mod arkadia;
mod num;
mod stats;
mod stats_utils;
mod str2;
mod utils;
use pyo3::{pymodule, types::PyModule, Bound, PyResult, Python};

#[pymodule]
#[pyo3(name = "_polars_ds")]
fn _polars_ds(_py: Python<'_>, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
