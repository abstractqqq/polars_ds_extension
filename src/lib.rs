#![feature(float_gamma)]

mod graph;
mod num;
mod stats;
mod stats_utils;
mod str2;
mod utils;
use pyo3::{pymodule, types::PyModule, PyResult, Python};

#[cfg(target_os = "linux")]
use jemallocator::Jemalloc;

#[global_allocator]
#[cfg(target_os = "linux")]
static ALLOC: Jemalloc = Jemalloc;

#[pymodule]
#[pyo3(name = "_polars_ds")]
fn _polars_ds(_py: Python<'_>, _m: &PyModule) -> PyResult<()> {
    Ok(())
}
