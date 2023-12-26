mod num_ext;
mod stats;
mod stats_ext;
mod str_ext;
mod utils;
use polars::{
    error::{PolarsError, PolarsResult},
    series::Series,
};
use pyo3::{pymodule, types::PyModule, PyResult, Python};

#[cfg(target_os = "linux")]
use jemallocator::Jemalloc;

#[global_allocator]
#[cfg(target_os = "linux")]
static ALLOC: Jemalloc = Jemalloc;

// #[inline]
// pub fn no_null_in_inputs(inputs: &[Series], err_msg: String) -> PolarsResult<()> {
//     for s in inputs {
//         if s.null_count() > 0 {
//             return Err(PolarsError::ComputeError(err_msg.into()));
//         }
//     }
//     Ok(())
// }

#[pymodule]
#[pyo3(name = "_polars_ds")]
fn _polars_ds(_py: Python<'_>, _m: &PyModule) -> PyResult<()> {
    Ok(())
}
