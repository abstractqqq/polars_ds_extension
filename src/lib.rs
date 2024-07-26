#![feature(float_gamma)]

mod arkadia;
mod linalg;
mod num;
mod stats;
mod stats_utils;
mod str2;
mod utils;

use pyo3::{pymodule, types::PyModule, Bound, PyResult, Python};

#[pymodule]
#[pyo3(name = "_polars_ds")]
fn _polars_ds(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    use numpy::{Ix1, PyArray, PyReadonlyArray2};

    // How do I factor out this? I don't want to put all code here.
    #[pyfn(m)]
    #[pyo3(name = "pds_linear_regression")]
    fn pds_linear_regression<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray2<f64>,
        has_bias: bool,
        method: String,
        lambda: f64,
        tol: f64,
    ) -> Bound<'py, PyArray<f64, Ix1>> {
        use crate::linalg::lstsq;

        let x = x.as_array();
        let y = y.as_array();
        // Add bias is done in Python.
        let coeffs = match lstsq::LRMethods::from(method) {
            lstsq::LRMethods::Normal => lstsq::faer_qr_lstsq(x, y),
            lstsq::LRMethods::L1 => lstsq::faer_lasso_regression(x, y, lambda, has_bias, tol),
            lstsq::LRMethods::L2 => lstsq::faer_cholskey_ridge_regression(x, y, lambda, has_bias),
        };

        let coeffs = coeffs.col_as_slice(0).to_vec();
        PyArray::from_vec_bound(py, coeffs)
    }

    Ok(())
}
