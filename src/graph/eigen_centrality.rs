use faer::{scale, IntoFaer, Mat, MatRef};
use ndarray::Array2;
use polars::prelude::*;
use pyo3_polars::{derive::polars_expr, export::polars_core::utils::arrow::array::PrimitiveArray};
use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub(crate) struct EigenKwargs {
    pub(crate) n_iter: usize,
    pub(crate) normalize: bool,
}

/// Use power iteration to find the dominant eigenvector
fn power_iteration(a: MatRef<f64>, n: usize, n_iter: usize) -> Vec<f64> {
    // n = a.shape[1]
    let mut b = Mat::<f64>::from_fn(n, 1, |_, _| (n as f64) / 2.0);
    for _ in 0..n_iter {
        b = a * b;
        let norm = b.norm_l2();
        b = scale(1.0 / norm) * b;
    }

    let mut out: Vec<f64> = Vec::with_capacity(n);
    for i in 0..b.nrows() {
        out.push(b.read(i, 0))
    }
    out
}

#[polars_expr(output_type=Float64)]
fn pl_eigen_centrality(inputs: &[Series], kwargs: EigenKwargs) -> PolarsResult<Series> {
    // Use this function for now. I think sparse matrices work better for such problems.
    // However, Faer's sparse module is too hard to use right now.

    let edges = inputs[0].list()?;
    let nrows = edges.len();
    // Set up kwargs
    let n_iter = kwargs.n_iter;
    let normalize = kwargs.normalize;

    // slow. Sparse probably would be better.
    let mut incident: Array2<f64> = Array2::from_elem((nrows, nrows), 0.);
    for arr in edges.downcast_iter() {
        for (i, op_arr) in arr.iter().enumerate() {
            if let Some(arr2) = op_arr {
                if let Some(indices) = arr2.as_any().downcast_ref::<PrimitiveArray<u64>>() {
                    for j in indices.non_null_values_iter() {
                        if let Some(pt) = incident.get_mut([i, j as usize]) {
                            *pt = 1.0;
                        }
                    }
                }
            }
        }
    }
    let faer_incident = incident.view().into_faer();
    let mut out = power_iteration(faer_incident, nrows, n_iter);
    if normalize {
        let s: f64 = out.iter().sum();
        out.iter_mut().for_each(|x| *x = *x / s);
    }
    Ok(Series::from_vec("", out))
}
