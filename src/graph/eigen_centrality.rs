use faer::{scale, IntoFaer, Mat, MatRef};
use ndarray::Array2;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
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

    let mut incident: Array2<f64> = Array2::from_elem((nrows, nrows), 0.);
    // slow. Sparse probably would be better.
    for (i, op_e) in edges.into_iter().enumerate() {
        if let Some(e) = op_e {
            let edges = e.u64()?;
            for op_j in edges.into_iter() {
                if let Some(j) = op_j {
                    match incident.get_mut([i, j as usize]) {
                        Some(pt) => {
                            *pt = 1.;
                        }
                        None => {
                            return Err(PolarsError::ComputeError(
                                "Edge list contains out-of-bounds index.".into(),
                            ))
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
