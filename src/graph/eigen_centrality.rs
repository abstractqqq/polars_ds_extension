use faer::{scale, sparse::SymbolicSparseRowMatRef, IntoFaer, Mat, MatRef};
use ndarray::Array2;
use polars::prelude::*;
use pyo3_polars::{derive::polars_expr, export::polars_core::utils::arrow::array::PrimitiveArray};
use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub(crate) struct EigenKwargs {
    pub(crate) n_iter: usize,
    pub(crate) normalize: bool,
    pub(crate) sparse: bool,
}

/// Use power iteration to find the dominant eigenvector
fn power_iteration(a: MatRef<f64>, nrows: usize, n_iter: usize) -> Vec<f64> {
    let mut b = Mat::<f64>::from_fn(nrows, 1, |_, _| (nrows as f64) / 2.0);
    for _ in 0..n_iter {
        b = a * b;
        let norm = b.norm_l2();
        b = scale(1.0 / norm) * b;
    }

    let mut out: Vec<f64> = Vec::with_capacity(nrows);
    for i in 0..b.nrows() {
        out.push(b.read(i, 0))
    }
    out
}

fn sparse_power_iteration(
    a: SymbolicSparseRowMatRef<'_, usize>,
    nrows: usize,
    n_iter: usize,
) -> Vec<f64> {
    // We don't need a SparseRowMatRef. Only a symbolic one is enough because
    // this matrix's only non-zero value is 1.
    let mut b = vec![nrows as f64 / 2.0; nrows];
    for _ in 0..n_iter {
        // update b, matrix multiplication
        let mut temp = (0..nrows)
            .map(|i| a.col_indices_of_row(i).fold(0., |acc, i| acc + b[i]))
            .collect::<Vec<f64>>();
        let norm = temp.iter().fold(0., |acc, x| acc + x * x).sqrt();
        // normalize
        temp.iter_mut().for_each(|v| *v = *v / norm);
        b = temp;
    }
    b
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
    let sparse = kwargs.sparse;

    let mut out = if sparse {
        let mut row_ptrs: Vec<usize> = Vec::with_capacity(nrows + 1);
        row_ptrs.push(0);
        let mut col_indices: Vec<usize> = Vec::new();
        // let mut values:Vec<u8> = Vec::new();
        for arr in edges.downcast_iter() {
            for op_arr in arr.iter() {
                if let Some(arr2) = op_arr {
                    if let Some(indices) = arr2.as_any().downcast_ref::<PrimitiveArray<u64>>() {
                        for j in indices.non_null_values_iter() {
                            col_indices.push(j as usize);
                            // values.push(1);
                        }
                    }
                }
                row_ptrs.push(col_indices.len());
            }
        }

        let syms =
            SymbolicSparseRowMatRef::new_checked(nrows, nrows, &row_ptrs, None, &col_indices);
        sparse_power_iteration(syms, nrows, n_iter)
    } else {
        let mut adj_mat: Array2<f64> = Array2::from_elem((nrows, nrows), 0.);
        for arr in edges.downcast_iter() {
            for (i, op_arr) in arr.iter().enumerate() {
                if let Some(arr2) = op_arr {
                    if let Some(indices) = arr2.as_any().downcast_ref::<PrimitiveArray<u64>>() {
                        for j in indices.non_null_values_iter() {
                            if let Some(pt) = adj_mat.get_mut([i, j as usize]) {
                                *pt = 1.0;
                            }
                        }
                    }
                }
            }
        }
        let faer_adj = adj_mat.view().into_faer();
        power_iteration(faer_adj, nrows, n_iter)
    };

    if normalize {
        let s: f64 = out.iter().sum();
        out.iter_mut().for_each(|x| *x = *x / s);
    }
    Ok(Series::from_vec(edges.name(), out))
}
