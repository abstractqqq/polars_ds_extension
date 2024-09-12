#![allow(non_snake_case)]
use crate::arkadia::{leaf::OwnedLeaf, OwnedKDT, SpatialQueries};
use crate::linalg::LinalgErrors;
use crate::utils::DIST;
use core::f64;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyclass(subclass)]
pub struct PyKDT {
    kdt: OwnedKDT<f64, usize>,
}

fn row_major_slice_to_leaves(sl: &[f64], row_size: usize) -> Vec<OwnedLeaf<f64, usize>> {
    sl.chunks_exact(row_size).enumerate().map(|pair| pair.into()).collect()
}

#[pymethods]
impl PyKDT {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(
        X,
        balanced = true,
        distance = "sql2".to_string(),
    ))]
    pub fn new(X: PyReadonlyArray2<f64>, balanced: bool, distance: String) -> PyResult<Self> {
        let ncols = X.shape()[1];
        let d = DIST::<f64>::new_from_str_informed(distance, ncols)
            .map_err(|e| PyValueError::new_err(e))?;

        if X.is_c_contiguous() && !X.is_empty() {
            let matrix_slice = X.as_slice().unwrap();
            let leaves = row_major_slice_to_leaves(matrix_slice, ncols);
            let kdt = OwnedKDT::from_leaves_unchecked(leaves, d, balanced.into());
            Ok(PyKDT { kdt: kdt })
        } else {
            Err(LinalgErrors::NotContiguousOrEmpty.into())
        }
    }

    // pub fn add<'py>(
    //     &mut self,
    //     py: Python<'py>,
    //     X: PyReadonlyArray2<f64>,
    // ) {

    //     if X.is_c_contiguous() && !X.is_empty() {
    //         let nrows = X.shape()[0];
    //         let ncols = X.shape()[1];
    //         let matrix_slice = X.as_slice().unwrap();
    //         if nrows != self.kdt.dim() {
    //             Err(LinalgErrors::DimensionMismatch.into())
    //         } else {

    //             for sl in matrix_slice.chunks_exact(ncols) {

    //             }
    //             self.kdt.add_unchecked(leaf, depth)


    //         }



    //     } else {
    //         Err(LinalgErrors::NotContiguousOrEmpty.into())
    //     }

    // }

    pub fn knn<'py>(
        &self,
        py: Python<'py>,
        X: PyReadonlyArray2<f64>,
        k: usize,
        epsilon: f64,
        max_dist_bound: f64,
        parallel: bool,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<usize>>)> {
        if X.is_c_contiguous() && !X.is_empty() {
            let nrows = X.shape()[0];
            let ncols = X.shape()[1];
            let matrix_slice = X.as_slice().unwrap();
            if k < 1 || ncols != self.kdt.dim {
                return Err(PyValueError::new_err(
                    "Input `k` cannot be 0 or dimension doesn't match.".to_string(),
                ));
            }

            let mut dist_vec = Vec::with_capacity(nrows * k);
            let mut idx_vec = Vec::with_capacity(nrows * k);
            if parallel && nrows >= 16 {
                let mut neighbors = Vec::with_capacity(nrows);
                (0..nrows)
                    .into_par_iter()
                    .map(|i| 
                        self.kdt
                            .knn_bounded_unchecked(
                                k, 
                                &matrix_slice[i * ncols..(i + 1) * ncols], 
                                max_dist_bound, 
                                epsilon
                            )
                    )
                    .collect_into_vec(&mut neighbors);

                for nbs in neighbors {
                    let n_found = nbs.len();
                    for nb in nbs {
                        let (d, id) = nb.to_pair();
                        dist_vec.push(d);
                        idx_vec.push(id);
                    }
                    let diff = k.abs_diff(n_found);
                    for _ in 0..diff {
                        dist_vec.push(f64::INFINITY);
                        idx_vec.push(nrows + 1);
                    }
                }
            } else {
                for pt in matrix_slice.chunks_exact(ncols) {
                    let result = self
                        .kdt
                        .knn_bounded_unchecked(k, pt, max_dist_bound, epsilon);
                    let n_found = result.len();
                    for nb in result {
                        let (d, id) = nb.to_pair();
                        dist_vec.push(d);
                        idx_vec.push(id);
                    }
                    let diff = k.abs_diff(n_found);
                    for _ in 0..diff {
                        dist_vec.push(f64::INFINITY);
                        idx_vec.push(nrows + 1);
                    }
                }
            }

            let dist_mat = dist_vec.into_pyarray_bound(py).reshape((nrows, k))?;
            let idx_mat = idx_vec.into_pyarray_bound(py).reshape((nrows, k))?;

            Ok((dist_mat, idx_mat))
        } else {
            Err(LinalgErrors::NotContiguousOrEmpty.into())
        }
    }

    pub fn within_idx_only<'py>(
        &self,
        X: PyReadonlyArray2<f64>,
        r: f64,
        sort:bool,
        parallel: bool,
    ) -> PyResult<Vec<Vec<u32>>> {

        if X.is_c_contiguous() && !X.is_empty() {
            let nrows = X.shape()[0];
            let ncols = X.shape()[1];
            let matrix_slice = X.as_slice().unwrap();

            if parallel && nrows > 16 {
                let mut output_vec = Vec::with_capacity(nrows);
                (0..nrows)
                    .into_par_iter()
                    .map(|i| {
                        self.kdt
                            .within(&matrix_slice[i * ncols..(i + 1) * ncols], r, sort)
                            .unwrap_or(vec![])
                            .into_iter()
                            .map(|nb| nb.to_item() as u32)
                            .collect::<Vec<_>>()
                    }).collect_into_vec(&mut output_vec);
                Ok(output_vec)
            } else {
                Ok(
                    matrix_slice.chunks_exact(ncols).map(
                    |pt| 
                    self.kdt
                        .within(pt, r, sort)
                        .unwrap_or(vec![])
                        .into_iter()
                        .map(|nb| nb.to_item() as u32)
                        .collect::<Vec<_>>()
                    ).collect::<Vec<_>>()
                )
            }

        } else {
            Err(LinalgErrors::NotContiguousOrEmpty.into())
        }
    }

    pub fn within_with_dist<'py>(
        &self,
        X: PyReadonlyArray2<f64>,
        r: f64,
        sort:bool,
        parallel: bool,
    ) -> PyResult<Vec<Vec<(u32, f64)>>> {

        if X.is_c_contiguous() && !X.is_empty() {
            let nrows = X.shape()[0];
            let ncols = X.shape()[1];
            let matrix_slice = X.as_slice().unwrap();

            if parallel && nrows > 16 {
                let mut output_vec = Vec::with_capacity(nrows);
                (0..nrows)
                    .into_par_iter()
                    .map(|i| {
                        self.kdt
                            .within(&matrix_slice[i * ncols..(i + 1) * ncols], r, sort)
                            .unwrap_or(vec![])
                            .into_iter()
                            .map(|nb|{
                                let (d, u) = nb.to_pair();
                                (u as u32, d)
                            })
                            .collect::<Vec<_>>()
                    }).collect_into_vec(&mut output_vec);
                Ok(output_vec)
            } else {
                Ok(
                    matrix_slice.chunks_exact(ncols).map(
                    |pt| 
                    self.kdt
                        .within(pt, r, sort)
                        .unwrap_or(vec![])
                        .into_iter()
                        .map(|nb| {
                            let (d, u) = nb.to_pair();
                            (u as u32, d)
                        })
                        .collect::<Vec<_>>()
                    ).collect::<Vec<_>>()
                )
            }
        } else {
            Err(LinalgErrors::NotContiguousOrEmpty.into())
        }
    }


    pub fn within_count<'py>(
        &self,
        py: Python<'py>,
        X: PyReadonlyArray2<f64>,
        r: f64,
        // parallel: bool,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {

        if X.is_c_contiguous() && !X.is_empty() {
            let ncols = X.shape()[1];
            let matrix_slice = X.as_slice().unwrap();
            let output:Vec<u32> = matrix_slice.chunks_exact(ncols).map(
                |sl| self.kdt.within_count(sl, r).unwrap_or(0)
            ).collect();
            Ok(output.into_pyarray_bound(py))
        } else {
            Err(LinalgErrors::NotContiguousOrEmpty.into())
        }

    }

}
