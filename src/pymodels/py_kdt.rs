#![allow(non_snake_case)]
use core::f64;
use crate::arkadia::{SpatialQueries, OwnedKDT, leaf::OwnedLeaf};
use crate::linalg::LinalgErrors;
use crate::utils::DIST;
use numpy::{ IntoPyArray, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use rayon::prelude::*;

#[pyclass(subclass)]
pub struct PyKDT {
    kdt: OwnedKDT<f64, usize>,
}

fn row_major_slice_to_leaves(sl:&[f64], row_size:usize) -> Vec<OwnedLeaf<f64, usize>> {
    (0..sl.len()).step_by(row_size).enumerate().map(|(i, j)| {
        let j_end = j + row_size;
        (i, &sl[j..j_end]).into()
    }).collect()
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
    pub fn new(X: PyReadonlyArray2<f64>, balanced:bool, distance: String) -> PyResult<Self> {
        
        let ncols = X.shape()[1];
        let d = DIST::<f64>::new_from_str_informed(distance, ncols)
            .map_err(|e| PyValueError::new_err(e))?;

        if X.is_c_contiguous() && !X.is_empty() {
            let matrix_slice = X.as_slice().unwrap();
            let leaves = row_major_slice_to_leaves(matrix_slice, ncols);
            let kdt = OwnedKDT::from_leaves_unchecked(
                leaves,
                d,
                balanced.into()
            );
            Ok(PyKDT{kdt: kdt})
        } else {
            Err(LinalgErrors::NotContiguousOrEmpty.into())
        }
    }

    pub fn knn<'py>(
        &self, 
        py: Python<'py>,
        X: PyReadonlyArray2<f64>, 
        k:usize, 
        epsilon:f64, 
        max_dist_bound:f64, 
        parallel: bool,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<usize>>)> {

        if X.is_c_contiguous() && !X.is_empty() {

            let nrows = X.shape()[0];
            let ncols = X.shape()[1];
            let matrix_slice = X.as_slice().unwrap();
            if k < 1 || ncols != self.kdt.dim {
                return Err(PyValueError::new_err("Input `k` cannot be 0 or dimension doesn't match.".to_string()))
            }

            let mut dist_vec = Vec::with_capacity(nrows * k);
            let mut idx_vec = Vec::with_capacity(nrows * k);
            if parallel {
                let mut neighbors = Vec::with_capacity(nrows);
                (0..nrows).into_par_iter().map(|i| {
                    let pt = &matrix_slice[i*ncols..(i+1)*ncols];
                    self.kdt.knn_bounded_unchecked(k, pt, max_dist_bound, epsilon)
                }).collect_into_vec(&mut neighbors);

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
                    let result = self.kdt.knn_bounded_unchecked(k, pt, max_dist_bound, epsilon);
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
}




