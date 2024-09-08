#![allow(non_snake_case)]
use crate::arkadia::{SpatialQueries, OwnedKDT, utils::matrix_to_leaves_w_row_num_owned};
use crate::linalg::LinalgErrors;
use crate::utils::DIST;
use numpy::{ PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use rayon::prelude::*;

#[pyclass(subclass)]
pub struct PyKDT {
    kdt: OwnedKDT<f64, usize>,
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
        
        let d = DIST::<f64>::new_from_str_informed(distance, X.shape()[0])
            .map_err(|e| PyValueError::new_err(e))?;

        if X.is_c_contiguous() && X.shape()[0] > 0 {
            let matrix = X.as_array();
            let leaves = matrix_to_leaves_w_row_num_owned(matrix);
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

    pub fn knn(
        &self, 
        X: PyReadonlyArray2<f64>, 
        k:usize, 
        epsilon:f64, 
        max_dist_bound:f64, 
        parallel: bool,
    ) -> PyResult<Vec<Option<Vec<(f64, usize)>>>> {

        if X.is_c_contiguous() {
            let matrix = X.as_array();
            let nrows = matrix.nrows();
            if parallel {
                let mut output_vec = Vec::with_capacity(nrows);

                (0..nrows).into_par_iter().map(|i| {
                    let row = matrix.row(i);
                    let pt = row.as_slice().unwrap();
                    self.kdt
                        .knn_bounded(k, pt, max_dist_bound, epsilon)
                        .map(|nbs| 
                            nbs.into_iter().map(|nb| nb.to_pair()).collect::<Vec<_>>()
                        )
                }).collect_into_vec(&mut output_vec);
                Ok(output_vec)
            } else {
                Ok(
                    matrix.rows().into_iter().map(|row| {
                        let pt = row.as_slice().unwrap();
                        self.kdt
                            .knn_bounded(k, pt, max_dist_bound, epsilon)
                            .map(|nbs| 
                                nbs.into_iter().map(|nb| nb.to_pair()).collect::<Vec<_>>()
                            )
                    }).collect::<Vec<Option<Vec<(f64, usize)>>>>()
                )
            }
        } else {
            Err(LinalgErrors::NotContiguousOrEmpty.into())
        }

    }
}




