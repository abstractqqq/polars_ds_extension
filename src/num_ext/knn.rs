/// Performs KNN related search queries, classification and regression, and
/// other features/entropies that require KNN to be efficiently computed.
use super::which_distance;
use crate::no_null_in_inputs;
use itertools::Itertools;
use kdtree::KdTree;
use ndarray::{ArrayView2, Axis};
use polars::prelude::*;
use pyo3_polars::export::polars_core::utils::rayon::iter::{
    FromParallelIterator, IntoParallelIterator, ParallelBridge,
};
use pyo3_polars::{
    derive::polars_expr,
    export::polars_core::utils::rayon::iter::{IndexedParallelIterator, ParallelIterator},
};
use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub(crate) struct KdtreeKwargs {
    pub(crate) k: usize,
    pub(crate) leaf_size: usize,
    pub(crate) metric: String,
    pub(crate) parallel: bool,
}

#[inline]
pub fn build_standard_kdtree<'a>(
    dim: usize,
    leaf_size: usize,
    data: &'a ArrayView2<f64>,
) -> Result<KdTree<f64, usize, &'a [f64]>, PolarsError> {
    // Building the tree
    let mut tree = KdTree::with_capacity(dim, leaf_size);
    for (i, p) in data.rows().into_iter().enumerate() {
        let s = p.to_slice().unwrap(); // C order makes sure rows are contiguous
        let _is_ok = tree
            .add(s, i)
            .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    }
    Ok(tree)
}

pub fn knn_index_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "ids",
        DataType::List(Box::new(DataType::UInt64)),
    ))
}

#[polars_expr(output_type_func=knn_index_output)]
fn pl_knn_ptwise(inputs: &[Series], kwargs: KdtreeKwargs) -> PolarsResult<Series> {
    // Make sure no null input
    let _ = no_null_in_inputs(inputs, "KNN: Input contains null.".into())?;

    // Set up params
    let id = inputs[0].u64()?;
    let data = DataFrame::new(inputs[1..].to_vec())?.agg_chunks();
    let dim = inputs[1..].len();
    if dim == 0 {
        return Err(PolarsError::ComputeError(
            "KNN: No column to decide distance from.".into(),
        ));
    }

    let k = kwargs.k;
    let leaf_size = kwargs.leaf_size;
    let parallel = kwargs.parallel;
    let dist_func = which_distance(kwargs.metric.as_str(), dim)?;

    // Need to use C order because C order is row-contiguous
    let data = data.to_ndarray::<Float64Type>(IndexOrder::C)?;

    // Building the tree
    let binding = data.view();
    let tree = build_standard_kdtree(dim, leaf_size, &binding)?;

    // Building output
    let mut builder =
        ListPrimitiveChunkedBuilder::<UInt64Type>::new("", id.len(), k + 1, DataType::UInt64);
    if parallel {
        let mut nbs: Vec<Option<Vec<u64>>> = Vec::with_capacity(id.len());
        data.axis_iter(Axis(0))
            .into_par_iter()
            .map(|p| {
                let s = p.to_slice().unwrap(); // C order makes sure rows are contiguous
                if let Ok(v) = tree.iter_nearest(s, &dist_func) {
                    // By construction, this unwrap is safe
                    Some(
                        v.map(|(_, i)| id.get(*i).unwrap())
                            .take(k + 1)
                            .collect_vec(),
                    )
                } else {
                    None
                }
            })
            .collect_into_vec(&mut nbs);
        for op_s in nbs {
            if let Some(s) = op_s {
                builder.append_slice(&s);
            } else {
                builder.append_null();
            }
        }
    } else {
        for p in data.rows() {
            let s = p.to_slice().unwrap(); // C order makes sure rows are contiguous
            if let Ok(v) = tree.iter_nearest(s, &dist_func) {
                // By construction, this unwrap is safe
                let w: Vec<u64> = v
                    .map(|(_, i)| id.get(*i).unwrap())
                    .take(k + 1)
                    .collect_vec();
                builder.append_slice(w.as_slice());
            } else {
                builder.append_null();
            }
        }
    }
    let ca = builder.finish();
    Ok(ca.into_series())
}

/// Find all the rows that are the k-nearest neighbors to the point given.
/// Only k points will be returned as true.
#[polars_expr(output_type=Boolean)]
fn pl_knn_pt(inputs: &[Series], kwargs: KdtreeKwargs) -> PolarsResult<Series> {
    // Make sure no null input
    let _ = no_null_in_inputs(inputs, "KNN: Input contains null.".into())?;

    // Check len
    let pt = inputs[0].f64()?;
    let dim = inputs[1..].len();
    if dim == 0 || pt.len() != dim {
        return Err(PolarsError::ComputeError(
            "KNN: There has to be at least one column in `others` and input point \
            must be the same length as the number of columns in `others`."
                .into(),
        ));
    }
    // Set up the point to query
    let binding = pt.rechunk();
    let p = binding.to_ndarray()?;
    let p = p.as_slice().unwrap(); // Rechunked, so safe to unwrap

    // Set up params
    let data = DataFrame::new(inputs[1..].to_vec())?.agg_chunks();
    let height = data.height();
    let dim = inputs[1..].len();
    let k = kwargs.k;
    let leaf_size = kwargs.leaf_size;
    let dist_func = which_distance(kwargs.metric.as_str(), dim)?;

    // Need to use C order because C order is row-contiguous
    let data = data.to_ndarray::<Float64Type>(IndexOrder::C)?;

    // Building the tree
    let binding = data.view();
    let tree = build_standard_kdtree(dim, leaf_size, &binding)?;

    // Building the output
    let mut out: Vec<bool> = vec![false; height];
    match tree.iter_nearest(p, &dist_func) {
        Ok(v) => {
            for (_, i) in v.into_iter().take(k) {
                out[*i] = true;
            }
        },
        Err(e) => {
            return Err(PolarsError::ComputeError(
                ("KNN: ".to_owned() + e.to_string().as_str()).into()
            ));
        }
    }
    Ok(BooleanChunked::from_slice("", &out).into_series())
}

/// Neighbor count query
#[inline]
pub fn query_nb_cnt(
    tree: &KdTree<f64, usize, &[f64]>,
    data: ArrayView2<f64>,
    dist_func: &fn(&[f64], &[f64]) -> f64,
    r: &f64,
    parallel: bool,
) -> UInt32Chunked {
    if parallel {
        UInt32Chunked::from_par_iter(data.axis_iter(Axis(0)).into_par_iter().map(|pt| {
            let s = pt.to_slice().unwrap(); // C order makes sure rows are contiguous
            if let Ok(v) = tree.iter_nearest(s, dist_func) {
                Some(v.take_while(|(d, _)| d <= &r).count() as u32)
            } else {
                None
            }
        }))
    } else {
        UInt32Chunked::from_iter(data.axis_iter(Axis(0)).map(|pt| {
            let s = pt.to_slice().unwrap(); // C order makes sure rows are contiguous
            if let Ok(v) = tree.iter_nearest(s, dist_func) {
                Some(v.take_while(|(d, _)| d <= &r).count() as u32)
            } else {
                None
            }
        }))
    }
}

/// For every point in this dataframe, find the number of neighbors within radius r
/// The point itself is always considered as a neighbor to itself.
#[polars_expr(output_type=UInt32)]
fn pl_nb_cnt(inputs: &[Series], kwargs: KdtreeKwargs) -> PolarsResult<Series> {
    // Make sure no null input
    let _ = no_null_in_inputs(inputs, "KNN: Input contains null.".into())?;

    // Set up radius
    let radius = inputs[0].f64()?;

    // Set up params
    let data = DataFrame::new(inputs[1..].to_vec())?.agg_chunks();
    let dim = inputs[1..].len();
    if dim == 0 {
        return Err(PolarsError::ComputeError(
            "KNN: No column to decide distance from.".into(),
        ));
    }
    let parallel = kwargs.parallel;
    let leaf_size = kwargs.leaf_size;
    let dist_func = which_distance(kwargs.metric.as_str(), dim)?;

    // Need to use C order because C order is row-contiguous
    let height = data.height();
    let data = data.to_ndarray::<Float64Type>(IndexOrder::C)?;

    // Building the tree
    let binding = data.view();
    let tree = build_standard_kdtree(dim, leaf_size, &binding)?;

    if radius.len() == 1 {
        let r = radius.get(0).unwrap();
        let ca = query_nb_cnt(&tree, data.view(), &dist_func, &r, parallel);
        Ok(ca.into_series())
    } else if radius.len() == height {
        if parallel {
            let ca = UInt32Chunked::from_par_iter(
                radius
                    .into_iter()
                    .zip(data.axis_iter(Axis(0)))
                    .par_bridge()
                    .map(|(rad, pt)| {
                        let r = rad?;
                        let s = pt.to_slice().unwrap(); // C order makes sure rows are contiguous
                        if let Ok(v) = tree.iter_nearest(s, &dist_func) {
                            Some(v.take_while(|(d, _)| d <= &r).count() as u32)
                        } else {
                            None
                        }
                    }),
            );
            Ok(ca.into_series())
        } else {
            let ca = UInt32Chunked::from_iter(radius.into_iter().zip(data.axis_iter(Axis(0))).map(
                |(rad, pt)| {
                    let r = rad?;
                    let s = pt.to_slice().unwrap(); // C order makes sure rows are contiguous
                    if let Ok(v) = tree.iter_nearest(s, &dist_func) {
                        Some(v.take_while(|(d, _)| d <= &r).count() as u32)
                    } else {
                        None
                    }
                },
            ));
            Ok(ca.into_series())
        }
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length or one of them must be a scalar.".into(),
        ))
    }
}
