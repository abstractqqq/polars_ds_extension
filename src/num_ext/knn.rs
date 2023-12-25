/// Performs KNN related search queries, classification and regression, and
/// other features/entropies that require KNN to be efficiently computed.
use super::which_distance;
use crate::no_null_in_inputs;
use itertools::Itertools;
use kdtree::KdTree;
use ndarray::{Array2, Axis};
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
struct KnnKwargs {
    k: usize,
    leaf_size: usize,
    metric: String,
    parallel: bool,
}

#[derive(Deserialize, Debug)]
struct RadiusKwargs {
    radius: f64,
    leaf_size: usize,
    metric: String,
    parallel: bool,
}

#[inline]
fn build_standard_kdtree<'a>(
    dim: usize,
    leaf_size: usize,
    data: &'a Array2<f64>,
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
fn pl_knn_ptwise(inputs: &[Series], kwargs: KnnKwargs) -> PolarsResult<Series> {
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
    let tree = build_standard_kdtree(dim, leaf_size, &data)?;

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

#[polars_expr(output_type=Boolean)]
fn pl_knn_pt(inputs: &[Series], kwargs: KnnKwargs) -> PolarsResult<Series> {
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
    let tree = build_standard_kdtree(dim, leaf_size, &data)?;

    // Building the output
    let mut out: Vec<bool> = vec![false; height];
    if let Ok(v) = tree.iter_nearest(p, &dist_func) {
        for (_, i) in v.into_iter().take(k) {
            out[*i] = true;
        }
    }
    Ok(BooleanChunked::from_slice("", &out).into_series())
}

// /// For a given point, find out how many neighbors are within that radius
// #[polars_expr(output_type=Boolean)]
// fn pl_query_radius(inputs: &[Series], kwargs: RadiusKwargs) -> PolarsResult<Series> {
//     // Make sure no null input
//     let _ = no_null_in_inputs(inputs, "KNN: Input contains null.".into())?;

//     // Set up radius
//     let pt = inputs[0].f64()?;
//     let pt = pt.rechunk();
//     let pt = pt.to_ndarray().unwrap(); // safe because we rechunked
//     let pt = pt.as_slice().unwrap();

//     // Set up params
//     let radius = kwargs.radius;
//     let data = DataFrame::new(inputs[1..].to_vec())?.agg_chunks();
//     let dim = inputs[1..].len();
//     if dim == 0 {
//         return Err(PolarsError::ComputeError("KNN: No column to decide distance from.".into()));
//     }
//     let leaf_size = kwargs.leaf_size;
//     let dist_func = which_distance(kwargs.metric.as_str(), dim)?;

//     // Need to use C order because C order is row-contiguous
//     let height = data.height();
//     let data = data.to_ndarray::<Float64Type>(IndexOrder::C)?;

//     // Building the tree
//     let tree = build_standard_kdtree(dim, leaf_size, &data)?;

//     // Build output
//     let mut output:Vec<bool> = vec![false; height];
//     if let Ok(v) = tree.iter_nearest(pt, &dist_func) {
//         for (_, i) in v.take_while(|(d, _)| d <= &radius) {
//             output[*i] = true;
//         }
//     }

//     Ok(BooleanChunked::from_slice("", &output).into_series())

// }
