use super::which_distance;
/// Performs KNN related search queries, classification and regression, and
/// other features/entropies that require KNN to be efficiently computed.
use itertools::Itertools;
use kdtree::KdTree;
use ndarray::Axis;
use polars::prelude::*;
use pyo3_polars::export::polars_core::utils::rayon::iter::IntoParallelIterator;
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

pub fn knn_index_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "ids",
        DataType::List(Box::new(DataType::UInt64)),
    ))
}

#[polars_expr(output_type_func=knn_index_output)]
fn pl_knn_ptwise(inputs: &[Series], kwargs: KnnKwargs) -> PolarsResult<Series> {
    // Make sure no null input
    for s in inputs {
        if s.null_count() > 0 {
            return Err(PolarsError::ComputeError(
                "KNN: Input contains null.".into(),
            ));
        }
    }

    // Set up params
    let id = inputs[0].u64()?;
    let data = DataFrame::new(inputs[1..].to_vec())?.agg_chunks();
    let dim = inputs[1..].len();

    let k = kwargs.k;
    let leaf_size = kwargs.leaf_size;
    let parallel = kwargs.parallel;
    let dist_func = which_distance(kwargs.metric.as_str(), dim)
        .map_err(|e| PolarsError::ComputeError(e.into()))?;

    // Need to use C order because C order is row-contiguous
    let matrix = data.to_ndarray::<Float64Type>(IndexOrder::C)?;

    // Building the tree
    let mut tree = KdTree::with_capacity(dim, leaf_size);
    for (i, p) in matrix.rows().into_iter().enumerate() {
        let s = p.to_slice().unwrap(); // C order makes sure rows are contiguous
        let _is_ok = tree
            .add(s, i)
            .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    }

    // Building output
    let mut builder =
        ListPrimitiveChunkedBuilder::<UInt64Type>::new("", id.len(), k, DataType::UInt64);
    if parallel {
        let mut nbs: Vec<Option<Vec<u64>>> = Vec::with_capacity(id.len());
        matrix
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|p| {
                let s = p.to_slice().unwrap(); // C order makes sure rows are contiguous
                if let Ok(v) = tree.iter_nearest(s, &dist_func) {
                    // By construction, this unwrap is safe
                    Some(
                        v.map(|(_, i)| id.get(*i).unwrap())
                            .skip(1)
                            .take(k)
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
        for p in matrix.rows() {
            let s = p.to_slice().unwrap(); // C order makes sure rows are contiguous
            if let Ok(v) = tree.iter_nearest(s, &dist_func) {
                // By construction, this unwrap is safe
                let w: Vec<u64> = v
                    .map(|(_, i)| id.get(*i).unwrap())
                    .skip(1)
                    .take(k)
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
