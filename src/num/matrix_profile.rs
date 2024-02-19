use crate::utils::split_offsets;
use itertools::Itertools;
use kdtree::KdTree;
use ndarray::parallel;
use num::ToPrimitive;
use polars::prelude::*;
use pyo3_polars::{
    derive::{polars_expr, CallerContext},
    export::polars_core::{
        utils::rayon::prelude::{IntoParallelIterator, ParallelIterator},
        POOL,
    },
};
use serde::Deserialize;

// pub fn matrix_profile_output(_: &[Field]) -> PolarsResult<Field> {
//     let dist = Field::new("squared_znorm", DataType::List(Box::new(DataType::Float64)));
//     let idx = Field::new("idx", DataType::List(Box::new(DataType::UInt32)));
//     let v = vec![dist, idx];
//     Ok(Field::new("profile", DataType::Struct(v)))
// }

#[derive(Deserialize, Debug)]
pub(crate) struct MatrixProfileKwargs {
    pub(crate) window_size: usize,
    pub(crate) leaf_size: usize,
    pub(crate) parallel: bool,
}

#[polars_expr(output_type=Float64)]
fn pl_matrix_profile(
    inputs: &[Series],
    context: CallerContext,
    kwargs: MatrixProfileKwargs,
) -> PolarsResult<Series> {
    let ts = inputs[0].f64()?;
    let m = kwargs.window_size;
    let mf64 = m as f64;
    let leaf_size = kwargs.leaf_size;
    let parallel = kwargs.parallel;
    let can_parallel = parallel && !context.parallel();
    if ts.null_count() > 0 || ts.len() <= m || m < 4 {
        return Err(PolarsError::ComputeError(
            "Matrix Profile: Input must not contain nulls and must have length > window size > 4."
                .into(),
        ));
    }
    let ts = ts.rechunk();
    let ts = ts.cont_slice()?;
    let output_size = ts.len() - m + 1;
    // Build the kd-tree
    let mut points: Vec<f64> = Vec::with_capacity(output_size * m);

    // Creating Z-normalized Points
    // Rolling std is complicated and I am being lazy
    // Init for the rolling stats
    let mut rolling_mean: f64 = ts[..m].iter().sum::<f64>() / mf64;
    let mut first: f64 = ts[0];
    let std = (ts[..m]
        .iter()
        .fold(0., |acc, x| acc + (x - rolling_mean).powi(2))
        / (mf64 - 1.0))
        .sqrt();
    points.extend(ts[..m].iter().map(|x| (x - rolling_mean) / std));
    // Build the rolling stats
    for sl in ts[1..].windows(m) {
        let last = sl.last().unwrap();
        rolling_mean += (last - first) / mf64;
        first = sl[0];
        let std = (sl
            .iter()
            .fold(0., |acc, x| acc + (x - rolling_mean).powi(2))
            / (mf64 - 1.0))
            .sqrt();
        points.extend(sl.iter().map(|x| (x - rolling_mean) / std))
    }
    // Inserting points into tree
    let mut tree: KdTree<f64, usize, &[f64]> = KdTree::with_capacity(m, leaf_size);
    for (i, p) in points.chunks_exact(m).enumerate() {
        tree.add(p, i)
            .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    }

    // Query and build output
    // Exclusion zone default is: i +- ceil(m/4)
    let exclusion = (mf64 / 4.0).ceil().to_usize().unwrap();
    // need at least exclusion * 2 + 2 points to ensure at least 1 pt to be outside exclusion. Pigeon hole.
    let k = (exclusion << 1) + 2;

    let ca = if can_parallel {
        POOL.install(|| {
            let n_threads = POOL.current_num_threads();
            let splits = split_offsets(output_size, n_threads);
            let chunks: Vec<_> = splits
                .into_par_iter()
                .map(|(offset, len)| {
                    let start_idx = offset * m;
                    let end_idx = (offset + len) * m;
                    let piece = &points[start_idx..end_idx];
                    let out = Float64Chunked::from_iter_options(
                        "",
                        piece.chunks_exact(m).enumerate().map(|(i, pt)| {
                            let mut dist: Option<f64> = None;
                            if let Ok(v) = tree.nearest(&pt, k, &super::squared_euclidean) {
                                for (d, j) in v.into_iter() {
                                    if (i + offset).abs_diff(*j) > exclusion {
                                        dist = Some(d);
                                        break;
                                    }
                                }
                            }
                            dist
                        }),
                    );
                    out.downcast_iter().cloned().collect::<Vec<_>>()
                })
                .collect();
            Float64Chunked::from_chunk_iter("squared_znorm", chunks.into_iter().flatten())
        })
    } else {
        let mut dist_builder: PrimitiveChunkedBuilder<Float64Type> =
            PrimitiveChunkedBuilder::new("squared_znorm", output_size);
        // let mut idx_builder:PrimitiveChunkedBuilder<UInt32Type> = PrimitiveChunkedBuilder::new("idx", output_size);
        for (i, pt) in points.chunks_exact(m).enumerate() {
            let mut dist: Option<f64> = None;
            // let mut id: Option<u32> = None;
            if let Ok(v) = tree.nearest(&pt, k, &super::squared_euclidean) {
                for (d, j) in v.into_iter() {
                    if i.abs_diff(*j) > exclusion {
                        dist = Some(d);
                        // id = Some(*j as u32);
                        break;
                    } // <= means in exclusion zone. > means good
                }
            }
            dist_builder.append_option(dist);
            // idx_builder.append_option(id);
        }
        dist_builder.finish()
    };

    Ok(ca.into_series())

    // let out_dist = dist_builder.finish();
    // let out_id = idx_builder.finish();
    // let s_dist = out_dist.into_series();
    // let s_id = out_id.into_series();
    // let out = StructChunked::new("profile", &[s_dist, s_id])?;
    // Ok(out.into_series())
}
