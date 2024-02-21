/// Creating matrix profile for a time series.
/// This is significantly faster than STUMPY when m (window size) is small, and
/// when n (total data points) is large. But is significantly slower when m is large.
use crate::utils::split_offsets;
use kdtree::KdTree;
use num::ToPrimitive;
use polars::prelude::*;
use pyo3_polars::{
    derive::{polars_expr, CallerContext},
    export::polars_core::{
        utils::rayon::prelude::{IntoParallelIterator, ParallelIterator},
        POOL,
    },
};
use rand::distributions::Uniform;
use rand::Rng;
use serde::Deserialize;

pub fn matrix_profile_output(_: &[Field]) -> PolarsResult<Field> {
    let dist = Field::new("squared_znorm", DataType::Float64);
    let idx = Field::new("index", DataType::UInt32);
    let v = vec![dist, idx];
    Ok(Field::new("profile", DataType::Struct(v)))
}

#[derive(Deserialize, Debug)]
pub(crate) struct MatrixProfileKwargs {
    pub(crate) window_size: usize,
    pub(crate) leaf_size: usize,
    pub(crate) sample: f64,
    pub(crate) parallel: bool,
}

// Comments on Performance
// Can be made much faster if kd-tree has add_unchecked, or other ways that bypass the point's finite checks.
// Right now kd-tree is very slow when window-size is large. I wonder why.

#[inline(always)]
fn equiv_dist(a: &[f64], b: &[f64]) -> f64 {
    let d = a.iter().zip(b.iter()).fold(0., |acc, (x, y)| acc + x * y);
    2.0 * (a.len() as f64 - d)
}

#[polars_expr(output_type_func=matrix_profile_output)]
fn pl_matrix_profile(
    inputs: &[Series],
    context: CallerContext,
    kwargs: MatrixProfileKwargs,
) -> PolarsResult<Series> {
    // Set up
    let ts = inputs[0].f64()?;
    let m = kwargs.window_size;
    let mf64 = m as f64;
    let leaf_size = kwargs.leaf_size;
    let sample = kwargs.sample;
    let parallel = kwargs.parallel;
    let can_parallel = parallel && !context.parallel();
    if ts.null_count() > 0 || ts.len() <= m || m < 4 || sample.abs() > 1.0 || sample <= 0. {
        return Err(PolarsError::ComputeError(
            "Matrix Profile: Input must not contain nulls and must have length > window size > 4. If approximate, the % must be in (0, 1]."
                .into(),
        ));
    }
    let ts = ts.rechunk();
    let ts = ts.cont_slice().unwrap();
    let output_size = ts.len() - m + 1;

    // Build the kd-tree
    let mut points: Vec<f64> = Vec::with_capacity(output_size * m);
    // // Creating Z-normalized Points
    // // Init for the rolling stats, mean, var
    let mut rolling = [0f64, 0f64];
    rolling[0] = ts[..m].iter().sum::<f64>() / mf64;
    rolling[1] = ts[..m]
        .iter()
        .fold(0., |acc, x| acc + (x - rolling[0]).powi(2))
        / mf64;
    let std = rolling[1].sqrt();
    points.extend(ts[..m].iter().map(|x| (x - rolling[0]) / std));
    let mut old: f64 = ts[0];
    // // Build the rolling stats
    for sl in ts[1..].windows(m) {
        let new = sl[m - 1];
        let old_mean = rolling[0].clone();
        rolling[0] += (new - old) / mf64;
        rolling[1] += (new - old) * (new - rolling[0] + old - old_mean) / mf64;
        old = sl[0];
        let std = rolling[1].sqrt();
        points.extend(sl.iter().map(|x| (x - rolling[0]) / std))
    }
    // // Inserting points into tree
    let mut tree: KdTree<f64, usize, &[f64]> = KdTree::with_capacity(m, leaf_size);
    if sample < 1. {
        // don't add all points to sample, only some
        let dist = Uniform::new(0., 1.);
        let mut rng = rand::thread_rng();
        for (i, p) in points.chunks_exact(m).enumerate() {
            if rng.sample(dist) < sample {
                tree.add(p, i)
                    .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
            }
        }
    } else {
        for (i, p) in points.chunks_exact(m).enumerate() {
            tree.add(p, i)
                .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
        }
    }

    // Query and build output
    // Exclusion zone default is: i +- ceil(m/4)
    let exclusion = (mf64 / 4.0).ceil().to_usize().unwrap();

    // Need at least exclusion * 2 + 2 points to ensure at least 1 pt to be outside exclusion. Pigeon hole.
    // We can be faster if we can make k smaller.
    let k = exclusion * 2 + 2;

    let (ca_dist, ca_id) = if can_parallel {
        POOL.install(|| {
            let n_threads = POOL.current_num_threads();
            let splits = split_offsets(output_size, n_threads);
            let chunks: (Vec<_>, Vec<_>) = splits
                .into_par_iter()
                .map(|(offset, len)| {
                    let start_idx = offset * m;
                    let end_idx = (offset + len) * m;
                    let piece = &points[start_idx..end_idx];
                    let mut dist_builder: PrimitiveChunkedBuilder<Float64Type> =
                        PrimitiveChunkedBuilder::new("", len);
                    let mut idx_builder: PrimitiveChunkedBuilder<UInt32Type> =
                        PrimitiveChunkedBuilder::new("", len);
                    for (i, pt) in piece.chunks_exact(m).enumerate() {
                        let mut dist: Option<f64> = None;
                        let mut id: Option<u32> = None;
                        if let Ok(v) = tree.nearest(&pt, k, &equiv_dist) {
                            for (d, j) in v {
                                if (i + offset).abs_diff(*j) > exclusion {
                                    dist = Some(d);
                                    id = Some(*j as u32);
                                    break;
                                } // <= means in exclusion zone. > means good
                            }
                        }
                        dist_builder.append_option(dist);
                        idx_builder.append_option(id);
                    }
                    let ca1 = dist_builder.finish();
                    let ca2 = idx_builder.finish();
                    (
                        ca1.downcast_iter().cloned().collect::<Vec<_>>(),
                        ca2.downcast_iter().cloned().collect::<Vec<_>>(),
                    )
                })
                .collect();

            (
                Float64Chunked::from_chunk_iter("squared_znorm", chunks.0.into_iter().flatten()),
                UInt32Chunked::from_chunk_iter("index", chunks.1.into_iter().flatten()),
            )
        })
    } else {
        let mut dist_builder: PrimitiveChunkedBuilder<Float64Type> =
            PrimitiveChunkedBuilder::new("squared_znorm", output_size);
        let mut idx_builder: PrimitiveChunkedBuilder<UInt32Type> =
            PrimitiveChunkedBuilder::new("index", output_size);
        for (i, pt) in points.chunks_exact(m).enumerate() {
            let mut dist: Option<f64> = None;
            let mut id: Option<u32> = None;
            if let Ok(v) = tree.nearest(&pt, k, &equiv_dist) {
                for (d, j) in v {
                    if i.abs_diff(*j) > exclusion {
                        dist = Some(d);
                        id = Some(*j as u32);
                        break;
                    } // <= means in exclusion zone. > means good
                }
            }
            dist_builder.append_option(dist);
            idx_builder.append_option(id);
        }
        (dist_builder.finish(), idx_builder.finish())
    };

    let s_dist = ca_dist.into_series();
    let s_id = ca_id.into_series();
    let out = StructChunked::new("profile", &[s_dist, s_id])?;
    Ok(out.into_series())
}
