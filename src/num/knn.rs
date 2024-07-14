/// Performs KNN related search queries, classification and regression, and
/// other features/entropies that require KNN to be efficiently computed.
// use super::which_distance;
// use kdtree::KdTree;
// use crate::utils::get_common_float_dtype;
use crate::{
    arkadia::{
        matrix_to_empty_leaves, matrix_to_empty_leaves_w_norm, matrix_to_leaves,
        matrix_to_leaves_w_norm, AnyKDT, Leaf, LeafWithNorm, SplitMethod, DIST, KDT, KDTQ,
    },
    utils::{list_u32_output, series_to_ndarray, split_offsets},
};

use ndarray::{s, ArrayView2, Axis};
use num::Float;
use polars::prelude::*;
use pyo3_polars::{
    derive::{polars_expr, CallerContext},
    export::polars_core::{
        utils::rayon::prelude::{IntoParallelIterator, ParallelIterator},
        POOL,
    },
};
use serde::Deserialize;

pub fn knn_full_output(_: &[Field]) -> PolarsResult<Field> {
    let idx = Field::new("idx", DataType::List(Box::new(DataType::UInt32)));
    let dist = Field::new("dist", DataType::List(Box::new(DataType::Float64)));
    let v = vec![idx, dist];
    Ok(Field::new("knn_dist", DataType::Struct(v)))
}

#[derive(Deserialize, Debug)]
pub(crate) struct KDTKwargs {
    pub(crate) k: usize,
    pub(crate) leaf_size: usize,
    pub(crate) metric: String,
    pub(crate) parallel: bool,
    pub(crate) skip_eval: bool,
    pub(crate) skip_data: bool,
}

#[derive(Deserialize, Debug)]
pub(crate) struct KDTRadiusKwargs {
    pub(crate) r: f64,
    pub(crate) leaf_size: usize,
    pub(crate) metric: String,
    pub(crate) parallel: bool,
    pub(crate) sort: bool,
}

pub fn matrix_to_leaves_filtered<'a, T: Float + 'static, A: Copy>(
    matrix: &'a ArrayView2<'a, T>,
    values: &'a [A],
    filter: &BooleanChunked,
) -> Vec<Leaf<'a, T, A>> {
    filter
        .into_iter()
        .zip(values.iter().copied().zip(matrix.rows()))
        .filter(|(f, _)| f.unwrap_or(false))
        .map(|(_, pair)| pair.into())
        .collect::<Vec<_>>()
}

pub fn matrix_to_leaves_w_norm_filtered<'a, T: Float + 'static, A: Copy>(
    matrix: &'a ArrayView2<'a, T>,
    values: &'a [A],
    filter: &BooleanChunked,
) -> Vec<LeafWithNorm<'a, T, A>> {
    filter
        .into_iter()
        .zip(values.iter().copied().zip(matrix.rows()))
        .filter(|(f, _)| f.unwrap_or(false))
        .map(|(_, pair)| pair.into())
        .collect::<Vec<_>>()
}

// used in all cases but squared l2 (multiple queries)
pub fn dist_from_str<T: Float + 'static>(dist_str: &str) -> Result<DIST<T>, String> {
    match dist_str {
        "l1" => Ok(DIST::L1),
        "l2" => Ok(DIST::L2),
        "sql2" => Ok(DIST::SQL2),
        "linf" | "inf" => Ok(DIST::LINF),
        "cosine" => Ok(DIST::ANY(super::cosine_dist)),
        _ => Err("Unknown distance metric.".into()),
    }
}

pub fn knn_ptwise<'a, Kdt>(
    tree: Kdt,
    eval_mask: Vec<bool>,
    data: ArrayView2<'a, f64>,
    k: usize,
    can_parallel: bool,
    epsilon: f64,
) -> ListChunked
where
    Kdt: KDTQ<'a, f64, u32> + std::marker::Sync,
{
    let nrows = data.nrows();
    if can_parallel {
        let n_threads = POOL.current_num_threads();
        let splits = split_offsets(nrows, n_threads);
        let chunks_iter = splits.into_par_iter().map(|(offset, len)| {
            let mut builder =
                ListPrimitiveChunkedBuilder::<UInt32Type>::new("", len, k + 1, DataType::UInt32);

            let piece = data.slice(s![offset..offset + len, ..]);
            let mask = &eval_mask[offset..offset + len];
            for (b, p) in mask.iter().zip(piece.rows()) {
                if *b {
                    match tree.knn(k + 1, p.to_slice().unwrap(), epsilon) {
                        Some(nbs) => {
                            let v = nbs.into_iter().map(|nb| nb.to_item()).collect::<Vec<u32>>();
                            builder.append_slice(&v);
                        }
                        None => builder.append_null(),
                    }
                } else {
                    builder.append_null();
                };
            }
            let ca = builder.finish();
            ca.downcast_iter().cloned().collect::<Vec<_>>()
        });
        let chunks = POOL.install(|| chunks_iter.collect::<Vec<_>>());
        ListChunked::from_chunk_iter("knn", chunks.into_iter().flatten())
    } else {
        let mut builder = ListPrimitiveChunkedBuilder::<UInt32Type>::new(
            "knn",
            eval_mask.len(),
            k + 1,
            DataType::UInt32,
        );
        for (b, p) in eval_mask.into_iter().zip(data.rows()) {
            if b {
                match tree.knn(k + 1, p.to_slice().unwrap(), epsilon) {
                    Some(nbs) => {
                        let v = nbs.into_iter().map(|nb| nb.to_item()).collect::<Vec<u32>>();
                        builder.append_slice(&v);
                    }
                    None => builder.append_null(),
                }
            } else {
                builder.append_null();
            }
        }
        builder.finish()
    }
}

#[polars_expr(output_type_func=list_u32_output)]
fn pl_knn_ptwise(
    inputs: &[Series],
    context: CallerContext,
    kwargs: KDTKwargs,
) -> PolarsResult<Series> {
    // Set up params
    let k = kwargs.k;
    // let leaf_size = kwargs.leaf_size;
    let can_parallel = kwargs.parallel && !context.parallel();
    let skip_eval = kwargs.skip_eval;
    let skip_data = kwargs.skip_data;

    let mut inputs_offset = 0;
    let id = inputs[inputs_offset].u32().unwrap();
    let id = id.cont_slice()?;
    let nrows = id.len();

    let eval_mask = if skip_eval {
        inputs_offset += 1; // True means we need a new eval list
        let eval_mask = inputs[inputs_offset].bool().unwrap();
        eval_mask
            .iter()
            .map(|b| b.unwrap_or(false))
            .collect::<Vec<bool>>()
    } else {
        vec![true; nrows]
    };

    inputs_offset += skip_data as usize;
    let data = series_to_ndarray(&inputs[inputs_offset + 1..], IndexOrder::C)?;
    let binding = data.view();

    let ca = match dist_from_str::<f64>(kwargs.metric.as_str()) {
        Ok(d) => {
            if d == DIST::L2 {
                // This kdtree will be faster because norms are cached
                let mut leaves = if skip_data {
                    let data_mask = inputs[inputs_offset].bool().unwrap();
                    matrix_to_leaves_w_norm_filtered(&binding, id, data_mask)
                } else {
                    matrix_to_leaves_w_norm(&binding, id)
                };
                KDT::from_leaves(&mut leaves, SplitMethod::MIDPOINT)
                    .map(|tree| knn_ptwise(tree, eval_mask, binding, k, can_parallel, 0.))
            } else {
                let mut leaves = if skip_data {
                    let data_mask = inputs[inputs_offset].bool().unwrap();
                    matrix_to_leaves_filtered(&binding, id, data_mask)
                } else {
                    matrix_to_leaves(&binding, id)
                };
                AnyKDT::from_leaves(&mut leaves, SplitMethod::MIDPOINT, d)
                    .map(|tree| knn_ptwise(tree, eval_mask, binding, k, can_parallel, 0.))
            }
        }
        Err(e) => Err(e),
    }
    .map_err(|err| PolarsError::ComputeError(err.into()))?;

    Ok(ca.into_series())
}

pub fn knn_ptwise_w_dist<'a, Kdt>(
    tree: Kdt,
    eval_mask: Vec<bool>,
    data: ArrayView2<'a, f64>,
    k: usize,
    can_parallel: bool,
    epsilon: f64,
) -> (ListChunked, ListChunked)
where
    Kdt: KDTQ<'a, f64, u32> + std::marker::Sync,
{
    let nrows = data.nrows();
    if can_parallel {
        POOL.install(|| {
            let n_threads = POOL.current_num_threads();
            let splits = split_offsets(nrows, n_threads);
            let chunks: (Vec<_>, Vec<_>) = splits
                .into_par_iter()
                .map(|(offset, len)| {
                    let mut builder = ListPrimitiveChunkedBuilder::<UInt32Type>::new(
                        "",
                        len,
                        k + 1,
                        DataType::UInt32,
                    );
                    let mut dist_builder = ListPrimitiveChunkedBuilder::<Float64Type>::new(
                        "",
                        len,
                        k + 1,
                        DataType::Float64,
                    );
                    let piece = data.slice(s![offset..offset + len, ..]);
                    let mask = &eval_mask[offset..offset + len];
                    for (b, p) in mask.iter().zip(piece.rows()) {
                        if *b {
                            match tree.knn(k + 1, p.to_slice().unwrap(), epsilon) {
                                Some(nbs) => {
                                    let mut distances = Vec::with_capacity(nbs.len());
                                    let mut neighbors = Vec::with_capacity(nbs.len());
                                    for (d, id) in nbs.into_iter().map(|nb| nb.to_pair()) {
                                        distances.push(d);
                                        neighbors.push(id);
                                    }
                                    builder.append_slice(&neighbors);
                                    dist_builder.append_slice(&distances);
                                }
                                None => {
                                    builder.append_null();
                                    dist_builder.append_null();
                                }
                            }
                        } else {
                            builder.append_null();
                            dist_builder.append_null();
                        }
                    }
                    let ca_nb = builder.finish();
                    let ca_dist = dist_builder.finish();
                    (
                        ca_nb.downcast_iter().cloned().collect::<Vec<_>>(),
                        ca_dist.downcast_iter().cloned().collect::<Vec<_>>(),
                    )
                })
                .collect();

            let ca_nb = ListChunked::from_chunk_iter("idx", chunks.0.into_iter().flatten());
            let ca_dist = ListChunked::from_chunk_iter("dist", chunks.1.into_iter().flatten());
            (ca_nb, ca_dist)
        })
    } else {
        let mut builder =
            ListPrimitiveChunkedBuilder::<UInt32Type>::new("idx", nrows, k + 1, DataType::UInt32);
        let mut dist_builder = ListPrimitiveChunkedBuilder::<Float64Type>::new(
            "dist",
            nrows,
            k + 1,
            DataType::Float64,
        );
        for (b, p) in eval_mask.into_iter().zip(data.rows()) {
            if b {
                match tree.knn(k + 1, p.to_slice().unwrap(), epsilon) {
                    Some(nbs) => {
                        // let v = nbs.into_iter().map(|nb| nb.to_item()).collect::<Vec<u32>>();
                        // builder.append_slice(&v);
                        let mut distances = Vec::with_capacity(nbs.len());
                        let mut neighbors = Vec::with_capacity(nbs.len());
                        for (d, id) in nbs.into_iter().map(|nb| nb.to_pair()) {
                            distances.push(d);
                            neighbors.push(id);
                        }
                        builder.append_slice(&neighbors);
                        dist_builder.append_slice(&distances);
                    }
                    None => {
                        builder.append_null();
                        dist_builder.append_null();
                    }
                }
            } else {
                builder.append_null();
                dist_builder.append_null();
            }
        }
        (builder.finish(), dist_builder.finish())
    }
}

#[polars_expr(output_type_func=knn_full_output)]
fn pl_knn_ptwise_w_dist(
    inputs: &[Series],
    context: CallerContext,
    kwargs: KDTKwargs,
) -> PolarsResult<Series> {
    // Set up params
    let k = kwargs.k;
    // let leaf_size = kwargs.leaf_size;
    let can_parallel = kwargs.parallel && !context.parallel();
    let skip_eval = kwargs.skip_eval;
    let skip_data = kwargs.skip_data;

    let mut inputs_offset = 0;
    let id = inputs[inputs_offset].u32().unwrap();
    let id = id.cont_slice()?;
    let nrows = id.len();

    let eval_mask = if skip_eval {
        inputs_offset += 1; // True means we need a new eval list
        let eval_mask = inputs[inputs_offset].bool().unwrap();
        eval_mask
            .iter()
            .map(|b| b.unwrap_or(false))
            .collect::<Vec<bool>>()
    } else {
        vec![true; nrows]
    };

    inputs_offset += skip_data as usize;
    let data = series_to_ndarray(&inputs[inputs_offset + 1..], IndexOrder::C)?;
    let binding = data.view();

    let (ca_nb, ca_dist) = match dist_from_str::<f64>(kwargs.metric.as_str()) {
        Ok(d) => {
            if d == DIST::L2 {
                // This kdtree will be faster because norms are cached
                let mut leaves = if skip_data {
                    let data_mask = inputs[inputs_offset].bool().unwrap();
                    matrix_to_leaves_w_norm_filtered(&binding, id, data_mask)
                } else {
                    matrix_to_leaves_w_norm(&binding, id)
                };
                KDT::from_leaves(&mut leaves, SplitMethod::MIDPOINT)
                    .map(|tree| knn_ptwise_w_dist(tree, eval_mask, binding, k, can_parallel, 0.))
            } else {
                let mut leaves = if skip_data {
                    let data_mask = inputs[inputs_offset].bool().unwrap();
                    matrix_to_leaves_filtered(&binding, id, data_mask)
                } else {
                    matrix_to_leaves(&binding, id)
                };
                AnyKDT::from_leaves(&mut leaves, SplitMethod::MIDPOINT, d)
                    .map(|tree| knn_ptwise_w_dist(tree, eval_mask, binding, k, can_parallel, 0.))
            }
        }
        Err(e) => Err(e),
    }
    .map_err(|err| PolarsError::ComputeError(err.into()))?;

    let out = StructChunked::new("knn_dist", &[ca_nb.into_series(), ca_dist.into_series()])?;
    Ok(out.into_series())
}

pub fn query_radius_ptwise<'a, Kdt>(
    tree: Kdt,
    data: ArrayView2<'a, f64>,
    r: f64,
    can_parallel: bool,
    sort: bool,
) -> ListChunked
where
    Kdt: KDTQ<'a, f64, u32> + std::marker::Sync,
{
    if can_parallel {
        let nrows = data.nrows();
        let n_threads = POOL.current_num_threads();
        let splits = split_offsets(nrows, n_threads);
        let chunks_iter = splits.into_par_iter().map(|(offset, len)| {
            let mut builder =
                ListPrimitiveChunkedBuilder::<UInt32Type>::new("", len, 16, DataType::UInt32);
            let piece = data.slice(s![offset..offset + len, ..]);
            for p in piece.rows() {
                let sl = p.to_slice().unwrap();
                if let Some(v) = tree.within(sl, r, sort) {
                    let out: Vec<u32> = v.into_iter().map(|nb| nb.to_item()).collect();
                    builder.append_slice(&out);
                } else {
                    builder.append_null();
                }
            }
            let ca = builder.finish();
            ca.downcast_iter().cloned().collect::<Vec<_>>()
        });
        let chunks = POOL.install(|| chunks_iter.collect::<Vec<_>>());
        ListChunked::from_chunk_iter("", chunks.into_iter().flatten())
    } else {
        let mut builder =
            ListPrimitiveChunkedBuilder::<UInt32Type>::new("", data.len(), 16, DataType::UInt32);

        for p in data.rows() {
            let sl = p.to_slice().unwrap(); // C order makes sure rows are contiguous
            if let Some(v) = tree.within(sl, r, sort) {
                let out: Vec<u32> = v.into_iter().map(|nb| nb.to_item()).collect();
                builder.append_slice(&out);
            } else {
                builder.append_null();
            }
        }
        builder.finish()
    }
}

#[polars_expr(output_type_func=list_u32_output)]
fn pl_query_radius_ptwise(
    inputs: &[Series],
    context: CallerContext,
    kwargs: KDTRadiusKwargs,
) -> PolarsResult<Series> {
    // Set up params
    let can_parallel = kwargs.parallel && !context.parallel();
    let radius = kwargs.r;
    let sort = kwargs.sort;

    let id = inputs[0].u32()?;
    let id = id.cont_slice()?;

    let data = series_to_ndarray(&inputs[1..], IndexOrder::C)?;
    let binding = data.view();
    // Building output

    let ca = match dist_from_str::<f64>(kwargs.metric.as_str()) {
        Ok(d) => {
            if d == DIST::L2 {
                // This kdtree will be faster because norms are cached
                let mut leaves = matrix_to_leaves_w_norm(&binding, id);
                KDT::from_leaves(&mut leaves, SplitMethod::MIDPOINT)
                    .map(|tree| query_radius_ptwise(tree, binding, radius, can_parallel, sort))
            } else {
                let mut leaves = matrix_to_leaves(&binding, id);
                AnyKDT::from_leaves(&mut leaves, SplitMethod::MIDPOINT, d)
                    .map(|tree| query_radius_ptwise(tree, binding, radius, can_parallel, sort))
            }
        }
        Err(e) => Err(e),
    }
    .map_err(|err| PolarsError::ComputeError(err.into()))?;
    Ok(ca.into_series())
}

#[inline]
pub fn query_nb_cnt<'a, Kdt>(
    tree: Kdt,
    data: ArrayView2<'a, f64>,
    r: f64,
    can_parallel: bool,
) -> UInt32Chunked
where
    Kdt: KDTQ<'a, f64, ()> + std::marker::Sync,
{
    // as_slice.unwrap() is safe because when we create the matrices, we specified C order.
    let nrows = data.nrows();
    let ncols = data.ncols();
    if can_parallel {
        let splits = split_offsets(nrows, POOL.current_num_threads());
        let chunks_iter = splits.into_par_iter().map(|(offset, len)| {
            let mut builder: PrimitiveChunkedBuilder<UInt32Type> =
                PrimitiveChunkedBuilder::new("", nrows);

            let piece = data.slice(s![offset..offset + len, ..ncols]);
            for row in piece.rows() {
                builder.append_option(tree.within_count(row.to_slice().unwrap(), r));
            }
            let ca = builder.finish();
            ca.downcast_iter().cloned().collect::<Vec<_>>()
        });
        let chunks = POOL.install(|| chunks_iter.collect::<Vec<_>>());
        UInt32Chunked::from_chunk_iter("cnt", chunks.into_iter().flatten())
    } else {
        UInt32Chunked::from_iter_options(
            "cnt",
            data.rows()
                .into_iter()
                .map(|row| tree.within_count(row.to_slice().unwrap(), r)),
        )
    }
}

#[inline]
pub fn query_nb_cnt_w_radius<'a, Kdt>(
    tree: Kdt,
    data: ArrayView2<'a, f64>,
    radius: &Float64Chunked,
    can_parallel: bool,
) -> UInt32Chunked
where
    Kdt: KDTQ<'a, f64, ()> + std::marker::Sync,
{
    if can_parallel {
        let radius = radius.to_vec();
        let nrows = data.nrows();
        let n_threads = POOL.current_num_threads();
        let splits = split_offsets(nrows, n_threads);
        let chunks_iter = splits.into_par_iter().map(|(offset, len)| {
            let mut builder: PrimitiveChunkedBuilder<UInt32Type> =
                PrimitiveChunkedBuilder::new("", nrows);
            let piece = data.slice(s![offset..offset + len, ..]);
            let rad = &radius[offset..offset + len];
            for (row, r) in piece.rows().into_iter().zip(rad.iter()) {
                match r {
                    Some(r) => {
                        builder.append_option(tree.within_count(row.as_slice().unwrap(), *r))
                    }
                    None => builder.append_null(),
                }
            }
            let ca = builder.finish();
            ca.downcast_iter().cloned().collect::<Vec<_>>()
        });
        let chunks = POOL.install(|| chunks_iter.collect::<Vec<_>>());
        UInt32Chunked::from_chunk_iter("cnt", chunks.into_iter().flatten())
    } else {
        UInt32Chunked::from_iter_options(
            "cnt",
            data.rows()
                .into_iter()
                .zip(radius.into_iter())
                .map(|(row, r)| match r {
                    Some(r) => tree.within_count(row.as_slice().unwrap(), r),
                    None => None,
                }),
        )
    }
}

/// For every point in this dataframe, find the number of neighbors within radius r
/// The point itself is always considered as a neighbor to itself.
#[polars_expr(output_type=UInt32)]
fn pl_nb_cnt(inputs: &[Series], context: CallerContext, kwargs: KDTKwargs) -> PolarsResult<Series> {
    // Set up params
    // let leaf_size = kwargs.leaf_size;
    // Set up radius

    let radius = inputs[0].f64()?;
    let can_parallel = kwargs.parallel && !context.parallel();

    let data = series_to_ndarray(&inputs[1..], IndexOrder::C)?;
    let nrows = data.nrows();

    let binding = data.view();
    if radius.len() == 1 {
        let r = radius.get(0).unwrap();
        let ca = match dist_from_str::<f64>(kwargs.metric.as_str()) {
            Ok(d) => {
                if d == DIST::L2 {
                    // This kdtree will be faster because norms are cached
                    let mut leaves = matrix_to_empty_leaves_w_norm(&binding);
                    KDT::from_leaves(&mut leaves, SplitMethod::MIDPOINT)
                        .map(|tree| query_nb_cnt(tree, data.view(), r, can_parallel))
                } else {
                    let mut leaves = matrix_to_empty_leaves(&binding);
                    AnyKDT::from_leaves(&mut leaves, SplitMethod::MIDPOINT, d)
                        .map(|tree| query_nb_cnt(tree, data.view(), r, can_parallel))
                }
            }
            Err(e) => Err(e),
        }
        .map_err(|err| PolarsError::ComputeError(err.into()))?;
        Ok(ca.with_name("cnt").into_series())
    } else if radius.len() == nrows {
        let ca = match dist_from_str::<f64>(kwargs.metric.as_str()) {
            Ok(d) => {
                if d == DIST::L2 {
                    // This kdtree will be faster because norms are cached
                    let mut leaves = matrix_to_empty_leaves_w_norm(&binding);
                    KDT::from_leaves(&mut leaves, SplitMethod::MIDPOINT)
                        .map(|tree| query_nb_cnt_w_radius(tree, data.view(), radius, can_parallel))
                } else {
                    let mut leaves = matrix_to_empty_leaves(&binding);
                    AnyKDT::from_leaves(&mut leaves, SplitMethod::MIDPOINT, d)
                        .map(|tree| query_nb_cnt_w_radius(tree, data.view(), radius, can_parallel))
                }
            }
            Err(e) => Err(e),
        }
        .map_err(|err| PolarsError::ComputeError(err.into()))?;
        Ok(ca.with_name("cnt").into_series())
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length or one of them must be a scalar.".into(),
        ))
    }
}
