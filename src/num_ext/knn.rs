/// Performs KNN related search queries, classification and regression, and
/// other features/entropies that require KNN to be efficiently computed.
use crate::{
    arkadia::{
        utils::{slice_to_empty_leaves, slice_to_leaves},
        KNNMethod, KNNRegressor, Leaf, SpatialQueries, KDT,
    },
    utils::{list_u32_output, series_to_row_major_slice, split_offsets, DIST},
};

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
    let idx = Field::new("idx".into(), DataType::List(Box::new(DataType::UInt32)));
    let dist = Field::new("dist".into(), DataType::List(Box::new(DataType::Float64)));
    let v = vec![idx, dist];
    Ok(Field::new("knn_dist".into(), DataType::Struct(v)))
}

#[derive(Deserialize)]
pub(crate) struct KNNAvgKwargs {
    // pub(crate) leaf_size: usize,
    pub(crate) k: usize,
    pub(crate) metric: String,
    #[serde(default)]
    pub(crate) weighted: bool,
    #[serde(default)]
    pub(crate) parallel: bool,
    pub(crate) min_bound: f64,
    #[serde(default = "_max_bound")]
    pub(crate) max_bound: f64,
}

#[derive(Deserialize)]
pub(crate) struct KDTKwargs {
    // pub(crate) leaf_size: usize,
    pub(crate) k: usize,
    pub(crate) metric: String,
    #[serde(default)]
    pub(crate) parallel: bool,
    #[serde(default)]
    pub(crate) skip_eval: bool,
    #[serde(default = "_max_bound")]
    pub(crate) max_bound: f64,
    #[serde(default)]
    pub(crate) epsilon: f64,
}

fn _max_bound() -> f64 {
    f64::max_value()
}

#[derive(Deserialize, Debug)]
pub(crate) struct KDTRadiusKwargs {
    // pub(crate) leaf_size: usize,
    pub(crate) r: f64,
    pub(crate) metric: String,
    pub(crate) parallel: bool,
    pub(crate) sort: bool,
}

pub fn row_major_slice_to_leaves_filtered<'a, T: Float + 'static, A: Copy>(
    slice: &'a [T],
    row_len: usize,
    values: &'a [A],
    filter: &BooleanChunked,
) -> Vec<Leaf<'a, T, A>> {
    filter
        .into_iter()
        .zip(values.iter().copied().zip(slice.chunks_exact(row_len)))
        .filter(|(f, _)| f.unwrap_or(false))
        .map(|(_, pair)| pair.into())
        .collect::<Vec<_>>()
}

/// KNN Regression
/// Always do k + 1 because this operation is in-dataframe, and this means
/// that the point itself is always a neighbor to itself.
#[polars_expr(output_type=Float64)]
fn pl_knn_avg(
    inputs: &[Series],
    context: CallerContext,
    kwargs: KNNAvgKwargs,
) -> PolarsResult<Series> {
    // Set up params

    // let leaf_size = kwargs.leaf_size;
    let k = kwargs.k;
    let can_parallel = kwargs.parallel && !context.parallel();
    let max_bound = kwargs.max_bound;
    let min_bound = kwargs.min_bound;
    let method = KNNMethod::new(kwargs.weighted, min_bound);

    let id = inputs[0].f64().unwrap();
    let id = id.cont_slice()?;
    let null_mask = inputs[1].bool().unwrap();
    let nrows = null_mask.len();

    // let data = series_to_ndarray(&inputs[2..], IndexOrder::C)?;

    let ncols = inputs[2..].len();
    let data = series_to_row_major_slice::<Float64Type>(&inputs[2..])?;
    let mut leaves = row_major_slice_to_leaves_filtered(&data, ncols, id, &null_mask);

    let tree = match DIST::<f64>::new_from_str_informed(kwargs.metric, ncols) {
        Ok(d) => Ok(KDT::from_leaves_unchecked(&mut leaves, d)),
        Err(e) => Err(e),
    }
    .map_err(|err| PolarsError::ComputeError(err.into()))?;

    let ca = if can_parallel {
        let n_threads = POOL.current_num_threads();
        let splits = split_offsets(nrows, n_threads);
        let chunks_iter = splits.into_par_iter().map(|(offset, len)| {
            let subslice = &data[offset * ncols..(offset + len) * ncols];
            let out = Float64Chunked::from_iter_options(
                "".into(),
                subslice
                    .chunks_exact(ncols)
                    .map(|row| tree.knn_regress(k + 1, row, min_bound, max_bound, method)),
            );
            out.downcast_iter().cloned().collect::<Vec<_>>()
        });
        let chunks = POOL.install(|| chunks_iter.collect::<Vec<_>>());
        Float64Chunked::from_chunk_iter("".into(), chunks.into_iter().flatten())
    } else {
        Float64Chunked::from_iter_options(
            "".into(),
            data.chunks_exact(ncols)
                .map(|row| tree.knn_regress(k + 1, row, min_bound, max_bound, method)),
        )
    };

    Ok(ca.into_series())
}

/// KNN Point-wise
/// Always do k + 1 because this operation is in-dataframe, and this means
/// that the point itself is always a neighbor to itself.
/// Eval mask determines which values will be evaluated. Some can be skipped (Null will be returned) if user desires
pub fn knn_ptwise<'a, Kdt>(
    tree: Kdt,
    eval_mask: Vec<bool>,
    data: &'a [f64],
    k: usize,
    can_parallel: bool,
    max_bound: f64,
    epsilon: f64,
) -> ListChunked
where
    Kdt: SpatialQueries<'a, f64, u32> + std::marker::Sync,
{
    let ncols = tree.dim();
    let nrows = data.len() / ncols;
    if can_parallel {
        let n_threads = POOL.current_num_threads();
        let splits = split_offsets(nrows, n_threads);
        let chunks_iter = splits.into_par_iter().map(|(offset, len)| {
            let mut builder = ListPrimitiveChunkedBuilder::<UInt32Type>::new(
                "".into(),
                len,
                k + 1,
                DataType::UInt32,
            );

            let subslice = &data[offset * ncols..(offset + len) * ncols];
            let mask = &eval_mask[offset..offset + len];
            for (b, p) in mask.iter().zip(subslice.chunks_exact(ncols)) {
                if *b {
                    match tree.knn_bounded(k + 1, p, max_bound, epsilon) {
                        Some(nbs) => {
                            builder.append_values_iter(nbs.into_iter().map(|nb| nb.to_item()));
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
        ListChunked::from_chunk_iter("knn".into(), chunks.into_iter().flatten())
    } else {
        let mut builder = ListPrimitiveChunkedBuilder::<UInt32Type>::new(
            "knn".into(),
            eval_mask.len(),
            k + 1,
            DataType::UInt32,
        );
        for (b, p) in eval_mask.into_iter().zip(data.chunks_exact(ncols)) {
            if b {
                match tree.knn_bounded(k + 1, p, max_bound, epsilon) {
                    Some(nbs) => builder.append_values_iter(nbs.into_iter().map(|nb| nb.to_item())),
                    None => builder.append_null(),
                }
            } else {
                builder.append_null();
            }
        }
        builder.finish()
    }
}

#[polars_expr(output_type=Float64)]
fn pl_dist_from_kth_nb(
    inputs: &[Series],
    context: CallerContext,
    kwargs: KDTKwargs,
) -> PolarsResult<Series> {
    // Set up params
    let k = kwargs.k;
    let can_parallel = kwargs.parallel && !context.parallel();
    let ncols = inputs.len();
    let data = series_to_row_major_slice::<Float64Type>(inputs)?;
    match DIST::<f64>::new_from_str_informed(kwargs.metric, ncols) {
        Ok(d) => {
            let mut leaves: Vec<Leaf<f64, ()>> =
                data.chunks_exact(ncols).map(|sl| ((), sl).into()).collect();
            // let mut leaves = row_major_slice_to_leaves(&data, ncols, id, null_mask);
            let tree = KDT::from_leaves_unchecked(&mut leaves, d);
            let nrows = data.len() / ncols;
            let ca = if can_parallel {
                let n_threads = POOL.current_num_threads();
                let splits = split_offsets(nrows, n_threads);
                let chunks_iter = splits.into_par_iter().map(|(i, n)| {
                    let start = i * ncols;
                    let end = (i + n) * ncols;
                    let slice = &data[start..end];
                    let mut builder: PrimitiveChunkedBuilder<Float64Type> =
                        PrimitiveChunkedBuilder::new("".into(), n);

                    for point in slice.chunks_exact(ncols) {
                        match tree.knn_bounded(k + 1, point, kwargs.max_bound, kwargs.epsilon) {
                            Some(mut nbs) => {
                                builder.append_option(nbs.pop().map(|nb| nb.to_dist()));
                            }
                            _ => builder.append_null(),
                        }
                    }
                    let ca = builder.finish();
                    ca.downcast_iter().cloned().collect::<Vec<_>>()
                });
                let chunks = POOL.install(|| chunks_iter.collect::<Vec<_>>());
                Float64Chunked::from_chunk_iter("".into(), chunks.into_iter().flatten())
            } else {
                let mut builder: PrimitiveChunkedBuilder<Float64Type> =
                    PrimitiveChunkedBuilder::new("".into(), nrows);
                for point in data.chunks_exact(ncols) {
                    match tree.knn_bounded(k + 1, point, kwargs.max_bound, kwargs.epsilon) {
                        Some(mut nbs) => {
                            builder.append_option(nbs.pop().map(|nb| nb.to_dist()));
                        }
                        _ => builder.append_null(),
                    }
                }
                builder.finish()
            };
            Ok(ca.into_series())
        }
        Err(e) => Err(PolarsError::ComputeError(e.into())),
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

    let id = inputs[0].u32().unwrap();
    let id = id.cont_slice()?;
    let nrows = id.len();

    // True means keep
    let keep_mask = inputs[1].bool().unwrap();

    let mut inputs_offset = 2;
    let eval_mask = if skip_eval {
        let eval_mask = inputs[2].bool().unwrap();
        inputs_offset = 3;
        eval_mask
            .iter()
            .map(|b| b.unwrap_or(false))
            .collect::<Vec<bool>>()
    } else {
        vec![true; nrows]
    };

    let ncols = inputs[inputs_offset..].len();
    let data = series_to_row_major_slice::<Float64Type>(&inputs[inputs_offset..])?;

    match DIST::<f64>::new_from_str_informed(kwargs.metric, ncols) {
        Ok(d) => {
            let mut leaves = row_major_slice_to_leaves_filtered(&data, ncols, id, keep_mask);
            let tree = KDT::from_leaves_unchecked(&mut leaves, d);
            Ok(knn_ptwise(
                tree,
                eval_mask,
                &data,
                k,
                can_parallel,
                kwargs.max_bound,
                kwargs.epsilon,
            )
            .into_series())
        }
        Err(e) => Err(PolarsError::ComputeError(e.into())),
    }
}

pub fn knn_ptwise_w_dist<'a, Kdt>(
    tree: Kdt,
    eval_mask: Vec<bool>,
    data: &'a [f64],
    k: usize,
    can_parallel: bool,
    max_bound: f64,
    epsilon: f64,
) -> (ListChunked, ListChunked)
where
    Kdt: SpatialQueries<'a, f64, u32> + std::marker::Sync,
{
    let ncols = tree.dim();
    let nrows = data.len() / ncols;
    if can_parallel {
        POOL.install(|| {
            let n_threads = POOL.current_num_threads();
            let splits = split_offsets(nrows, n_threads);
            let chunks: (Vec<_>, Vec<_>) = splits
                .into_par_iter()
                .map(|(offset, len)| {
                    let mut builder = ListPrimitiveChunkedBuilder::<UInt32Type>::new(
                        "".into(),
                        len,
                        k + 1,
                        DataType::UInt32,
                    );
                    let mut dist_builder = ListPrimitiveChunkedBuilder::<Float64Type>::new(
                        "".into(),
                        len,
                        k + 1,
                        DataType::Float64,
                    );
                    let subslice = &data[offset * ncols..(offset + len) * ncols];
                    let mask = &eval_mask[offset..offset + len];
                    for (b, p) in mask.iter().zip(subslice.chunks_exact(ncols)) {
                        if *b {
                            match tree.knn_bounded(k + 1, p, max_bound, epsilon) {
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

            let ca_nb = ListChunked::from_chunk_iter("idx".into(), chunks.0.into_iter().flatten());
            let ca_dist =
                ListChunked::from_chunk_iter("dist".into(), chunks.1.into_iter().flatten());
            (ca_nb, ca_dist)
        })
    } else {
        let mut builder = ListPrimitiveChunkedBuilder::<UInt32Type>::new(
            "idx".into(),
            nrows,
            k + 1,
            DataType::UInt32,
        );
        let mut dist_builder = ListPrimitiveChunkedBuilder::<Float64Type>::new(
            "dist".into(),
            nrows,
            k + 1,
            DataType::Float64,
        );
        for (b, p) in eval_mask.into_iter().zip(data.chunks_exact(ncols)) {
            if b {
                match tree.knn_bounded(k + 1, p, max_bound, epsilon) {
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

    let id = inputs[0].u32().unwrap();
    let id = id.cont_slice()?;
    let nrows = id.len();

    let null_mask = inputs[1].bool().unwrap();

    let mut inputs_offset = 2;
    let eval_mask = if skip_eval {
        let eval_mask = inputs[2].bool().unwrap();
        inputs_offset = 3;
        eval_mask
            .iter()
            .map(|b| b.unwrap_or(false))
            .collect::<Vec<bool>>()
    } else {
        vec![true; nrows]
    };

    let ncols = inputs[inputs_offset..].len();
    let data = series_to_row_major_slice::<Float64Type>(&inputs[inputs_offset..])?;
    let (ca_nb, ca_dist) = match DIST::<f64>::new_from_str_informed(kwargs.metric, ncols) {
        Ok(d) => {
            let mut leaves = row_major_slice_to_leaves_filtered(&data, ncols, id, null_mask);
            let tree = KDT::from_leaves_unchecked(&mut leaves, d);
            Ok(knn_ptwise_w_dist(
                tree,
                eval_mask,
                &data,
                k,
                can_parallel,
                kwargs.max_bound,
                kwargs.epsilon,
            ))
        }
        Err(e) => Err(PolarsError::ComputeError(e.into())),
    }?;

    let out = StructChunked::from_series(
        "knn_dist".into(),
        ca_nb.len(),
        [&ca_nb.into_series(), &ca_dist.into_series()].into_iter(),
    )?;
    Ok(out.into_series())
}

pub fn query_radius_ptwise<'a, Kdt>(
    tree: Kdt,
    data: &'a [f64],
    r: f64,
    can_parallel: bool,
    sort: bool,
) -> ListChunked
where
    Kdt: SpatialQueries<'a, f64, u32> + std::marker::Sync,
{
    let ncols = tree.dim();
    let nrows = data.len() / ncols;
    if can_parallel {
        let n_threads = POOL.current_num_threads();
        let splits = split_offsets(nrows, n_threads);
        let chunks_iter = splits.into_par_iter().map(|(offset, len)| {
            let mut builder = ListPrimitiveChunkedBuilder::<UInt32Type>::new(
                "".into(),
                len,
                16,
                DataType::UInt32,
            );

            let subslice = &data[offset * ncols..(offset + len) * ncols];
            for p in subslice.chunks_exact(ncols) {
                if let Some(v) = tree.within(p, r, sort) {
                    builder.append_values_iter(v.into_iter().map(|nb| nb.to_item()));
                } else {
                    builder.append_null();
                }
            }
            let ca = builder.finish();
            ca.downcast_iter().cloned().collect::<Vec<_>>()
        });
        let chunks = POOL.install(|| chunks_iter.collect::<Vec<_>>());
        ListChunked::from_chunk_iter("".into(), chunks.into_iter().flatten())
    } else {
        let mut builder = ListPrimitiveChunkedBuilder::<UInt32Type>::new(
            "".into(),
            data.len(),
            16,
            DataType::UInt32,
        );

        for p in data.chunks_exact(ncols) {
            // C order (row major) makes sure rows are contiguous
            if let Some(v) = tree.within(p, r, sort) {
                builder.append_values_iter(v.into_iter().map(|nb| nb.to_item()));
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

    let ncols = inputs[1..].len();
    let data = series_to_row_major_slice::<Float64Type>(&inputs[1..])?;
    // Building output
    match DIST::<f64>::new_from_str_informed(kwargs.metric, ncols) {
        Ok(d) => {
            let mut leaves = slice_to_leaves(&data, ncols, id);
            let tree = KDT::from_leaves_unchecked(&mut leaves, d);
            Ok(query_radius_ptwise(tree, &data, radius, can_parallel, sort).into_series())
        }
        Err(e) => Err(PolarsError::ComputeError(e.into())),
    }
}

#[inline]
pub fn query_nb_cnt<'a, Kdt>(
    tree: Kdt,
    data: &'a [f64],
    r: f64,
    can_parallel: bool,
) -> UInt32Chunked
where
    Kdt: SpatialQueries<'a, f64, ()> + std::marker::Sync,
{
    // as_slice.unwrap() is safe because when we create the matrices, we specified C order.
    let ncols = tree.dim();
    let nrows = data.len() / ncols;
    if can_parallel {
        let splits = split_offsets(nrows, POOL.current_num_threads());
        let chunks_iter = splits.into_par_iter().map(|(offset, len)| {
            let mut builder: PrimitiveChunkedBuilder<UInt32Type> =
                PrimitiveChunkedBuilder::new("".into(), nrows);

            let subslice = &data[offset * ncols..(offset + len) * ncols];
            for row in subslice.chunks_exact(ncols) {
                builder.append_option(tree.within_count(row, r));
            }
            let ca = builder.finish();
            ca.downcast_iter().cloned().collect::<Vec<_>>()
        });
        let chunks = POOL.install(|| chunks_iter.collect::<Vec<_>>());
        UInt32Chunked::from_chunk_iter("cnt".into(), chunks.into_iter().flatten())
    } else {
        UInt32Chunked::from_iter_options(
            "cnt".into(),
            data.chunks_exact(ncols)
                .map(|row| tree.within_count(row, r)),
        )
    }
}

#[inline]
pub fn query_nb_cnt_w_radius<'a, Kdt>(
    tree: Kdt,
    data: &'a [f64],
    radius: &Float64Chunked,
    can_parallel: bool,
) -> UInt32Chunked
where
    Kdt: SpatialQueries<'a, f64, ()> + std::marker::Sync,
{
    let ncols = tree.dim();
    let nrows = data.len() / ncols;
    if can_parallel {
        let radius = radius.to_vec();
        let n_threads = POOL.current_num_threads();
        let splits = split_offsets(nrows, n_threads);
        let chunks_iter = splits.into_par_iter().map(|(offset, len)| {
            let mut builder: PrimitiveChunkedBuilder<UInt32Type> =
                PrimitiveChunkedBuilder::new("".into(), nrows);
            let subslice = &data[offset * ncols..(offset + len) * ncols];
            let rad = &radius[offset..offset + len];
            for (row, r) in subslice.chunks_exact(ncols).zip(rad.iter()) {
                match r {
                    Some(r) => builder.append_option(tree.within_count(row, *r)),
                    None => builder.append_null(),
                }
            }
            let ca = builder.finish();
            ca.downcast_iter().cloned().collect::<Vec<_>>()
        });
        let chunks = POOL.install(|| chunks_iter.collect::<Vec<_>>());
        UInt32Chunked::from_chunk_iter("cnt".into(), chunks.into_iter().flatten())
    } else {
        UInt32Chunked::from_iter_options(
            "cnt".into(),
            data.chunks_exact(ncols)
                .zip(radius.into_iter())
                .map(|(row, r)| match r {
                    Some(r) => tree.within_count(row, r),
                    None => None,
                }),
        )
    }
}

/// For every point in this dataframe, find the number of neighbors within radius r
/// The point itself is always considered as a neighbor to itself.
#[polars_expr(output_type=UInt32)]
fn pl_nb_cnt(inputs: &[Series], context: CallerContext, kwargs: KDTKwargs) -> PolarsResult<Series> {
    let radius = inputs[0].f64()?;
    let can_parallel = kwargs.parallel && !context.parallel();

    let ncols = inputs[1..].len();
    let data = series_to_row_major_slice::<Float64Type>(&inputs[1..])?;
    let nrows = data.len() / ncols;

    if radius.len() == 1 {
        let r = radius.get(0).unwrap();
        match DIST::<f64>::new_from_str_informed(kwargs.metric, ncols) {
            Ok(d) => {
                let mut leaves = slice_to_empty_leaves(&data, ncols);
                let tree = KDT::from_leaves_unchecked(&mut leaves, d);
                Ok(query_nb_cnt(tree, &data, r, can_parallel)
                    .with_name("cnt".into())
                    .into_series())
            }
            Err(e) => Err(PolarsError::ComputeError(e.into())),
        }
    } else if radius.len() == nrows {
        match DIST::<f64>::new_from_str_informed(kwargs.metric, ncols) {
            Ok(d) => {
                let mut leaves = slice_to_empty_leaves(&data, ncols);
                let tree = KDT::from_leaves_unchecked(&mut leaves, d);
                Ok(query_nb_cnt_w_radius(tree, &data, radius, can_parallel)
                    .with_name("cnt".into())
                    .into_series())
            }
            Err(e) => Err(PolarsError::ComputeError(e.into())),
        }
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length or one of them must be a scalar.".into(),
        ))
    }
}
