use crate::utils::squared_l2_distance;
/// Subsequence similarity related queries
use polars::prelude::*;
use pyo3_polars::{
    derive::{polars_expr, CallerContext},
    export::polars_core::{
        runtime::RAYON as POOL,
        utils::rayon::{
            iter::{IntoParallelIterator, ParallelIterator},
            slice::ParallelSlice,
        },
    },
};
use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub(crate) struct SubseqQueryKwargs {
    pub(crate) threshold: f64,
    pub(crate) parallel: bool,
}

#[polars_expr(output_type=UInt32)]
fn pl_subseq_sim_cnt_l2(
    inputs: &[Series],
    context: CallerContext,
    kwargs: SubseqQueryKwargs,
) -> PolarsResult<Series> {
    let binding_seq = inputs[0].rechunk();
    let seq = binding_seq.f64()?;
    let seq = seq.cont_slice().unwrap();
    let binding_query = inputs[1].rechunk();
    let query = binding_query.f64()?;
    let query = query.cont_slice().unwrap();

    if query.len() > seq.len() {
        return Err(PolarsError::ComputeError(
            "query length cannot be greater than sequence length".into(),
        ));
    }

    let threshold = kwargs.threshold;
    let par = kwargs.parallel && !context.parallel();
    let window_size = query.len();

    let n = if par {
        seq.par_windows(window_size)
            .map(|w| (squared_l2_distance(query, w) < threshold) as u32)
            .sum()
    } else {
        if window_size < 16 {
            seq.windows(window_size).fold(0u32, |acc, w| {
                let d = w
                    .into_iter()
                    .copied()
                    .zip(query.into_iter().copied())
                    .fold(0., |acc, (x, y)| acc + (x - y) * (x - y));
                acc + (d < threshold) as u32
            })
        } else {
            seq.windows(window_size).fold(0u32, |acc, w| {
                acc + (squared_l2_distance(query, w) < threshold) as u32
            })
        }
    };

    let output = UInt32Chunked::from_slice("".into(), &[n]);
    Ok(output.into_series())
}

#[polars_expr(output_type=UInt32)]
fn pl_subseq_sim_cnt_zl2(
    inputs: &[Series],
    context: CallerContext,
    kwargs: SubseqQueryKwargs,
) -> PolarsResult<Series> {
    let binding_seq = inputs[0].rechunk();
    let seq = binding_seq.f64()?;
    let seq = seq.cont_slice().unwrap();
    let binding_query = inputs[1].rechunk();
    let query = binding_query.f64()?; // is already z normalized
    let query = query.cont_slice().unwrap();

    let binding_mean = inputs[2].rechunk();
    let rolling_mean = binding_mean.f64()?;
    let rolling_mean = rolling_mean.cont_slice()?;
    let binding_var = inputs[3].rechunk();
    let rolling_var = binding_var.f64()?;
    let rolling_var = rolling_var.cont_slice()?;

    let threshold = kwargs.threshold;
    let par = kwargs.parallel && !context.parallel();
    let window_size = query.len();

    let total_windows = seq.len() + 1 - window_size;

    let n = if par {
        let n_threads = POOL.current_num_threads();
        let windows = seq.windows(window_size).collect::<Vec<_>>();
        let splits = crate::utils::split_offsets(total_windows, n_threads);
        splits
            .into_par_iter()
            .map(|(offset, len)| {
                let mut acc: u32 = 0;
                for (i, &w) in windows[offset..offset + len].iter().enumerate() {
                    let actual_i = i + offset;
                    let normalized = w
                        .iter()
                        .map(|x| (x - rolling_mean[actual_i]) / rolling_var[actual_i].sqrt())
                        .collect::<Vec<_>>();
                    acc += (squared_l2_distance(query, &normalized) < threshold) as u32;
                }
                acc
            })
            .sum()
    } else {
        seq.windows(window_size)
            .enumerate()
            .fold(0u32, |acc, (i, w)| {
                let normalized = w
                    .iter()
                    .map(|x| (x - rolling_mean[i]) / rolling_var[i].sqrt())
                    .collect::<Vec<_>>();
                acc + (squared_l2_distance(query, &normalized) < threshold) as u32
            })
    };

    let output = UInt32Chunked::from_slice("".into(), &[n]);
    Ok(output.into_series())
}
