use cfavml;
/// Subsequence similarity related queries
use polars::prelude::*;
use pyo3_polars::{
    derive::{polars_expr, CallerContext},
    export::polars_core::{
        utils::rayon::{
            iter::{IntoParallelIterator, ParallelIterator},
            slice::ParallelSlice,
        },
        POOL,
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
    let seq = inputs[0].f64()?;
    let seq = seq.cont_slice().unwrap();
    let query = inputs[1].f64()?;
    let query = query.cont_slice().unwrap();

    if query.len() > seq.len() {
        return Err(PolarsError::ComputeError(
            "Not enough data points for the query.".into(),
        ));
    }

    let threshold = kwargs.threshold;
    let par = kwargs.parallel && !context.parallel();
    let window_size = query.len();

    let n = if par {
        seq.par_windows(window_size)
            .map(|w| (cfavml::squared_euclidean(query, w) < threshold) as u32)
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
                acc + (cfavml::squared_euclidean(query, w) < threshold) as u32
            })
        }
    };

    let output = UInt32Chunked::from_slice("", &[n]);
    Ok(output.into_series())
}

#[polars_expr(output_type=UInt32)]
fn pl_subseq_sim_cnt_zl2(
    inputs: &[Series],
    context: CallerContext,
    kwargs: SubseqQueryKwargs,
) -> PolarsResult<Series> {
    let seq = inputs[0].f64()?;
    let seq = seq.cont_slice().unwrap();
    let query = inputs[1].f64()?; // is already z normalized
    let query = query.cont_slice().unwrap();

    let rolling_mean = inputs[2].f64()?;
    let rolling_mean = rolling_mean.cont_slice()?;
    let rolling_var = inputs[3].f64()?;
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
                for (i, w) in windows[offset..offset + len].iter().enumerate() {
                    let actual_i = i + offset;
                    let normalized = w
                        .iter()
                        .map(|x| (x - rolling_mean[actual_i]) / rolling_var[actual_i].sqrt())
                        .collect::<Vec<_>>();
                    acc += (cfavml::squared_euclidean(query, &normalized) < threshold) as u32;
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
                acc + (cfavml::squared_euclidean(query, &normalized) < threshold) as u32
            })
    };

    let output = UInt32Chunked::from_slice("", &[n]);
    Ok(output.into_series())
}
