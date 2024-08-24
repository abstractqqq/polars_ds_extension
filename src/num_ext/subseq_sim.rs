/// Subsequence similarity related queries

use polars::prelude::*;
use pyo3_polars::{
    derive::{polars_expr, CallerContext},
    export::polars_core::utils::rayon::{
        iter::{IndexedParallelIterator, ParallelIterator},
        slice::ParallelSlice,
    },
};
use cfavml;
use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub(crate) struct SubseqQueryKwargs {
    pub(crate) threshold:f64,
    pub(crate) parallel: bool,
}

#[polars_expr(output_type=UInt32)]
fn pl_subseq_sim_cnt_l2(inputs: &[Series], context: CallerContext, kwargs: SubseqQueryKwargs) -> PolarsResult<Series> {

    let seq = inputs[0].f64()?;
    let seq = seq.cont_slice().unwrap();
    let query = inputs[1].f64()?;
    let query = query.cont_slice().unwrap();
    let threshold = kwargs.threshold;
    let par = kwargs.parallel && !context.parallel();
    let window_size = query.len();

    let n = if par {
        let mut is_close = vec![false; seq.len() - window_size + 1];
        seq.par_windows(window_size)
            .map(|w| cfavml::squared_euclidean(query, w) < threshold)
            .collect_into_vec(&mut is_close);

        is_close.into_iter().fold(0u32, |acc, yes| acc + yes as u32)

    } else {
        if window_size < 16 {
            seq.windows(window_size).fold(0u32, |acc, w| {
                let d = w.into_iter().copied().zip(query.into_iter().copied())
                    .fold(0., |acc, (x, y)| acc + (x - y) * (x - y));
                acc + (d < threshold) as u32
            })
        } else {
            seq
                .windows(window_size)
                .fold(0u32, |acc, w| acc + (cfavml::squared_euclidean(query, w) < threshold) as u32)
        }
    };

    let output = UInt32Chunked::from_slice("", &[n]);
    Ok(output.into_series())
}