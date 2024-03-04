use super::str_set_sim_helper;
use crate::utils::split_offsets;
use polars::prelude::{arity::binary_elementwise_values, *};
use pyo3_polars::{
    derive::{polars_expr, CallerContext},
    export::polars_core::{
        utils::rayon::prelude::{IntoParallelIterator, ParallelIterator},
        POOL,
    },
};

// Todo.
// Can optimize the case when ca1 is scalar. No need to regenerate the hashset in that case.

#[inline(always)]
fn tversky_sim(w1: &str, w2: &str, ngram: usize, alpha: f64, beta: f64) -> f64 {
    let (s1, s2, intersect) = str_set_sim_helper(w1, w2, ngram);
    let s1ms2 = s1.abs_diff(intersect) as f64;
    let s2ms1 = s2.abs_diff(intersect) as f64;
    (intersect as f64) / (intersect as f64 + alpha * s1ms2 + beta * s2ms1)
}

#[polars_expr(output_type=Float64)]
fn pl_tversky_sim(inputs: &[Series], context: CallerContext) -> PolarsResult<Series> {
    let ca1 = inputs[0].str()?;
    let ca2 = inputs[1].str()?;

    // ngram size
    let ngram = inputs[2].u32()?;
    let ngram = ngram.get(0).unwrap() as usize;

    // Alpha and beta params
    let alpha = inputs[3].f64()?;
    let alpha = alpha.get(0).unwrap();
    let beta = inputs[4].f64()?;
    let beta = beta.get(0).unwrap();
    // parallel
    let parallel = inputs[5].bool()?;
    let parallel = parallel.get(0).unwrap();
    let can_parallel = parallel && !context.parallel();

    if ca2.len() == 1 {
        let r = ca2.get(0).unwrap();
        let out: Float64Chunked = if can_parallel {
            let n_threads = POOL.current_num_threads();
            let splits = split_offsets(ca1.len(), n_threads);
            let chunks_iter = splits.into_par_iter().map(|(offset, len)| {
                let s1 = ca1.slice(offset as i64, len);
                let out: Float64Chunked = s1.apply_nonnull_values_generic(DataType::Float64, |s| {
                    tversky_sim(s, r, ngram, alpha, beta)
                });
                out.downcast_iter().cloned().collect::<Vec<_>>()
            });

            let chunks = POOL.install(|| chunks_iter.collect::<Vec<_>>());
            Float64Chunked::from_chunk_iter(ca1.name(), chunks.into_iter().flatten())
        } else {
            ca1.apply_nonnull_values_generic(DataType::Float64, |s| {
                tversky_sim(s, r, ngram, alpha, beta)
            })
        };
        Ok(out.into_series())
    } else if ca1.len() == ca2.len() {
        let out: Float64Chunked = if can_parallel {
            let n_threads = POOL.current_num_threads();
            let splits = split_offsets(ca1.len(), n_threads);
            let chunks_iter = splits.into_par_iter().map(|(offset, len)| {
                let s1 = ca1.slice(offset as i64, len);
                let s2 = ca2.slice(offset as i64, len);
                let out: Float64Chunked = binary_elementwise_values(&s1, &s2, |x, y| {
                    tversky_sim(x, y, ngram, alpha, beta)
                });
                out.downcast_iter().cloned().collect::<Vec<_>>()
            });

            let chunks = POOL.install(|| chunks_iter.collect::<Vec<_>>());
            Float64Chunked::from_chunk_iter(ca1.name(), chunks.into_iter().flatten())
        } else {
            binary_elementwise_values(ca1, ca2, |x, y| tversky_sim(x, y, ngram, alpha, beta))
        };
        Ok(out.into_series())
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length or one of them must be a scalar.".into(),
        ))
    }
}
