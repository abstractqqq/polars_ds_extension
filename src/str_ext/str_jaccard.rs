use super::{str_set_sim_helper, str_to_hashset};
use crate::utils::split_offsets;
use polars::prelude::{arity::binary_elementwise_values, *};
use pyo3_polars::{
    derive::{polars_expr, CallerContext},
    export::polars_core::{
        error::PolarsError,
        utils::rayon::prelude::{IntoParallelIterator, ParallelIterator},
        POOL,
    },
};

#[inline(always)]
fn str_jaccard(w1: &str, w2: &str, ngram: usize) -> f64 {
    let (s1, s2, intersect) = str_set_sim_helper(w1, w2, ngram);
    (intersect as f64) / ((s1 + s2 - intersect) as f64)
}

#[inline(always)]
fn str_jaccard_cached(w1: &str, cached_s2: &foldhash::HashSet<&[u8]>, ngram: usize) -> f64 {
    let s1 = str_to_hashset(w1, ngram);
    let intersection = s1.intersection(cached_s2).count();
    (intersection as f64) / ((s1.len() + cached_s2.len() - intersection) as f64)
}

#[polars_expr(output_type=Float64)]
fn pl_str_jaccard(inputs: &[Series], context: CallerContext) -> PolarsResult<Series> {
    let ca1 = inputs[0].str()?;
    let ca2 = inputs[1].str()?;

    // ngram size
    let ngram = inputs[2].u32()?;
    let ngram = ngram.get(0).unwrap() as usize;
    // parallel
    let parallel = inputs[3].bool()?;
    let parallel = parallel.get(0).unwrap();
    let can_parallel = parallel && !context.parallel();

    if ca2.len() == 1 {
        let r = ca2.get(0).unwrap(); // .unwrap();
        let cached_r_set = str_to_hashset(r, ngram); // Cache the set for r
        let out: Float64Chunked = if can_parallel {
            let n_threads = POOL.current_num_threads();
            let splits = split_offsets(ca1.len(), n_threads);
            let chunks_iter = splits.into_par_iter().map(|(offset, len)| {
                let s1 = ca1.slice(offset as i64, len);
                let out: Float64Chunked = s1.apply_nonnull_values_generic(DataType::Float64, |s| {
                    str_jaccard_cached(s, &cached_r_set, ngram)
                });
                out.downcast_iter().cloned().collect::<Vec<_>>()
            });

            let chunks = POOL.install(|| chunks_iter.collect::<Vec<_>>());
            Float64Chunked::from_chunk_iter(ca1.name().clone(), chunks.into_iter().flatten())
        } else {
            ca1.apply_nonnull_values_generic(DataType::Float64, |s| {
                str_jaccard_cached(s, &cached_r_set, ngram)
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
                let out: Float64Chunked =
                    binary_elementwise_values(&s1, &s2, |x, y| str_jaccard(x, y, ngram));
                out.downcast_iter().cloned().collect::<Vec<_>>()
            });

            let chunks = POOL.install(|| chunks_iter.collect::<Vec<_>>());
            Float64Chunked::from_chunk_iter(ca1.name().clone(), chunks.into_iter().flatten())
        } else {
            binary_elementwise_values(ca1, ca2, |x, y| str_jaccard(x, y, ngram))
        };
        Ok(out.into_series())
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length or the second of them must be a scalar.".into(),
        ))
    }
}
