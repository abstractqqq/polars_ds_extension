use super::generic_str_distancer::{
    generic_batched_distance, generic_batched_sim, generic_binary_distance, generic_binary_sim,
};
use crate::utils::split_offsets;
use polars::prelude::{arity::binary_elementwise_values, *};
use pyo3_polars::{
    derive::{polars_expr, CallerContext},
    export::polars_core::{
        utils::rayon::prelude::{IntoParallelIterator, ParallelIterator},
        POOL,
    },
};
use rapidfuzz::distance::{damerau_levenshtein, levenshtein};

#[inline(always)]
fn levenshtein(s1: &str, s2: &str) -> u32 {
    levenshtein::distance(s1.chars(), s2.chars()) as u32
}

#[inline(always)]
fn levenshtein_within_bound(s1: &str, s2: &str, bound: usize) -> bool {
    levenshtein::distance_with_args(
        s1.chars(),
        s2.chars(),
        &levenshtein::Args::default().score_cutoff(bound),
    )
    .is_some()
}

#[inline(always)]
fn levenshtein_sim(s1: &str, s2: &str) -> f64 {
    levenshtein::normalized_similarity(s1.chars(), s2.chars())
}

#[inline(always)]
fn d_levenshtein(s1: &str, s2: &str) -> u32 {
    damerau_levenshtein::distance(s1.chars(), s2.chars()) as u32
}

#[inline(always)]
fn d_levenshtein_sim(s1: &str, s2: &str) -> f64 {
    damerau_levenshtein::normalized_similarity(s1.chars(), s2.chars())
}

#[polars_expr(output_type=UInt32)]
fn pl_levenshtein(inputs: &[Series], context: CallerContext) -> PolarsResult<Series> {
    let ca1 = inputs[0].str()?;
    let ca2 = inputs[1].str()?;
    let parallel = inputs[2].bool()?;
    let parallel = parallel.get(0).unwrap();
    let can_parallel = parallel && !context.parallel();
    if ca2.len() == 1 {
        let r = ca2.get(0).unwrap();
        let batched = levenshtein::BatchComparator::new(r.chars());
        Ok(generic_batched_distance(batched, ca1, can_parallel))
    } else if ca1.len() == ca2.len() {
        Ok(generic_binary_distance(levenshtein, ca1, ca2, can_parallel))
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length or one of them must be a scalar.".into(),
        ))
    }
}

#[polars_expr(output_type=Boolean)]
fn pl_levenshtein_filter(inputs: &[Series], context: CallerContext) -> PolarsResult<Series> {
    let ca1 = inputs[0].str()?;
    let ca2 = inputs[1].str()?;
    let bound = inputs[2].u32()?;
    let bound = bound.get(0).unwrap() as usize;
    let parallel = inputs[3].bool()?;
    let parallel = parallel.get(0).unwrap();
    let can_parallel = parallel && !context.parallel();
    if ca2.len() == 1 {
        let r = ca2.get(0).unwrap();
        let batched = levenshtein::BatchComparator::new(r.chars());
        let out: BooleanChunked = if can_parallel {
            let n_threads = POOL.current_num_threads();
            let splits = split_offsets(ca1.len(), n_threads);
            let chunks_iter = splits.into_par_iter().map(|(offset, len)| {
                let s1 = ca1.slice(offset as i64, len);
                let out: BooleanChunked = s1.apply_nonnull_values_generic(DataType::Boolean, |s| {
                    batched
                        .distance_with_args(
                            s.chars(),
                            &levenshtein::Args::default().score_cutoff(bound),
                        )
                        .is_some()
                });
                out.downcast_iter().cloned().collect::<Vec<_>>()
            });
            let chunks = POOL.install(|| chunks_iter.collect::<Vec<_>>());
            BooleanChunked::from_chunk_iter(ca1.name().clone(), chunks.into_iter().flatten())
        } else {
            ca1.apply_nonnull_values_generic(DataType::Boolean, |s| {
                batched
                    .distance_with_args(
                        s.chars(),
                        &levenshtein::Args::default().score_cutoff(bound),
                    )
                    .is_some()
            })
        };
        Ok(out.into_series())
    } else if ca1.len() == ca2.len() {
        let out: BooleanChunked = if parallel {
            let n_threads = POOL.current_num_threads();
            let splits = split_offsets(ca1.len(), n_threads);
            let chunks_iter = splits.into_par_iter().map(|(offset, len)| {
                let s1 = ca1.slice(offset as i64, len);
                let s2 = ca2.slice(offset as i64, len);
                let out: BooleanChunked = binary_elementwise_values(&s1, &s2, |x, y: &str| {
                    levenshtein_within_bound(x, y, bound)
                });
                out.downcast_iter().cloned().collect::<Vec<_>>()
            });
            let chunks = POOL.install(|| chunks_iter.collect::<Vec<_>>());
            BooleanChunked::from_chunk_iter(ca1.name().clone(), chunks.into_iter().flatten())
        } else {
            binary_elementwise_values(ca1, ca2, |x, y| levenshtein_within_bound(x, y, bound))
        };
        Ok(out.into_series())
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length or one of them must be a scalar.".into(),
        ))
    }
}

#[polars_expr(output_type=Float64)]
fn pl_levenshtein_sim(inputs: &[Series], context: CallerContext) -> PolarsResult<Series> {
    let ca1 = inputs[0].str()?;
    let ca2 = inputs[1].str()?;
    let parallel = inputs[2].bool()?;
    let parallel = parallel.get(0).unwrap();
    let can_parallel = parallel && !context.parallel();
    if ca2.len() == 1 {
        let r = ca2.get(0).unwrap();
        let batched = levenshtein::BatchComparator::new(r.chars());
        Ok(generic_batched_sim(batched, ca1, can_parallel))
    } else if ca1.len() == ca2.len() {
        Ok(generic_binary_sim(levenshtein_sim, ca1, ca2, can_parallel))
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length or one of them must be a scalar.".into(),
        ))
    }
}

#[polars_expr(output_type=UInt32)]
fn pl_d_levenshtein(inputs: &[Series], context: CallerContext) -> PolarsResult<Series> {
    let ca1 = inputs[0].str()?;
    let ca2 = inputs[1].str()?;
    let parallel = inputs[2].bool()?;
    let parallel = parallel.get(0).unwrap();
    let can_parallel = parallel && !context.parallel();
    if ca2.len() == 1 {
        let r = ca2.get(0).unwrap();
        let batched = damerau_levenshtein::BatchComparator::new(r.chars());
        Ok(generic_batched_distance(batched, ca1, can_parallel))
    } else if ca1.len() == ca2.len() {
        Ok(generic_binary_distance(
            d_levenshtein,
            ca1,
            ca2,
            can_parallel,
        ))
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length or one of them must be a scalar.".into(),
        ))
    }
}

#[polars_expr(output_type=Float64)]
fn pl_d_levenshtein_sim(inputs: &[Series], context: CallerContext) -> PolarsResult<Series> {
    let ca1 = inputs[0].str()?;
    let ca2 = inputs[1].str()?;
    let parallel = inputs[2].bool()?;
    let parallel = parallel.get(0).unwrap();
    let can_parallel = parallel && !context.parallel();
    if ca2.len() == 1 {
        let r = ca2.get(0).unwrap();
        let batched = damerau_levenshtein::BatchComparator::new(r.chars());
        Ok(generic_batched_sim(batched, ca1, can_parallel))
    } else if ca1.len() == ca2.len() {
        Ok(generic_binary_sim(
            d_levenshtein_sim,
            ca1,
            ca2,
            can_parallel,
        ))
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length or one of them must be a scalar.".into(),
        ))
    }
}
