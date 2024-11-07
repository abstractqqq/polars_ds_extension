use super::generic_str_distancer::{
    generic_batched_distance, generic_batched_sim, generic_binary_distance, generic_binary_sim,
};
use polars::prelude::*;
use pyo3_polars::derive::{polars_expr, CallerContext};
use rapidfuzz::distance::osa;

#[inline(always)]
fn osa(s1: &str, s2: &str) -> u32 {
    osa::distance(s1.chars(), s2.chars()) as u32
}

#[inline(always)]
fn osa_sim(s1: &str, s2: &str) -> f64 {
    osa::normalized_similarity(s1.chars(), s2.chars())
}

#[polars_expr(output_type=UInt32)]
fn pl_osa(inputs: &[Series], context: CallerContext) -> PolarsResult<Series> {
    let ca1 = inputs[0].str()?;
    let ca2 = inputs[1].str()?;
    let parallel = inputs[2].bool()?;
    let parallel = parallel.get(0).unwrap();
    let can_parallel = parallel && !context.parallel();
    if ca2.len() == 1 {
        let r = ca2.get(0).unwrap();
        let batched = osa::BatchComparator::new(r.chars());
        Ok(generic_batched_distance(batched, ca1, can_parallel))
    } else if ca1.len() == ca2.len() {
        Ok(generic_binary_distance(osa, ca1, ca2, can_parallel))
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length or one of them must be a scalar.".into(),
        ))
    }
}

#[polars_expr(output_type=Float64)]
fn pl_osa_sim(inputs: &[Series], context: CallerContext) -> PolarsResult<Series> {
    let ca1 = inputs[0].str()?;
    let ca2 = inputs[1].str()?;
    let parallel = inputs[2].bool()?;
    let parallel = parallel.get(0).unwrap();
    let can_parallel = parallel && !context.parallel();
    if ca2.len() == 1 {
        let r = ca2.get(0).unwrap();
        let batched = osa::BatchComparator::new(r.chars());
        Ok(generic_batched_sim(batched, ca1, can_parallel))
    } else if ca1.len() == ca2.len() {
        Ok(generic_binary_sim(osa_sim, ca1, ca2, can_parallel))
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length or one of them must be a scalar.".into(),
        ))
    }
}
