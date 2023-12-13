/// Optimal String Alignment distance
use polars::prelude::{arity::binary_elementwise_values, *};
use pyo3_polars::{
    derive::polars_expr,
    export::polars_core::utils::rayon::prelude::{IndexedParallelIterator, ParallelIterator},
};
use rapidfuzz::distance::osa;

#[inline]
fn osa(s1: &str, s2: &str) -> u32 {
    osa::distance(s1.chars(), s2.chars()) as u32
}

#[inline]
fn osa_sim(s1: &str, s2: &str) -> f64 {
    osa::normalized_similarity(s1.chars(), s2.chars())
}

#[polars_expr(output_type=UInt32)]
fn pl_osa(inputs: &[Series]) -> PolarsResult<Series> {
    let ca1 = inputs[0].utf8()?;
    let ca2 = inputs[1].utf8()?;
    let parallel = inputs[2].bool()?;
    let parallel = parallel.get(0).unwrap();
    if ca2.len() == 1 {
        let r = ca2.get(0).unwrap();
        let batched = osa::BatchComparator::new(r.chars());
        let out: UInt32Chunked = if parallel {
            ca1.par_iter()
                .map(|op_s| {
                    let s = op_s?;
                    Some(batched.distance(s.chars()) as u32)
                })
                .collect()
        } else {
            ca1.apply_nonnull_values_generic(DataType::UInt32, |s| {
                batched.distance(s.chars()) as u32
            })
        };
        Ok(out.into_series())
    } else if ca1.len() == ca2.len() {
        let out: UInt32Chunked = if parallel {
            ca1.par_iter_indexed()
                .zip(ca2.par_iter_indexed())
                .map(|(op_w1, op_w2)| {
                    let (w1, w2) = (op_w1?, op_w2?);
                    Some(osa(w1, w2))
                })
                .collect()
        } else {
            binary_elementwise_values(ca1, ca2, osa)
        };
        Ok(out.into_series())
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length.".into(),
        ))
    }
}

#[polars_expr(output_type=Float64)]
fn pl_osa_sim(inputs: &[Series]) -> PolarsResult<Series> {
    let ca1 = inputs[0].utf8()?;
    let ca2 = inputs[1].utf8()?;
    let parallel = inputs[2].bool()?;
    let parallel = parallel.get(0).unwrap();
    if ca2.len() == 1 {
        let r = ca2.get(0).unwrap();
        let batched = osa::BatchComparator::new(r.chars());
        let out: Float64Chunked = if parallel {
            ca1.par_iter()
                .map(|op_s| {
                    let s = op_s?;
                    Some(batched.normalized_similarity(s.chars()))
                })
                .collect()
        } else {
            ca1.apply_nonnull_values_generic(DataType::Float64, |s| {
                batched.normalized_similarity(s.chars())
            })
        };
        Ok(out.into_series())
    } else if ca1.len() == ca2.len() {
        let out: Float64Chunked = if parallel {
            ca1.par_iter_indexed()
                .zip(ca2.par_iter_indexed())
                .map(|(op_w1, op_w2)| {
                    let (w1, w2) = (op_w1?, op_w2?);
                    Some(osa_sim(w1, w2))
                })
                .collect()
        } else {
            binary_elementwise_values(ca1, ca2, osa_sim)
        };
        Ok(out.into_series())
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length.".into(),
        ))
    }
}
