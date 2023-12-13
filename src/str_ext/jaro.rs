use polars::prelude::{arity::binary_elementwise_values, *};
use pyo3_polars::{
    derive::polars_expr,
    export::polars_core::utils::rayon::prelude::{IndexedParallelIterator, ParallelIterator},
};
use rapidfuzz::distance::{jaro, jaro_winkler};

#[inline]
fn jaro_sim(s1: &str, s2: &str) -> f64 {
    jaro::normalized_similarity(s1.chars(), s2.chars())
}

#[inline]
fn jw_sim(s1: &str, s2: &str, weight: f64) -> f64 {
    jaro_winkler::normalized_similarity_with_args(
        s1.chars(),
        s2.chars(),
        &jaro_winkler::Args::default().prefix_weight(weight),
    )
}

#[polars_expr(output_type=Float64)]
fn pl_jaro(inputs: &[Series]) -> PolarsResult<Series> {
    let ca1 = inputs[0].utf8()?;
    let ca2 = inputs[1].utf8()?;
    let parallel = inputs[2].bool()?;
    let parallel = parallel.get(0).unwrap();
    if ca2.len() == 1 {
        let r = ca2.get(0).unwrap();
        let batched = jaro::BatchComparator::new(r.chars());
        let out: Float64Chunked = if parallel {
            ca1.par_iter()
                .map(|op_s| {
                    let s = op_s?;
                    Some(batched.similarity(s.chars()))
                })
                .collect()
        } else {
            ca1.apply_nonnull_values_generic(DataType::Float64, |s| batched.similarity(s.chars()))
        };
        Ok(out.into_series())
    } else if ca1.len() == ca2.len() {
        let out: Float64Chunked = if parallel {
            ca1.par_iter_indexed()
                .zip(ca2.par_iter_indexed())
                .map(|(op_w1, op_w2)| {
                    let (w1, w2) = (op_w1?, op_w2?);
                    Some(jaro_sim(w1, w2))
                })
                .collect()
        } else {
            binary_elementwise_values(ca1, ca2, |x, y| jaro_sim(x, y))
        };
        Ok(out.into_series())
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length.".into(),
        ))
    }
}

#[polars_expr(output_type=Float64)]
fn pl_jw(inputs: &[Series]) -> PolarsResult<Series> {
    let ca1 = inputs[0].utf8()?;
    let ca2 = inputs[1].utf8()?;
    let weight = inputs[2].f64()?;
    let weight = weight.get(0).unwrap_or(0.1);
    let parallel = inputs[3].bool()?;
    let parallel = parallel.get(0).unwrap();
    if ca2.len() == 1 {
        let r = ca2.get(0).unwrap();
        let batched = jaro_winkler::BatchComparator::new(r.chars());
        let out: Float64Chunked = if parallel {
            ca1.par_iter()
                .map(|op_s| {
                    let s = op_s?;
                    Some(batched.similarity_with_args(
                        s.chars(),
                        &jaro_winkler::Args::default().prefix_weight(weight),
                    ))
                })
                .collect()
        } else {
            ca1.apply_nonnull_values_generic(DataType::Float64, |s| {
                batched.similarity_with_args(
                    s.chars(),
                    &jaro_winkler::Args::default().prefix_weight(weight),
                )
            })
        };
        Ok(out.into_series())
    } else if ca1.len() == ca2.len() {
        let out: Float64Chunked = if parallel {
            ca1.par_iter_indexed()
                .zip(ca2.par_iter_indexed())
                .map(|(op_w1, op_w2)| {
                    let (w1, w2) = (op_w1?, op_w2?);
                    Some(jw_sim(w1, w2, weight))
                })
                .collect()
        } else {
            binary_elementwise_values(ca1, ca2, |x, y| jw_sim(x, y, weight))
        };
        Ok(out.into_series())
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length.".into(),
        ))
    }
}
