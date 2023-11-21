use polars::prelude::{arity::binary_elementwise_values, *};
use pyo3_polars::{
    derive::polars_expr,
    export::polars_core::utils::rayon::prelude::{IndexedParallelIterator, ParallelIterator},
};
use strsim::{damerau_levenshtein, normalized_damerau_levenshtein, normalized_levenshtein};

// A slightly faster version than strsim's Levenshtein by dropping some abstractions
#[inline]
fn _levenshtein(a: &str, b: &str) -> u32 {
    let (aa, bb) = super::remove_common_prefix(&a, &b);
    let (aa, bb) = super::remove_common_suffix(&aa, &bb);

    let b_len = bb.len() as u32;
    let a_len = aa.len() as u32;
    if (a_len == 0) || (b_len == 0) {
        return a_len.max(b_len);
    }

    let mut cache: Vec<u32> = (1..(b_len + 1)).collect();
    let mut result: u32 = 0;

    for (i, a_elem) in (0..a_len).zip(aa.chars()) {
        result = i + 1;
        let mut distance_b = i;
        for (j, b_elem) in bb.chars().enumerate() {
            if a_elem == b_elem {
                let distance_a = distance_b;
                distance_b = cache[j];
                result = distance_a;
            } else {
                let distance_a = distance_b + 1;
                distance_b = cache[j];
                result = (result + 1).min(distance_a.min(distance_b + 1));
            }
            cache[j] = result;
        }
    }
    result
}

#[inline]
fn optional_damerau_levenshtein(op_s1: Option<&str>, op_s2: Option<&str>) -> Option<u32> {
    let s1 = op_s1?;
    let s2 = op_s2?;
    Some(damerau_levenshtein(s1, s2) as u32)
}

#[inline]
fn optional_damerau_levenshtein_sim(op_s1: Option<&str>, op_s2: Option<&str>) -> Option<f64> {
    let s1 = op_s1?;
    let s2 = op_s2?;
    Some(normalized_damerau_levenshtein(s1, s2))
}

#[inline]
fn optional_levenshtein(op_s1: Option<&str>, op_s2: Option<&str>) -> Option<u32> {
    let s1 = op_s1?;
    let s2 = op_s2?;
    Some(_levenshtein(s1, s2))
}

#[inline]
fn optional_levenshtein_sim(op_s1: Option<&str>, op_s2: Option<&str>) -> Option<f64> {
    let s1 = op_s1?;
    let s2 = op_s2?;
    Some(normalized_levenshtein(s1, s2))
}

#[polars_expr(output_type=UInt32)]
fn pl_levenshtein(inputs: &[Series]) -> PolarsResult<Series> {
    let ca1 = inputs[0].utf8()?;
    let ca2 = inputs[1].utf8()?;
    let parallel = inputs[2].bool()?;
    let parallel = parallel.get(0).unwrap();
    if ca2.len() == 1 {
        let r = ca2.get(0);
        let out: UInt32Chunked = if parallel {
            ca1.par_iter()
                .map(|op_s| optional_levenshtein(op_s, r))
                .collect()
        } else {
            let r = r.unwrap();
            ca1.apply_nonnull_values_generic(DataType::UInt32, |x| _levenshtein(x, r))
        };
        Ok(out.into_series())
    } else if ca1.len() == ca2.len() {
        let out: UInt32Chunked = if parallel {
            ca1.par_iter_indexed()
                .zip(ca2.par_iter_indexed())
                .map(|(op_w1, op_w2)| optional_levenshtein(op_w1, op_w2))
                .collect()
        } else {
            binary_elementwise_values(ca1, ca2, |x, y| _levenshtein(x, y) as u32)
        };
        Ok(out.into_series())
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length.".into(),
        ))
    }
}

#[polars_expr(output_type=Float64)]
fn pl_levenshtein_sim(inputs: &[Series]) -> PolarsResult<Series> {
    let ca1 = inputs[0].utf8()?;
    let ca2 = inputs[1].utf8()?;
    let parallel = inputs[2].bool()?;
    let parallel = parallel.get(0).unwrap();
    if ca2.len() == 1 {
        let r = ca2.get(0);
        let out: Float64Chunked = if parallel {
            ca1.par_iter()
                .map(|op_s| optional_levenshtein_sim(op_s, r))
                .collect()
        } else {
            let r = r.unwrap();
            ca1.apply_nonnull_values_generic(DataType::Float64, |x| normalized_levenshtein(x, r))
        };
        Ok(out.into_series())
    } else if ca1.len() == ca2.len() {
        let out: Float64Chunked = if parallel {
            ca1.par_iter_indexed()
                .zip(ca2.par_iter_indexed())
                .map(|(op_w1, op_w2)| optional_levenshtein_sim(op_w1, op_w2))
                .collect()
        } else {
            binary_elementwise_values(ca1, ca2, |x, y| normalized_levenshtein(x, y))
        };
        Ok(out.into_series())
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length.".into(),
        ))
    }
}

#[polars_expr(output_type=UInt32)]
fn pl_d_levenshtein(inputs: &[Series]) -> PolarsResult<Series> {
    let ca1 = inputs[0].utf8()?;
    let ca2 = inputs[1].utf8()?;
    let parallel = inputs[2].bool()?;
    let parallel = parallel.get(0).unwrap();
    if ca2.len() == 1 {
        let r = ca2.get(0);
        let out: UInt32Chunked = if parallel {
            ca1.par_iter()
                .map(|op_s| optional_damerau_levenshtein(op_s, r))
                .collect()
        } else {
            let r = r.unwrap();
            ca1.apply_nonnull_values_generic(DataType::UInt32, |x| damerau_levenshtein(x, r) as u32)
        };
        Ok(out.into_series())
    } else if ca1.len() == ca2.len() {
        let out: UInt32Chunked = if parallel {
            ca1.par_iter_indexed()
                .zip(ca2.par_iter_indexed())
                .map(|(op_w1, op_w2)| optional_damerau_levenshtein(op_w1, op_w2))
                .collect()
        } else {
            binary_elementwise_values(ca1, ca2, |x, y| damerau_levenshtein(x, y) as u32)
        };
        Ok(out.into_series())
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length.".into(),
        ))
    }
}

#[polars_expr(output_type=Float64)]
fn pl_d_levenshtein_sim(inputs: &[Series]) -> PolarsResult<Series> {
    let ca1 = inputs[0].utf8()?;
    let ca2 = inputs[1].utf8()?;
    let parallel = inputs[2].bool()?;
    let parallel = parallel.get(0).unwrap();
    if ca2.len() == 1 {
        let r = ca2.get(0);
        let out: Float64Chunked = if parallel {
            ca1.par_iter()
                .map(|op_s| optional_damerau_levenshtein_sim(op_s, r))
                .collect()
        } else {
            let r = r.unwrap();
            ca1.apply_nonnull_values_generic(DataType::Float64, |x| {
                normalized_damerau_levenshtein(x, r)
            })
        };
        Ok(out.into_series())
    } else if ca1.len() == ca2.len() {
        let out: Float64Chunked = if parallel {
            ca1.par_iter_indexed()
                .zip(ca2.par_iter_indexed())
                .map(|(op_w1, op_w2)| optional_damerau_levenshtein_sim(op_w1, op_w2))
                .collect()
        } else {
            binary_elementwise_values(ca1, ca2, |x, y| normalized_damerau_levenshtein(x, y))
        };
        Ok(out.into_series())
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length.".into(),
        ))
    }
}
