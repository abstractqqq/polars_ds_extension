use polars::prelude::{arity::binary_elementwise_values, *};
use pyo3_polars::{
    derive::polars_expr,
    export::polars_core::utils::rayon::prelude::{IndexedParallelIterator, ParallelIterator},
};
use strsim::{
    damerau_levenshtein, levenshtein, normalized_damerau_levenshtein, normalized_levenshtein,
};

#[inline]
fn optional_damerau_levenshtein(op_s1: Option<&str>, op_s2: Option<&str>) -> Option<u32> {
    if let (Some(s1), Some(s2)) = (op_s1, op_s2) {
        Some(damerau_levenshtein(s1, s2) as u32)
    } else {
        None
    }
}

#[inline]
fn optional_damerau_levenshtein_sim(op_s1: Option<&str>, op_s2: Option<&str>) -> Option<f64> {
    if let (Some(s1), Some(s2)) = (op_s1, op_s2) {
        Some(normalized_damerau_levenshtein(s1, s2))
    } else {
        None
    }
}

#[inline]
fn optional_levenshtein(op_s1: Option<&str>, op_s2: Option<&str>) -> Option<u32> {
    if let (Some(s1), Some(s2)) = (op_s1, op_s2) {
        Some(levenshtein(s1, s2) as u32)
    } else {
        None
    }
}

#[inline]
fn optional_levenshtein_sim(op_s1: Option<&str>, op_s2: Option<&str>) -> Option<f64> {
    if let (Some(s1), Some(s2)) = (op_s1, op_s2) {
        Some(normalized_levenshtein(s1, s2))
    } else {
        None
    }
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
            ca1.apply_nonnull_values_generic(DataType::UInt32, |x| {
                levenshtein(x, r.unwrap()) as u32
            })
        };
        Ok(out.into_series())
    } else if ca1.len() == ca2.len() {
        let out: UInt32Chunked = if parallel {
            ca1.par_iter_indexed()
                .zip(ca2.par_iter_indexed())
                .map(|(op_w1, op_w2)| optional_levenshtein(op_w1, op_w2))
                .collect()
        } else {
            binary_elementwise_values(ca1, ca2, |x, y| levenshtein(x, y) as u32)
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
            ca1.apply_nonnull_values_generic(DataType::Float64, |x| {
                normalized_levenshtein(x, r.unwrap())
            })
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
            ca1.apply_nonnull_values_generic(DataType::UInt32, |x| {
                damerau_levenshtein(x, r.unwrap()) as u32
            })
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
            ca1.apply_nonnull_values_generic(DataType::Float64, |x| {
                normalized_damerau_levenshtein(x, r.unwrap())
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
