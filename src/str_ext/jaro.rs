use polars::prelude::{arity::binary_elementwise_values, *};
use pyo3_polars::{
    derive::polars_expr,
    export::polars_core::utils::rayon::prelude::{IndexedParallelIterator, ParallelIterator},
};
// Jaro winkler seems to have a bug related to prefix length
use strsim::jaro;

#[inline]
fn optional_jaro(op_s1: Option<&str>, op_s2: Option<&str>) -> Option<f64> {
    if let (Some(s1), Some(s2)) = (op_s1, op_s2) {
        Some(jaro(s1, s2))
    } else {
        None
    }
}

// #[inline]
// fn optional_jaro_winkler(op_s1: Option<&str>, op_s2: Option<&str>) -> Option<f64> {
//     if let (Some(s1), Some(s2)) = (op_s1, op_s2) {
//         Some(jaro_winkler(s1, s2))
//     } else {
//         None
//     }
// }

#[polars_expr(output_type=UInt32)]
fn pl_jaro(inputs: &[Series]) -> PolarsResult<Series> {
    let ca1 = inputs[0].utf8()?;
    let ca2 = inputs[1].utf8()?;
    let parallel = inputs[2].bool()?;
    let parallel = parallel.get(0).unwrap();
    if ca2.len() == 1 {
        let r = ca2.get(0);
        let out: Float64Chunked = if parallel {
            ca1.par_iter().map(|op_s| optional_jaro(op_s, r)).collect()
        } else {
            ca1.apply_nonnull_values_generic(DataType::Float64, |x| jaro(x, r.unwrap()))
        };
        Ok(out.into_series())
    } else if ca1.len() == ca2.len() {
        let out: Float64Chunked = if parallel {
            ca1.par_iter_indexed()
                .zip(ca2.par_iter_indexed())
                .map(|(op_w1, op_w2)| optional_jaro(op_w1, op_w2))
                .collect()
        } else {
            binary_elementwise_values(ca1, ca2, |x, y| jaro(x, y))
        };
        Ok(out.into_series())
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length.".into(),
        ))
    }
}
