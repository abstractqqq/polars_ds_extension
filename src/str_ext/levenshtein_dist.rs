use polars::prelude::{*, arity::binary_elementwise_values};
use pyo3_polars::{
    derive::polars_expr, 
    export::polars_core::utils::rayon::prelude::{ParallelIterator, IndexedParallelIterator}
};

pub fn levenshtein_dist(s1: &str, s2: &str) -> u32 {
    //
    // https://en.wikipedia.org/wiki/Wagner%E2%80%93Fischer_algorithm

    let (len1, len2) = (s1.len(), s2.len());
    let mut dp: Vec<Vec<u32>> = vec![vec![0; len2 + 1]; len1 + 1];

    // Initialize the first row and first column
    for i in 0..=len1 {
        dp[i][0] = i as u32;
    }

    for j in 0..=len2 {
        dp[0][j] = j as u32;
    }

    // Fill the dp matrix using dynamic programming
    for (i, char1) in s1.chars().enumerate() {
        for (j, char2) in s2.chars().enumerate() {
            if char1 == char2 {
                dp[i + 1][j + 1] = dp[i][j];
            } else {
                dp[i + 1][j + 1] = 1 + dp[i][j].min(dp[i][j + 1].min(dp[i + 1][j]));
            }
        }
    }
    dp[len1][len2]
}


fn optional_levenshtein(op_w1: Option<&str>, op_w2: Option<&str>) -> Option<u32> {
    if let (Some(w1), Some(w2)) = (op_w1, op_w2) {
        Some(levenshtein_dist(w1, w2))
    } else {
        None
    }
}

#[polars_expr(output_type=UInt32)]
fn pl_levenshtein_dist(inputs: &[Series]) -> PolarsResult<Series> {
    let ca1 = inputs[0].utf8()?;
    let ca2 = inputs[1].utf8()?;
    let parallel = inputs[2].bool()?;
    let parallel = parallel.get(0).unwrap();
    if ca2.len() == 1 {
        let r = ca2.get(0).unwrap();
        let out: UInt32Chunked = if parallel {
            let op = |op_s| {
                if let Some(s) = op_s {
                    Some(levenshtein_dist(s, r))
                } else {
                    None
                }
            };
            ca1.par_iter().map(|op_s| op(op_s)).collect()
        } else {
            ca1.apply_nonnull_values_generic(DataType::UInt32, |x| levenshtein_dist(x, r))
        };
        Ok(out.into_series())
    } else if ca1.len() == ca2.len() {
        let out: UInt32Chunked = if parallel {
            ca1.par_iter_indexed()
                .zip(ca2.par_iter_indexed())
                .map(|(op_w1, op_w2)| optional_levenshtein(op_w1, op_w2))
                .collect()
        } else {
            binary_elementwise_values(ca1, ca2, levenshtein_dist)
        };
        Ok(out.into_series())
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length.".into(),
        ))
    }
}