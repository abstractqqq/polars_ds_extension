use super::snowball::{algorithms, SnowballEnv};
use crate::str_ext::consts::EN_STOPWORDS;
use hashbrown::HashSet;
use polars::prelude::*;
use polars_core::{
    prelude::arity::binary_elementwise_values,
    utils::rayon::prelude::{IndexedParallelIterator, ParallelIterator},
};
use pyo3_polars::derive::polars_expr;
use std::str;

#[inline]
pub fn snowball_stem(word: Option<&str>, no_stopwords: bool) -> Option<String> {
    match word {
        Some(w) => {
            if (no_stopwords) & (EN_STOPWORDS.binary_search(&w).is_ok()) {
                None
            } else if w.parse::<f64>().is_ok() {
                None
            } else {
                let mut env: SnowballEnv<'_> = SnowballEnv::create(w);
                algorithms::english_stemmer::stem(&mut env);
                Some(env.get_current().to_string())
            }
        }
        _ => None,
    }
}

#[inline]
pub fn hamming_dist(s1: &str, s2: &str) -> Option<u32> {
    if s1.len() != s2.len() {
        return None;
    }
    Some(
        s1.chars()
            .zip(s2.chars())
            .fold(0, |a, (b, c)| a + (b != c) as u32),
    )
}

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

// Wrapper for Polars Extension
#[polars_expr(output_type=Utf8)]
fn pl_snowball_stem(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].utf8()?;
    let no_stop = inputs[1].bool()?;
    let no_stop = no_stop.get(0).unwrap();
    let parallel = inputs[2].bool()?;
    let parallel = parallel.get(0).unwrap();
    let out: Utf8Chunked = if parallel {
        ca.par_iter()
            .map(|op_s| snowball_stem(op_s, no_stop))
            .collect()
    } else {
        // have to do apply_generic, not apply_values because snowball may return None for string inputs.
        ca.apply_generic(|op_s| snowball_stem(op_s, no_stop))
    };
    Ok(out.into_series())
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

fn str_jaccard(w1: &str, w2: &str, n: usize) -> f64 {
    let w1_len = w1.len();
    let w2_len = w2.len();
    let s1: HashSet<&str> = if w1_len < n {
        HashSet::from_iter([w1])
    } else {
        HashSet::from_iter(
            w1.as_bytes()
                .windows(n)
                .map(|sl| str::from_utf8(sl).unwrap()),
        )
    };
    let s2: HashSet<&str> = if w2_len < n {
        HashSet::from_iter([w2])
    } else {
        HashSet::from_iter(
            w2.as_bytes()
                .windows(n)
                .map(|sl| str::from_utf8(sl).unwrap()),
        )
    };
    let intersection = s1.intersection(&s2).count();
    (intersection as f64) / ((s1.len() + s2.len() - intersection) as f64)
}

fn optional_str_jaccard(op_w1: Option<&str>, op_w2: Option<&str>, n: usize) -> Option<f64> {
    if let (Some(w1), Some(w2)) = (op_w1, op_w2) {
        Some(str_jaccard(w1, w2, n))
    } else {
        None
    }
}

#[polars_expr(output_type=Float64)]
fn pl_str_jaccard(inputs: &[Series]) -> PolarsResult<Series> {
    let ca1 = inputs[0].utf8()?;
    let ca2 = inputs[1].utf8()?;

    // gauranteed to have 4 input series by the input from Python side.
    // The 3rd input is size of substring length
    let n = inputs[2].u32()?;
    let n = n.get(0).unwrap() as usize;
    // parallel
    let parallel = inputs[3].bool()?;
    let parallel = parallel.get(0).unwrap();

    if ca2.len() == 1 {
        let r = ca2.get(0).unwrap();
        let s2: HashSet<&str> = if r.len() > n {
            HashSet::from_iter(
                r.as_bytes()
                    .windows(n)
                    .map(|sl| str::from_utf8(sl).unwrap()),
            )
        } else {
            HashSet::from_iter([r])
        };
        let op = |op_s: Option<&str>| {
            if let Some(s) = op_s {
                let s1: HashSet<&str> = if s.len() > n {
                    HashSet::from_iter(
                        s.as_bytes()
                            .windows(n)
                            .map(|sl| str::from_utf8(sl).unwrap()),
                    )
                } else {
                    HashSet::from_iter([s])
                };
                let intersection = s1.intersection(&s2).count();
                Some((intersection as f64) / ((s1.len() + s2.len() - intersection) as f64))
            } else {
                None
            }
        };
        let out: Float64Chunked = if parallel {
            ca1.par_iter().map(|op_s| op(op_s)).collect()
        } else {
            ca1.apply_generic(op)
        };
        Ok(out.into_series())
    } else if ca1.len() == ca2.len() {
        let out: Float64Chunked = if parallel {
            ca1.par_iter_indexed()
                .zip(ca2.par_iter_indexed())
                .map(|(op_w1, op_w2)| optional_str_jaccard(op_w1, op_w2, n))
                .collect()
        } else {
            binary_elementwise_values(ca1, ca2, |x, y| str_jaccard(x, y, n))
        };
        Ok(out.into_series())
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length.".into(),
        ))
    }
}

fn optional_hamming(op_w1: Option<&str>, op_w2: Option<&str>) -> Option<u32> {
    if let (Some(w1), Some(w2)) = (op_w1, op_w2) {
        hamming_dist(w1, w2)
    } else {
        None
    }
}

#[polars_expr(output_type=UInt32)]
fn pl_hamming_dist(inputs: &[Series]) -> PolarsResult<Series> {
    let ca1 = inputs[0].utf8()?;
    let ca2 = inputs[1].utf8()?;
    let parallel = inputs[2].bool()?;
    let parallel = parallel.get(0).unwrap();

    if ca2.len() == 1 {
        let r = ca2.get(0).unwrap();
        let op = |op_s| {
            if let Some(w) = op_s {
                hamming_dist(w, r)
            } else {
                None
            }
        };
        let out: UInt32Chunked = if parallel {
            ca1.par_iter().map(|op_s| op(op_s)).collect()
        } else {
            ca1.apply_generic(op)
        };
        Ok(out.into_series())
    } else if ca1.len() == ca2.len() {
        let out: UInt32Chunked = if parallel {
            ca1.par_iter_indexed()
                .zip(ca2.par_iter_indexed())
                .map(|(op_w1, op_w2)| optional_hamming(op_w1, op_w2))
                .collect()
        } else {
            binary_elementwise_values(ca1, ca2, hamming_dist)
        };
        Ok(out.into_series())
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length.".into(),
        ))
    }
}
