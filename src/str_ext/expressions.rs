use polars::prelude::*;
use hashbrown::HashSet;
// use polars::chunked_array::ops::arity::binary_elementwise;
use polars_core::utils::rayon::prelude::{ParallelIterator, IndexedParallelIterator};
use crate::str_ext::consts::EN_STOPWORDS;
use pyo3_polars::derive::polars_expr;
use super::snowball::{algorithms, SnowballEnv};
use std::str;


#[inline]
pub fn snowball_stem(word:Option<&str>, no_stopwords:bool) -> Option<String> {
    
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
        },
        _ => None
    }
}


#[inline]
pub fn hamming_dist(s1:&str, s2:&str) -> Option<u32> {
    if s1.len() != s2.len() {
        return None
    }
    Some(
        s1.chars()
        .zip(s2.chars())
        .fold(0, |a, (b, c)| a + (b != c) as u32)
    )
}


#[inline]
pub fn levenshtein_dist(s1:&str, s2:&str) -> u32 {
    // It is possible to go faster by not using a matrix to represent the 
    // data structure it seems.

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
    let out: Utf8Chunked = ca.par_iter()
        .map(|op_s| snowball_stem(op_s, true)).collect();
    Ok(out.into_series())
}


#[polars_expr(output_type=UInt32)]
fn pl_levenshtein_dist(inputs: &[Series]) -> PolarsResult<Series> {
    let ca1 = inputs[0].utf8()?;
    let ca2 = inputs[1].utf8()?;

    if ca2.len() == 1 {
        let r = ca2.get(0).unwrap();
        let out: UInt32Chunked = ca1.par_iter().map(
            |op_s| {
                if let Some(s) = op_s {
                    Some(levenshtein_dist(s, r))
                } else {
                    None
                }
            }
        ).collect();
        Ok(out.into_series())
    } else if ca1.len() == ca2.len() {

        let out: UInt32Chunked = ca1.par_iter_indexed()
            .zip(ca2.par_iter_indexed())
            .map(|(op_w1, op_w2)| {
                if let (Some(w1), Some(w2)) = (op_w1, op_w2) {
                    Some(levenshtein_dist(w1, w2))
                } else {
                    None
                }
            }).collect();
        Ok(out.into_series())
    } else {
        Err(PolarsError::ComputeError("Inputs must have the same length.".into()))
    }
}

// #[polars_expr(output_type=UInt32)]
// fn pl_levenshtein_dist2(inputs: &[Series]) -> PolarsResult<Series> {
//     let ca1 = inputs[0].utf8()?;
//     let ca2 = inputs[1].utf8()?;

//     if ca2.len() == 1 {
//         let r = ca2.get(0).unwrap();
//         let op = |x:Option<&str>| {
//             if let Some(s) = x {
//                 Some(levenshtein_dist(s, r))
//             } else {
//                 None
//             }
//         };
//         // ca1.apply_generic(op)
//         let out: UInt32Chunked = ca1.apply_generic(op);
//         Ok(out.into_series())
//     } else if ca1.len() == ca2.len() {

//         let op = |x:Option<&str>,y:Option<&str>| {
//             if let (Some(s1), Some(s2)) = (x,y) {
//                 Some(levenshtein_dist(s1, s2))
//             } else {
//                 None
//             }
//         };
//         let out:UInt32Chunked = binary_elementwise(
//             ca1,
//             ca2,
//             op
//         );
//         Ok(out.into_series())
//     } else {
//         Err(PolarsError::ComputeError("Inputs must have the same length.".into()))
//     }
// }


//         binary_elementwise()

#[polars_expr(output_type=Float64)]
fn pl_str_jaccard(inputs: &[Series]) -> PolarsResult<Series> {
    let ca1 = inputs[0].utf8()?;
    let ca2 = inputs[1].utf8()?;
    // gauranteed to have 3 input series by the input from Python side.
    let n = inputs[2].u32()?;
    let n = n.get(0).unwrap() as usize;

    if ca2.len() == 1 {

        let r = ca2.get(0).unwrap();
        let s2: HashSet<&str> = if r.len() > n {
            HashSet::from_iter(
                r.as_bytes().windows(n).map(|sl| str::from_utf8(sl).unwrap()
            )
        )} else {
            HashSet::from_iter([r])
        };
        let out: Float64Chunked = ca1.par_iter().map(|op_s| {
            if let Some(s) = op_s {
                let s1: HashSet<&str> = if s.len() > n {
                    HashSet::from_iter(
                        s.as_bytes().windows(n).map(|sl| str::from_utf8(sl).unwrap())
                    )
                } else {
                    HashSet::from_iter([s])
                };
                let intersection = s1.intersection(&s2).count();
                Some(
                    (intersection as f64) / ((s1.len() + s2.len() - intersection) as f64)
                )
            } else {
                None
            }
        }).collect();
        Ok(out.into_series())
    } else if ca1.len() == ca2.len() {

        let out: Float64Chunked = ca1.par_iter_indexed()
            .zip(ca2.par_iter_indexed())
            .map(|(op_w1, op_w2)| {
                if let (Some(w1), Some(w2)) = (op_w1, op_w2) {
                    if (w1.len() >= n) & (w2.len() >= n) {
                        let s1: HashSet<&str> = HashSet::from_iter(
                            w1.as_bytes().windows(n).map(|sl| str::from_utf8(sl).unwrap())
                        );
                        let s2: HashSet<&str> = HashSet::from_iter(
                            w2.as_bytes().windows(n).map(|sl| str::from_utf8(sl).unwrap())
                        );
                        let intersection = s1.intersection(&s2).count();
                        Some(
                            (intersection as f64) / ((s1.len() + s2.len() - intersection) as f64)
                        )
                    } else if (w1.len() < n) & (w2.len() < n) {
                        Some(((w1 == w2) as u8) as f64)
                    } else {
                        Some(0.)
                    }
                } else {
                    None
                }
            }).collect();
        Ok(out.into_series())
    } else {
        Err(PolarsError::ComputeError("Inputs must have the same length.".into()))
    }
}


#[polars_expr(output_type=UInt32)]
fn pl_hamming_dist(inputs: &[Series]) -> PolarsResult<Series> {
    let ca1 = inputs[0].utf8()?;
    let ca2 = inputs[1].utf8()?;

    if ca2.len() == 1 {
        let r = ca2.get(0).unwrap();
        let out: UInt32Chunked = ca1.par_iter().map(|op_s| {
            if let Some(w) = op_s {
                hamming_dist(w, r)
            } else {
                None
            }            
        }).collect();
        Ok(out.into_series())
    } else if ca1.len() == ca2.len() {
        let out: UInt32Chunked = ca1.par_iter_indexed()
            .zip(ca2.par_iter_indexed())
            .map(|(op_w1, op_w2)| {
                if let (Some(w1), Some(w2)) = (op_w1, op_w2) {
                    hamming_dist(w1, w2)
                } else {
                    None
                }
            }).collect();
        Ok(out.into_series())
    } else {
        Err(PolarsError::ComputeError("Inputs must have the same length.".into()))
    }
}