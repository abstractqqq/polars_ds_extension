use crate::utils::split_offsets;

use super::generic_str_distancer::{
    generic_batched_distance, generic_batched_sim, generic_binary_distance, generic_binary_sim,
};
use polars::prelude::{arity::binary_elementwise_values, *};
use pyo3_polars::{
    derive::{polars_expr, CallerContext},
    export::polars_core::{
        utils::rayon::iter::{IntoParallelIterator, ParallelIterator},
        POOL,
    },
};

use rapidfuzz::distance::lcs_seq;

#[inline(always)]
fn lcs_seq(s1: &str, s2: &str) -> u32 {
    lcs_seq::distance(s1.chars(), s2.chars()) as u32
}

#[inline(always)]
fn lcs_seq_sim(s1: &str, s2: &str) -> f64 {
    lcs_seq::normalized_similarity(s1.chars(), s2.chars())
}

/// Finds the longest common subsequence and extracts it.
pub fn lcs_subseq_extract(s1: &str, s2: &str) -> String {
    // Convert string slices to character vectors for easier indexing.
    // This handles multi-byte UTF-8 characters correctly.
    let chars1: Vec<char> = s1.chars().collect();
    let chars2: Vec<char> = s2.chars().collect();

    let len1 = chars1.len();
    let len2 = chars2.len();

    // Create a 2D dynamic programming table (matrix).
    // dp[i][j] will store the length of the LCS of chars1[0...i-1] and chars2[0...j-1].
    // Dimensions are (len1 + 1) x (len2 + 1) to handle empty prefixes.
    let mut dp = vec![vec![0i32; len2 + 1]; len1 + 1];

    // Fill the dp table.
    // i iterates through chars1, j iterates through chars2.
    // Note: The loop indices (i, j) correspond to the lengths of prefixes,
    // so chars1[i-1] and chars2[j-1] are used to access the actual characters.
    for i in 1..=len1 {
        for j in 1..=len2 {
            // If the current characters match, extend the LCS from the diagonal element.
            if chars1[i - 1] == chars2[j - 1] {
                dp[i][j] = 1 + dp[i - 1][j - 1];
            } else {
                // If characters do not match, take the maximum LCS length
                // from either skipping a character from s1 or skipping a character from s2.
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }

    // Backtrack through the DP table to reconstruct the LCS.
    let mut lcs_chars = Vec::new();
    let mut i = len1;
    let mut j = len2;

    while i > 0 && j > 0 {
        // If the current characters match, this character is part of the LCS.
        // Add it and move diagonally up-left.
        if chars1[i - 1] == chars2[j - 1] {
            lcs_chars.push(chars1[i - 1]);
            i -= 1;
            j -= 1;
        } else if dp[i - 1][j] > dp[i][j - 1] {
            // If the character from s1 was skipped (dp[i-1][j] was larger), move up.
            i -= 1;
        } else {
            // If the character from s2 was skipped (dp[i][j-1] was larger), move left.
            j -= 1;
        }
    }

    // The LCS characters were collected in reverse order, so reverse them.
    lcs_chars.into_iter().rev().collect::<String>()
}

#[polars_expr(output_type=String)]
fn pl_lcs_subseq(inputs: &[Series], context: CallerContext) -> PolarsResult<Series> {
    let ca1 = inputs[0].str()?;
    let ca2 = inputs[1].str()?;
    let parallel = inputs[2].bool()?;
    let parallel = parallel.get(0).unwrap();
    let can_parallel = parallel && !context.parallel();
    if ca2.len() == 1 {
        let r = ca2.get(0).unwrap();
        if can_parallel {
            let ca = ca1
                .par_iter()
                .map(|ss| ss.map(|s| lcs_subseq_extract(s, r)))
                .collect::<StringChunked>();
            Ok(ca.into_series())
        } else {
            let ca = ca1.apply_values(|s| lcs_subseq_extract(s, r).into());
            Ok(ca.into_series())
        }
    } else if ca1.len() == ca2.len() {
        if can_parallel {
            let n_threads = POOL.current_num_threads();
            let splits = split_offsets(ca1.len(), n_threads);
            let chunks_iter = splits.into_par_iter().map(|(offset, len)| {
                let s1 = ca1.slice(offset as i64, len);
                let s2 = ca2.slice(offset as i64, len);
                let out: StringChunked = binary_elementwise_values(&s1, &s2, lcs_subseq_extract);
                out.downcast_iter().cloned().collect::<Vec<_>>()
            });
            let chunks = POOL.install(|| chunks_iter.collect::<Vec<_>>());
            let ca =
                StringChunked::from_chunk_iter(ca1.name().clone(), chunks.into_iter().flatten());
            Ok(ca.into_series())
        } else {
            let ca: StringChunked = binary_elementwise_values(ca1, ca2, lcs_subseq_extract);
            Ok(ca.into_series())
        }
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length or one of them must be a scalar.".into(),
        ))
    }
}

#[polars_expr(output_type=UInt32)]
fn pl_lcs_subseq_dist(inputs: &[Series], context: CallerContext) -> PolarsResult<Series> {
    let ca1 = inputs[0].str()?;
    let ca2 = inputs[1].str()?;
    let parallel = inputs[2].bool()?;
    let parallel = parallel.get(0).unwrap();
    let can_parallel = parallel && !context.parallel();
    if ca2.len() == 1 {
        let r = ca2.get(0).unwrap();
        let batched = lcs_seq::BatchComparator::new(r.chars());
        Ok(generic_batched_distance(batched, ca1, can_parallel))
    } else if ca1.len() == ca2.len() {
        Ok(generic_binary_distance(lcs_seq, ca1, ca2, can_parallel))
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length or one of them must be a scalar.".into(),
        ))
    }
}

#[polars_expr(output_type=Float64)]
fn pl_lcs_subseq_sim(inputs: &[Series], context: CallerContext) -> PolarsResult<Series> {
    let ca1 = inputs[0].str()?;
    let ca2 = inputs[1].str()?;
    let parallel = inputs[2].bool()?;
    let parallel = parallel.get(0).unwrap();
    let can_parallel = parallel && !context.parallel();
    if ca2.len() == 1 {
        let r = ca2.get(0).unwrap();
        let batched = lcs_seq::BatchComparator::new(r.chars());
        Ok(generic_batched_sim(batched, ca1, can_parallel))
    } else if ca1.len() == ca2.len() {
        Ok(generic_binary_sim(lcs_seq_sim, ca1, ca2, can_parallel))
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length or one of them must be a scalar.".into(),
        ))
    }
}
