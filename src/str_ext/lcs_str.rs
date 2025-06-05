use crate::utils::split_offsets;
use polars::prelude::{arity::binary_elementwise_values, *};
use pyo3_polars::{
    derive::{polars_expr, CallerContext},
    export::polars_core::{
        utils::rayon::iter::{IntoParallelIterator, ParallelIterator},
        POOL,
    },
};

/// Finds the longest common substring between two input strings.
///
/// This function uses a space-optimized dynamic programming approach to solve the
/// longest common substring problem. It constructs a 2D matrix conceptually,
/// but only stores two rows (`prev_row` and `curr_row`) at any given time,
/// where `curr_row[j]` stores the length of the longest common suffix
/// of `s1[0...i-1]` and `s2[0...j-1]`.
///
/// The space complexity is reduced from O(len1 * len2) to O(min(len1, len2)).
/// The time complexity remains O(len1 * len2).
///
/// See Wikipedia for more details
pub fn lcs_substr_extract(s1: &str, s2: &str) -> String {
    // Convert string slices to character vectors for easier indexing.
    // This handles multi-byte UTF-8 characters correctly.
    let c1: Vec<char> = s1.chars().collect();
    let c2: Vec<char> = s2.chars().collect();

    let (longer_chars, shorter_chars) = if c1.len() >= c2.len() {
        (c1, c2)
    } else {
        (c2, c1)
    };

    let longer_len = longer_chars.len();
    let shorter_len = shorter_chars.len();

    // Handle edge cases where one or both strings are empty.
    if longer_len == 0 || shorter_len == 0 {
        return String::new();
    }

    // Variables to keep track of the maximum length found and the
    // ending index of the longest common substring within `longer_chars`.
    let mut max_len = 0;
    let mut end_index_in_longer_chars = 0;

    // Use two rows for the dynamic programming table: `dp_prev` and `dp_curr`.
    // Each row's size is based on the length of the shorter string + 1.
    // This ensures space complexity is O(min(len1, len2)).
    let mut dp_prev = vec![0; shorter_len + 1];
    let mut dp_curr = vec![0; shorter_len + 1];

    // Fill the dp table using only two rows.
    // The outer loop iterates through the `longer_chars`.
    // The inner loop iterates through the `shorter_chars`.
    for i in 1..=longer_len {
        for j in 1..=shorter_len {
            // If the current characters from both strings match,
            // the length of the common suffix is extended.
            // It's 1 plus the value from the diagonal element in the `dp_prev` row.
            if longer_chars[i - 1] == shorter_chars[j - 1] {
                dp_curr[j] = 1 + dp_prev[j - 1];

                // If the newly calculated length is greater than the current maximum,
                // update `max_len` and store the ending index in `longer_chars`.
                if dp_curr[j] > max_len {
                    max_len = dp_curr[j];
                    end_index_in_longer_chars = i - 1; // 0-based index of the last char in `longer_chars`
                }
            } else {
                // If characters do not match, there is no common suffix ending at these positions,
                // so the length is 0.
                dp_curr[j] = 0;
            }
        }
        // After processing the current row, copy its contents to `dp_prev`
        dp_prev.copy_from_slice(&dp_curr);
    }

    // If `max_len` is 0, it means no common substring was found.
    if max_len == 0 {
        return String::new();
    }

    // Extract the longest common substring from `longer_chars`
    // using the calculated `max_len` and `end_index_in_longer_chars`.
    // The starting index is derived by subtracting `max_len` and adding 1.
    let start_index_in_longer_chars = end_index_in_longer_chars - max_len + 1;
    longer_chars[start_index_in_longer_chars..=end_index_in_longer_chars]
        .iter()
        .collect::<String>()
}

#[polars_expr(output_type=String)]
fn pl_lcs_substr(inputs: &[Series], context: CallerContext) -> PolarsResult<Series> {
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
                .map(|ss| ss.map(|s| lcs_substr_extract(s, r)))
                .collect::<StringChunked>();
            Ok(ca.into_series())
        } else {
            let ca = ca1.apply_values(|s| lcs_substr_extract(s, r).into());
            Ok(ca.into_series())
        }
    } else if ca1.len() == ca2.len() {
        if can_parallel {
            let n_threads = POOL.current_num_threads();
            let splits = split_offsets(ca1.len(), n_threads);
            let chunks_iter = splits.into_par_iter().map(|(offset, len)| {
                let s1 = ca1.slice(offset as i64, len);
                let s2 = ca2.slice(offset as i64, len);
                let out: StringChunked = binary_elementwise_values(&s1, &s2, lcs_substr_extract);
                out.downcast_iter().cloned().collect::<Vec<_>>()
            });
            let chunks = POOL.install(|| chunks_iter.collect::<Vec<_>>());
            let ca =
                StringChunked::from_chunk_iter(ca1.name().clone(), chunks.into_iter().flatten());
            Ok(ca.into_series())
        } else {
            let ca: StringChunked = binary_elementwise_values(ca1, ca2, lcs_substr_extract);
            Ok(ca.into_series())
        }
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length or one of them must be a scalar.".into(),
        ))
    }
}
