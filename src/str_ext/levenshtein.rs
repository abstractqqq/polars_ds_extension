use polars::prelude::{arity::binary_elementwise_values, *};
use pyo3_polars::{
    derive::polars_expr,
    export::polars_core::utils::rayon::prelude::{IndexedParallelIterator, ParallelIterator},
};
use strsim::{damerau_levenshtein, normalized_damerau_levenshtein, normalized_levenshtein};

// A slightly faster version than strsim's Levenshtein by dropping some abstractions
#[inline]
fn levenshtein(s1: &str, s2: &str) -> u32 {
    // Use chars to avoid mistakes for multi-byte characters
    let a: Vec<char> = s1.chars().collect();
    let b: Vec<char> = s2.chars().collect();

    let a_slice = a.as_slice();
    let b_slice = b.as_slice();

    let (a, b) = super::strip_common(a_slice, b_slice);

    let mut l1 = a.len();
    let mut l2 = b.len();

    if l1 == 0 {
        return l2 as u32;
    }
    if l2 == 0 {
        return l1 as u32;
    }

    let (a, b) = if l1 > l2 { (b, a) } else { (a, b) };
    (l1, l2) = (a.len(), b.len());

    let width = l2 + 1;
    // Using 1 buffer to represent two rows
    // Because we can keep updating the one buffer
    let mut buffer: Vec<usize> = (0..width).collect();
    // mid point of the verticle axis of the edit matrix
    let v_mid = (l1 + 1) >> 1; // (l1 + 1) / 2

    // buf[0]/buf[j] = cell to the left in the edit graph
    // buf[1]/buf[j+1] = the cell above
    // tmp = the cell in the upper left
    for i in 1..l1 + 1 {
        let mut tmp = buffer[0]; // cell to upper left
        buffer[0] = i; // cell to the left

        // Use Ukkonen's trick to reduce computations for the inner loop
        // I computed and proved (empirically) these start and end point values
        let (start, end) = (
            // if i < v_mid {1} else {1 + i % v_mid},
            1 + (i >= v_mid) as usize * (i % v_mid),
            width.min(width - (l1 >> 1) + i - 1),
        );
        // This cuts out the triangles in the upper right and lower left of the edit matrix
        for j in start..end {
            if a[i - 1] == b[j - 1] {
                // Equal, swap. buf[j] = cell above
                // Swap them. buf[j] will be tmp in the next round
                std::mem::swap(&mut tmp, &mut buffer[j]);
            } else {
                // char index is j - 1
                // buf[j] = cell above, buf[j-1] = cell to the left
                // tmp = cell upper left
                let val = buffer[j].min(buffer[j - 1]).min(tmp) + 1;
                tmp = buffer[j];
                buffer[j] = val;
            }
        }
    }
    buffer[l2] as u32
}

#[inline]
fn levenshtein_within(s1: &str, s2: &str, k: usize) -> bool {
    // Use chars to avoid mistakes for multi-byte characters
    let a: Vec<char> = s1.chars().collect();
    let b: Vec<char> = s2.chars().collect();

    let a_slice = a.as_slice();
    let b_slice = b.as_slice();

    let (a, b) = super::strip_common(a_slice, b_slice);

    let mut l1 = a.len();
    let mut l2 = b.len();

    if l1 == 0 {
        return l2 <= k;
    }
    if l2 == 0 {
        return l1 <= k;
    }
    if k < l1.abs_diff(l2) {
        return false;
    }

    let (a, b) = if l1 > l2 { (b, a) } else { (a, b) };
    (l1, l2) = (a.len(), b.len());

    let width = l2 + 1;

    let mut buffer: Vec<usize> = (0..width).collect();
    let v_mid = (l1 + 1) >> 1; // (l1 + 1) / 2

    for i in 1..l1 + 1 {
        let mut tmp = buffer[0];
        buffer[0] = i;

        let (start, end) = (
            1 + (i >= v_mid) as usize * (i % v_mid),
            width.min(width - (l1 >> 1) + i - 1),
        );
        let mut must_exceed: bool = true;
        for j in start..end {
            if a[i - 1] == b[j - 1] {
                std::mem::swap(&mut tmp, &mut buffer[j]);
            } else {
                let val = buffer[j].min(buffer[j - 1]).min(tmp) + 1;
                tmp = buffer[j];
                buffer[j] = val;
            }
            must_exceed &= buffer[j] > k; // if one false, we continue
        }
        // If all numbers in the row is > k, then return early
        if must_exceed {
            return false;
        }
    }
    buffer[l2] <= k
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
    Some(levenshtein(s1, s2))
}

#[inline]
fn optional_levenshtein_within(op_s1: Option<&str>, op_s2: Option<&str>, k: usize) -> Option<bool> {
    let s1 = op_s1?;
    let s2 = op_s2?;
    Some(levenshtein_within(s1, s2, k))
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
            ca1.apply_nonnull_values_generic(DataType::UInt32, |x| levenshtein(x, r))
        };
        Ok(out.into_series())
    } else if ca1.len() == ca2.len() {
        let out: UInt32Chunked = if parallel {
            ca1.par_iter_indexed()
                .zip(ca2.par_iter_indexed())
                .map(|(op_w1, op_w2)| optional_levenshtein(op_w1, op_w2))
                .collect()
        } else {
            binary_elementwise_values(ca1, ca2, |x, y| levenshtein(x, y))
        };
        Ok(out.into_series())
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length.".into(),
        ))
    }
}

#[polars_expr(output_type=Boolean)]
fn pl_levenshtein_within(inputs: &[Series]) -> PolarsResult<Series> {
    let ca1 = inputs[0].utf8()?;
    let ca2 = inputs[1].utf8()?;
    let bound = inputs[2].u32()?;
    let bound = bound.get(0).unwrap() as usize;
    let parallel = inputs[3].bool()?;
    let parallel = parallel.get(0).unwrap();
    if ca2.len() == 1 {
        let r = ca2.get(0);
        let out: BooleanChunked = if parallel {
            ca1.par_iter()
                .map(|op_s| optional_levenshtein_within(op_s, r, bound))
                .collect()
        } else {
            let r = r.unwrap();
            ca1.apply_nonnull_values_generic(DataType::Boolean, |x| levenshtein_within(x, r, bound))
        };
        Ok(out.into_series())
    } else if ca1.len() == ca2.len() {
        let out: BooleanChunked = if parallel {
            ca1.par_iter_indexed()
                .zip(ca2.par_iter_indexed())
                .map(|(op_w1, op_w2)| optional_levenshtein_within(op_w1, op_w2, bound))
                .collect()
        } else {
            binary_elementwise_values(ca1, ca2, |x, y| levenshtein_within(x, y, bound))
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
