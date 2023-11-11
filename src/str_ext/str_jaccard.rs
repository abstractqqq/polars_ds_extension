use hashbrown::HashSet;
use polars::prelude::{*, arity::binary_elementwise_values};
use pyo3_polars::{
    derive::polars_expr, 
    export::polars_core::utils::rayon::prelude::{ParallelIterator, IndexedParallelIterator}
};
use std::str;

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
