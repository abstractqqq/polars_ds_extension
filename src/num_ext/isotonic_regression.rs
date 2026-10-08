use itertools::Either;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

// Reference: https://github.com/scipy/scipy/blob/v1.14.1/scipy/optimize/_isotonic.py
// https://github.com/scipy/scipy/blob/v1.14.1/scipy/optimize/_pava/pava_pybind.cpp
// https://www.jstatsoft.org/article/view/v102c01

#[derive(Deserialize, Debug)]
pub(crate) struct IsotonicRegKwargs {
    pub(crate) has_weights: bool,
    pub(crate) increasing: bool,
}

fn isotonic_regression(x: &mut [f64], w: &mut [f64], r: &mut [usize]) {
    let n = x.len();
    r[0] = 0;
    r[1] = 1;
    let mut b: usize = 0;
    let mut xb_pre = x[b];
    let mut wb_pre = w[b];

    let mut i: usize = 1;

    while i < n {
        b += 1;
        let mut xb = x[i];
        let mut wb = w[i];
        if xb_pre >= xb {
            b -= 1;
            let mut sb = wb_pre * xb_pre + wb * xb;
            wb += wb_pre;
            xb = sb / wb;
            while i + 1 < n && xb >= x[i + 1] {
                i += 1;
                sb += w[i] * x[i];
                wb += w[i];
                xb = sb / wb;
            }
            while b > 0 && x[b - 1] >= xb {
                b -= 1;
                sb += w[b] * x[b];
                wb += w[b];
                xb = sb / wb;
            }
        }
        xb_pre = xb;
        x[b] = xb;

        wb_pre = wb;
        w[b] = wb;

        r[b + 1] = i + 1;
        i += 1;
    }

    let mut f = n - 1;
    for k in (0..=b).rev() {
        // println!("{}", k);
        let t = r[k];
        let xk = x[k];
        for i in t..=f {
            x[i] = xk;
        }
        // Avoid usize underflow on the final iteration when t == 0; f is unused after the loop.
        if k == 0 {
            break;
        }
        f = t - 1;
    }
}

// Unweighted PAVA: every element has weight 1.0, so the weight of block `k`
// spanning [r[k], r[k+1]) is identically `(r[k+1] - r[k]) as f64`.
// This avoids allocating a dummy `vec![1.0; n]` weight vector of size N * 8 bytes.
fn isotonic_regression_unweighted(x: &mut [f64], r: &mut [usize]) {
    let n = x.len();
    r[0] = 0;
    r[1] = 1;
    let mut b: usize = 0;
    let mut xb_pre = x[b];
    let mut wb_pre = 1.0f64;

    let mut i: usize = 1;

    while i < n {
        b += 1;
        let mut xb = x[i];
        let mut wb = 1.0f64;
        if xb_pre >= xb {
            b -= 1;
            let mut sb = wb_pre * xb_pre + xb;
            wb += wb_pre;
            xb = sb / wb;
            while i + 1 < n && xb >= x[i + 1] {
                i += 1;
                sb += x[i];
                wb += 1.0;
                xb = sb / wb;
            }
            while b > 0 && x[b - 1] >= xb {
                b -= 1;
                let prev_w = (r[b + 1] - r[b]) as f64;
                sb += prev_w * x[b];
                wb += prev_w;
                xb = sb / wb;
            }
        }
        xb_pre = xb;
        x[b] = xb;

        wb_pre = wb;

        r[b + 1] = i + 1;
        i += 1;
    }

    let mut f = n - 1;
    for k in (0..=b).rev() {
        let t = r[k];
        let xk = x[k];
        for i in t..=f {
            x[i] = xk;
        }
        if k == 0 {
            break;
        }
        f = t - 1;
    }
}

#[polars_expr(output_type=Float64)]
fn pl_isotonic_regression(inputs: &[Series], kwargs: IsotonicRegKwargs) -> PolarsResult<Series> {
    // Not sure why increasing = False doesn't give the right result
    let y = inputs[0].f64()?;
    let increasing = kwargs.increasing;

    if y.len() <= 1 {
        return Ok(y.clone().into_series());
    }

    let mut y = match y.to_vec_null_aware() {
        Either::Left(v) => Ok(v),
        Either::Right(_) => Err(PolarsError::ComputeError(
            "null values found in data".into(),
        )),
    }?;

    let has_weights = kwargs.has_weights; // True then weights are given, false then use 1s.
    if !increasing {
        y.reverse();
    }

    let mut r = vec![0; y.len() + 1];
    if has_weights {
        let weight = inputs[1].f64()?;
        if weight.len() != y.len() {
            return Err(PolarsError::ComputeError(
                "weights length does not match number of samples".into(),
            ));
        }
        let mut w = match weight.to_vec_null_aware() {
            Either::Left(mut w) => {
                if w.iter().any(|x| *x <= 0.) {
                    return Err(PolarsError::ComputeError(
                        "weights must be positive".into(),
                    ));
                }
                if !increasing {
                    w.reverse();
                }
                w
            }
            Either::Right(_) => {
                return Err(PolarsError::ComputeError(
                    "null values found in data".into(),
                ))
            }
        };
        isotonic_regression(&mut y, &mut w, &mut r);
    } else {
        // Zero-allocation unweighted path: avoids creating an N * 8 bytes dummy weight vector
        isotonic_regression_unweighted(&mut y, &mut r);
    }

    if !increasing {
        y.reverse();
    }

    let output = Float64Chunked::from_vec("".into(), y);
    Ok(output.into_series())
}
