/// Student's t test and Welch's t test.
use super::{
    generic_stats_output, simple_stats_output, Alternative,
};
use crate::{stats, stats_utils::beta};
use core::f64;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;


// Under these conditions, the output of a welch_t test will be bad.
// However, if any of the following is true, we would get NaN in the calculation below,
// So no need to branch. ALso, returning NaN should tell the user that there are numeric 
// issues encountered in the process.
// if m1.is_nan()
//     || m1.is_infinite()
//     || m2.is_nan()
//     || m2.is_infinite()
//     || v1.is_nan()
//     || v1.is_infinite()
//     || v2.is_nan()
//     || v2.is_infinite()
//     || v1 <= 0.0
//     || v2 <= 0.0
//     || n1 == 0.0
//     || n2 == 0.0


#[inline]
fn ttest_ind(m1: f64, m2: f64, v1: f64, v2: f64, n: f64, alt: Alternative) -> (f64, f64) {
    let num = m1 - m2;
    // ((var1 + var2) / 2 ).sqrt() * (2./n).sqrt() can be simplified as below
    let denom = ((v1 + v2) / n).sqrt();

    let t = num / denom;
    let df = 2. * n - 2.;
    let p = match alt {
        Alternative::Less => beta::student_t_sf(-t, df).unwrap_or(f64::NAN),
        Alternative::Greater => beta::student_t_sf(t, df).unwrap_or(f64::NAN),
        Alternative::TwoSided => match beta::student_t_sf(t.abs(), df) {
            Ok(p) => 2.0 * p,
            Err(_) => f64::NAN,
        },
    };
    (t, p)
}


#[inline]
fn ttest_1samp(
    mean: f64,
    pop_mean: f64,
    var: f64,
    n: f64,
    alt: Alternative,
) -> (f64, f64) {

    let num = mean - pop_mean;
    let denom = (var / n).sqrt();
    let t = num / denom;
    let df = n - 1.;
    let p = match alt {
        Alternative::Less => beta::student_t_sf(-t, df).unwrap_or(f64::NAN),
        Alternative::Greater => beta::student_t_sf(t, df).unwrap_or(f64::NAN),
        Alternative::TwoSided => match beta::student_t_sf(t.abs(), df) {
            Ok(p) => 2.0 * p,
            Err(_) => f64::NAN,
        },
    };
    (t, p)
}

#[inline]
fn welch_t(
    m1: f64,
    m2: f64,
    v1: f64,
    v2: f64,
    n1: f64,
    n2: f64,
    alt: Alternative,
) -> (f64, f64) {

    let num = m1 - m2;
    let vn1 = v1 / n1;
    let vn2 = v2 / n2;
    let denom = (vn1 + vn2).sqrt();
    let t = num / denom;
    let df = (vn1 + vn2).powi(2) / (vn1.powi(2) / (n1 - 1.) + (vn2.powi(2) / (n2 - 1.)));
    let p = match alt {
        // the distribution is approximately student t
        Alternative::Less => beta::student_t_sf(-t, df).unwrap_or(f64::NAN),
        Alternative::Greater => beta::student_t_sf(t, df).unwrap_or(f64::NAN),
        Alternative::TwoSided => match beta::student_t_sf(t.abs(), df) {
            Ok(p) => 2.0 * p,
            Err(_) => f64::NAN,
        },
    };

    (t, p)

}

#[polars_expr(output_type_func=simple_stats_output)]
fn pl_ttest_2samp(inputs: &[Series]) -> PolarsResult<Series> {
    let mean1 = inputs[0].f64()?;
    let mean1 = mean1.get(0).unwrap_or(f64::NAN);
    let mean2 = inputs[1].f64()?;
    let mean2 = mean2.get(0).unwrap_or(f64::NAN);
    let var1 = inputs[2].f64()?;
    let var1 = var1.get(0).unwrap_or(f64::NAN);
    let var2 = inputs[3].f64()?;
    let var2 = var2.get(0).unwrap_or(f64::NAN);
    let n = inputs[4].u64()?;
    let n = n.get(0).unwrap() as f64;

    let alt = inputs[5].str()?;
    let alt = alt.get(0).unwrap();
    let alt = stats::Alternative::from(alt);

    // See comment above for why there is no finiteness or n > 0 checks

    let (t, p) = ttest_ind(mean1, mean2, var1, var2, n, alt);
    generic_stats_output(t, p)

}

#[polars_expr(output_type_func=simple_stats_output)]
fn pl_welch_t(inputs: &[Series]) -> PolarsResult<Series> {
    let mean1 = inputs[0].f64()?;
    let mean1 = mean1.get(0).unwrap_or(f64::NAN);
    let mean2 = inputs[1].f64()?;
    let mean2 = mean2.get(0).unwrap_or(f64::NAN);
    let var1 = inputs[2].f64()?;
    let var1 = var1.get(0).unwrap_or(f64::NAN);
    let var2 = inputs[3].f64()?;
    let var2 = var2.get(0).unwrap_or(f64::NAN);
    let n1 = inputs[4].u64()?;
    let n1 = n1.get(0).map_or(f64::NAN, |x| x as f64);
    let n2 = inputs[5].u64()?;
    let n2 = n2.get(0).map_or(f64::NAN, |x| x as f64);

    let alt = inputs[6].str()?;
    let alt = alt.get(0).unwrap();
    let alt = stats::Alternative::from(alt);

    // See comment above for why there is no finiteness or n > 0 checks

    let (s, p) = welch_t(mean1, mean2, var1, var2, n1, n2, alt);
    generic_stats_output(s, p)
}

#[polars_expr(output_type_func=simple_stats_output)]
fn pl_ttest_1samp(inputs: &[Series]) -> PolarsResult<Series> {
    let mean = inputs[0].f64()?;
    let mean = mean.get(0).unwrap_or(f64::NAN);
    let pop_mean = inputs[1].f64()?;
    let pop_mean = pop_mean.get(0).unwrap_or(f64::NAN);
    let var = inputs[2].f64()?;
    let var = var.get(0).unwrap_or(f64::NAN);
    let n = inputs[3].u64()?;
    let n = n.get(0).map_or(f64::NAN, |x| x as f64);

    let alt = inputs[4].str()?;
    let alt = alt.get(0).unwrap();
    let alt = stats::Alternative::from(alt);

    // See comment above for why there is no finiteness or n > 0 checks

    let (t, p) = ttest_1samp(mean, pop_mean, var, n, alt);
    generic_stats_output(t, p)
}
