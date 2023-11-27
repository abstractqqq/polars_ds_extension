use super::{simple_stats_output, Alternative, StatsResult};
use crate::stats::beta;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

#[inline]
fn ttest_ind(
    m1: f64,
    m2: f64,
    v1: f64,
    v2: f64,
    n: f64,
    alt: Alternative,
) -> Result<StatsResult, String> {
    let num = m1 - m2;
    // ((var1 + var2) / 2 ).sqrt() * (2./n).sqrt() can be simplified as below
    let denom = ((v1 + v2) / n).sqrt();
    if denom == 0. {
        Ok(StatsResult::new(f64::INFINITY, f64::NAN))
    } else {
        let t = num / denom;
        let df = 2. * n - 2.;
        let p = match alt {
            Alternative::Less => beta::student_t_sf(-t, df),
            Alternative::Greater => beta::student_t_sf(t, df),
            Alternative::TwoSided => match beta::student_t_sf(t.abs(), df) {
                Ok(p) => Ok(2.0 * p),
                Err(e) => Err(e),
            },
        };
        let p = p?;
        Ok(StatsResult::new(t, p))
    }
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
) -> Result<StatsResult, String> {
    let num = m1 - m2;
    let vn1 = v1 / n1;
    let vn2 = v2 / n2;
    let denom = (vn1 + vn2).sqrt();
    if denom == 0. {
        Ok(StatsResult::new(f64::INFINITY, f64::NAN))
    } else {
        let t = num / denom;
        let df = (vn1 + vn2).powi(2) / (vn1.powi(2) / (n1 - 1.) + (vn2.powi(2) / (n2 - 1.)));
        let p = match alt {
            // the distribution is approximately student t
            Alternative::Less => beta::student_t_sf(-t, df),
            Alternative::Greater => beta::student_t_sf(t, df),
            Alternative::TwoSided => match beta::student_t_sf(t.abs(), df) {
                Ok(p) => Ok(2.0 * p),
                Err(e) => Err(e),
            },
        };
        let p = p?;
        Ok(StatsResult::new(t, p))
    }
}

#[polars_expr(output_type_func=simple_stats_output)]
fn pl_student_t_2samp(inputs: &[Series]) -> PolarsResult<Series> {
    let mean1 = inputs[0].f64()?;
    let mean1 = mean1.get(0).unwrap();
    let mean2 = inputs[1].f64()?;
    let mean2 = mean2.get(0).unwrap();
    let var1 = inputs[2].f64()?;
    let var1 = var1.get(0).unwrap();
    let var2 = inputs[3].f64()?;
    let var2 = var2.get(0).unwrap();
    let n = inputs[4].u64()?;
    let n = n.get(0).unwrap() as f64;

    let alt = inputs[5].utf8()?;
    let alt = alt.get(0).unwrap();
    let alt = super::Alternative::from(alt);

    let valid = mean1.is_finite() && mean2.is_finite() && var1.is_finite() && var2.is_finite();
    if !valid {
        return Err(PolarsError::ComputeError(
            "T Test: Sample Mean/Std is found to be NaN or Inf.".into(),
        ));
    }

    let res = ttest_ind(mean1, mean2, var1, var2, n, alt)
        .map_err(|e| PolarsError::ComputeError(e.into()));
    let res = res?;

    let s = Series::from_vec("statistic", vec![res.statistic]);
    let pchunked = Float64Chunked::from_iter_options("pvalue", [res.p].into_iter());
    let p = pchunked.into_series();
    let out = StructChunked::new("", &[s, p])?;
    Ok(out.into_series())
}

#[polars_expr(output_type_func=simple_stats_output)]
fn pl_welch_t(inputs: &[Series]) -> PolarsResult<Series> {
    let mean1 = inputs[0].f64()?;
    let mean1 = mean1.get(0).unwrap();
    let mean2 = inputs[1].f64()?;
    let mean2 = mean2.get(0).unwrap();
    let var1 = inputs[2].f64()?;
    let var1 = var1.get(0).unwrap();
    let var2 = inputs[3].f64()?;
    let var2 = var2.get(0).unwrap();
    let n1 = inputs[4].u64()?;
    let n1 = n1.get(0).unwrap() as f64;
    let n2 = inputs[5].u64()?;
    let n2 = n2.get(0).unwrap() as f64;

    let alt = inputs[6].utf8()?;
    let alt = alt.get(0).unwrap();
    let alt = super::Alternative::from(alt);

    let valid = mean1.is_finite() && mean2.is_finite() && var1.is_finite() && var2.is_finite();
    if !valid {
        return Err(PolarsError::ComputeError(
            "T Test: Sample Mean/Std is found to be NaN or Inf.".into(),
        ));
    }

    let res = welch_t(mean1, mean2, var1, var2, n1, n2, alt)
        .map_err(|e| PolarsError::ComputeError(e.into()));
    let res = res?;

    let s = Series::from_vec("statistic", vec![res.statistic]);
    let pchunked = Float64Chunked::from_iter_options("pvalue", [res.p].into_iter());
    let p = pchunked.into_series();
    let out = StructChunked::new("", &[s, p])?;
    Ok(out.into_series())
}
