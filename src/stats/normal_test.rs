/// Here we implement the test as in SciPy:
/// https://github.com/scipy/scipy/blob/v1.11.4/scipy/stats/_stats_py.py#L1836-L1996
///
/// It is a method based on Kurtosis and Skew, and the Chi-2 distribution.
///
/// References:
/// [1] D'Agostino, R. B. (1971), "An omnibus test of normality for
///     moderate and large sample size", Biometrika, 58, 341-348
/// [2] https://www.stata.com/manuals/rsktest.pdf
///
/// I chose this over the Shapiro Francia test because the distribution is unknown and would require Monte Carlo
use super::{simple_stats_output, StatsResult};
use crate::stats_utils::{gamma, is_zero};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

/// Returns the skew test statistic, no pvalue, add p value if needed
fn skew_test_statistic(skew: f64, n: usize) -> Result<StatsResult, String> {
    let n = n as f64;
    let y = skew * ((n + 1.) * (n + 3.) / (6. * (n - 2.))).sqrt();
    let beta2 = 3. * (n.powi(2) + 27. * n - 70.) * (n + 1.) * (n + 3.)
        / ((n - 2.) * (n + 5.) * (n + 7.) * (n + 9.));
    let w2 = (2. * (beta2 - 1.)).sqrt() - 1.;
    let alpha = (2. / (w2 - 1.)).sqrt();

    let tmp = y / alpha;
    let z = (tmp + (tmp.powi(2) + 1.).sqrt()).ln() / (w2.ln() * 0.5).sqrt();
    Ok(StatsResult::from_stats(z))
}

/// Returns the kurtosis test statistic, no pvalue, add p value if needed
fn kurtosis_test_statistic(kur: f64, n: usize) -> Result<StatsResult, String> {
    let n = n as f64;
    let e = 3.0 * (n - 1.) / (n + 1.);
    let var = 24.0 * n * (n - 2.) * (n - 3.) / ((n + 1.).powi(2) * (n + 3.) * (n + 5.));
    let x = (kur - e) / var.sqrt();
    let root_beta_1 = 6. * (n.powi(2) - 5. * n + 2.) / ((n + 7.) * (n + 9.));
    let root_beta_2 = (6. * (n + 3.) * (n + 5.) / (n * (n - 2.) * (n - 3.))).sqrt();
    let root_beta = root_beta_1 * root_beta_2;

    let a = 6. + (8. / root_beta) * (2. / root_beta + (1. + 4. / root_beta.powi(2)).sqrt());

    let tmp = 2. / (9. * a);
    let denom = 1. + x * (2. / (a - 4.)).sqrt();
    if is_zero(denom) {
        Err("Kurtosis test: Division by 0 encountered.".to_owned())
    } else {
        let term1 = 1. - tmp;
        let term2 = ((1. - 2. / a) / denom.abs()).cbrt();
        let z = (term1 - term2) / tmp.sqrt();
        Ok(StatsResult::from_stats(z))
    }
}

#[polars_expr(output_type_func=simple_stats_output)]
fn pl_normal_test(inputs: &[Series]) -> PolarsResult<Series> {
    let skew = inputs[0].f64()?;
    let skew = skew.get(0).unwrap();

    let kurtosis = inputs[1].f64()?;
    let kurtosis = kurtosis.get(0).unwrap();

    let n = inputs[2].u64()?;
    let n = n.get(0).unwrap() as usize;

    if n < 20 {
        return Err(PolarsError::ComputeError(
            "Normal Test: Input should have non-null length >= 20.".into(),
        ));
    }

    let s = skew_test_statistic(skew, n).map_err(|err| PolarsError::ComputeError(err.into()));
    let k =
        kurtosis_test_statistic(kurtosis, n).map_err(|err| PolarsError::ComputeError(err.into()));

    let s = s?.statistic; // the statistics
    let k = k?.statistic; // the statistics

    let k2 = s * s + k * k; // the statistics

    // Define gamma
    // Shape = (degree of freedom (2) / 2, rate = 0.5)
    let (shape, rate) = (1., 0.5);
    let p = gamma::sf(k2, shape, rate).map_err(|e| PolarsError::ComputeError(e.into()))?;

    let s = Series::from_vec("statistic", vec![k2]);
    let p = Series::from_vec("pvalue", vec![p]);
    let out = StructChunked::new("", &[s, p])?;
    Ok(out.into_series())
}
