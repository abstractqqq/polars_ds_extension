use super::{gamma, normal, PREC_ACC};
use num::Zero;

/// Calculates the cumulative distribution function for the student's
/// t-distribution
/// at `x`
///
/// # Formula
///
/// ```ignore
/// if x < μ {
///     1 - (1 / 2) * I(t, v / 2, 1 / 2)
/// } else {
///     (1 / 2) * I(t, v / 2, 1 / 2)
/// }
/// ```
///
/// where `t = v / (v + k^2)`, `k = (x - μ) / σ`, `μ` is the location,
/// `σ` is the scale, `v` is the freedom, and `I` is the regularized
/// incomplete beta function
///
/// Since I am only working with standard student t's distribution,
/// location = 0 and scale = 1.
pub fn student_t_sf(x: f64, df: f64) -> Result<f64, String> {
    if df.is_infinite() {
        Ok(normal::sf_unchecked(x, 0., 1.))
    } else {
        // let k = (x - location) / scale;
        let h = df / (df + x * x); // freedom / (freedom + k * k);
        let ib = 0.5 * checked_beta_reg(df / 2.0, 0.5, h)?;
        if x <= 0. {
            Ok(1.0 - ib)
        } else {
            Ok(ib)
        }
    }
}

/// Calculates the survival function for the fisher-snedecor
/// distribution at `x`
///
/// # Formula
///
/// ```ignore
/// I_(1 - ((d1 * x) / (d1 * x + d2))(d2 / 2, d1 / 2)
/// ```
///
/// where `d1` is the first degree of freedom, `d2` is
/// the second degree of freedom, and `I` is the regularized incomplete
/// beta function
pub fn fisher_snedecor_sf(x: f64, freedom_1: f64, freedom_2: f64) -> Result<f64, String> {
    if x < 0.0 {
        Err("F stats found to be < 0. This should be impossible.".into())
    } else if x.is_infinite() {
        Ok(0.)
    } else {
        let t = freedom_1 * x;
        checked_beta_reg(freedom_2 / 2.0, freedom_1 / 2.0, 1. - (t / (t + freedom_2)))
    }
}

fn checked_beta_reg(a: f64, b: f64, x: f64) -> Result<f64, String> {
    // a, degree of freedom
    // b, shape parameter
    if a <= 0.0 {
        Err("Beta: Shape parameter alpha must be positive.".into())
    } else if b <= 0.0 {
        Err("Beta: Shape parameter beta must be positive.".into())
    } else if !(0.0..=1.0).contains(&x) {
        Err("Beta: Input x must be between 0 and 1.".into())
    } else {
        let bt = if x.is_zero() || (x == 1.0) {
            0.0
        } else {
            (gamma::ln_gamma(a + b) - gamma::ln_gamma(a) - gamma::ln_gamma(b)
                + a * x.ln()
                + b * (1.0 - x).ln())
            .exp()
        };
        let symm_transform = x >= (a + 1.0) / (a + b + 2.0);
        let eps = PREC_ACC;
        let fpmin = f64::MIN_POSITIVE / eps;

        let mut a = a;
        let mut b = b;
        let mut x = x;
        if symm_transform {
            std::mem::swap(&mut a, &mut b);
            x = 1.0 - x;
            // let swap = a;
            // a = b;
            // b = swap;
        }

        let qab = a + b;
        let qap = a + 1.0;
        let qam = a - 1.0;
        let mut c = 1.0;
        let mut d = 1.0 - qab * x / qap;

        if d.abs() < fpmin {
            d = fpmin;
        }
        d = 1.0 / d;
        let mut h = d;

        // Maybe one day I will research into these magic numbers a bit more...
        for m in 1..141 {
            let m = f64::from(m);
            let m2 = m * 2.0;
            let mut aa = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1.0 + aa * d;

            if d.abs() < fpmin {
                d = fpmin;
            }

            c = 1.0 + aa / c;
            if c.abs() < fpmin {
                c = fpmin;
            }

            d = 1.0 / d;
            h = h * d * c;
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1.0 + aa * d;

            if d.abs() < fpmin {
                d = fpmin;
            }

            c = 1.0 + aa / c;

            if c.abs() < fpmin {
                c = fpmin;
            }

            d = 1.0 / d;
            let del = d * c;
            h *= del;

            if (del - 1.0).abs() <= eps {
                return if symm_transform {
                    Ok(1.0 - bt * h / a)
                } else {
                    Ok(bt * h / a)
                };
            }
        }

        if symm_transform {
            Ok(1.0 - bt * h / a)
        } else {
            Ok(bt * h / a)
        }
    }
}
