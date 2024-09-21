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

/// Computes the natural logarithm
/// of the beta function
/// where `a` is the first beta parameter
/// and `b` is the second beta parameter
/// and `a > 0`, `b > 0`.
pub fn ln_beta(a: f64, b: f64) -> f64 {
    if a <= 0.0 || b <= 0.0 {
        f64::NAN
    } else {
        gamma::ln_gamma(a) + gamma::ln_gamma(b) - gamma::ln_gamma(a + b)
    }
}

/// Computes the inverse of the regularized incomplete beta function
// This code is based on the implementation in the ["special"][1] crate,
// which in turn is based on a [C implementation][2] by John Burkardt. The
// original algorithm was published in Applied Statistics and is known as
// [Algorithm AS 64][3] and [Algorithm AS 109][4].
//
// [1]: https://docs.rs/special/0.8.1/
// [2]: http://people.sc.fsu.edu/~jburkardt/c_src/asa109/asa109.html
// [3]: http://www.jstor.org/stable/2346798
// [4]: http://www.jstor.org/stable/2346887
//
// > Copyright 2014–2019 The special Developers
// >
// > Permission is hereby granted, free of charge, to any person obtaining a copy of
// > this software and associated documentation files (the “Software”), to deal in
// > the Software without restriction, including without limitation the rights to
// > use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// > the Software, and to permit persons to whom the Software is furnished to do so,
// > subject to the following conditions:
// >
// > The above copyright notice and this permission notice shall be included in all
// > copies or substantial portions of the Software.
// >
// > THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// > IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// > FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// > COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// > IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// > CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
pub fn inv_beta_reg(mut a: f64, mut b: f64, mut x: f64) -> f64 {
    // Algorithm AS 64
    // http://www.jstor.org/stable/2346798
    //
    // An approximation x₀ to x if found from (cf. Scheffé and Tukey, 1944)
    //
    // 1 + x₀   4p + 2q - 2
    // ------ = -----------
    // 1 - x₀      χ²(α)
    //
    // where χ²(α) is the upper α point of the χ² distribution with 2q
    // degrees of freedom and is obtained from Wilson and Hilferty’s
    // approximation (cf. Wilson and Hilferty, 1931)
    //
    // χ²(α) = 2q (1 - 1 / (9q) + y(α) sqrt(1 / (9q)))^3,
    //
    // y(α) being Hastings’ approximation (cf. Hastings, 1955) for the upper
    // α point of the standard normal distribution. If χ²(α) < 0, then
    //
    // x₀ = 1 - ((1 - α)q B(p, q))^(1 / q).
    //
    // Again if (4p + 2q - 2) / χ²(α) does not exceed 1, x₀ is obtained from
    //
    // x₀ = (αp B(p, q))^(1 / p).
    //
    // The final solution is obtained by the Newton–Raphson method from the
    // relation
    //
    //                    f(x[i - 1])
    // x[i] = x[i - 1] - ------------
    //                   f'(x[i - 1])
    //
    // where
    //
    // f(x) = I(x, p, q) - α.
    let ln_beta = ln_beta(a, b);

    // Remark AS R83
    // http://www.jstor.org/stable/2347779
    const SAE: i32 = -30;
    const FPU: f64 = 1e-30; // 10^SAE

    debug_assert!((0.0..=1.0).contains(&x) && a > 0.0 && b > 0.0);

    if x == 0.0 {
        return 0.0;
    }
    if x == 1.0 {
        return 1.0;
    }

    let mut p;
    let mut q;

    let flip = 0.5 < x;
    if flip {
        p = a;
        a = b;
        b = p;
        x = 1.0 - x;
    }

    p = (-(x * x).ln()).sqrt();
    q = p - (2.30753 + 0.27061 * p) / (1.0 + (0.99229 + 0.04481 * p) * p);

    if 1.0 < a && 1.0 < b {
        // Remark AS R19 and Algorithm AS 109
        // http://www.jstor.org/stable/2346887
        //
        // For a and b > 1, the approximation given by Carter (1947), which
        // improves the Fisher–Cochran formula, is generally better. For
        // other values of a and b en empirical investigation has shown that
        // the approximation given in AS 64 is adequate.
        let r = (q * q - 3.0) / 6.0;
        let s = 1.0 / (2.0 * a - 1.0);
        let t = 1.0 / (2.0 * b - 1.0);
        let h = 2.0 / (s + t);
        let w = q * (h + r).sqrt() / h - (t - s) * (r + 5.0 / 6.0 - 2.0 / (3.0 * h));
        p = a / (a + b * (2.0 * w).exp());
    } else {
        let mut t = 1.0 / (9.0 * b);
        t = 2.0 * b * (1.0 - t + q * t.sqrt()).powf(3.0);
        if t <= 0.0 {
            p = 1.0 - ((((1.0 - x) * b).ln() + ln_beta) / b).exp();
        } else {
            t = 2.0 * (2.0 * a + b - 1.0) / t;
            if t <= 1.0 {
                p = (((x * a).ln() + ln_beta) / a).exp();
            } else {
                p = 1.0 - 2.0 / (t + 1.0);
            }
        }
    }

    p = p.clamp(0.0001, 0.9999);

    // Remark AS R83
    // http://www.jstor.org/stable/2347779
    let e = (-5.0 / a / a - 1.0 / x.powf(0.2) - 13.0) as i32;
    let acu = if e > SAE { f64::powi(10.0, e) } else { FPU };

    let mut pnext;
    let mut qprev = 0.0;
    let mut sq = 1.0;
    let mut prev = 1.0;

    'outer: loop {
        // Remark AS R19 and Algorithm AS 109
        // http://www.jstor.org/stable/2346887
        q = checked_beta_reg(a, b, p).unwrap();
        q = (q - x) * (ln_beta + (1.0 - a) * p.ln() + (1.0 - b) * (1.0 - p).ln()).exp();

        // Remark AS R83
        // http://www.jstor.org/stable/2347779
        if q * qprev <= 0.0 {
            prev = if sq > FPU { sq } else { FPU };
        }

        // Remark AS R19 and Algorithm AS 109
        // http://www.jstor.org/stable/2346887
        let mut g = 1.0;
        loop {
            loop {
                let adj = g * q;
                sq = adj * adj;

                if sq < prev {
                    pnext = p - adj;
                    if pnext >= 0. && pnext <= 1. {
                        break;
                    }
                }
                g /= 3.0;
            }

            if prev <= acu || q * q <= acu {
                p = pnext;
                break 'outer;
            }

            if pnext != 0.0 && pnext != 1.0 {
                break;
            }

            g /= 3.0;
        }

        if pnext == p {
            break;
        }

        p = pnext;
        qprev = q;
    }

    if flip {
        1.0 - p
    } else {
        p
    }
}

/// Calculates the inverse cumulative distribution function for the
/// Student's T-distribution at `x`
pub fn student_t_ppf(x: f64, df: f64) -> f64 {
    // first calculate inverse_cdf for normal Student's T
    assert!((0.0..=1.0).contains(&x));
    let x1 = if x >= 0.5 { 1.0 - x } else { x };
    let a = 0.5 * df;
    let b = 0.5;
    let mut y = inv_beta_reg(a, b, 2.0 * x1);
    y = (df * (1. - y) / y).sqrt();
    if x >= 0.5 {
        y
    } else {
        -y
    }
    // generalised Student's T is related to normal Student's T by `Y = μ + σ X`
    // where `X` is distributed as Student's T, so this result has to be scaled and shifted back
    // formally: F_Y(t) = P(Y <= t) = P(X <= (t - μ) / σ) = F_X((t - μ) / σ)
    // F_Y^{-1}(p) = inf { t' | F_Y(t') >= p } = inf { t' = μ + σ t | F_X((t' - μ) / σ) >= p }
    // because scale is positive: loc + scale * t is strictly monotonic function
    // = μ + σ inf { t | F_X(t) >= p } = μ + σ F_X^{-1}(p)

    // In our case, use location = 0, scale = 1
    // self.location + self.scale * y
}
