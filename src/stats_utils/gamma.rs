use num::Zero;

use super::{LN_2_SQRT_E_OVER_PI, LN_PI, PREC_ACC};

const GAMMA_R: f64 = 10.900511;

const GAMMA_DK: &[f64] = &[
    2.48574089138753565546e-5,
    1.05142378581721974210,
    -3.45687097222016235469,
    4.51227709466894823700,
    -2.98285225323576655721,
    1.05639711577126713077,
    -1.95428773191645869583e-1,
    1.70970543404441224307e-2,
    -5.71926117404305781283e-4,
    4.63399473359905636708e-6,
    -2.71994908488607703910e-9,
];

/// Calculates the survival function for the gamma
/// distribution at `x`
///
/// # Formula
///
/// ```ignore
/// (1 / Γ(α)) * γ(α, β * x)
/// ```
///
/// where `α` is the shape, `β` is the rate, `Γ` is the gamma function,
/// and `γ` is the upper incomplete gamma function
pub fn sf(x: f64, shape: f64, rate: f64) -> Result<f64, String> {
    if x <= 0.0 {
        Ok(1.0)
    } else if (x == shape) && rate.is_infinite() {
        Ok(0.0)
    } else if rate.is_infinite() {
        Ok(1.0)
    } else if x.is_infinite() {
        Ok(0.0)
    } else {
        checked_gamma_ur(shape, x * rate)
    }
}

/// Computes the logarithm of the gamma function
/// with an accuracy of 16 floating point digits.
/// The implementation is derived from
/// "An Analysis of the Lanczos Gamma Approximation",
/// Glendon Ralph Pugh, 2004 p. 116
pub fn ln_gamma(x: f64) -> f64 {
    if x < 0.5 {
        let s = GAMMA_DK
            .iter()
            .enumerate()
            .skip(1)
            .fold(GAMMA_DK[0], |s, t| s + t.1 / (t.0 as f64 - x));

        LN_PI
            - (std::f64::consts::PI * x).sin().ln()
            - s.ln()
            - LN_2_SQRT_E_OVER_PI
            - (0.5 - x) * ((0.5 - x + GAMMA_R) / std::f64::consts::E).ln()
    } else {
        let s = GAMMA_DK
            .iter()
            .enumerate()
            .skip(1)
            .fold(GAMMA_DK[0], |s, t| s + t.1 / (x + t.0 as f64 - 1.0));

        s.ln() + LN_2_SQRT_E_OVER_PI + (x - 0.5) * ((x - 0.5 + GAMMA_R) / std::f64::consts::E).ln()
    }
}

fn checked_gamma_lr(a: f64, x: f64) -> Result<f64, String> {
    if a.is_nan() || x.is_nan() {
        return Ok(f64::NAN);
    }
    if a <= 0.0 || a == f64::INFINITY {
        return Err("Gamma: Shape parameter alpha must be positive and not infinity.".into());
    }
    if x <= 0.0 || x == f64::INFINITY {
        return Err("Gamma: Input x must be positive and not infinity.".into());
    }

    let eps = 0.000000000000001;
    let big = 4503599627370496.0;
    let big_inv = 2.22044604925031308085e-16;

    if a.abs() < PREC_ACC {
        return Ok(1.0);
    }
    if x.abs() < PREC_ACC {
        return Ok(0.0);
    }

    let ax = a * x.ln() - x - ln_gamma(a);
    if ax < -709.78271289338399 {
        if a < x {
            return Ok(1.0);
        }
        return Ok(0.0);
    }
    if x <= 1.0 || x <= a {
        let mut r2 = a;
        let mut c2 = 1.0;
        let mut ans2 = 1.0;
        loop {
            r2 += 1.0;
            c2 *= x / r2;
            ans2 += c2;

            if c2 / ans2 <= eps {
                break;
            }
        }
        return Ok(ax.exp() * ans2 / a);
    }

    let mut y = 1.0 - a;
    let mut z = x + y + 1.0;
    let mut c = 0;

    let mut p3 = 1.0;
    let mut q3 = x;
    let mut p2 = x + 1.0;
    let mut q2 = z * x;
    let mut ans = p2 / q2;

    loop {
        y += 1.0;
        z += 2.0;
        c += 1;
        let yc = y * f64::from(c);

        let p = p2 * z - p3 * yc;
        let q = q2 * z - q3 * yc;

        p3 = p2;
        p2 = p;
        q3 = q2;
        q2 = q;

        if p.abs() > big {
            p3 *= big_inv;
            p2 *= big_inv;
            q3 *= big_inv;
            q2 *= big_inv;
        }

        if q != 0.0 {
            let nextans = p / q;
            let error = ((ans - nextans) / nextans).abs();
            ans = nextans;

            if error <= eps {
                break;
            }
        }
    }
    Ok(1.0 - ax.exp() * ans)
}

/// Upper incomplete gamma function
fn checked_gamma_ur(a: f64, x: f64) -> Result<f64, String> {
    if a.is_nan() || x.is_nan() {
        return Ok(f64::NAN);
    }
    if a <= 0.0 || a == f64::INFINITY {
        return Err("Gamma: Shape parameter alpha must be positive and not infinity.".into());
    }
    if x <= 0.0 || x == f64::INFINITY {
        return Err("Gamma: Input x must be positive and not infinity.".into());
    }

    let eps = 0.000000000000001;
    let big = 4503599627370496.0;
    let big_inv = 2.22044604925031308085e-16;

    if x < 1.0 || x <= a {
        return Ok(1.0 - checked_gamma_lr(a, x).unwrap());
    }

    let mut ax = a * x.ln() - x - ln_gamma(a);
    if ax < -709.78271289338399 {
        return if a < x { Ok(0.0) } else { Ok(1.0) };
    }

    ax = ax.exp();
    let mut y = 1.0 - a;
    let mut z = x + y + 1.0;
    let mut c = 0.0;
    let mut pkm2 = 1.0;
    let mut qkm2 = x;
    let mut pkm1 = x + 1.0;
    let mut qkm1 = z * x;
    let mut ans = pkm1 / qkm1;
    loop {
        y += 1.0;
        z += 2.0;
        c += 1.0;
        let yc = y * c;
        let pk = pkm1 * z - pkm2 * yc;
        let qk = qkm1 * z - qkm2 * yc;

        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;

        if pk.abs() > big {
            pkm2 *= big_inv;
            pkm1 *= big_inv;
            qkm2 *= big_inv;
            qkm1 *= big_inv;
        }

        if !qk.is_zero() {
            let r = pk / qk;
            let t = ((ans - r) / r).abs();
            ans = r;

            if t <= eps {
                break;
            }
        }
    }
    Ok(ans * ax)
}
