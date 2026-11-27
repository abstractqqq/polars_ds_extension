//! Smoothed pinball majorization-minimization (IRLS), with a numerical dual-gap check.
use faer::{linalg::solvers::SolveLstsq, Mat, MatRef};

/// Fit a full-column-rank design. See docs/expr_linear.md for the surrogate and dual.
/// Columns and the response are scaled internally; the intercept, if any, is just
/// the last design column, supplied by the expression adapter.
pub fn quantile_regression(
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    quantile: f64,
    tol: f64,
    max_iter: usize,
) -> Result<Vec<f64>, String> {
    let (n, p) = x.shape();
    if n == 0 || p == 0 || n < p || y.nrows() != n || y.ncols() != 1 {
        return Err("quantile regression requires nonempty data and samples >= features".into());
    }
    if !(0.0 < quantile && quantile < 1.0) || !tol.is_finite() || tol <= 0.0 || max_iter == 0 {
        return Err("invalid quantile regression quantile, tol or max_iter".into());
    }
    let mut scales = vec![0.0_f64; p];
    let mut y_scale = 0.0_f64;
    for i in 0..n {
        if !y[(i, 0)].is_finite() {
            return Err("quantile regression requires finite input values".into());
        }
        y_scale = y_scale.max(y[(i, 0)].abs());
        for j in 0..p {
            if !x[(i, j)].is_finite() {
                return Err("quantile regression requires finite input values".into());
            }
            scales[j] = scales[j].max(x[(i, j)].abs());
        }
    }
    if scales.contains(&0.0) {
        return Err("quantile regression requires a full-column-rank design".into());
    }
    y_scale = if y_scale == 0.0 { 1.0 } else { y_scale };
    let x = Mat::from_fn(n, p, |i, j| x[(i, j)] / scales[j]);
    let y = Mat::from_fn(n, 1, |i, _| y[(i, 0)] / y_scale);
    let qr = x.col_piv_qr();
    let r = qr.R();
    let rank_tol = f64::EPSILON * n.max(p) as f64 * r[(0, 0)].abs();
    if (0..p).any(|j| r[(j, j)].abs() <= rank_tol) {
        return Err("quantile regression requires a full-column-rank design".into());
    }
    // A thin Q avoids squaring the condition number in the dual projection.
    let basis = qr.compute_thin_Q();
    let mut beta = qr.solve_lstsq(&y);
    let smoothing = (0.05 * tol).max(1e-12);
    let mut smooth_abs = vec![0.0_f64; n];
    let mut dual = vec![0.0_f64; n];
    let mut weighted_x = Mat::zeros(n, p);
    let mut working_y = Mat::zeros(n, 1);

    // Check the initial least-squares fit and then each of max_iter IRLS updates.
    for iteration in 0..=max_iter {
        let mut primal = 0.0;
        for i in 0..n {
            let pred: f64 = (0..p).map(|j| x[(i, j)] * beta[(j, 0)]).sum();
            let residual = y[(i, 0)] - pred;
            let s = residual.hypot(smoothing);
            if !s.is_finite() {
                return Err("quantile regression encountered a non-finite iterate".into());
            }
            smooth_abs[i] = s;
            primal += (quantile * residual).max((quantile - 1.0) * residual) / n as f64;
            dual[i] = quantile - 0.5 + 0.5 * residual / s;
        }
        // Project the smoothed score onto ker(X^T). Reorthogonalize to reduce
        // cancellation, then scale toward zero to satisfy q-1 <= dual <= q.
        for _ in 0..2 {
            for j in 0..p {
                let projection: f64 = (0..n).map(|i| basis[(i, j)] * dual[i]).sum();
                for i in 0..n {
                    dual[i] -= basis[(i, j)] * projection;
                }
            }
        }
        let mut factor = 1.0_f64;
        for &u in &dual {
            if u > quantile {
                factor = factor.min(quantile / u);
            } else if u < quantile - 1.0 {
                factor = factor.min((quantile - 1.0) / u);
            }
        }
        factor *= 1.0 - 8.0 * f64::EPSILON;
        let lower: f64 = (0..n)
            .map(|i| y[(i, 0)] * (factor * dual[i]) / n as f64)
            .sum();
        let gap = primal - lower;
        let roundoff = 128.0 * f64::EPSILON * (1.0 + primal);
        if !gap.is_finite() || gap < -roundoff {
            return Err("quantile regression failed its numerical dual-gap check".into());
        }
        if gap <= tol * (1.0 + primal) {
            let mut coefficients = Vec::with_capacity(p);
            for j in 0..p {
                let ratio = y_scale / scales[j];
                let coefficient = if ratio.is_finite() && ratio != 0.0 {
                    beta[(j, 0)] * ratio
                } else {
                    (beta[(j, 0)] * y_scale) / scales[j]
                };
                if !coefficient.is_finite() || (coefficient == 0.0 && beta[(j, 0)] != 0.0) {
                    return Err("quantile regression coefficients exceed the Float64 range".into());
                }
                coefficients.push(coefficient);
            }
            return Ok(coefficients);
        }
        if iteration == max_iter {
            break;
        }
        // sqrt(r^2 + delta^2) has a tangent majorizer in r^2. Completing
        // the square yields weights 1/s and target y + (2q-1)s.
        for i in 0..n {
            let weight = 1.0 / smooth_abs[i].sqrt();
            working_y[(i, 0)] = (y[(i, 0)] + (2.0 * quantile - 1.0) * smooth_abs[i]) * weight;
            for j in 0..p {
                weighted_x[(i, j)] = x[(i, j)] * weight;
            }
        }
        beta = weighted_x.col_piv_qr().solve_lstsq(&working_y);
    }
    Err(format!("quantile regression did not converge within {max_iter} iterations; increase max_iter or relax tol"))
}
