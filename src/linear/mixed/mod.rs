#![allow(non_snake_case)]
//! Random-intercept linear mixed model, fit via restricted maximum likelihood (REML).
//!
//! The model is
//!
//!     y = X @ beta + Z @ u + e,   u ~ N(0, sigma_g^2 I),   e ~ N(0, sigma_e^2 I)
//!
//! where `Z` is the indicator (dummy) design matrix for the grouping factor, i.e. one
//! random intercept per group level. Estimation profiles the restricted log-likelihood
//! over the variance ratio `gamma = sigma_g^2 / sigma_e^2` with a golden-section search,
//! then reports the generalized least squares (GLS) solution for `beta` at the optimum.
//!
//! `Z` is never materialized: since `Z' Z = diag(group_count)`, the Woodbury identity
//! turns every `H^-1 = (I + gamma Z Z')^-1` apply and `slogdet(H)` into an O(n) pass over
//! group sums, rather than the O(n^3) dense `n x n` inverse this would otherwise require.

use faer::{linalg::solvers::Solve, mat::Mat, prelude::*, MatRef, Side};

pub struct MixedModelResult {
    pub coeffs: Vec<f64>,
    pub std_errors: Vec<f64>,
    pub dfs: Vec<f64>,
    pub gamma: f64,
    pub resid_variance: f64,
}

/// Per-group counts and the `1 / (1 + gamma * count)` terms needed to apply `H^-1`
/// via Woodbury without ever building `Z` or `H`.
struct GroupInfo {
    codes: Vec<usize>,
    counts: Vec<f64>,
}

impl GroupInfo {
    fn new(codes: &[usize], n_groups: usize) -> Self {
        let mut counts = vec![0.0; n_groups];
        for &g in codes {
            counts[g] += 1.0;
        }
        GroupInfo {
            codes: codes.to_vec(),
            counts,
        }
    }

    /// Applies `(I + gamma * Z Z')^-1` to a vector via Woodbury:
    /// `Hi v = v - gamma * Z (I + gamma Z'Z)^-1 Z' v`, and `Z'Z = diag(counts)`.
    fn apply_hi(&self, v: &[f64], gamma: f64) -> Vec<f64> {
        let mut group_sum = vec![0.0; self.counts.len()];
        for (i, &vi) in v.iter().enumerate() {
            group_sum[self.codes[i]] += vi;
        }
        let scaled: Vec<f64> = group_sum
            .iter()
            .zip(self.counts.iter())
            .map(|(&gs, &cnt)| gamma * gs / (1.0 + gamma * cnt))
            .collect();
        v.iter()
            .enumerate()
            .map(|(i, &vi)| vi - scaled[self.codes[i]])
            .collect()
    }

    fn apply_hi_mat(&self, x: MatRef<f64>, gamma: f64) -> Mat<f64> {
        let cols: Vec<Vec<f64>> = (0..x.ncols())
            .map(|j| {
                let col: Vec<f64> = x.col(j).iter().copied().collect();
                self.apply_hi(&col, gamma)
            })
            .collect();
        Mat::from_fn(x.nrows(), x.ncols(), |i, j| cols[j][i])
    }

    /// `sum_g log(1 + gamma * count_g)` == `slogdet(I + gamma * Z Z')`.
    fn slogdet_h(&self, gamma: f64) -> f64 {
        self.counts
            .iter()
            .map(|&cnt| (1.0 + gamma * cnt).ln())
            .sum()
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn to_mat(v: &[f64]) -> Mat<f64> {
    Mat::from_fn(v.len(), 1, |i, _| v[i])
}

fn to_vec(m: MatRef<f64>) -> Vec<f64> {
    (0..m.nrows()).map(|i| m[(i, 0)]).collect()
}

/// Rank of `x` via rank-revealing (column-pivoted) QR: the number of `|R_ii|` no
/// smaller than `tol * |R_00|`.
fn matrix_rank(x: MatRef<f64>, tol: f64) -> usize {
    if x.nrows() == 0 || x.ncols() == 0 {
        return 0;
    }
    let qr = x.col_piv_qr();
    let r = qr.R();
    let r00 = r[(0, 0)].abs();
    if r00 == 0.0 {
        return 0;
    }
    let cutoff = r00 * tol;
    (0..r.nrows().min(r.ncols()))
        .take_while(|&i| r[(i, i)].abs() > cutoff)
        .count()
}

/// One profiled REML deviance evaluation at a fixed `gamma`, plus everything needed to
/// finalize `beta` / residual variance once the optimal `gamma` is found.
struct Profiled {
    beta: Mat<f64>,
    resid_var: f64,
    deviance: f64,
}

fn profile(x: MatRef<f64>, y: &[f64], info: &GroupInfo, gamma: f64) -> Result<Profiled, String> {
    let n = x.nrows() as f64;
    let p = x.ncols() as f64;

    let hix = info.apply_hi_mat(x, gamma);
    let hiy = info.apply_hi(y, gamma);

    let xt_hi_x = x.transpose() * &hix;
    let xt_hi_y = x.transpose() * to_mat(&hiy);

    let llt = xt_hi_x
        .llt(Side::Lower)
        .map_err(|_| "X'HiX is not positive definite; design may be rank-deficient.".to_string())?;
    let beta = llt.solve(&xt_hi_y);

    let fitted = x * &beta;
    let r: Vec<f64> = y
        .iter()
        .zip(to_vec(fitted.as_ref()))
        .map(|(a, b)| a - b)
        .collect();
    let hir = info.apply_hi(&r, gamma);
    let rhir = dot(&r, &hir);
    let resid_var = rhir / (n - p);
    if !(resid_var > 0.0) {
        return Err("Residual variance estimate is non-positive.".to_string());
    }

    let slogdet_xt_hi_x: f64 = 2.0
        * llt
            .L()
            .diagonal()
            .column_vector()
            .iter()
            .map(|v| v.abs().ln())
            .sum::<f64>();

    let deviance = (n - p) * resid_var.ln() + info.slogdet_h(gamma) + slogdet_xt_hi_x;

    Ok(Profiled {
        beta,
        resid_var,
        deviance,
    })
}

/// Fits a random-intercept model via REML.
///
/// `x` is the fixed-effect design (intercept column included), `y` the response,
/// `group_codes` a dense `0..n_groups` grouping per row, and `between_idx` the column
/// indices of `x` that are constant within every group level (used for containment
/// degrees of freedom).
pub fn fit_reml(
    x: MatRef<f64>,
    y: &[f64],
    group_codes: &[usize],
    n_groups: usize,
    between_idx: &[usize],
    max_iter: usize,
    tol: f64,
) -> Result<MixedModelResult, String> {
    let n = x.nrows();
    let p = x.ncols();
    if n != y.len() || n != group_codes.len() {
        return Err("X, y, and group must have the same number of rows.".to_string());
    }
    if n <= p {
        return Err(
            "Not enough rows to fit a mixed model with this many fixed effects.".to_string(),
        );
    }

    let info = GroupInfo::new(group_codes, n_groups);

    let phi = (5f64.sqrt() - 1.0) / 2.0;
    let (mut lo, mut hi) = (0.0f64, 1e6f64);
    let mut c = hi - phi * (hi - lo);
    let mut e = lo + phi * (hi - lo);
    let mut fc = profile(x, y, &info, c)?.deviance;
    let mut fe = profile(x, y, &info, e)?.deviance;
    for _ in 0..max_iter {
        if hi - lo < tol {
            break;
        }
        if fc < fe {
            hi = e;
            e = c;
            fe = fc;
            c = hi - phi * (hi - lo);
            fc = profile(x, y, &info, c)?.deviance;
        } else {
            lo = c;
            c = e;
            fc = fe;
            e = lo + phi * (hi - lo);
            fe = profile(x, y, &info, e)?.deviance;
        }
    }
    let gamma = (lo + hi) / 2.0;

    let fitted = profile(x, y, &info, gamma)?;
    let xt_hi_x = x.transpose() * info.apply_hi_mat(x, gamma);
    let cov = xt_hi_x
        .llt(Side::Lower)
        .map_err(|_| "X'HiX is not positive definite at the REML optimum.".to_string())?
        .solve(Mat::<f64>::identity(p, p));

    let coeffs = to_vec(fitted.beta.as_ref());
    let std_errors: Vec<f64> = (0..p)
        .map(|j| (fitted.resid_var * cov[(j, j)]).sqrt())
        .collect();

    // Containment degrees of freedom: fixed effects constant within every group level
    // (including the intercept) are tested against the group ("between") stratum,
    // everything else against the residual ("within") stratum.
    let rank_tol = 1e-9;
    let between_cols: Vec<usize> = between_idx.to_vec();
    let x_between = Mat::from_fn(n, between_cols.len(), |i, j| x[(i, between_cols[j])]);
    let rank_between = matrix_rank(x_between.as_ref(), rank_tol);
    let ddf_between = (n_groups as f64) - (rank_between as f64);

    let x_and_z = Mat::from_fn(n, p + n_groups, |i, j| {
        if j < p {
            x[(i, j)]
        } else if group_codes[i] == j - p {
            1.0
        } else {
            0.0
        }
    });
    let rank_combined = matrix_rank(x_and_z.as_ref(), rank_tol);
    let ddf_within = (n as f64) - (rank_combined as f64);

    let between_set: std::collections::HashSet<usize> = between_cols.into_iter().collect();
    let dfs: Vec<f64> = (0..p)
        .map(|j| {
            if between_set.contains(&j) {
                ddf_between
            } else {
                ddf_within
            }
        })
        .collect();

    Ok(MixedModelResult {
        coeffs,
        std_errors,
        dfs,
        gamma,
        resid_variance: fitted.resid_var,
    })
}
