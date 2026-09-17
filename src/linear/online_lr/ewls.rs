//! Finite-window exponentially weighted least squares.
//!
//! For rho = 2^(-1 / half_life), maintain A = sum(w xx') and b = sum(w xy):
//! A[t] = rho A[t-1] + x[t]x[t]' - rho^W x[t-W]x[t-W]', and likewise for b.
//! Invalid rows contribute zero but retain their original position. See
//! maths/rolling_ewls.md for the derivation, rank policy and rebuilding costs.

use faer::{linalg::solvers::Solve, Mat, MatRef, Side};
use faer_traits::RealField;
use num::Float;

struct CrossProducts {
    gram: Mat<f64>,
    rhs: Vec<f64>,
}

impl CrossProducts {
    fn new(p: usize) -> Self {
        Self {
            gram: Mat::zeros(p, p),
            rhs: vec![0.0; p],
        }
    }

    fn scale(&mut self, factor: f64) -> bool {
        if factor == 0.0 {
            // Fully decayed state contributes nothing, even after overflow.
            self.clear();
            return false;
        }
        let mut nonfinite = false;
        for i in 0..self.rhs.len() {
            self.rhs[i] *= factor;
            nonfinite |= !self.rhs[i].is_finite();
            for j in 0..=i {
                let value = *self.gram.get(i, j) * factor;
                nonfinite |= !value.is_finite();
                *self.gram.get_mut(i, j) = value;
                *self.gram.get_mut(j, i) = value;
            }
        }
        nonfinite
    }

    fn clear(&mut self) {
        self.rhs.fill(0.0);
        for i in 0..self.rhs.len() {
            for j in 0..self.rhs.len() {
                *self.gram.get_mut(i, j) = 0.0;
            }
        }
    }

    fn add<T: RealField + Float + Into<f64>>(
        &mut self,
        x: MatRef<T>,
        y: MatRef<T>,
        row: usize,
        weight: f64,
    ) {
        if weight == 0.0 {
            return;
        }
        let target: f64 = (*y.get(row, 0)).into();
        for i in 0..self.rhs.len() {
            let xi: f64 = (*x.get(row, i)).into();
            self.rhs[i] += (weight * xi) * target;
            for j in 0..=i {
                let xj: f64 = (*x.get(row, j)).into();
                let value = *self.gram.get(i, j) + (weight * xi) * xj;
                *self.gram.get_mut(i, j) = value;
                *self.gram.get_mut(j, i) = value;
            }
        }
    }

    fn rebuild<T: RealField + Float + Into<f64>>(
        &mut self,
        x: MatRef<T>,
        y: MatRef<T>,
        valid: &[bool],
        end: usize,
        window: usize,
        half_life: f64,
    ) {
        self.clear();
        for row in end + 1 - window..=end {
            if valid[row] {
                let weight = (-((end - row) as f64) / half_life).exp2();
                self.add(x, y, row, weight);
            }
        }
    }

    // A large outgoing observation can erase meaningful digits in a downdate.
    // Rebuild when less than 1e-4 of an entry survives, leaving headroom for the
    // coefficient solve and for absolute-error checks near zero coefficients.
    fn remove<T: RealField + Float + Into<f64>>(
        &mut self,
        x: MatRef<T>,
        y: MatRef<T>,
        row: usize,
        weight: f64,
    ) -> bool {
        if weight == 0.0 {
            return false;
        }
        let target: f64 = (*y.get(row, 0)).into();
        let mut cancellation = false;
        for i in 0..self.rhs.len() {
            let xi: f64 = (*x.get(row, i)).into();
            let diagonal = *self.gram.get(i, i);
            let rhs = self.rhs[i];
            let new_diagonal = diagonal - (weight * xi) * xi;
            let new_rhs = rhs - (weight * xi) * target;
            cancellation |= !new_diagonal.is_finite()
                || !new_rhs.is_finite()
                || (diagonal != 0.0 && new_diagonal.abs() < 1e-4 * diagonal.abs())
                || (rhs != 0.0 && new_rhs.abs() < 1e-4 * rhs.abs());
        }
        self.add(x, y, row, -weight);
        cancellation
    }

    fn solve(&self, rank_tol: f64) -> Option<Vec<f64>> {
        let p = self.rhs.len();
        let mut scales = Vec::with_capacity(p);
        for i in 0..p {
            let diagonal = *self.gram.get(i, i);
            if !diagonal.is_finite() || diagonal <= 0.0 || !self.rhs[i].is_finite() {
                return None;
            }
            scales.push(diagonal.sqrt());
        }
        // Diagonal scaling keeps the rank test invariant to column units. Use
        // successive divisions to avoid overflowing/underflowing scale[i]*scale[j].
        let normalized = Mat::from_fn(p, p, |i, j| *self.gram.get(i, j) / scales[i] / scales[j]);
        if !normalized.is_all_finite() {
            return None;
        }
        let factor = normalized.llt(Side::Lower).ok()?;
        // Same relative-determinant convention as faer_solve_lr_gated, evaluated
        // in log space and reusing the solve's single factorization.
        let log_det: f64 = factor
            .L()
            .diagonal()
            .column_vector()
            .iter()
            .map(|d| 2.0 * d.ln())
            .sum();
        if !log_det.is_finite() || log_det <= rank_tol.ln() {
            return None;
        }
        let rhs = Mat::from_fn(p, 1, |i, _| self.rhs[i] / scales[i]);
        let solution = factor.solve(rhs);
        let coefficients: Vec<f64> = (0..p).map(|i| *solution.get(i, 0) / scales[i]).collect();
        coefficients
            .iter()
            .all(|v| v.is_finite())
            .then_some(coefficients)
    }
}

/// Return one optional coefficient vector per input row, including warm-up nulls.
/// The plugin validates dimensions and parameters and supplies a joint X/y mask.
/// Both f32 and f64 inputs accumulate and solve in f64; the plugin casts outputs.
pub fn faer_rolling_ewls<T: RealField + Float + Into<f64>>(
    x: MatRef<T>,
    y: MatRef<T>,
    valid: &[bool],
    window: usize,
    min_rows: usize,
    half_life: f64,
    rank_tol: f64,
) -> Vec<Option<Vec<f64>>> {
    let n = x.nrows();
    let mut output = Vec::with_capacity(n);
    output.resize_with(n.min(window - 1), || None);
    if n < window {
        return output;
    }

    let decay = (-1.0 / half_life).exp2();
    // Compute expiry directly from the half-life, avoiding accumulated error
    // from raising a rounded decay factor to a large power.
    let expiry = (-(window as f64) / half_life).exp2();
    let mut products = CrossProducts::new(x.ncols());
    let mut count = valid[..window].iter().filter(|&&v| v).count();
    let mut previous_fit = false;
    for t in window - 1..n {
        let mut rebuilt = (t + 1) % window == 0;
        if t >= window {
            count -= usize::from(valid[t - window]);
            count += usize::from(valid[t]);
            if !rebuilt {
                rebuilt = products.scale(decay);
                if valid[t - window] {
                    rebuilt |= products.remove(x, y, t - window, expiry);
                }
                if valid[t] {
                    products.add(x, y, t, 1.0);
                }
            }
        }
        if count == 0 {
            products.clear();
        } else if rebuilt {
            // One O(W p^2) rebuild per W rows gives O(p^2) amortized work.
            products.rebuild(x, y, valid, t, window, half_life);
        }
        if count < min_rows.max(x.ncols()) {
            output.push(None);
            previous_fit = false;
            continue;
        }
        let mut fit = products.solve(rank_tol);
        if fit.is_none() && previous_fit && !rebuilt {
            // Check a transition to degeneracy once against fresh statistics.
            // Persistently singular windows do not trigger an O(W) rescan per row.
            products.rebuild(x, y, valid, t, window, half_life);
            fit = products.solve(rank_tol);
        }
        previous_fit = fit.is_some();
        output.push(fit);
    }
    output
}
