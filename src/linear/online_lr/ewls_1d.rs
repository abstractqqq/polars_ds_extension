//! Stable finite-window, one-predictor EWLS with an intercept.

use faer_traits::{math_utils::from_f64, RealField};
use num::Float;
use std::ops::Range;

const SW: usize = 0;
const SX: usize = 1;
const SY: usize = 2;
const SXX: usize = 3;
const SXY: usize = 4;

struct Moments<T: RealField + Float> {
    values: [T; 5],
}

impl<T: RealField + Float> Moments<T> {
    fn new() -> Self {
        Self {
            values: [T::zero(); 5],
        }
    }

    fn clear(&mut self) {
        self.values.fill(T::zero());
    }

    fn scale(&mut self, factor: T) -> bool {
        if factor == T::zero() {
            // A fully decayed overflowed state contributes nothing.
            self.clear();
            return false;
        }
        let mut nonfinite = false;
        for value in &mut self.values {
            *value *= factor;
            nonfinite |= !value.is_finite();
        }
        nonfinite
    }

    fn contribution(x: T, y: T, weight: T) -> [T; 5] {
        [
            weight,
            weight * x,
            weight * y,
            (weight * x) * x,
            (weight * x) * y,
        ]
    }

    fn add(&mut self, x: T, y: T, weight: T) {
        if weight != T::zero() {
            for (value, term) in self.values.iter_mut().zip(Self::contribution(x, y, weight)) {
                *value += term;
            }
        }
    }

    fn remove(&mut self, x: T, y: T, weight: T, cancellation_tol: T) -> bool {
        if weight == T::zero() {
            return false;
        }
        let mut cancellation = false;
        for (value, term) in self.values.iter_mut().zip(Self::contribution(x, y, weight)) {
            let next = *value - term;
            cancellation |= !next.is_finite()
                || (*value != T::zero() && next.abs() < cancellation_tol * value.abs());
            *value = next;
        }
        cancellation
    }

    fn rebuild(
        &mut self,
        x: &[T],
        y: &[T],
        valid: &[bool],
        rows: Range<usize>,
        anchor: usize,
        half_life: f64,
    ) {
        self.clear();
        for row in rows {
            if valid[row] {
                let weight = from_f64((-((anchor - row) as f64) / half_life).exp2());
                self.add(x[row], y[row], weight);
            }
        }
    }

    fn solve(&self, rank_tol: T) -> Option<[T; 2]> {
        let v = &self.values;
        if v[SW] <= T::zero() || v[SXX] <= T::zero() || !v.iter().all(|v| v.is_finite()) {
            return None;
        }
        // Scale each column separately: multiplying their scales can overflow
        // even when the normalized Gram matrix and coefficients are finite.
        let scale_x = v[SXX].sqrt();
        let scale_w = v[SW].sqrt();
        let d0 = v[SXX] / scale_x / scale_x;
        let l00 = d0.sqrt();
        let l10 = (v[SX] / scale_w / scale_x) / l00;
        let pivot = v[SW] / scale_w / scale_w - l10 * l10;
        if !pivot.is_finite() || d0 * pivot <= rank_tol {
            return None;
        }
        let l11 = pivot.sqrt();
        let z0 = (v[SXY] / scale_x) / l00;
        let z1 = (v[SY] / scale_w - l10 * z0) / l11;
        let b1 = z1 / l11;
        let slope = (z0 - l10 * b1) / l00 / scale_x;
        let intercept = b1 / scale_w;
        (slope.is_finite() && intercept.is_finite()).then_some([slope, intercept])
    }
}

/// Emit one fit per original row, including warm-up and degenerate nulls.
/// The adapter validates dimensions/parameters and maps Arrow nulls to NaNs.
/// Only complete finite observations contribute; their original ages are kept.
pub(crate) fn for_each_rolling_ewls_1d<T: RealField + Float>(
    x: &[T],
    y: &[T],
    window: usize,
    min_rows: usize,
    half_life: f64,
    rank_tol: T,
    mut emit: impl FnMut(usize, Option<[T; 2]>),
) {
    let nrows = x.len();
    for row in 0..nrows.min(window - 1) {
        emit(row, None);
    }
    if nrows < window {
        return;
    }
    let valid: Vec<bool> = x
        .iter()
        .zip(y)
        .map(|(x, y)| x.is_finite() && y.is_finite())
        .collect();
    let mut moments = Moments::new();
    let decay = from_f64::<T>((-1.0 / half_life).exp2());
    let expiry = from_f64::<T>((-(window as f64) / half_life).exp2());
    let cancellation_tol = T::epsilon().cbrt().max(from_f64(1e-4));
    let mut count = valid[..window].iter().filter(|&&v| v).count();
    // Normalize to the newest valid observation, retaining original ages.
    let mut anchor = valid[..window].iter().rposition(|&v| v).unwrap_or(0);
    let mut previous_fit = false;
    for t in window - 1..nrows {
        let previous_anchor = anchor;
        if valid[t] {
            anchor = t;
        }
        let mut rebuilt = (t + 1) % window == 0;
        if t >= window {
            count -= usize::from(valid[t - window]);
            count += usize::from(valid[t]);
            if !rebuilt {
                if anchor != previous_anchor {
                    let factor = if anchor - previous_anchor == 1 {
                        decay
                    } else {
                        from_f64((-((anchor - previous_anchor) as f64) / half_life).exp2())
                    };
                    rebuilt = moments.scale(factor);
                }
                if valid[t - window] {
                    let weight = if anchor == t {
                        expiry
                    } else {
                        from_f64((-((anchor - (t - window)) as f64) / half_life).exp2())
                    };
                    rebuilt |=
                        moments.remove(x[t - window], y[t - window], weight, cancellation_tol);
                }
                if valid[t] {
                    moments.add(x[t], y[t], T::one());
                }
            }
        }
        if count == 0 {
            moments.clear();
        } else if rebuilt {
            moments.rebuild(x, y, &valid, t + 1 - window..t + 1, anchor, half_life);
        }
        let mut fit = if count >= min_rows.max(2) {
            moments.solve(rank_tol)
        } else {
            None
        };
        if fit.is_none() && previous_fit && !rebuilt && count >= min_rows.max(2) {
            moments.rebuild(x, y, &valid, t + 1 - window..t + 1, anchor, half_life);
            fit = moments.solve(rank_tol);
        }
        previous_fit = fit.is_some();
        emit(t, fit);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expired_outlier<T: RealField + Float>(large: f64, tolerance: T, rank_tol: T) {
        let mut x: Vec<T> = (0..180)
            .map(|i| from_f64::<T>((((i * 37) % 101) as f64 - 50.0) / 31.0))
            .collect();
        let y: Vec<T> = x
            .iter()
            .map(|&x| from_f64::<T>(2.0) * x + from_f64::<T>(3.0))
            .collect();
        x[25] = from_f64::<T>(large);
        let mut emitted = 0;
        for_each_rolling_ewls_1d(&x, &y, 23, 2, 6.0, rank_tol, |row, fit| {
            emitted += 1;
            if row >= 48 {
                let beta = fit.expect("ordinary window must recover after the outlier expires");
                assert!(
                    (beta[0] - from_f64::<T>(2.0)).abs() < tolerance,
                    "row {row}"
                );
                assert!(
                    (beta[1] - from_f64::<T>(3.0)).abs() < tolerance,
                    "row {row}"
                );
            }
        });
        assert_eq!(emitted, x.len());
    }

    #[test]
    fn expired_f64_outliers_do_not_affect_later_windows() {
        expired_outlier::<f64>(1e12, 1e-9, 1e-12);
        expired_outlier::<f64>(1e200, 1e-9, 1e-12);
    }

    #[test]
    fn expired_f32_outliers_do_not_affect_later_windows() {
        expired_outlier::<f32>(1e12, 1e-4, 1e-6);
        expired_outlier::<f32>(1e30, 1e-4, 1e-6);
    }

    fn missing_tail<T: RealField + Float>(tolerance: T, rank_tol: T) {
        let x: Vec<T> = (0..3100)
            .map(|i| from_f64::<T>(if i % 2 == 0 { -1.0 } else { 1.0 }))
            .collect();
        let y: Vec<T> = x
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                if (1500..3000).contains(&i) {
                    T::nan()
                } else {
                    from_f64::<T>(2.0) * x + from_f64::<T>(3.0)
                }
            })
            .collect();
        for_each_rolling_ewls_1d(&x, &y, 1500, 2, 1.1, rank_tol, |row, fit| {
            if (1499..2998).contains(&row) || row >= 3001 {
                let beta = fit.expect("two surviving observations still identify the line");
                assert!(
                    (beta[0] - from_f64::<T>(2.0)).abs() < tolerance,
                    "row {row}"
                );
                assert!(
                    (beta[1] - from_f64::<T>(3.0)).abs() < tolerance,
                    "row {row}"
                );
            } else {
                assert!(fit.is_none(), "row {row}");
            }
        });
    }

    #[test]
    fn missing_tail_preserves_fits_in_both_precisions() {
        missing_tail::<f64>(1e-9, 1e-12);
        missing_tail::<f32>(1e-4, 1e-6);
    }
}
