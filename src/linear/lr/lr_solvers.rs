#![allow(non_snake_case)]
use crate::linear::{
    lr::{LRSolverMethods, LinearModel},
    LinalgErrors,
};
use crate::utils::parallelism::PARALLEL_MATMUL_THRESHOLD;
use faer::{linalg::solvers::Solve, mat::Mat, prelude::*, Side, zip, unzip};
use faer_traits::RealField;
use num::Float;

/// A struct that handles regular linear regression and Ridge regression.
pub struct LR<T: RealField + Float> {
    pub solver: LRSolverMethods,
    pub lambda: T,
    pub coefficients: Mat<T>, // n_features x 1 matrix, doesn't contain bias
    pub add_bias: bool,
}

impl<T: RealField + Float> LR<T> {
    pub fn new(solver: &str, lambda: T, add_bias: bool) -> Self {
        LR {
            solver: solver.into(),
            lambda: lambda,
            coefficients: Mat::new(),
            add_bias: add_bias,
        }
    }

    pub fn from_values(coeffs: &[T], bias: T) -> Self {
        let mut slf = LR {
            solver: LRSolverMethods::default(),
            lambda: T::zero(),
            coefficients: Mat::new(),
            add_bias: false,
        };
        slf.set_coeffs_and_bias(coeffs, bias);
        slf
    }

    pub fn set_coeffs_and_bias(&mut self, coeffs: &[T], bias: T) {
        self.add_bias = bias.abs() > T::epsilon();
        if self.add_bias {
            self.coefficients = Mat::from_fn(coeffs.len() + 1, 1, |i, _| {
                if i < coeffs.len() {
                    coeffs[i]
                } else {
                    bias
                }
            })
        } else {
            self.coefficients = ColRef::<T>::from_slice(coeffs).as_mat().to_owned();
        }
    }
}

impl<T: RealField + Float> LinearModel<T> for LR<T> {
    fn fitted_values(&'_ self) -> MatRef<'_, T> {
        self.coefficients.as_ref()
    }

    fn add_bias(&self) -> bool {
        self.add_bias
    }

    fn fit_unchecked(&mut self, X: MatRef<T>, y: MatRef<T>) {
        self.coefficients = if self.add_bias {
            let ones = Mat::full(X.nrows(), 1, T::one());
            let new = faer::concat![[X, ones]];
            faer_solve_lr(new.as_ref(), y, self.lambda, true, self.solver)
        } else {
            faer_solve_lr(X, y, self.lambda, false, self.solver)
        };
    }
}

pub struct ElasticNet<T: RealField + Float> {
    pub l1_reg: T,
    pub l2_reg: T,
    pub coefficients: Mat<T>, // n_features x 1 matrix, doesn't contain bias
    pub add_bias: bool,
    pub tol: T,
    pub max_iter: usize,
}

impl<T: RealField + Float> ElasticNet<T> {
    pub fn new(l1_reg: T, l2_reg: T, add_bias: bool, tol: T, max_iter: usize) -> Self {
        ElasticNet {
            l1_reg: l1_reg,
            l2_reg: l2_reg,
            coefficients: Mat::new(),
            add_bias: add_bias,
            tol: tol,
            max_iter: max_iter,
        }
    }

    pub fn from_values(coeffs: &[T], bias: T) -> Self {
        let mut slf = ElasticNet {
            l1_reg: T::nan(),
            l2_reg: T::nan(),
            coefficients: Mat::new(),
            add_bias: false,
            tol: T::from(1e-5).unwrap(),
            max_iter: 2000,
        };
        slf.set_coeffs_and_bias(coeffs, bias);
        slf
    }

    pub fn set_coeffs_and_bias(&mut self, coeffs: &[T], bias: T) {
        self.add_bias = bias.abs() > T::epsilon();
        self.coefficients = if self.add_bias {
            Mat::from_fn(coeffs.len() + 1, 1, |i, _| {
                if i < coeffs.len() {
                    coeffs[i]
                } else {
                    bias
                }
            })
        } else {
            ColRef::<T>::from_slice(coeffs).as_mat().to_owned()
        };
    }

    pub fn regularizers(&self) -> (T, T) {
        (self.l1_reg, self.l2_reg)
    }
}

impl<T: RealField + Float> LinearModel<T> for ElasticNet<T> {
    fn fitted_values(&'_ self) -> MatRef<'_, T> {
        self.coefficients.as_ref()
    }

    fn add_bias(&self) -> bool {
        self.add_bias
    }

    fn fit_unchecked(&mut self, X: MatRef<T>, y: MatRef<T>) {
        self.coefficients = if self.add_bias {
            let ones = Mat::full(X.nrows(), 1, T::one());
            let new_x = faer::concat![[X, ones]];
            faer_coordinate_descent(
                new_x.as_ref(),
                y,
                self.l1_reg,
                self.l2_reg,
                self.add_bias,
                self.tol,
                self.max_iter,
                false,
            )
        } else {
            faer_coordinate_descent(
                X,
                y,
                self.l1_reg,
                self.l2_reg,
                self.add_bias,
                self.tol,
                self.max_iter,
                false,
            )
        };
    }

    fn fit(&mut self, X: MatRef<T>, y: MatRef<T>) -> Result<(), LinalgErrors> {
        if X.nrows() != y.nrows() {
            return Err(LinalgErrors::DimensionMismatch);
        } else if X.nrows() == 0 || y.nrows() == 0 {
            return Err(LinalgErrors::NotEnoughData);
        } // Ok to have nrows < ncols
        self.fit_unchecked(X, y);
        Ok(())
    }
}

// ---------------------
// --- The functions ---
// ---------------------

#[inline(always)]
fn get_xtx_with_lambda<T: RealField + Float>(
    x: MatRef<T>,
    lambda: T,
    add_bias: bool,
) -> Mat<T> {
    let ncols = x.ncols();
    let mut xtx = Mat::zeros(ncols, ncols);
    let par = if x.nrows() * x.ncols() < PARALLEL_MATMUL_THRESHOLD {
        Par::Seq
    } else {
        Par::rayon(0)
    };
    faer::linalg::matmul::matmul(
        xtx.as_mut(),
        faer::Accum::Replace,
        x.transpose(),
        x,
        T::one(),
        par,
    );
    // xtx + diagonal of lambda. If has bias, don't add to the last diagonal.
    let n1 = ncols.abs_diff(add_bias as usize);
    if lambda > T::zero() && n1 >= 1 {
        unsafe {
            xtx.get_mut_unchecked(0..n1, 0..n1)
                .diagonal_mut()
                .for_each_mut(|d| {
                    *d += lambda;
                });
        }
    }
    xtx
}

/// Least square that sets all singular values below threshold to 0.
/// Returns the coefficients and the singular values
#[inline(always)]
pub fn faer_solve_lr_rcond<T: RealField + Float>(
    x: MatRef<'_, T>,
    y: MatRef<T>,
    lambda: T,
    add_bias: bool,
    rcond: T,
) -> Result<(Mat<T>, Vec<T>), String> {
    let xtx = get_xtx_with_lambda(x, lambda, add_bias);
    // need work here
    match xtx.thin_svd() {
        Ok(svd) => {
            let s = svd.S().column_vector();
            let singular_values = s.iter().copied().map(T::sqrt).collect::<Vec<_>>();
            let n = singular_values.len();
            let max_singular_value = singular_values[0]; // at least 1.
                                                         // singular_values.iter().copied().fold(T::min_value(), T::max);
            let threshold = rcond * max_singular_value;
            // Safe, because i <= n
            let mut s_inv = Mat::<T>::zeros(n, n);
            unsafe {
                for (i, v) in s.iter().copied().enumerate() {
                    *s_inv.get_mut_unchecked(i, i) =
                        if v >= threshold { v.recip() } else { T::zero() };
                }
            }
            let weights = svd.V() * s_inv * svd.U().transpose() * x.transpose() * y;
            Ok((weights, singular_values))
        }
        _ => Err("SVD failed.".to_string()),
    }
}

/// Builds X'y as an ncols x y.ncols() matrix, parallelizing the matmul above a threshold.
#[inline(always)]
fn build_xty<T: RealField + Float>(x: MatRef<T>, y: MatRef<T>) -> Mat<T> {
    let mut xty = Mat::zeros(x.ncols(), y.ncols());
    let par = if x.nrows() * y.ncols() < PARALLEL_MATMUL_THRESHOLD {
        Par::Seq
    } else {
        Par::rayon(0)
    };
    faer::linalg::matmul::matmul(
        xty.as_mut(),
        faer::Accum::Replace,
        x.transpose(),
        y,
        T::one(),
        par,
    );
    xty
}

/// Solves (X'X) b = X'y for b using the chosen factorization, falling back to QR.
#[inline(always)]
fn solve_xtx_xty<T: RealField + Float>(
    xtx: Mat<T>,
    xty: Mat<T>,
    how: LRSolverMethods,
) -> Mat<T> {
    match how {
        LRSolverMethods::SVD => match xtx.thin_svd() {
            Ok(svd) => svd.solve(xty),
            _ => xtx.col_piv_qr().solve(xty),
        },
        LRSolverMethods::Choleskey => match xtx.llt(Side::Lower) {
            Ok(llt) => llt.solve(xty),
            _ => xtx.col_piv_qr().solve(xty),
        },
        LRSolverMethods::QR => xtx.col_piv_qr().solve(xty),
    }
}

/// Returns the coefficients for lstsq with l2 (Ridge) regularization as a nrows x 1 matrix
/// If lambda is 0, then this is the regular lstsq
#[inline(always)]
pub fn faer_solve_lr<T: RealField + Float>(
    x: MatRef<T>,
    y: MatRef<T>,
    lambda: T,
    add_bias: bool,
    how: LRSolverMethods,
) -> Mat<T> {
    let xtx = get_xtx_with_lambda(x, lambda, add_bias);
    solve_xtx_xty(xtx, build_xty(x, y), how)
}

/// Relative determinant of X'X: |det(X'X)| / Π diag(X'X). Scale-invariant rank
/// check, ~1.0 for well-conditioned designs and ~0 for (near-)singular ones.
#[inline(always)]
fn rel_det<T: RealField + Float>(xtx: &Mat<T>) -> T {
    let det: T = xtx.as_ref().determinant().abs();
    let dprod = xtx
        .diagonal()
        .column_vector()
        .iter()
        .copied()
        .fold(T::one(), |acc, d| acc * d);
    if dprod > T::zero() {
        det / dprod
    } else {
        T::zero()
    }
}

/// Like `faer_solve_lr` but returns `None` when the design is rank-deficient, i.e.
/// `rel_det(X'X) <= tol`. Builds X'X once and skips the X'y build + solve when gated.
#[inline(always)]
pub fn faer_solve_lr_gated<T: RealField + Float>(
    x: MatRef<T>,
    y: MatRef<T>,
    lambda: T,
    add_bias: bool,
    how: LRSolverMethods,
    tol: T,
) -> Option<Mat<T>> {
    let xtx = get_xtx_with_lambda(x, lambda, add_bias);
    if rel_det(&xtx) <= tol {
        return None;
    }
    Some(solve_xtx_xty(xtx, build_xty(x, y), how))
}

/// Solves the weighted least square with weights given by the user
#[inline(always)]
pub fn faer_weighted_lr<T: RealField>(
    x: MatRef<T>,
    y: MatRef<T>,
    w: &[T],
    how: LRSolverMethods,
) -> Mat<T> {
    let weights = faer::ColRef::from_slice(w);
    let w = weights.as_diagonal();

    let xt = x.transpose();
    let xtw = xt * w;
    let xtwx = &xtw * x;
    match how {
        LRSolverMethods::SVD => match xtwx.thin_svd() {
            Ok(svd) => svd.solve(xtw * y),
            Err(_) => xtwx.col_piv_qr().solve(xtw * y),
        },
        LRSolverMethods::Choleskey => match xtwx.llt(Side::Lower) {
            Ok(llt) => llt.solve(xtw * y),
            _ => xtwx.col_piv_qr().solve(xtw * y),
        },
        LRSolverMethods::QR => xtwx.col_piv_qr().solve(xtw * y),
    }
}

#[inline(always)]
fn soft_threshold_l1<T: Float>(z: T, lambda: T) -> T {
    z.signum() * (z.abs() - lambda).max(T::zero())
}

/// Computes Lasso / Elastic Regression coefficients by the use of Coordinate Descent.
/// The current stopping criterion is based on L Inf norm of the changes in the
/// coordinates. A better alternative might be the dual gap.
///
/// Reference:
/// https://xavierbourretsicotte.github.io/lasso_implementation.html
/// https://www.stat.cmu.edu/~ryantibs/convexopt-F18/lectures/coord-desc.pdf
/// https://github.com/minatosato/Lasso/blob/master/coordinate_descent_lasso.py
/// https://en.wikipedia.org/wiki/Lasso_(statistics)
#[inline(always)]
pub fn faer_coordinate_descent<T: RealField + Float>(
    x: MatRef<T>,
    y: MatRef<T>,
    l1_reg: T,
    l2_reg: T,
    add_bias: bool,
    tol: T,
    max_iter: usize,
    positive: bool,
) -> Mat<T> {
    // Replace this in the future? L-BFGS can do this too. Too much math that I cannot remember.

    let m = T::from(x.nrows()).unwrap();
    let ncols = x.ncols();
    let n1 = ncols.abs_diff(add_bias as usize);

    let lambda_l1 = m * l1_reg;

    let mut beta: Mat<T> = Mat::zeros(ncols, 1);
    let mut converge = false;


    let mut xty = Mat::zeros(ncols, 1);
    let par_xty = if x.nrows() * y.ncols() < PARALLEL_MATMUL_THRESHOLD {
        Par::Seq
    } else {
        Par::rayon(0)
    };
    faer::linalg::matmul::matmul(
        xty.as_mut(),
        faer::Accum::Replace,
        x.transpose(),
        y,
        T::one(),
        par_xty,
    );

    let mut xtx = Mat::zeros(ncols, ncols);
    let par_xtx = if x.nrows() * x.ncols() < PARALLEL_MATMUL_THRESHOLD {
        Par::Seq
    } else {
        Par::rayon(0)
    };
    faer::linalg::matmul::matmul(
        xtx.as_mut(),
        faer::Accum::Replace,
        x.transpose(),
        x,
        T::one(),
        par_xtx,
    );

    // Compute column squared l2 norms from xtx diagonal
    let norms = (0..ncols)
        .map(|j| *unsafe { xtx.get_unchecked(j, j) } + m * l2_reg)
        .collect::<Vec<_>>();

    // Precompute sums for bias update optimization
    let y_sum = y.col(0).sum();
    let col_sums = (0..n1).map(
        |j| x.col(j).sum()
    ).collect::<Vec<_>>();
    
    // Random selection often leads to faster convergence?
    for _ in 0..max_iter {
        let mut max_change = T::zero();
        for j in 0..n1 {
            // temporary set beta(j, 0) to 0.
            // Safe. The index is valid and the value is initialized.
            let before = *unsafe { beta.get_unchecked(j, 0) };
            *unsafe { beta.get_mut_unchecked(j, 0) } = T::zero();

            // Compute dot product (X^T X * Beta)_j without allocating a 1x1 matrix.
            // Since X^T X is symmetric, col(j) is equivalent to row(j) but faster.
            let mut dot = T::zero();
            zip!(xtx.col(j), beta.col(0)).for_each(|unzip!(x_val, b_val)| {
                dot = dot + *x_val * *b_val;
            });
            let main_update = unsafe { *xty.get_unchecked(j, 0) - dot };

            // update beta(j, 0).
            let after = if positive && main_update < T::zero() {
                T::zero()
            } else {
                soft_threshold_l1(main_update, lambda_l1) / norms[j]
            };
            *unsafe { beta.get_mut_unchecked(j, 0) } = after;
            max_change = (after - before).abs().max(max_change);
        }
        // if add_bias, n1 = last index = ncols - 1 = column of bias. If add_bias is False, n = ncols
        // In positive case, there should be no positive constraint on the bias term
        if add_bias {
            // Precomputed sum optimization: ss = sum(y - X_no_bias * beta_no_bias) / m
            let mut dot_sums = T::zero();
            for j in 0..n1 {
                dot_sums = dot_sums + *unsafe { beta.get_unchecked(j, 0) } * col_sums[j];
            }
            let ss = (y_sum - dot_sums) / m;
            *unsafe { beta.get_mut_unchecked(n1, 0) } = ss;
        }
        // This stopping criterion is not perfect. There is something called a dual gap which
        // is better.
        converge = max_change < tol;
        if converge {
            break;
        }
    }

    if !converge {
        println!(
            "Lasso regression: Max number of iterations have passed and result hasn't converged."
        )
    }

    beta
}

/// Non-negative Lstsq. Returns the coefficients as a ncols x 1 faer matrix
/// Reference???
pub fn faer_nn_lr<T: RealField + Float>(
    x: MatRef<T>,
    y: MatRef<T>,
    add_bias: bool,
    tol: T,
    max_iter: usize,
) -> Mat<T> {
    // x: nrows x ncols
    let xtx: Mat<T> = x.transpose() * x;
    let mut beta: Mat<T> = Mat::zeros(x.ncols(), 1);
    // Initialize mu = -xty in-place to avoid allocation from Scale * Mat
    use crate::utils::parallelism::PARALLEL_MATMUL_THRESHOLD;
    let mut mu = Mat::zeros(x.ncols(), 1);
    let par = if x.nrows() * y.ncols() < PARALLEL_MATMUL_THRESHOLD {
        Par::Seq
    } else {
        Par::rayon(0)
    };
    faer::linalg::matmul::matmul(
    mu.as_mut(),
        faer::Accum::Replace,
        x.transpose(),
        y,
        T::one().neg(),
        par
    );

    // safe, all indices are in valid range
    unsafe {
        for _ in 0..max_iter {
            let criterion1 = mu.col(0).iter().all(|c| *c >= -tol);
            let mut criterion2 = true;
            for j in 0..beta.nrows() {
                if *beta.get_unchecked(j, 0) > T::zero() {
                    criterion2 &= *mu.get_unchecked(j, 0) <= tol;
                }
            }

            if criterion1 && criterion2 {
                break;
            };

            for k in 0..x.ncols() {
                let beta_k = *beta.get_unchecked(k, 0);
                let mut update = beta_k - *mu.get_unchecked(k, 0) / *xtx.get_unchecked(k, k);
                if !add_bias || k < x.ncols() - 1 {
                    update = update.max(T::zero());
                } // No need for max with 0 if this is the bias term
                *beta.get_mut_unchecked(k, 0) = update;
                let x_diff = update - beta_k;
                // This is a vector update (AXPY). zip! performs this in-place,
                // whereas the + operator allocates a new matrix every time.
                zip!(mu.as_mut(), xtx.col(k).as_mat()).for_each(
                    |unzip!(m, x)| *m = *m + x_diff * *x
                );
            }
        }
    }
    beta
}
