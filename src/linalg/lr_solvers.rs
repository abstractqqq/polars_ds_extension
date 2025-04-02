#![allow(non_snake_case)]
use super::{LRSolverMethods, LinalgErrors, LinearRegression};
use faer::{linalg::solvers::Solve, mat::Mat, prelude::*};
use faer_traits::RealField;
use num::Float;

/// A struct that handles regular linear regression and Ridge regression.
pub struct LR<T: RealField + Float> {
    pub solver: LRSolverMethods,
    pub lambda: T,
    pub coefficients: Mat<T>, // n_features x 1 matrix, doesn't contain bias
    pub has_bias: bool,
}

impl<T: RealField + Float> LR<T> {
    pub fn new(solver: &str, lambda: T, has_bias: bool) -> Self {
        LR {
            solver: solver.into(),
            lambda: lambda,
            coefficients: Mat::new(),
            has_bias: has_bias,
        }
    }

    pub fn from_values(coeffs: &[T], bias: T) -> Self {
        LR {
            solver: LRSolverMethods::default(),
            lambda: T::zero(),
            coefficients: faer::mat::Mat::from_fn(coeffs.len(), 1, |i, _| coeffs[i]),
            has_bias: bias.abs() > T::epsilon(),
        }
    }

    pub fn set_coeffs_and_bias(&mut self, coeffs: &[T], bias: T) {
        self.has_bias = bias.abs() > T::epsilon();
        if self.has_bias {
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

impl<T: RealField + Float> LinearRegression<T> for LR<T> {
    fn fitted_values(&self) -> MatRef<T> {
        self.coefficients.as_ref()
    }

    fn has_bias(&self) -> bool {
        self.has_bias
    }

    fn fit_unchecked(&mut self, X: MatRef<T>, y: MatRef<T>) {
        self.coefficients = if self.has_bias {
            let ones = Mat::full(X.nrows(), 1, T::one());
            let new = faer::concat![[X, ones]];
            faer_solve_lstsq(new.as_ref(), y, self.lambda, true, self.solver)
        } else {
            faer_solve_lstsq(X, y, self.lambda, false, self.solver)
        };
    }
}

pub struct ElasticNet<T: RealField + Float> {
    pub l1_reg: T,
    pub l2_reg: T,
    pub coefficients: Mat<T>, // n_features x 1 matrix, doesn't contain bias
    pub has_bias: bool,
    pub tol: T,
    pub max_iter: usize,
}

impl<T: RealField + Float> ElasticNet<T> {
    pub fn new(l1_reg: T, l2_reg: T, has_bias: bool, tol: T, max_iter: usize) -> Self {
        ElasticNet {
            l1_reg: l1_reg,
            l2_reg: l2_reg,
            coefficients: Mat::new(),
            has_bias: has_bias,
            tol: tol,
            max_iter: max_iter,
        }
    }

    pub fn from_values(coeffs: &[T], bias: T) -> Self {
        let has_bias = bias.abs() > T::epsilon();
        let coefficients = if has_bias {
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

        ElasticNet {
            l1_reg: T::nan(),
            l2_reg: T::nan(),
            coefficients: coefficients,
            has_bias: has_bias,
            tol: T::from(1e-5).unwrap(),
            max_iter: 2000,
        }
    }

    pub fn set_coeffs_and_bias(&mut self, coeffs: &[T], bias: T) {
        self.has_bias = bias.abs() > T::epsilon();
        self.coefficients = if self.has_bias {
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

impl<T: RealField + Float> LinearRegression<T> for ElasticNet<T> {
    fn fitted_values(&self) -> MatRef<T> {
        self.coefficients.as_ref()
    }

    fn has_bias(&self) -> bool {
        self.has_bias
    }

    fn fit_unchecked(&mut self, X: MatRef<T>, y: MatRef<T>) {
        self.coefficients = if self.has_bias {
            let ones = Mat::full(X.nrows(), 1, T::one());
            let new_x = faer::concat![[X, ones]];
            faer_coordinate_descent(
                new_x.as_ref(),
                y,
                self.l1_reg,
                self.l2_reg,
                self.has_bias,
                self.tol,
                self.max_iter,
            )
        } else {
            faer_coordinate_descent(
                X,
                y,
                self.l1_reg,
                self.l2_reg,
                self.has_bias,
                self.tol,
                self.max_iter,
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

/// Least square that sets all singular values below threshold to 0.
/// Returns the coefficients and the singular values
#[inline(always)]
pub fn faer_solve_lstsq_rcond<T: RealField + Float>(
    x: MatRef<T>,
    y: MatRef<T>,
    lambda: T,
    has_bias: bool,
    rcond: T,
) -> (Mat<T>, Vec<T>) {
    let n1 = x.ncols().abs_diff(has_bias as usize);
    let xt = x.transpose();
    let mut xtx = xt * x;
    // xtx + diagonal of lambda. If has bias, last diagonal element is 0.
    // Safe. Index is valid and value is initialized.
    if lambda >= T::zero() && n1 >= 1 {
        unsafe {
            for i in 0..n1 {
                *xtx.get_mut_unchecked(i, i) = *xtx.get_mut_unchecked(i, i) + lambda;
            }
        }
    }
    // need work here
    let svd = xtx.thin_svd().unwrap();
    let s = svd.S().column_vector();
    let singular_values = s.iter().copied().map(T::sqrt).collect::<Vec<_>>();

    let n = singular_values.len();

    let max_singular_value = singular_values.iter().copied().fold(T::min_value(), T::max);
    let threshold = rcond * max_singular_value;
    // Safe, because i <= n
    let mut s_inv = Mat::<T>::zeros(n, n);
    unsafe {
        for (i, v) in s.iter().copied().enumerate() {
            *s_inv.get_mut_unchecked(i, i) = if v >= threshold { v.recip() } else { T::zero() };
        }
    }

    let weights = svd.V() * s_inv * svd.U().transpose() * xt * y;
    (weights, singular_values)
}

/// Returns the coefficients for lstsq with l2 (Ridge) regularization as a nrows x 1 matrix
/// If lambda is 0, then this is the regular lstsq
#[inline(always)]
pub fn faer_solve_lstsq<T: RealField + Float>(
    x: MatRef<T>,
    y: MatRef<T>,
    lambda: T,
    has_bias: bool,
    how: LRSolverMethods,
) -> Mat<T> {
    // Add ridge SVD with rconditional number later.

    let n1 = x.ncols().abs_diff(has_bias as usize);
    let xt = x.transpose();
    let mut xtx = xt * x;
    // xtx + diagonal of lambda. If has bias, last diagonal element is 0.
    // Safe. Index is valid and value is initialized.
    if lambda >= T::zero() && n1 >= 1 {
        unsafe {
            for i in 0..n1 {
                *xtx.get_mut_unchecked(i, i) = *xtx.get_mut_unchecked(i, i) + lambda;
            }
        }
    }

    match how {
        LRSolverMethods::SVD => match xtx.thin_svd() {
            Ok(svd) => svd.solve(xt * y),
            _ => xtx.col_piv_qr().solve(xt * y),
        },
        LRSolverMethods::QR => xtx.col_piv_qr().solve(xt * y),
        LRSolverMethods::Choleskey => todo!(),
    }
}

/// Solves the weighted least square with weights given by the user
#[inline(always)]
pub fn faer_weighted_lstsq<T: RealField>(
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
        LRSolverMethods::QR => xtwx.col_piv_qr().solve(xtw * y),
        LRSolverMethods::Choleskey => todo!(),
    }
}

#[inline(always)]
fn soft_threshold_l1<T: Float>(z: T, lambda: T) -> T {
    z.signum() * (z.abs() - lambda).max(T::zero())
}

/// Computes Lasso/Elastic Regression coefficients by the use of Coordinate Descent.
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
    has_bias: bool,
    tol: T,
    max_iter: usize,
) -> Mat<T> {
    let m = T::from(x.nrows()).unwrap();
    let ncols = x.ncols();
    let n1 = ncols.abs_diff(has_bias as usize);

    let lambda_l1 = m * l1_reg;

    let mut beta: Mat<T> = Mat::zeros(ncols, 1);
    let mut converge = false;

    // compute column squared l2 norms.
    // (In the case of Elastic net, squared l2 norms + l2 regularization factor)
    let norms = x
        .col_iter()
        .map(|c| c.squared_norm_l2() + m * l2_reg)
        .collect::<Vec<_>>();

    let xty = x.transpose() * y;
    let xtx = x.transpose() * x;

    // Random selection often leads to faster convergence?
    for _ in 0..max_iter {
        let mut max_change = T::zero();
        for j in 0..n1 {
            // temporary set beta(j, 0) to 0.
            // Safe. The index is valid and the value is initialized.
            let before = *unsafe { beta.get_unchecked(j, 0) };
            *unsafe { beta.get_mut_unchecked(j, 0) } = T::zero();
            let xtx_j = unsafe { xtx.get_unchecked(j..j + 1, ..) };

            // Xi^t(y - X-i Beta-i)
            let main_update = *xty.get(j, 0) - *(xtx_j * &beta).get(0, 0);

            // update beta(j, 0).
            let after = soft_threshold_l1(main_update, lambda_l1) / norms[j];
            *unsafe { beta.get_mut_unchecked(j, 0) } = after;
            max_change = (after - before).abs().max(max_change);
        }
        // if has_bias, n1 = last index = ncols - 1 = column of bias. If has_bias is False, n = ncols
        if has_bias {
            // Safe. The index is valid and the value is initialized.
            let xx = unsafe { x.get_unchecked(.., 0..n1) };
            let bb = unsafe { beta.get_unchecked(0..n1, ..) };
            let ss = (y - xx * bb).as_ref().sum() / m;
            *unsafe { beta.get_mut_unchecked(n1, 0) } = ss;
        }
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
