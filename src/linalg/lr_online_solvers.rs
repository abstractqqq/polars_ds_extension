#![allow(non_snake_case)]
use core::f64;
use faer::{
    linalg::solvers::{DenseSolveCore, Solve},
    mat::Mat,
    prelude::*,
};
use faer_traits::{math_utils::is_nan, RealField};
use num::Float;

use super::{LinalgErrors, LinearRegression};

#[inline]
pub fn has_nan<T: RealField>(mat: MatRef<T>) -> bool {
    mat.col_iter().any(|col| col.iter().any(|x| is_nan(x)))
}

pub struct OnlineLR<T: RealField + Float> {
    pub lambda: T,
    pub has_bias: bool,
    pub coefficients: Mat<T>, // n_features x 1 matrix, or (n_features + ) x 1 if there is bias
    pub inv: Mat<T>,          // Current Inverse of X^t X
}

impl<T: RealField + Float> OnlineLR<T> {
    pub fn new(lambda: T, has_bias: bool) -> Self {
        OnlineLR {
            lambda: lambda,
            has_bias: has_bias,
            coefficients: Mat::new(),
            inv: Mat::new(),
        }
    }

    pub fn set_coeffs_bias_inverse(
        &mut self,
        coeffs: &[T],
        bias: T,
        inv: MatRef<T>,
    ) -> Result<(), LinalgErrors> {
        if coeffs.len() != inv.ncols() {
            Err(LinalgErrors::DimensionMismatch)
        } else {
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
            self.inv = inv.to_owned();
            Ok(())
        }
    }

    pub fn get_inv(&self) -> Result<MatRef<T>, LinalgErrors> {
        if self.inv.shape() == (0, 0) {
            Err(LinalgErrors::MatNotLearnedYet)
        } else {
            Ok(self.inv.as_ref())
        }
    }

    pub fn update_unchecked(&mut self, new_x: MatRef<T>, new_y: MatRef<T>, c: T) {
        if self.has_bias() {
            let ones = Mat::full(new_x.nrows(), 1, T::one());
            let new_new_x = faer::concat![[new_x, ones]];
            woodbury_step(
                self.inv.as_mut(),
                self.coefficients.as_mut(),
                new_new_x.as_ref(),
                new_y,
                c,
            );
        } else {
            woodbury_step(
                self.inv.as_mut(),
                self.coefficients.as_mut(),
                new_x,
                new_y,
                c,
            )
        }
    }

    pub fn update(&mut self, new_x: MatRef<T>, new_y: MatRef<T>, c: T) {
        if !(has_nan(new_x) || has_nan(new_y)) {
            self.update_unchecked(new_x, new_y, c)
        }
    }
}

impl<T: RealField + Float> LinearRegression<T> for OnlineLR<T> {
    fn fitted_values(&self) -> MatRef<T> {
        self.coefficients.as_ref()
    }

    fn has_bias(&self) -> bool {
        self.has_bias
    }

    fn fit_unchecked(&mut self, X: MatRef<T>, y: MatRef<T>) {
        if self.has_bias {
            let ones = Mat::full(X.nrows(), 1, T::one());
            let new_x = faer::concat![[X, ones]];
            let (inv, all_coefficients) =
                faer_qr_lstsq_with_inv(new_x.as_ref(), y, self.lambda, true);

            self.inv = inv;
            self.coefficients = all_coefficients;
        } else {
            (self.inv, self.coefficients) =
                faer_qr_lstsq_with_inv(X.as_ref(), y, self.lambda, false);
        }
    }
}

/// Returns the coefficients for lstsq as a nrows x 1 matrix together with the inverse of XtX
/// The uses QR (column pivot) decomposition as default method to compute inverse,
/// Column Pivot QR is chosen to deal with rank deficient cases. It is also slightly
/// faster compared to other methods.
#[inline(always)]
pub fn faer_qr_lstsq_with_inv<T: RealField + Copy>(
    x: MatRef<T>,
    y: MatRef<T>,
    lambda: T,
    has_bias: bool,
) -> (Mat<T>, Mat<T>) {
    let n1 = x.ncols().abs_diff(has_bias as usize);
    let xt = x.transpose();
    let mut xtx = xt * x;

    if lambda > T::zero() && n1 >= 1 {
        unsafe {
            for i in 0..n1 {
                *xtx.get_mut_unchecked(i, i) = *xtx.get_mut_unchecked(i, i) + lambda;
            }
        }
    }

    let qr = xtx.col_piv_qr();
    let inv = qr.inverse();
    let weights = qr.solve(xt * y);

    (inv, weights)
}

/// Given all data, we start running a lstsq starting at position n and compute new coefficients
/// recurisively.
/// This will return all coefficients for rows >= n. This will only be used in Polars Expressions.
pub fn faer_recursive_lstsq<T: RealField + Float>(
    x: MatRef<T>,
    y: MatRef<T>,
    n: usize,
    lambda: T,
) -> Vec<Mat<T>> {
    let xn = x.nrows();
    // x: size xn x m
    // y: size xn x 1
    // Vector of matrix of size m x 1
    let mut coefficients = Vec::with_capacity(xn - n + 1);
    // n >= 2, guaranteed by Python
    let x0 = x.get(..n, ..);
    let y0 = y.get(..n, ..);

    // This is because if add_bias, the 1 is added to
    // all data already. No need to let OnlineLR add the 1 for the user.
    let mut online_lr = OnlineLR::new(lambda, false);
    online_lr.fit_unchecked(x0, y0); // safe because things are checked in plugin / python functions.
    coefficients.push(online_lr.fitted_values().to_owned());
    for j in n..xn {
        let next_x = x.get(j..j + 1, ..); // 1 by m, m = # of columns
        let next_y = y.get(j..j + 1, ..); // 1 by 1
        online_lr.update(next_x, next_y, T::one());
        coefficients.push(online_lr.fitted_values().to_owned());
    }
    coefficients
}

/// Given all data, we start running a lstsq starting at position n and compute new coefficients recurisively.
/// This will return all coefficients for rows >= n. This will only be used in Polars Expressions.
/// This supports Normal or Ridge regression
pub fn faer_rolling_lstsq<T: RealField + Float>(
    x: MatRef<T>,
    y: MatRef<T>,
    n: usize,
    lambda: T,
) -> Vec<Mat<T>> {
    let xn = x.nrows();
    // x: size xn x m
    // y: size xn x 1
    // Vector of matrix of size m x 1
    let mut coefficients = Vec::with_capacity(xn - n + 1); // xn >= n is checked in Python

    let x0 = x.get(..n, ..);
    let y0 = y.get(..n, ..);

    // This is because if add_bias, the 1 is added to
    // all data already. No need to let OnlineLR add the 1 for the user.
    let mut online_lr = OnlineLR::new(lambda, false);
    online_lr.fit_unchecked(x0, y0);
    coefficients.push(online_lr.fitted_values().to_owned());

    for j in n..xn {
        let remove_x = x.get(j - n..j - n + 1, ..);
        let remove_y = y.get(j - n..j - n + 1, ..);
        online_lr.update(remove_x, remove_y, T::one().neg());

        let next_x = x.get(j..j + 1, ..); // 1 by m, m = # of columns
        let next_y = y.get(j..j + 1, ..); // 1 by 1
        online_lr.update(next_x, next_y, T::one());
        coefficients.push(online_lr.fitted_values().to_owned());
    }
    coefficients
}

/// Given all data, we start running a lstsq starting at position n and compute new coefficients recurisively.
/// This will return all coefficients for rows >= n. This will only be used in Polars Expressions.
/// If # of non-null rows in the window is < m, a Matrix with size (0, 0) will be returned.
/// This supports Normal or Ridge regression
pub fn faer_rolling_skipping_lstsq<T: RealField + Float>(
    x: MatRef<T>,
    y: MatRef<T>,
    n: usize,
    m: usize,
    lambda: T,
) -> Vec<Mat<T>> {
    let xn = x.nrows();
    let ncols = x.ncols();
    // x: size xn x m
    // y: size xn x 1
    // n is window size. m is min_window_size after skipping null rows. n >= m > 0.
    // Vector of matrix of size m x 1
    let mut coefficients = Vec::with_capacity(xn - n + 1); // xn >= n is checked in Python

    // Initialize the problem.
    let mut non_null_cnt_in_window = 0;
    let mut left = 0;
    let mut right = n;
    let mut x_slice: Vec<T> = Vec::with_capacity(n * ncols);
    let mut y_slice: Vec<T> = Vec::with_capacity(n);

    // This is because if add_bias, the 1 is added to
    // all data already. No need to let OnlineLR add the 1 for the user.
    let mut online_lr = OnlineLR::new(lambda, false);
    while right <= xn {
        // Somewhat redundant here.
        non_null_cnt_in_window = 0;
        x_slice.clear();
        y_slice.clear();
        for i in left..right {
            let x_i = x.get(i, ..);
            let y_i = y.get(i, ..);

            if !(x_i.iter().any(|x| is_nan(x)) || y_i.iter().any(|y| is_nan(y))) {
                non_null_cnt_in_window += 1;
                x_slice.extend(x_i.iter());
                y_slice.extend(y_i.iter());
            }
        }
        if non_null_cnt_in_window >= m {
            let x0 = MatRef::from_row_major_slice(&x_slice, y_slice.len(), ncols);
            // faer::mat::from_row_major_slice(&x_slice, y_slice.len(), ncols);
            let y0 = MatRef::from_column_major_slice(&y_slice, y_slice.len(), 1);
            // faer::mat::from_row_major_slice(&y_slice, y_slice.len(), 1);
            online_lr.fit_unchecked(x0, y0);
            coefficients.push(online_lr.fitted_values().to_owned());
            break;
        } else {
            left += 1;
            right += 1;
            coefficients.push(Mat::with_capacity(0, 0));
        }
    }

    if right >= xn {
        return coefficients;
    }
    // right < xn, the problem must have been initialized (inv and weights are defined.)
    for j in right..xn {
        let remove_x = x.get(j - n..j - n + 1, ..);
        let remove_y = y.get(j - n..j - n + 1, ..);

        if !(has_nan(remove_x) | has_nan(remove_y)) {
            non_null_cnt_in_window -= 1; // removed one non-null column
            online_lr.update_unchecked(remove_x, remove_y, T::one().neg()); // No need to check for nan
        }

        let next_x = x.get(j..j + 1, ..); // 1 by m, m = # of columns
        let next_y = y.get(j..j + 1, ..); // 1 by 1
        if !(has_nan(next_x) | has_nan(next_y)) {
            non_null_cnt_in_window += 1;
            online_lr.update_unchecked(next_x, next_y, T::one()); // No need to check for nan
        }

        if non_null_cnt_in_window >= m {
            coefficients.push(online_lr.fitted_values().to_owned());
        } else {
            coefficients.push(Mat::with_capacity(0, 0));
        }
    }
    coefficients
}

/// Update the inverse and the weights for one step in a Woodbury update.
/// Reference: https://cpb-us-w2.wpmucdn.com/sites.gatech.edu/dist/2/436/files/2017/07/22-notes-6250-f16.pdf
/// https://en.wikipedia.org/wiki/Woodbury_matrix_identity
#[inline(always)]
pub fn woodbury_step<T: RealField + Float>(
    inverse: MatMut<T>,
    weights: MatMut<T>,
    new_x: MatRef<T>,
    new_y: MatRef<T>,
    c: T, // +1 or -1, for a "update" and a "removal"
) {
    // It is truly amazing that the C in the Woodbury identity essentially controls the update and
    // and removal of a new record (rolling)... Linear regression seems to be designed by God to work so well

    let u = &inverse * new_x.transpose(); // corresponding to u in the reference
                                          // right = left.transpose() by the fact that if A is symmetric, invertible, A-1 is also symmetric
    let z = (c + *(new_x * &u).get(0, 0)).recip();
    // Update the information matrix's inverse. Page 56 of the gatech reference
    faer::linalg::matmul::matmul(
        inverse,
        faer::Accum::Add,
        &u,
        &u.transpose(),
        z.neg(),
        Par::rayon(0), //
    ); // inv is updated

    // Difference from estimate using prior weights vs. actual next y
    let y_diff = new_y - (new_x * &weights);
    // Update weights. Page 56, after 'Then',.. in gatech reference
    faer::linalg::matmul::matmul(
        weights,
        faer::Accum::Add,
        u,
        y_diff,
        z,
        Par::rayon(0), //
    ); // weights are updated
}
