#![allow(non_snake_case)]
use super::LinalgErrors;
use faer::{prelude::*, Side};
use faer_ext::IntoFaer;
use ndarray::ArrayView2;
use std::ops::Neg;

#[derive(Clone, Copy, Default)]
pub enum LRSolverMethods {
    SVD,
    Choleskey,
    #[default]
    QR,
}

impl From<&str> for LRSolverMethods {
    fn from(value: &str) -> Self {
        match value {
            "qr" => Self::QR,
            "svd" => Self::SVD,
            "choleskey" => Self::Choleskey,
            _ => Self::QR,
        }
    }
}

// add elastic net
#[derive(Clone, Copy, Default, PartialEq)]
pub enum LRMethods {
    #[default]
    Normal, // Normal. Normal Equation
    L1, // Lasso, L1 regularized
    L2, // Ridge, L2 regularized
}

impl From<&str> for LRMethods {
    fn from(value: &str) -> Self {
        match value {
            "l1" | "lasso" => Self::L1,
            "l2" | "ridge" => Self::L2,
            _ => Self::Normal,
        }
    }
}

// add elastic net
#[derive(Clone, Copy, Default, PartialEq)]
pub enum ClosedFormLRMethods {
    #[default]
    Normal, // Normal. Normal Equation
    L2, // Ridge, L2 regularized
}

impl From<&str> for ClosedFormLRMethods {
    fn from(value: &str) -> Self {
        match value {
            "l2" | "ridge" => Self::L2,
            _ => Self::Normal,
        }
    }
}

impl From<f64> for ClosedFormLRMethods {
    fn from(value: f64) -> Self {
        if value > 0. {
            Self::L2
        } else {
            Self::Normal
        }
    }
}

pub trait LinearModels {
    fn coefficients(&self) -> MatRef<f64>;

    /// Returns a copy of the coefficients
    fn get_coefficients(&self) -> Mat<f64> {
        self.coefficients().to_owned()
    }

    fn bias(&self) -> f64;

    fn has_bias(&self) -> bool;

    fn fit(&mut self, X: MatRef<f64>, y: MatRef<f64>) -> Result<(), LinalgErrors>;

    fn fit_unchecked(&mut self, X: MatRef<f64>, y: MatRef<f64>);

    fn is_fit(&self) -> bool {
        !(self.coefficients().shape() == (0, 0))
    }

    fn coeffs_as_vec(&self) -> Result<Vec<f64>, LinalgErrors> {
        match self.check_is_fit() {
            Ok(_) => Ok(self
                .coefficients()
                .col(0)
                .iter()
                .copied()
                .collect::<Vec<_>>()),
            Err(e) => Err(e),
        }
    }

    fn check_is_fit(&self) -> Result<(), LinalgErrors> {
        if self.is_fit() {
            Ok(())
        } else {
            Err(LinalgErrors::MatNotLearnedYet)
        }
    }

    fn predict(&self, X: MatRef<f64>) -> Result<Mat<f64>, LinalgErrors> {
        if X.ncols() != self.coefficients().nrows() {
            Err(LinalgErrors::DimensionMismatch)
        } else if !self.is_fit() {
            Err(LinalgErrors::MatNotLearnedYet)
        } else {
            if self.has_bias() && self.bias().abs() > f64::EPSILON {
                let temp = Mat::full(X.nrows(), 1, self.bias());
                Ok(X * self.coefficients() + temp)
            } else {
                Ok(X * self.coefficients())
            }
        }
    }
}

/// A struct that handles regular linear regression and Ridge regression.
pub struct LR {
    pub solver: LRSolverMethods,
    pub method: ClosedFormLRMethods,
    pub lambda: f64,
    pub coefficients: Mat<f64>, // n_features x 1 matrix, doesn't contain bias
    pub fit_bias: bool,
    pub bias: f64,
}

impl LR {
    pub fn new(solver: &str, lambda: f64, fit_bias: bool) -> Self {
        LR {
            solver: solver.into(),
            method: lambda.into(),
            lambda: lambda,
            coefficients: Mat::new(),
            fit_bias: fit_bias,
            bias: 0.,
        }
    }

    pub fn from_values(coeffs: &[f64], bias: f64) -> Self {
        LR {
            solver: LRSolverMethods::default(),
            method: ClosedFormLRMethods::default(),
            lambda: 0.,
            coefficients: faer::mat::from_row_major_slice(coeffs, coeffs.len(), 1).to_owned(),
            fit_bias: bias.abs() > f64::EPSILON,
            bias: bias,
        }
    }

    pub fn set_coeffs_and_bias(&mut self, coeffs: &[f64], bias: f64) {
        self.coefficients = (faer::mat::from_row_major_slice(coeffs, coeffs.len(), 1)).to_owned();
        self.bias = bias;
        self.fit_bias = bias.abs() > f64::EPSILON;
    }
}

impl LinearModels for LR {
    fn coefficients(&self) -> MatRef<f64> {
        self.coefficients.as_ref()
    }

    fn bias(&self) -> f64 {
        self.bias
    }

    fn has_bias(&self) -> bool {
        self.fit_bias
    }

    fn fit_unchecked(&mut self, X: MatRef<f64>, y: MatRef<f64>) {
        let all_coefficients = if self.fit_bias {
            let ones = Mat::full(X.nrows(), 1, 1.0);
            let new = faer::concat![[X, ones]];
            match self.method {
                ClosedFormLRMethods::Normal => faer_solve_lstsq(new.as_ref(), y, self.solver),
                ClosedFormLRMethods::L2 => {
                    faer_solve_ridge(new.as_ref(), y, self.lambda, self.fit_bias, self.solver)
                }
            }
        } else {
            match self.method {
                ClosedFormLRMethods::Normal => faer_solve_lstsq(X, y, self.solver),
                ClosedFormLRMethods::L2 => {
                    faer_solve_ridge(X, y, self.lambda, self.fit_bias, self.solver)
                }
            }
        };
        if self.fit_bias {
            let n = all_coefficients.nrows();
            let slice = all_coefficients.col_as_slice(0);
            self.coefficients =
                faer::mat::from_row_major_slice(&slice[..n - 1], n - 1, 1).to_owned();
            self.bias = slice[n - 1];
        } else {
            self.coefficients = all_coefficients;
        }
    }

    /// Fits the linear regression. Input X is any m x n matrix. Input y must be a m x 1 matrix.
    /// Note, if there is a bias term in the data, then it must be in the matrix X as the last
    /// column and has_bias must be true. This will not append a bias column to X.
    /// This doesn't check dimensions.
    fn fit(&mut self, X: MatRef<f64>, y: MatRef<f64>) -> Result<(), LinalgErrors> {
        if X.nrows() != y.nrows() {
            return Err(LinalgErrors::DimensionMismatch);
        } else if X.nrows() < X.ncols() {
            return Err(LinalgErrors::NotEnoughRows);
        }
        self.fit_unchecked(X, y);
        Ok(())
    }
}

/// A struct that handles online linear regression
pub struct OnlineLR {
    pub method: ClosedFormLRMethods,
    pub lambda: f64,
    pub fit_bias: bool,
    pub bias: f64,
    pub coefficients: Mat<f64>, // n_features x 1 matrix, doesn't contain bias
    pub inv: Mat<f64>,          // Current Inverse of X^t X
}

impl OnlineLR {
    pub fn new(lambda: f64, fit_bias: bool) -> Self {
        OnlineLR {
            method: lambda.into(),
            lambda: lambda,
            fit_bias: fit_bias,
            bias: 0.,
            coefficients: Mat::new(),
            inv: Mat::new(),
        }
    }

    pub fn set_coeffs_bias_inverse(
        &mut self,
        coeffs: &[f64],
        inv: ArrayView2<f64>,
        bias: f64,
    ) -> Result<(), LinalgErrors> {
        if coeffs.len() != inv.ncols() {
            Err(LinalgErrors::DimensionMismatch)
        } else {
            self.coefficients =
                (faer::mat::from_row_major_slice(coeffs, coeffs.len(), 1)).to_owned();
            self.inv = inv.into_faer().to_owned();
            self.bias = bias;
            self.fit_bias = bias.abs() > f64::EPSILON;
            Ok(())
        }
    }

    pub fn get_inv(&self) -> Result<MatRef<f64>, LinalgErrors> {
        if self.inv.shape() == (0, 0) {
            Err(LinalgErrors::MatNotLearnedYet)
        } else {
            Ok(self.inv.as_ref())
        }
    }

    pub fn update_unchecked(&mut self, new_x: MatRef<f64>, new_y: MatRef<f64>, c: f64) {
        woodbury_step(
            self.inv.as_mut(),
            self.coefficients.as_mut(),
            new_x,
            new_y,
            c,
        )
    }

    pub fn update(&mut self, new_x: MatRef<f64>, new_y: MatRef<f64>, c: f64) {
        if !(new_x.has_nan() || new_y.has_nan()) {
            woodbury_step(
                self.inv.as_mut(),
                self.coefficients.as_mut(),
                new_x,
                new_y,
                c,
            )
        }
    }
}

impl LinearModels for OnlineLR {
    fn coefficients(&self) -> MatRef<f64> {
        self.coefficients.as_ref()
    }

    fn bias(&self) -> f64 {
        self.bias
    }

    fn has_bias(&self) -> bool {
        self.fit_bias
    }

    fn fit_unchecked(&mut self, X: MatRef<f64>, y: MatRef<f64>) {
        (self.inv, self.coefficients) = match self.method {
            ClosedFormLRMethods::Normal => faer_qr_lstsq_with_inv(X, y),
            ClosedFormLRMethods::L2 => {
                faer_cholesky_ridge_with_inv(X, y, self.lambda, self.fit_bias)
            }
        };
    }

    /// The very initial fit of the Online Regression Problem.
    /// Defaults to Column Pivot QR for normal and Cholesky for Ridge
    fn fit(&mut self, X: MatRef<f64>, y: MatRef<f64>) -> Result<(), LinalgErrors> {
        if X.nrows() != y.nrows() {
            return Err(LinalgErrors::DimensionMismatch);
        } else if X.nrows() < X.ncols() {
            return Err(LinalgErrors::NotEnoughRows);
        }
        self.fit_unchecked(X, y);
        Ok(())
    }
}

//------------------------------------ The Basic Functions ---------------------------------------

/// Returns the coefficients for lstsq as a nrows x 1 matrix
#[inline(always)]
pub fn faer_solve_lstsq(x: MatRef<f64>, y: MatRef<f64>, how: LRSolverMethods) -> Mat<f64> {
    match how {
        LRSolverMethods::SVD => (x.transpose() * x).thin_svd().solve(x.transpose() * y),
        LRSolverMethods::QR => x.col_piv_qr().solve_lstsq(y),
        LRSolverMethods::Choleskey => match (x.transpose() * x).cholesky(Side::Lower) {
            Ok(cho) => cho.solve(x.transpose() * y),
            Err(_) => x.col_piv_qr().solve_lstsq(y),
        },
    }
}

/// Least square that sets all singular values below threshold to 0.
/// Returns the coefficients and the singular values
#[inline(always)]
pub fn faer_solve_lstsq_rcond(x: MatRef<f64>, y: MatRef<f64>, rcond:f64) -> (Mat<f64>, Vec<f64>) {

    let xt = x.transpose();
    let svd = (xt * x).thin_svd();

    let singular_values = svd
        .s_diagonal()
        .iter()
        .copied()
        .map(f64::sqrt)
        .collect::<Vec<_>>();

    let max_singular_value = singular_values.iter().copied().fold(f64::MIN, f64::max);
    let threshold = rcond * max_singular_value;

    let s_inv = svd
        .s_diagonal()
        .iter()
        .copied()
        .map(|x| if x >= threshold { x.recip() } else { 0. } )
        .collect::<Vec<_>>();

    let s_inv = faer::mat::from_row_major_slice(&s_inv, s_inv.len(), 1);
    let s_inv = s_inv.column_vector_as_diagonal();

    let weights = svd.v() * s_inv * svd.u().transpose() * xt * y;
    (weights, singular_values)    

}

/// Least square that sets all singular values below threshold to 0.
/// Returns the coefficients and the singular values
#[inline(always)]
pub fn faer_solve_ridge_rcond(
    x: MatRef<f64>, 
    y: MatRef<f64>, 
    lambda: f64,
    has_bias: bool,
    rcond:f64
) -> (Mat<f64>, Vec<f64>) {


    let n1 = x.ncols().abs_diff(has_bias as usize);
    let xt = x.transpose();
    let mut xtx_plus = xt * x;
    // xtx + diagonal of lambda. If has bias, last diagonal element is 0.
    // Safe. Index is valid and value is initialized.
    for i in 0..n1 {
        *unsafe { xtx_plus.get_mut_unchecked(i, i) } += lambda;
    }

    let svd = xtx_plus.thin_svd();
    let singular_values = svd
        .s_diagonal()
        .iter()
        .copied()
        .map(f64::sqrt)
        .collect::<Vec<_>>();

    let max_singular_value = singular_values.iter().copied().fold(f64::MIN, f64::max);
    let threshold = rcond * max_singular_value;

    let s_inv = svd
        .s_diagonal()
        .iter()
        .copied()
        .map(|x| if x >= threshold { x.recip() } else { 0. } )
        .collect::<Vec<_>>();

    let s_inv = faer::mat::from_row_major_slice(&s_inv, s_inv.len(), 1);
    let s_inv = s_inv.column_vector_as_diagonal();

    let weights = svd.v() * s_inv * svd.u().transpose() * xt * y;
    (weights, singular_values)    

}



/// Returns the coefficients for lstsq with l2 (Ridge) regularization as a nrows x 1 matrix
#[inline(always)]
pub fn faer_solve_ridge(
    x: MatRef<f64>,
    y: MatRef<f64>,
    lambda: f64,
    has_bias: bool,
    how: LRSolverMethods,
) -> Mat<f64> {
    // Add ridge SVD with rconditional number later.

    let n1 = x.ncols().abs_diff(has_bias as usize);
    let xt = x.transpose();
    let mut xtx_plus = xt * x;
    // xtx + diagonal of lambda. If has bias, last diagonal element is 0.
    // Safe. Index is valid and value is initialized.
    for i in 0..n1 {
        *unsafe { xtx_plus.get_mut_unchecked(i, i) } += lambda;
    }

    match how {
        LRSolverMethods::Choleskey => match xtx_plus.cholesky(Side::Lower) {
            Ok(cho) => cho.solve(xt * y),
            Err(_) => xtx_plus.thin_svd().solve(xt * y),
        },
        LRSolverMethods::SVD => xtx_plus.thin_svd().solve(xt * y),
        LRSolverMethods::QR => xtx_plus.col_piv_qr().solve(xt * y),
    }
}

/// Returns the coefficients for lstsq as a nrows x 1 matrix together with the inverse of XtX
/// The uses QR (column pivot) decomposition as default method to compute inverse,
/// Column Pivot QR is chosen to deal with rank deficient cases. It is also slightly
/// faster compared to other methods.
#[inline(always)]
pub fn faer_qr_lstsq_with_inv(x: MatRef<f64>, y: MatRef<f64>) -> (Mat<f64>, Mat<f64>) {
    let xt = x.transpose();
    let qr = (xt * x).col_piv_qr();
    let inv = qr.inverse();
    let weights = qr.solve(xt * y);
    (inv, weights)
}

/// Returns the coefficients for lstsq with l2 (Ridge) regularization as a nrows x 1 matrix
/// By default this uses Choleskey to solve the system, and in case the matrix is not positive
/// definite, it falls back to SVD. (I suspect the matrix in this case is always positive definite!)
#[inline(always)]
pub fn faer_cholesky_ridge_with_inv(
    x: MatRef<f64>,
    y: MatRef<f64>,
    lambda: f64,
    has_bias: bool,
) -> (Mat<f64>, Mat<f64>) {
    let n1 = x.ncols().abs_diff(has_bias as usize);

    let xt = x.transpose();
    let mut xtx_plus = xt * x;
    // xtx + diagonal of lambda. If has bias, last diagonal element is 0.
    // Safe. Index is valid and value is initialized.
    for i in 0..n1 {
        *unsafe { xtx_plus.get_mut_unchecked(i, i) } += lambda;
    }

    match xtx_plus.cholesky(Side::Lower) {
        Ok(cho) => {
            let inv = cho.inverse();
            (inv, cho.solve(xt * y))
        }
        Err(_) => {
            let svd = xtx_plus.thin_svd();
            (svd.inverse(), svd.solve(xt * y))
        }
    }
}

/// Solves the weighted least square with weights given by the user
#[inline(always)]
pub fn faer_weighted_lstsq(x: MatRef<f64>, y: MatRef<f64>, w: &[f64], how:LRSolverMethods) -> Mat<f64> {
    let weights = faer::mat::from_row_major_slice(w, x.nrows(), 1);
    let w = weights.column_vector_as_diagonal();
    let xt = x.transpose();
    let xtw = xt * w;
    match how {
        LRSolverMethods::SVD => {
            let svd = (&xtw * x).thin_svd();
            svd.solve(xtw * y)
        },
        LRSolverMethods::QR => {
            let qr = (&xtw * x).col_piv_qr();
            qr.solve(xtw * y)
        },
        LRSolverMethods::Choleskey => match (&xtw * x).cholesky(Side::Lower) {
            Ok(cho) => cho.solve(xtw * y),
            Err(_) => {
                let svd = (&xtw * x).thin_svd();
                svd.solve(xtw * y)
            },
        },
    }
}

#[inline(always)]
fn soft_threshold_l1(rho: f64, lambda: f64) -> f64 {
    rho.signum() * (rho.abs() - lambda).max(0f64)
}

/// Computes Lasso Regression coefficients by the use of Coordinate Descent.
/// The current stopping criterion is based on L Inf norm of the changes in the
/// coordinates. A better alternative might be the dual gap.
///
/// Reference:
/// https://xavierbourretsicotte.github.io/lasso_implementation.html
/// https://github.com/minatosato/Lasso/blob/master/coordinate_descent_lasso.py
/// https://en.wikipedia.org/wiki/Lasso_(statistics)
#[inline(always)]
pub fn faer_lasso_regression(
    x: MatRef<f64>,
    y: MatRef<f64>,
    lambda: f64,
    has_bias: bool,
    tol: f64,
) -> Mat<f64> {
    let m = x.nrows() as f64;
    let ncols = x.ncols();
    let n1 = ncols.abs_diff(has_bias as usize);

    let lambda_new = m * lambda;

    let mut beta: Mat<f64> = Mat::zeros(ncols, 1);
    let mut converge = false;

    // compute column squared l2 norms.
    let norms = x
        .col_iter()
        .map(|c| c.squared_norm_l2())
        .collect::<Vec<_>>();

    let xty = x.transpose() * y;
    let xtx = x.transpose() * x;

    // Random selection often leads to faster convergence?
    for _ in 0..2000 {
        let mut max_change = 0f64;
        for j in 0..n1 {
            // temporary set beta(j, 0) to 0.
            // Safe. The index is valid and the value is initialized.
            let before = *unsafe { beta.get_unchecked(j, 0) };
            *unsafe { beta.get_mut_unchecked(j, 0) } = 0f64;
            let xtx_j = unsafe { xtx.get_unchecked(j..j + 1, ..) };

            let dot = xty.read(j, 0) - (xtx_j * &beta).read(0, 0);

            // update beta(j, 0).
            let after = soft_threshold_l1(dot, lambda_new) / norms[j];
            *unsafe { beta.get_mut_unchecked(j, 0) } = after;
            max_change = (after - before).abs().max(max_change);
        }
        // if has_bias, n1 = last index = ncols - 1 = column of bias. If has_bias is False, n = ncols
        if has_bias {
            // Safe. The index is valid and the value is initialized.
            let xx = unsafe { x.get_unchecked(.., 0..n1) };
            let bb = unsafe { beta.get_unchecked(0..n1, ..) };
            let ss = (y - xx * bb).sum() / m;
            *unsafe { beta.get_mut_unchecked(n1, 0) } = ss;
        }
        converge = max_change < tol;
        if converge {
            break;
        }
    }

    if !converge {
        println!("Lasso regression: 2000 iterations have passed and result hasn't converged.")
    }

    beta
}

/// Given all data, we start running a lstsq starting at position n and compute new coefficients recurisively.
/// This will return all coefficients for rows >= n. This will only be used in Polars Expressions.
pub fn faer_recursive_lstsq(
    x: MatRef<f64>,
    y: MatRef<f64>,
    n: usize,
    lambda: f64,
    has_bias: bool,
) -> Vec<Mat<f64>> {
    let xn = x.nrows();
    // x: size xn x m
    // y: size xn x 1
    // Vector of matrix of size m x 1
    let mut coefficients = Vec::with_capacity(xn - n + 1);
    // n >= 2, guaranteed by Python
    let x0 = x.get(..n, ..);
    let y0 = y.get(..n, ..);

    let mut online_lr = OnlineLR::new(lambda, has_bias);
    online_lr.fit_unchecked(x0, y0); // safe because things are checked in plugin / python functions.
    coefficients.push(online_lr.get_coefficients());
    for j in n..xn {
        let next_x = x.get(j..j + 1, ..); // 1 by m, m = # of columns
        let next_y = y.get(j..j + 1, ..); // 1 by 1
        online_lr.update(next_x, next_y, 1.0);
        coefficients.push(online_lr.get_coefficients());
    }
    coefficients
}

/// Given all data, we start running a lstsq starting at position n and compute new coefficients recurisively.
/// This will return all coefficients for rows >= n. This will only be used in Polars Expressions.
/// This supports Normal or Ridge regression
pub fn faer_rolling_lstsq(
    x: MatRef<f64>,
    y: MatRef<f64>,
    n: usize,
    lambda: f64,
    has_bias: bool,
) -> Vec<Mat<f64>> {
    let xn = x.nrows();
    // x: size xn x m
    // y: size xn x 1
    // Vector of matrix of size m x 1
    let mut coefficients = Vec::with_capacity(xn - n + 1); // xn >= n is checked in Python

    let x0 = x.get(..n, ..);
    let y0 = y.get(..n, ..);

    let mut online_lr = OnlineLR::new(lambda, has_bias);
    online_lr.fit_unchecked(x0, y0);
    coefficients.push(online_lr.get_coefficients());

    for j in n..xn {
        let remove_x = x.get(j - n..j - n + 1, ..);
        let remove_y = y.get(j - n..j - n + 1, ..);
        online_lr.update(remove_x, remove_y, -1.0);

        let next_x = x.get(j..j + 1, ..); // 1 by m, m = # of columns
        let next_y = y.get(j..j + 1, ..); // 1 by 1
        online_lr.update(next_x, next_y, 1.0);
        coefficients.push(online_lr.get_coefficients());
    }
    coefficients
}

/// Given all data, we start running a lstsq starting at position n and compute new coefficients recurisively.
/// This will return all coefficients for rows >= n. This will only be used in Polars Expressions.
/// If # of non-null rows in the window is < m, a Matrix with size (0, 0) will be returned.
/// This supports Normal or Ridge regression
pub fn faer_rolling_skipping_lstsq(
    x: MatRef<f64>,
    y: MatRef<f64>,
    n: usize,
    m: usize,
    lambda: f64,
    has_bias: bool,
) -> Vec<Mat<f64>> {
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
    let mut x_slice: Vec<f64> = Vec::with_capacity(n * ncols);
    let mut y_slice: Vec<f64> = Vec::with_capacity(n);

    let mut online_lr = OnlineLR::new(lambda, has_bias);
    while right <= xn {
        // Somewhat redundant here.
        non_null_cnt_in_window = 0;
        x_slice.clear();
        y_slice.clear();
        for i in left..right {
            let x_i = x.get(i, ..);
            let y_i = y.get(i, ..);
            if !(x_i.has_nan() | y_i.has_nan()) {
                non_null_cnt_in_window += 1;
                x_slice.extend(x_i.iter());
                y_slice.extend(y_i.iter());
            }
        }
        if non_null_cnt_in_window >= m {
            let x0 = faer::mat::from_row_major_slice(&x_slice, y_slice.len(), ncols);
            let y0 = faer::mat::from_row_major_slice(&y_slice, y_slice.len(), 1);
            online_lr.fit_unchecked(x0, y0);
            coefficients.push(online_lr.get_coefficients());
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
        if !(remove_x.has_nan() || remove_y.has_nan()) {
            non_null_cnt_in_window -= 1; // removed one non-null column
            online_lr.update_unchecked(remove_x, remove_y, -1.0); // No need to check for nan
        }

        let next_x = x.get(j..j + 1, ..); // 1 by m, m = # of columns
        let next_y = y.get(j..j + 1, ..); // 1 by 1
        if !(next_x.has_nan() || next_y.has_nan()) {
            non_null_cnt_in_window += 1;
            online_lr.update_unchecked(next_x, next_y, 1.0); // No need to check for nan
        }

        if non_null_cnt_in_window >= m {
            coefficients.push(online_lr.get_coefficients());
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
fn woodbury_step(
    inverse: MatMut<f64>,
    weights: MatMut<f64>,
    new_x: MatRef<f64>,
    new_y: MatRef<f64>,
    c: f64, // Should be +1 or -1, for a "update" and a "removal"
) {
    // It is truly amazing that the C in the Woodbury identity essentially controls the update and
    // and removal of a new record (rolling)... Linear regression seems to be designed by God to work so well

    let left = &inverse * new_x.transpose(); // corresponding to u in the reference
                                             // right = left.transpose() by the fact that if A is symmetric, invertible, A-1 is also symmetric
    let z = (c + (new_x * &left).read(0, 0)).recip();
    // Update the inverse
    faer::linalg::matmul::matmul(
        inverse,
        &left,
        left.transpose(),
        Some(1.0),
        z.neg(),
        faer::Parallelism::Rayon(0), //
    ); // inv is updated

    // Difference from esitmate using prior weights vs. actual next y
    let y_diff = new_y - (new_x * &weights);
    // Update weights
    faer::linalg::matmul::matmul(
        weights,
        left,
        y_diff,
        Some(1.0),
        z,
        faer::Parallelism::Rayon(0), //
    ); // weights are updated
}
