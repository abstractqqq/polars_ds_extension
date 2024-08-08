use faer::{prelude::*, scale, Side};
use polars::prelude::len;
use std::ops::Neg;

// add elastic net
pub enum LRMethods {
    Normal, // Normal. Normal Equation
    L1,     // Lasso, L1 regularized
    L2,     // Ridge, L2 regularized
}

impl From<String> for LRMethods {
    fn from(value: String) -> Self {
        match value.as_str() {
            "l1" | "lasso" => LRMethods::L1,
            "l2" | "ridge" => LRMethods::L2,
            _ => LRMethods::Normal,
        }
    }
}

#[inline(always)]
fn soft_threshold_l1(rho: f64, lambda: f64) -> f64 {
    rho.signum() * (rho.abs() - lambda).max(0f64)
}

/// Returns the coefficients for lstsq as a nrows x 1 matrix
/// The uses QR (column pivot) decomposition as default method to compute inverse,
/// Column Pivot QR is chosen to deal with rank deficient cases. It is also slightly
/// faster compared to other methods.
#[inline(always)]
pub fn faer_qr_lstsq(x: MatRef<f64>, y: MatRef<f64>) -> Mat<f64> {
    x.col_piv_qr().solve_lstsq(y)
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

/// Returns the coefficients for lstsq with l2 (Ridge) regularization as a nrows x 1 matrix
/// By default this uses Choleskey to solve the system, and in case the matrix is not positive
/// definite, it falls back to SVD. (I suspect the matrix in this case is always positive definite!)
#[inline(always)]
pub fn faer_cholskey_ridge_regression(
    x: MatRef<f64>,
    y: MatRef<f64>,
    lambda: f64,
    has_bias: bool,
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

    match xtx_plus.cholesky(Side::Lower) {
        Ok(cho) => cho.solve(xt * y),
        Err(_) => xtx_plus.thin_svd().solve(xt * y),
    }
}

/// Given all data, we start running a lstsq starting at position n and compute new coefficients recurisively.
/// This will return all coefficients for rows >= n. This will only be used in Polars Expressions.
pub fn faer_recursive_lstsq(x: MatRef<f64>, y: MatRef<f64>, n: usize) -> Vec<Mat<f64>> {
    let xn = x.nrows();
    // x: size xn x m
    // y: size xn x 1
    // Vector of matrix of size m x 1
    let mut coefficients = Vec::with_capacity(xn - n * ((n > 1) as usize));

    let nn = n.max(1); // if n = 0, start with first row of data, which is the same as n = 1.
    let x0 = x.get(..nn, ..);
    let y0 = y.get(..nn, ..);

    let x0t = x0.transpose();
    let qr = (x0t * x0).col_piv_qr();
    let mut inv = qr.inverse();
    let mut weights = qr.solve(x0t * y0); //

    coefficients.push(weights.to_owned());
    for j in nn..xn {
        let next_x = x.get(j..j + 1, ..); // 1 by m, m = # of columns
        let next_y = y.get(j..j + 1, ..); // 1 by 1
        woodbury_step(inv.as_mut(), weights.as_mut(), next_x, next_y, 1.0);
        coefficients.push(weights.to_owned());
    }
    coefficients
}

/// Given all data, we start running a lstsq starting at position n and compute new coefficients recurisively.
/// This will return all coefficients for rows >= n. This will only be used in Polars Expressions.
pub fn faer_recursive_ridge(
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
    let mut coefficients = Vec::with_capacity(xn - n * ((n > 1) as usize));

    let nn = n.max(1); // if n = 0, start with first row of data, which is the same as n = 1.
    let x0 = x.get(..nn, ..);
    let y0 = y.get(..nn, ..);

    let x0t = x0.transpose();
    let mut x0tx0_plus = x0t * x0;

    let n1 = x.ncols().abs_diff(has_bias as usize);
    // No bias at this moment
    for i in 0..n1 {
        *unsafe { x0tx0_plus.get_mut_unchecked(i, i) } += lambda;
    }

    let (mut inv, mut weights) = match x0tx0_plus.cholesky(Side::Lower) {
        Ok(cho) => (cho.inverse(), cho.solve(x0t * y0)),
        Err(_) => {
            let svd = x0tx0_plus.thin_svd();
            (svd.inverse(), svd.solve(x0t * y0))
        }
    };

    coefficients.push(weights.to_owned());
    for j in nn..xn {
        let next_x = x.get(j..j + 1, ..); // 1 by m, m = # of columns
        let next_y = y.get(j..j + 1, ..); // 1 by 1
        woodbury_step(inv.as_mut(), weights.as_mut(), next_x, next_y, 1.0);
        coefficients.push(weights.to_owned());
    }
    coefficients
}

// Need to Re-think the structure if I want to expose this to Python and NumPy. The biggest difficulty, however,
// is more about Faer interop with NumPy at this moment.

/// Given all data, we start running a lstsq starting at position n and compute new coefficients recurisively.
/// This will return all coefficients for rows >= n. This will only be used in Polars Expressions.
pub fn faer_rolling_lstsq(x: MatRef<f64>, y: MatRef<f64>, n: usize) -> Vec<Mat<f64>> {
    let xn = x.nrows();
    // x: size xn x m
    // y: size xn x 1
    // Vector of matrix of size m x 1
    let mut coefficients = Vec::with_capacity(xn - n + 1); // xn >= n is checked in Python

    let x0 = x.get(..n, ..);
    let y0 = y.get(..n, ..);
    let x0t = x0.transpose();
    let qr = (x0t * x0).col_piv_qr();
    let mut inv = qr.inverse();
    let mut weights = qr.solve(x0t * y0); //
    coefficients.push(weights.to_owned());

    for j in n..xn {
        let remove_x = x.get(j - n..j - n + 1, ..);
        let remove_y = y.get(j - n..j - n + 1, ..);
        woodbury_step(inv.as_mut(), weights.as_mut(), remove_x, remove_y, -1.0);

        let next_x = x.get(j..j + 1, ..); // 1 by m, m = # of columns
        let next_y = y.get(j..j + 1, ..); // 1 by 1
        woodbury_step(inv.as_mut(), weights.as_mut(), next_x, next_y, 1.0);
        coefficients.push(weights.to_owned());
    }
    coefficients
}

/// Given all data, we start running a lstsq starting at position n and compute new coefficients recurisively.
/// This will return all coefficients for rows >= n. This will only be used in Polars Expressions.
/// If # of non-null rows in the window is < m, a Matrix with size (0, 0) will be returned.
pub fn faer_rolling_skipping_lstsq(
    x: MatRef<f64>,
    y: MatRef<f64>,
    n: usize,
    m: usize,
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
    let mut inv = Mat::with_capacity(ncols, ncols);
    let mut weights = Mat::with_capacity(ncols, 1);

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
            let x0t = x0.transpose();
            let qr = (x0t * x0).col_piv_qr();
            inv = qr.inverse();
            weights = qr.solve(x0t * y0); //
            coefficients.push(weights.to_owned());
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
        if !(remove_x.has_nan() | remove_y.has_nan()) {
            non_null_cnt_in_window -= 1; // removed one non-null column
            woodbury_step(inv.as_mut(), weights.as_mut(), remove_x, remove_y, -1.0);
        }

        let next_x = x.get(j..j + 1, ..); // 1 by m, m = # of columns
        let next_y = y.get(j..j + 1, ..); // 1 by 1
        if !(next_x.has_nan() | next_y.has_nan()) {
            non_null_cnt_in_window += 1;
            woodbury_step(inv.as_mut(), weights.as_mut(), next_x, next_y, 1.0);
        }

        if non_null_cnt_in_window >= m {
            coefficients.push(weights.to_owned());
        } else {
            coefficients.push(Mat::with_capacity(0, 0));
        }
    }
    coefficients
}

/// Given all data, we start running a lstsq starting at position n and compute new coefficients recurisively.
/// This will return all coefficients for rows >= n. This will only be used in Polars Expressions.
pub fn faer_rolling_ridge(
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
    let mut coefficients = Vec::with_capacity(xn - n); // n >= 2 guaranteed in Python

    let x0 = x.get(..n, ..);
    let y0 = y.get(..n, ..);

    let x0t = x0.transpose();
    let mut x0tx0_plus = x0t * x0;

    let n1 = x.ncols().abs_diff(has_bias as usize);
    // No bias at this moment
    for i in 0..n1 {
        *unsafe { x0tx0_plus.get_mut_unchecked(i, i) } += lambda;
    }

    let (mut inv, mut weights) = match x0tx0_plus.cholesky(Side::Lower) {
        Ok(cho) => (cho.inverse(), cho.solve(x0t * y0)),
        Err(_) => {
            let svd = x0tx0_plus.thin_svd();
            (svd.inverse(), svd.solve(x0t * y0))
        }
    };

    coefficients.push(weights.to_owned());
    for j in n..xn {
        let remove_x = x.get(j - n..j - n + 1, ..);
        let remove_y = y.get(j - n..j - n + 1, ..);
        woodbury_step(inv.as_mut(), weights.as_mut(), remove_x, remove_y, -1.0);

        let next_x = x.get(j..j + 1, ..); // 1 by m, m = # of columns
        let next_y = y.get(j..j + 1, ..); // 1 by 1
        woodbury_step(inv.as_mut(), weights.as_mut(), next_x, next_y, 1.0);
        coefficients.push(weights.to_owned());
    }
    coefficients
}

/// Given all data, we start running a lstsq starting at position n and compute new coefficients recurisively.
/// This will return all coefficients for rows >= n. This will only be used in Polars Expressions.
/// If # of non-null rows in the window is < m, a Matrix with size (0, 0) will be returned.
pub fn faer_rolling_skipping_ridge(
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

    // Refactor and make some parts common :)

    // Initialize the problem.
    let mut non_null_cnt_in_window = 0;
    let mut left = 0;
    let mut right = n;
    let mut x_slice: Vec<f64> = Vec::with_capacity(n * ncols);
    let mut y_slice: Vec<f64> = Vec::with_capacity(n);
    let mut inv = Mat::with_capacity(ncols, ncols);
    let mut weights = Mat::with_capacity(ncols, 1);

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
            let x0t = x0.transpose();
            let mut x0tx0_plus = x0t * x0;
            let n1 = x.ncols().abs_diff(has_bias as usize);
            // No bias at this moment
            for i in 0..n1 {
                *unsafe { x0tx0_plus.get_mut_unchecked(i, i) } += lambda;
            }
            (inv, weights) = match x0tx0_plus.cholesky(Side::Lower) {
                Ok(cho) => (cho.inverse(), cho.solve(x0t * y0)),
                Err(_) => {
                    let svd = x0tx0_plus.thin_svd();
                    (svd.inverse(), svd.solve(x0t * y0))
                }
            };
            coefficients.push(weights.to_owned());
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
        if !(remove_x.has_nan() | remove_y.has_nan()) {
            non_null_cnt_in_window -= 1; // removed one non-null column
            woodbury_step(inv.as_mut(), weights.as_mut(), remove_x, remove_y, -1.0);
        }

        let next_x = x.get(j..j + 1, ..); // 1 by m, m = # of columns
        let next_y = y.get(j..j + 1, ..); // 1 by 1
        if !(next_x.has_nan() | next_y.has_nan()) {
            non_null_cnt_in_window += 1;
            woodbury_step(inv.as_mut(), weights.as_mut(), next_x, next_y, 1.0);
        }

        if non_null_cnt_in_window >= m {
            coefficients.push(weights.to_owned());
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
