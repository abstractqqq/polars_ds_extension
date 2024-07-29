use std::ops::Neg;

use faer::{prelude::*, scale, Side};
use faer_ext::IntoFaer;
use ndarray::ArrayView2;

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
    let qr = x.col_piv_qr();
    qr.solve_lstsq(y)
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

    for _ in 0..2000 {
        let mut max_change = 0f64;
        // Random selection often leads to faster convergence?
        for j in 0..n1 {
            // temporary set beta(j, 0) to 0.
            // Safe. The index is valid and the value is initialized.
            let before = *unsafe { beta.get_unchecked(j, 0) };
            *unsafe { beta.get_mut_unchecked(j, 0) } = 0f64;
            let xtx_j = unsafe { xtx.get_unchecked(j..j + 1, ..) };

            // let x_j = j th column of x
            // let dot = x_j.transpose() * (y - x * &beta); // Slow.
            // xty.read(j, 0) = x_j.transpose() * y,  xtx_j * &beta = x_j.transpose() * x * &beta
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
pub fn faer_recursive_lstsq(    
    x: MatRef<f64>,
    y: MatRef<f64>,
    n: usize,
) -> Vec<Mat<f64>>{

    let xn = x.nrows();
    // x: size xn x m
    // y: size xn x 1
    // Vector of matrix of size m x 1
    let mut coefficients = Vec::with_capacity(xn-n); // xn >= n is checked in Python

    let x0 = x.get(..n, ..);
    let y0 = y.get(..n, ..);

    let x0t = x0.transpose();
    let qr = (x0t * x0).qr();
    let mut inv = qr.inverse();
    // The y part of the update (The non Sherman-Morrison part)
    let mut weights = &inv * x0t * y0;
    coefficients.push(weights.to_owned());
    for j in n..xn {

        let next_x = x.get(j..j+1, ..); // 1 by m, m = # of columns
        let next_y = y.get(j..j+1, ..); // 1 by 1

        let left = &inv * next_x.transpose();
        let right = next_x * &inv;
        let denominator = 1f64 + (&right * next_x.transpose()).read(0, 0);
        // Update the inverse
        faer::linalg::matmul::matmul(
            inv.as_mut(), 
            left, 
            right, 
            Some(1.0), 
            denominator.recip().neg(), 
            faer::Parallelism::Rayon(0) // 
        ); // inv is updated

        let scaler = (next_y - (next_x * &weights)).read(0, 0);
        // Update weights
        faer::linalg::matmul::matmul(
            weights.as_mut(), 
            &inv, 
            next_x.transpose(), 
            Some(1.0), 
            scaler, 
            faer::Parallelism::Rayon(0) // 
        ); // weights is updated

        coefficients.push(weights.to_owned());
    }
    coefficients
}

// /// Initial fit for a recursive lstsq.
// /// This will return the inverse matrix (XtX) and the coefficients.
// pub fn faer_recursive_lstsq_init(    
//     x: MatRef<f64>, // n x m matrix, n >= 1
//     y: MatRef<f64>, // n x 1 matrix)
// ) -> (Mat<f64>, Mat<f64>) {

//     let xtx = x.transpose() * x;
//     let qr = xtx.col_piv_qr();
//     let inv = qr.inverse();
//     let coeffs = &inv * x.transpose() * y;
//     (inv, coeffs)
// }

// /// Batch update of the resursive lstsq
// /// This performs inplace updates and returns the inverse matrix (XtX) and the coefficients.
// pub fn faer_recursive_lstsq_batch_update(    
//     x: MatRef<f64>, // n x m matrix, n >= 1
//     y: MatRef<f64>, // n x 1 matrix
//     prev_weight: MatRef<f64>, // m x 1 matrix
//     prev_inv: MatRef<f64> // m x m matrix
// ) -> (Mat<f64>, Mat<f64>) {

//     let mut inv = prev_inv.to_owned();
//     let mut weights = prev_weight.to_owned();

//     for j in 0..x.nrows() {

//         let next_x = x.get(j..j+1, ..); // 1 by m, m = # of columns
//         let next_y = y.get(j..j+1, ..); // 1 by 1

//         let left = &inv * next_x.transpose();
//         let right = next_x * &inv;
//         let denominator = 1f64 + (&right * next_x.transpose()).read(0, 0);
//         // Update the inverse
//         faer::linalg::matmul::matmul(
//             inv.as_mut(), 
//             left, 
//             right, 
//             Some(1.0), 
//             denominator.recip().neg(), 
//             faer::Parallelism::Rayon(0) // 
//         ); // inv is updated

//         let scaler = (next_y - (next_x * &weights)).read(0, 0);
//         // Update weights
//         faer::linalg::matmul::matmul(
//             weights.as_mut(), 
//             &inv, 
//             next_x.transpose(), 
//             Some(1.0), 
//             scaler, 
//             faer::Parallelism::Rayon(0) // 
//         ); // weights is updated
//     }
//     (inv, weights)
// }