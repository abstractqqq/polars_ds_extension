use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use faer::{ColRef, Mat, Par, Side, prelude::Solve, sparse::{SparseColMat, Triplet}};

use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub(crate) struct SplineKwargs {
    pub(crate) lambda: f64,
}

/// Computes a cubic smoothing spline for 1D data using a fixed penalty.
///
/// This algorithm minimizes the penalized least-squares functional, balancing 
/// the fidelity to the data with the smoothness of the curve (measured by the 
/// integral of the squared second derivative).
///
/// Mathematically, it solves the linear system: (I + lambda * Q * R^-1 * Q^T) g = y
/// where `Q` represents the second difference operator and `R` represents the 
/// basis function integrations.
///
/// For more details, see the maths/ folder of the project. This implementation
/// is outlined by AI and the code is original.
///
/// # Arguments
/// * `x` - The strictly increasing independent variables.
/// * `y` - The dependent variables.
/// * `lambda` - The smoothing penalty (0.0 = perfect interpolation, higher = straighter line).
pub fn faer_smooth_spline(x: &[f64], y: &[f64], lambda: f64) -> Result<Mat<f64>, String> {

    let n = x.len();
    if x.len() != y.len() {
        return Err("Input lengths are not the same.".into())
    }
    if n < 3 {
        return Err("Must have >= 3 points.".into())
    }

    let y_ref = ColRef::from_slice(y).as_mat();

    // diffs
    let h = x.windows(2)
        .map(|w| w[1] - w[0])
        .collect::<Vec<_>>();

    if h.iter().any(|&x| x <= 0f64) {
        return Err("Input is not increasing.".into())
    }

    // Matrix Q (size n x n-2)
    // Maps second derivatives to the differences in function values
    let mut q = Mat::zeros(n, n - 2);
    for j in 0..(n - 2) {
        let hj_r = 1f64 / h[j];
        let hj1_r = 1f64 / h[j + 1];
        unsafe {
            *q.get_mut_unchecked(j, j) = hj_r;
            *q.get_mut_unchecked(j + 1, j) = -(hj_r + hj1_r);
            *q.get_mut_unchecked(j + 2, j) = hj1_r;
        }
    }

    let mut triplets = Vec::with_capacity(3 * (n - 2) - 2);
    // Matrix R (size n-2 x n-2)
    // Represents the integrated squared second derivatives
    for i in 0..(n - 3) {
        triplets.push(Triplet::new(i, i, (h[i] + h[i+1]) / 3f64));
        triplets.push(Triplet::new(i, i + 1, h[i+1] / 6f64));
        triplets.push(Triplet::new(i + 1, i, h[i+1] / 6f64));
    }
    triplets.push(Triplet::new(n-3, n-3, (h[n-3] + h[n-2]) / 3f64));
    let r = SparseColMat::<usize, f64>::try_new_from_triplets(
        n - 2, 
        n - 2, 
        &triplets
    ).unwrap();
    // We first solve R * xx = Q^T
    // if input x is sorted, which it should be, we can unwrap.
    let r_cholesky = r.sp_cholesky(Side::Lower).map_err(
        |e| e.to_string()
    )?;
    let xx = r_cholesky.solve(q.transpose());
    
    // Compute the penalty matrix: P = Q * (R^-1 * Q^T) = Q * xx
    // let penalty = q * xx;
    // Build the final system matrix: M = I + lambda * P = I + lambda * Q * XX
    let mut m = Mat::identity(n, n);
    faer::linalg::matmul::matmul(
        m.as_mut(),
        faer::Accum::Add,
        q,
        xx,
        lambda,
        Par::rayon(0), //
    ); // inv is updated

    // should be positive semidefinite, but add a fail path
    match m.llt(Side::Lower) {
        Ok(cho) => Ok(cho.solve(y_ref))
        , Err(_) => Ok(m.col_piv_qr().solve(y_ref)),
    }
}

#[polars_expr(output_type=Float64)]
fn pl_smooth_spline(inputs: &[Series], kwargs: SplineKwargs) -> PolarsResult<Series> {
    let x = inputs[0].f64()?; 
    let y = inputs[1].f64()?;

    if x.has_nulls() | y.has_nulls() {
        return Err(PolarsError::ComputeError("Input x or y has nulls.".into()))
    }
    // Make sure they are 1 chunk in Python
    let x_slice = x.cont_slice().unwrap();
    let y_slice = y.cont_slice().unwrap();
    let lambda = kwargs.lambda;

    let y_smoothed = 
        faer_smooth_spline(x_slice, y_slice, lambda)
        .map_err(|e| PolarsError::ComputeError(e.into()))?;

    let a = y_smoothed.col_as_slice(0);
    // if we go unsafe, no copy is needed.
    let ca = Float64Chunked::from_vec("".into(), a.to_vec());
    Ok(ca.into_series())
}