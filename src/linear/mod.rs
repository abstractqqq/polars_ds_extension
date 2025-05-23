#![allow(non_snake_case)]
pub mod glm;
pub mod lr;
pub mod online_lr;

use faer::{Mat, MatRef, Par};
use faer_traits::RealField;
use num::Float;

/// Errors encountered during linear algebra operations.
/// Since the ultimate goal is to propagate the error to Python, this error
/// implements to_string.
#[derive(Debug)]
pub enum LinalgErrors {
    DimensionMismatch,
    NotContiguousArray,
    NotEnoughData,
    MatNotLearnedYet,
    NotContiguousOrEmpty,
    NotConvergent,
    Other(String),
}

impl LinalgErrors {
    pub fn to_string(self) -> String {
        match self {
            Self::DimensionMismatch => "Dimension mismatch.".to_string(),
            Self::NotContiguousArray => "Input array is not contiguous.".to_string(),
            Self::MatNotLearnedYet => "Matrix is not learned yet.".to_string(),
            Self::NotEnoughData => "Not enough rows / columns.".to_string(),
            Self::NotContiguousOrEmpty => "Input is not contiguous or is empty".to_string(),
            Self::NotConvergent => "The algorithm failed to converge.".to_string(),
            LinalgErrors::Other(s) => s,
        }
    }
}

/// A trait for linear models. E.g. models of the form
/// y ~ beta_1 x_1 + ... + beta_n x_n + alpha
/// where the coefficients are beta_i, and alpha is the bias/intercept term.
pub trait LinearModel<T: RealField + Float> {
    /// Typically coefficients + the bias as a single matrix (n x 1) (single slice)
    fn fitted_values(&self) -> MatRef<T>;

    fn add_bias(&self) -> bool;

    fn bias(&self) -> T {
        if self.add_bias() {
            let n = self.fitted_values().nrows() - 1;
            *self.fitted_values().get(n, 0)
        } else {
            T::zero()
        }
    }

    /// Returns the coefficients as a MatRef
    /// If the model is not fitted yet, empty array will be returned.
    fn coefficients(&self) -> MatRef<T> {
        if self.is_fit() {
            let n = self
                .fitted_values()
                .nrows()
                .abs_diff(self.add_bias() as usize);
            self.fitted_values().get(0..n, ..)
        } else {
            self.fitted_values()
        }
    }

    fn fit_unchecked(&mut self, X: MatRef<T>, y: MatRef<T>);

    /// Fits the linear regression. Input X is any m x n matrix. Input y must be a m x 1 matrix.
    /// Note, if there is a bias term in the data, then it must be in the matrix X as the last
    /// column and has_bias must be true. This will not append a bias column to X.
    fn fit(&mut self, X: MatRef<T>, y: MatRef<T>) -> Result<(), LinalgErrors> {
        if X.nrows() != y.nrows() {
            return Err(LinalgErrors::DimensionMismatch);
        } else if X.nrows() < X.ncols() || X.nrows() == 0 || y.nrows() == 0 {
            return Err(LinalgErrors::NotEnoughData);
        }
        self.fit_unchecked(X, y);
        Ok(())
    }

    // If coefficient is an empty Mat, e.g. shape = (0 , 0), then it is not fitted yet.
    fn is_fit(&self) -> bool {
        self.fitted_values().nrows() > 0
    }

    fn coeffs_as_vec(&self) -> Result<Vec<T>, LinalgErrors> {
        match self.is_fit() {
            true => Ok(self
                .coefficients()
                .col(0)
                .iter()
                .copied()
                .collect::<Vec<_>>()),
            false => Err(LinalgErrors::MatNotLearnedYet),
        }
    }

    /// Run the linear regression to predict the target variable.
    /// In the case of GLM, this runs the linear predictor.
    fn predict(&self, X: MatRef<T>) -> Result<Mat<T>, LinalgErrors> {
        if X.ncols() != self.coefficients().nrows() {
            Err(LinalgErrors::DimensionMismatch)
        } else if !self.is_fit() {
            Err(LinalgErrors::MatNotLearnedYet)
        } else {
            let mut pred = Mat::full(X.nrows(), 1, {
                if self.add_bias() {
                    self.bias()
                } else {
                    T::zero()
                }
            });
            // Result is 0 if no bias, result is [bias] (ncols x 1) if has bias.
            // result = result + 1.0 * (X * coeffs)
            faer::linalg::matmul::matmul(
                pred.as_mut(),
                faer::Accum::Add,
                X,
                self.coefficients(),
                T::one(),
                Par::rayon(0),
            );
            Ok(pred)
        }
    }
}

/// A trait for Generalized Linear Models
/// eta ~ beta_1 x_1 + ... + beta_n x_n + alpha
/// y = link_function(^-1)(eta)
pub trait GeneralizedLinearModel<T: RealField + Float>: LinearModel<T> {
    fn glm_predict(&self, X: MatRef<T>) -> Result<Mat<T>, LinalgErrors>;
}
