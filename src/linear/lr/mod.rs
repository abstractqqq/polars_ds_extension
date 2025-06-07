pub mod lr_solvers;

use super::LinalgErrors;
use faer::{Mat, MatRef, Par};
use faer_traits::RealField;
use num::Float;

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
    ElasticNet,
}

impl From<&str> for LRMethods {
    fn from(value: &str) -> Self {
        match value {
            "l1" | "lasso" => Self::L1,
            "l2" | "ridge" => Self::L2,
            "elastic" => Self::ElasticNet,
            _ => Self::Normal,
        }
    }
}

/// Converts a 2-tuple of floats into LRMethods
/// The first entry is assumed to the l1 regularization factor, and
/// the second is assumed to be the l2 regularization factor
impl From<(f64, f64)> for LRMethods {
    fn from(value: (f64, f64)) -> Self {
        if value.0 > 0. && value.1 <= 0. {
            LRMethods::L1
        } else if value.0 <= 0. && value.1 > 0. {
            LRMethods::L2
        } else if value.0 > 0. && value.1 > 0. {
            LRMethods::ElasticNet
        } else {
            LRMethods::Normal
        }
    }
}

impl From<(f32, f32)> for LRMethods {
    fn from(value: (f32, f32)) -> Self {
        if value.0 > 0. && value.1 <= 0. {
            LRMethods::L1
        } else if value.0 <= 0. && value.1 > 0. {
            LRMethods::L2
        } else if value.0 > 0. && value.1 > 0. {
            LRMethods::ElasticNet
        } else {
            LRMethods::Normal
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
    /// column and add_bias must be true. This will not append a bias column to X.
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
