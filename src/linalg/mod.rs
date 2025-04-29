#![allow(non_snake_case)]
pub mod lr_online_solvers;
pub mod lr_solvers;
pub mod glm_solvers;
pub mod link_functions;

use faer::{Mat, MatRef, Par};
use faer_traits::RealField;
use num::Float;


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

#[derive(Clone, Copy, Default)]
pub enum GLMSolverMethods {
    LBFGS, // Limited-memory BFGS Not Implemented
    #[default]
    IRLS, // Iteratively Reweighted Least Squares
}

impl From<&str> for GLMSolverMethods {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "irls" => Self::IRLS,
            "lbfgs" => panic!("LBFGS not available"), // lbfgs not available
            _ => Self::IRLS,
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
        (value.0 as f64, value.1 as f64).into()
    }
}

pub trait LinearRegression<T: RealField + Float> {
    /// Typically coefficients + the bias as a single matrix (n x 1) (single slice)
    fn fitted_values(&self) -> MatRef<T>;

    fn has_bias(&self) -> bool;

    fn bias(&self) -> T {
        if self.has_bias() {
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
            let n = self.fitted_values().nrows().abs_diff(self.has_bias() as usize);
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
            let mut pred = Mat::full(
                X.nrows()
                , 1
                , {
                    if self.has_bias() {
                        self.bias()
                    } else {
                        T::zero()
                    }
                }
            );
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

    fn coeffs_as_vec(&self) -> Result<Vec<T>, LinalgErrors> {
        if !self.is_fit() {
            return Err(LinalgErrors::MatNotLearnedYet);
        }

        let coeffs = self.fitted_values();
        let n = if self.has_bias() {
            coeffs.nrows() - 1
        } else {
            coeffs.nrows()
        };

        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            result.push(*coeffs.get(i, 0));
        }

        Ok(result)
    }

    fn bias(&self) -> T {
        if !self.is_fit() || !self.has_bias() {
            T::zero()
        } else {
            let coeffs = self.fitted_values();
            *coeffs.get(coeffs.nrows() - 1, 0)
        }
    }
}

// // Ndarray and Faer Interop. Copied from faer-ext.
// pub trait IntoFaer {
//     type Faer;
//     fn into_faer(self) -> Self::Faer;
// }

// pub trait IntoNdarray {
//     type Ndarray;
//     fn into_ndarray(self) -> Self::Ndarray;
// }

// impl<'py> IntoFaer for PyReadonlyArray2<'py, f64> {
//     type Faer = MatRef<'py, f64>;

//     fn into_faer(self) -> Self::Faer {
//         let shape = self.shape();
//         let nrows = shape[0];
//         let ncols = shape[1];

//         let strides: [isize; 2] = self.strides().try_into().unwrap();

//         let py_ptr = self.as_array_ptr();
//         let ptr = py_ptr.cast::<f64>();
//         unsafe { 
//             MatRef::from_raw_parts(
//                 ptr, 
//                 nrows, 
//                 ncols, 
//                 strides[0], 
//                 strides[1]
//             ) 
//         }
//     }
// }

// impl<'a, T: RealField> IntoNdarray for MatRef<'a, T> {
//     type Ndarray = ArrayView<'a, T, Ix2>;

//     fn into_ndarray(self) -> Self::Ndarray {
//         let nrows = self.nrows();
//         let ncols = self.ncols();
//         let row_stride: usize = self.row_stride().try_into().unwrap();
//         let col_stride: usize = self.col_stride().try_into().unwrap();
//         let ptr = self.as_ptr();
//         unsafe {
//             ArrayView::<'_, T, Ix2>::from_shape_ptr(
//                 (nrows, ncols).strides((row_stride, col_stride)),
//                 ptr,
//             )
//         }
//     }
// }
