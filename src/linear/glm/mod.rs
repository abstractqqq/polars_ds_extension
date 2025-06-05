pub mod glm_solvers;
pub mod link_functions;

use crate::linear::{lr::LinearModel, LinalgErrors};
use faer::{Mat, MatRef};
use faer_traits::RealField;
use num::Float;

#[derive(Clone, Copy, Default)]
pub enum GLMSolverMethods {
    LBFGS, // Limited-memory BFGS Not Implemented
    #[default]
    IRLS, // Iteratively Reweighted Least Squares
}

impl From<&str> for GLMSolverMethods {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "irls" => GLMSolverMethods::IRLS,
            "lbfgs" => panic!("LBFGS not available"), // lbfgs not available
            _ => GLMSolverMethods::IRLS,
        }
    }
}

/// A trait for Generalized Linear Models
/// eta ~ beta_1 x_1 + ... + beta_n x_n + alpha
/// y = link_function(^-1)(eta)
pub trait GeneralizedLinearModel<T: RealField + Float>: LinearModel<T> {
    fn glm_predict(&self, X: MatRef<T>) -> Result<Mat<T>, LinalgErrors>;
}
