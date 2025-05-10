pub mod lr_solvers;

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
