pub mod glm_solvers;
pub mod link_functions;

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
