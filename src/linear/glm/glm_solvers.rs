#![allow(non_snake_case)]
use crate::linear::{
    glm::{
        link_functions::{LinkFunction, VarianceFunction},
        GLMSolverMethods, GeneralizedLinearModel,
    },
    lr::{lr_solvers::faer_weighted_lstsq, LRSolverMethods, LinearModel},
    LinalgErrors,
};
use faer::{diag::DiagRef, mat::Mat, MatRef, Par};
use faer_traits::RealField;
use itertools::Itertools;
use num::Float;

/// Methods supported by the GLM solver
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GLMFamily {
    Gaussian,
    Poisson,
    Binomial,
    Gamma,
    // Custom is hard. Let's save it for later
    // Custom(LinkFunction, VarianceFunction),
}

impl GLMFamily {
    pub fn link_function(&self) -> LinkFunction {
        match self {
            GLMFamily::Gaussian => LinkFunction::Identity,
            GLMFamily::Poisson => LinkFunction::Log,
            GLMFamily::Binomial => LinkFunction::Logit,
            GLMFamily::Gamma => LinkFunction::Inverse,
            // GLMFamily::Custom(link, _) => *link,
        }
    }

    pub fn variance_function(&self) -> VarianceFunction {
        match self {
            GLMFamily::Gaussian => VarianceFunction::Gaussian,
            GLMFamily::Poisson => VarianceFunction::Poisson,
            GLMFamily::Binomial => VarianceFunction::Binomial,
            GLMFamily::Gamma => VarianceFunction::Gamma,
            // GLMFamily::Custom(_, var) => *var,
        }
    }
}

impl From<&str> for GLMFamily {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "gaussian" | "normal" => GLMFamily::Gaussian,
            "poisson" => GLMFamily::Poisson,
            "binomial" | "logistic" => GLMFamily::Binomial,
            "gamma" => GLMFamily::Gamma,
            _ => GLMFamily::Gaussian, // Default to Gaussian
        }
    }
}

/// Parameters for the IRLS algorithm
pub struct GLMParams<T: RealField + Float> {
    pub tol: T,
    pub max_iter: usize,
}

impl<T: RealField + Float> Default for GLMParams<T> {
    fn default() -> Self {
        GLMParams {
            tol: T::from(1e-8).unwrap(),
            max_iter: 100,
        }
    }
}

impl<T: RealField + Float> GLMParams<T> {
    pub fn new(max_iter: usize, tol: T) -> Self {
        GLMParams {
            tol: tol,
            max_iter: max_iter,
        }
    }
}

/// Generalized Linear Model implementation supporting various link and variance functions
pub struct GLM<T: RealField + Float> {
    pub solver: GLMSolverMethods,
    pub lambda: T,
    pub fitted_values: Mat<T>, // n_features x 1 matrix, (n_features + 1) x 1 if add_bias
    pub add_bias: bool,
    pub link: LinkFunction,
    pub variance: VarianceFunction,
    pub glm_params: GLMParams<T>,
}

impl<T: RealField + Float> ToString for GLM<T> {
    fn to_string(&self) -> String {
        format!("GLM:\nLink: {:?}\nVariance: {:?}", self.link, self.variance)
    }
}

impl<T: RealField + Float> GLM<T> {
    pub fn new(
        solver: &str,
        lambda: T,
        add_bias: bool,
        link: LinkFunction,
        variance: VarianceFunction,
        glm_params: GLMParams<T>,
    ) -> Self {
        GLM {
            solver: solver.into(),
            lambda,
            fitted_values: Mat::new(),
            add_bias,
            link,
            variance,
            glm_params: glm_params,
        }
    }

    /// Create a new GLM with a specified family that determines the link and variance functions
    pub fn new_with_family(solver: &str, lambda: T, add_bias: bool, family: GLMFamily) -> Self {
        GLM {
            solver: solver.into(),
            lambda,
            fitted_values: Mat::new(),
            add_bias,
            link: family.link_function(),
            variance: family.variance_function(),
            glm_params: GLMParams::default(),
        }
    }

    pub fn set_link_function(&mut self, link: LinkFunction) {
        self.link = link;
    }

    pub fn set_variance_function(&mut self, variance: VarianceFunction) {
        self.variance = variance;
    }

    pub fn set_family(&mut self, family: GLMFamily) {
        self.link = family.link_function();
        self.variance = family.variance_function();
    }

    pub fn set_solver(&mut self, solver: &str) {
        self.solver = solver.into();
    }

    pub fn set_irls_params(&mut self, tol: T, max_iter: usize) {
        self.glm_params.tol = tol;
        self.glm_params.max_iter = max_iter;
    }

    pub fn set_coeffs_and_bias(&mut self, coeffs: &[T], bias: T) {
        self.add_bias = bias.abs() > T::epsilon();
        if self.add_bias {
            self.fitted_values = Mat::from_fn(coeffs.len() + 1, 1, |i, _| {
                if i < coeffs.len() {
                    coeffs[i]
                } else {
                    bias
                }
            })
        } else {
            self.fitted_values = faer::ColRef::<T>::from_slice(coeffs).as_mat().to_owned();
        }
    }

    /// Get the current link function
    pub fn link_function(&self) -> LinkFunction {
        self.link
    }

    /// Get the current variance function
    pub fn variance_function(&self) -> VarianceFunction {
        self.variance
    }
}

impl<T: RealField + Float> LinearModel<T> for GLM<T> {
    fn fitted_values(&self) -> MatRef<T> {
        self.fitted_values.as_ref()
    }

    fn add_bias(&self) -> bool {
        self.add_bias
    }

    fn fit_unchecked(&mut self, X: MatRef<T>, y: MatRef<T>) {
        // This looks somewhat awkward but if we don't, we are forced to have owned Mat
        // in the no bias branch which means we do an additional copy for no reason.
        if self.add_bias() {
            let ones = Mat::full(X.nrows(), 1, T::one());
            let new_x = faer::concat![[X, ones]];
            self.fitted_values = faer_irls(
                new_x.as_ref(),
                y,
                self.link,
                self.variance,
                self.lambda,
                LRSolverMethods::QR,
                // self.add_bias,
                &self.glm_params,
            );
        } else {
            self.fitted_values = faer_irls(
                X,
                y,
                self.link,
                self.variance,
                self.lambda,
                LRSolverMethods::QR,
                // self.add_bias,
                &self.glm_params,
            );
        }
    }
}

impl<T: RealField + Float> GeneralizedLinearModel<T> for GLM<T> {
    fn glm_predict(&self, X: MatRef<T>) -> Result<Mat<T>, LinalgErrors> {
        let mut result = self.predict(X)?;
        let result_slice = result.col_as_slice_mut(0);
        result_slice.iter_mut().for_each(|v| *v = self.link.inv(*v));
        Ok(result)
    }
}

/// Implements the Iteratively Reweighted Least Squares algorithm for GLMs
///
/// # Arguments
/// * `X` - Design matrix
/// * `y` - Response vector
/// * `link` - Link function
/// * `variance` - Variance function
/// * `lambda` - L2 regularization parameter (Ridge penalty)
/// * `wls_solver_method` - Method to solve the weighted least squares problem
/// * `add_bias` - Whether to include a bias term
/// * `params` - IRLS parameters (tolerance and max iterations)
///
/// # Returns
/// * Coefficients matrix (including bias if requested)
#[inline(always)]
pub fn faer_irls<T: RealField + Float>(
    X: MatRef<T>,
    y: MatRef<T>,
    link: LinkFunction,
    variance: VarianceFunction,
    lambda: T,
    wls_solver_method: LRSolverMethods,
    // add_bias: bool,
    params: &GLMParams<T>,
) -> Mat<T> {
    let n_samples = X.nrows();
    // let n_features = X.ncols().abs_diff(add_bias as usize);
    let mut beta: Mat<T> = Mat::zeros(X.ncols(), 1);

    // let epsilon = T::from(1e-8).unwrap();
    // Initialized mu based on variance
    let point_5 = T::one() / (T::one() + T::one());
    let mut mu = match variance {
        VarianceFunction::Binomial => y
            .col(0)
            .iter()
            .map(|yy| (*yy + point_5) * point_5)
            .collect_vec(),
        _ => {
            let mean = y.sum() / T::from(n_samples).unwrap();
            y.col(0)
                .iter()
                .map(|yy| (*yy + mean) * point_5)
                .collect_vec()
        }
    };

    // IRLS
    let mut converged = false;
    let mut weights = vec![T::one(); n_samples];
    let mut d_mu = vec![T::zero(); n_samples];

    // All the get_unchecked / get_mut_unchecked are safe
    unsafe {
        // Initial linear prediction
        let mut eta = Mat::from_fn(mu.len(), 1, |i, _| link.link(mu[i]));

        for _ in 0..params.max_iter {
            // Figure out weights

            for i in 0..n_samples {
                let mu_i = mu[i];
                d_mu[i] = link.deriv(mu_i);
                weights[i] = (d_mu[i].powi(2) * variance.variance(mu_i)).recip()
                // .max(epsilon)
            }

            // Update Response
            // Reuse eta as Z, the working response
            // This computation is vectorized
            // let z = eta + diag_d_mu * (y - mu.as_mat());
            let diag_d_mu = DiagRef::from_slice(&d_mu);
            let mu_mat = MatRef::from_column_major_slice(&mu, mu.len(), 1);
            eta += diag_d_mu * (y - mu_mat); // This is now Z

            // Solve weighted least squares
            let beta_new = faer_weighted_lstsq(
                X,
                eta.as_ref(), // Eta is now Z, the working response
                &weights,
                wls_solver_method,
            );

            // Update Eta (eta = X @ new_beta)
            faer::linalg::matmul::matmul(
                eta.as_mut(),
                faer::Accum::Replace,
                X,
                beta_new.as_ref(),
                T::one(),
                Par::rayon(0),
            );

            // Update mu: mu = g^-1(eta)
            for i in 0..mu.len() {
                mu[i] = link.inv(*eta.get_unchecked(i, 0));
            }

            // Check for convergence
            // Use L Inf norm. Ignore diff from bias/intercept term???
            let diff = beta - &beta_new;
            let max_diff = diff.norm_max();
            // Update beta
            beta = beta_new;
            // Check convergence
            if max_diff < params.tol {
                converged = true;
                break;
            }
            println!("Max Diff: {:?}", max_diff);
        }
    }

    //
    if !converged {
        println!("IRLS algorithm did not converge within maximum iterations");
    }

    beta
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     // Helper function to check if two f64 values are approximately equal
//     fn approx_eq(a: f64, b: f64, epsilon: f64) -> bool {
//         (a - b).abs() < epsilon
//     }

//     // Helper function to check if vectors are approximately equal
//     fn assert_approx_eq_vec(actual: &[f64], expected: &[f64], epsilon: f64) {
//         assert_eq!(
//             actual.len(),
//             expected.len(),
//             "Vectors have different lengths"
//         );
//         for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
//             assert!(
//                 approx_eq(*a, *e, epsilon),
//                 "Elements at position {} differ significantly: {} vs {} (epsilon: {})",
//                 i,
//                 a,
//                 e,
//                 epsilon
//             );
//         }
//     }

//     // Helper function to create Poisson data
//     fn create_poisson_test_data<T: RealField + Float>(n_samples: usize) -> (Mat<T>, Mat<T>) {
//         let mut X = Mat::zeros(n_samples, 2);
//         let mut y = Mat::zeros(n_samples, 1);

//         let beta_true = [T::from(0.5).unwrap(), T::from(-0.3).unwrap()];
//         let intercept = T::from(1.0).unwrap();

//         for i in 0..n_samples {
//             let x1 = T::from(i % 10).unwrap() / T::from(10).unwrap();
//             let x2 = T::from((i * 7) % 13).unwrap() / T::from(13).unwrap();

//             *unsafe { X.get_mut_unchecked(i, 0) } = x1;
//             *unsafe { X.get_mut_unchecked(i, 1) } = x2;

//             // Generate log(lambda) = intercept + beta1*x1 + beta2*x2
//             let log_lambda = intercept + beta_true[0] * x1 + beta_true[1] * x2;
//             let lambda = log_lambda.exp();

//             let count = lambda.round();
//             *unsafe { y.get_mut_unchecked(i, 0) } = count;
//         }

//         (X, y)
//     }

//     #[test]
//     fn test_glm_poisson() {
//         let (X, y) = create_poisson_test_data::<f64>(100);

//         let mut glm = GLM::new_with_family("irls", 0.0, true, GLMFamily::Poisson);

//         glm.set_irls_params(1e-8, 200);

//         let result = glm.fit(X.as_ref(), y.as_ref());
//         assert!(result.is_ok());

//         assert!(glm.is_fit());

//         let coeffs = glm.coeffs_as_vec().unwrap();
//         let bias = glm.bias();

//         assert_approx_eq_vec(&coeffs, &[0.5, -0.3], 0.8);
//         approx_eq(bias, 1.0, 0.8);

//         let linear_pred = glm.predict(X.as_ref()).unwrap();

//         let mut expected_counts = Mat::zeros(linear_pred.nrows(), 1);
//         for i in 0..linear_pred.nrows() {
//             *unsafe { expected_counts.get_mut_unchecked(i, 0) } = linear_pred.get(i, 0).exp();
//         }

//         for i in 0..expected_counts.nrows() {
//             assert!(*expected_counts.get(i, 0) >= 0.0);
//         }
//     }
// }
