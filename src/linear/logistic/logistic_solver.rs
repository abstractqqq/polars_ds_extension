use core::f64;
use std::f64::EPSILON;
use rand::{Rng, SeedableRng};
use rand_distr::Normal;
use rand::rngs::StdRng;
use argmin::core::{CostFunction, Error, Executor, Gradient};
use argmin::solver::{
    quasinewton::LBFGS
    , linesearch::MoreThuenteLineSearch
};
use faer::{Mat, MatRef, Scale};

#[inline(always)]
pub fn stable_sigmoid(x: f64) -> f64 {
    let abs_x = x.abs();
    let res = 1.0 / (1.0 + (-abs_x).exp());
    if x >= 0.0 { res } else { 1.0 - res }
}

/// y: actual
/// z: predicted, before applying sigmoid
#[inline(always)]
pub fn stable_log_loss(y: f64, z: f64) -> f64 {
    z.max(0.0) - y * z + (1.0 + (-z.abs()).exp()).ln()
}

/// LogRegProblem
/// x: the matrix X, n rows x m cols.
/// y: the target. We use a slice to simplify some operations
/// add_bias: whether a bias term has been added or not. This will not add a column of 1s to x!
/// l2_reg: whether to add a l2 regularization term.
struct LogRegProblem<'a> {
    x: MatRef<'a, f64>, // Features. n x m
    y: MatRef<'a, f64>, // Labels (0 or 1). m x 1
    add_bias: bool,
    l2_reg: Option<f64>
}

impl <'a> CostFunction for LogRegProblem<'a> {
    type Param = Mat<f64>; // Weights/coefficients
    type Output = f64;

    fn cost(&self, w: &Self::Param) -> Result<Self::Output, Error> {
        let m = self.y.nrows() as f64;
        let out = self.x * w; // predicted value before sigmoid
        let total_loss = 
            self.y.col(0).iter().copied()
            .zip(out.col(0).iter().copied())
            .map(|(y, z)| stable_log_loss(y, z))
            .sum::<f64>();

        // last term is the bias
        match self.l2_reg {
            Some(lambda) => {
                let n_params = w.nrows() - self.add_bias as usize;
                // safe because the indices are within bounds
                let l2_penalty = unsafe {
                    w.get_unchecked(0..n_params, 0)
                    .iter()
                    .map(|x| x.powi(2))
                    .sum::<f64>() * 0.5 * lambda
                };

                Ok(total_loss / m + l2_penalty)
            },
            None => Ok(total_loss / m),
        }
    }
}

impl <'a> Gradient for LogRegProblem<'a> {
    type Param = Mat<f64>;
    type Gradient = Mat<f64>;

    fn gradient(&self, w: &Self::Param) -> Result<Self::Gradient, Error> {
        let m = self.y.nrows() as f64;
        let mut diff = self.x * w;
        diff.col_as_slice_mut(0).iter_mut().for_each(
            |v| *v = stable_sigmoid(*v)
        );
        diff = Scale(1.0 / m) * (diff - self.y);
        // Gradient of log loss: X^T * (y_hat - y) / m
        let mut grad = self.x.transpose() * diff;
        // If l2_reg, add lambda * w_i
        if let Some(lambda) = self.l2_reg {
            let n_params = w.nrows() - self.add_bias as usize;
            unsafe {
                (0..n_params).for_each(
                    |i| {
                        *grad.get_mut_unchecked(i, 0) += lambda * *w.get_unchecked(i, 0);
                    }
                );
            }
        }
        Ok(grad)
    }
}

/// Solving logistic regression with faer. The bias (intercept) should already be added in x
pub fn faer_logistic_reg(
    x: MatRef<f64>,
    y: MatRef<f64>,
    add_bias: bool,
    l1_reg: Option<f64>,
    l2_reg: Option<f64>,
    tol: f64,
    max_iters: usize,
) -> Mat<f64> {

    // This seems very slow at this moment.
    let problem = LogRegProblem {x: x, y: y, add_bias: add_bias, l2_reg: l2_reg};
    let normal = Normal::new(0.0, 0.01).unwrap();
    let mut rng = StdRng::seed_from_u64(42);
    // let w = Mat::full(x.ncols(), 1, 0.0);
    let w = Mat::from_fn(x.ncols(), 1, |_, _| rng.sample(normal));
    let nrows = w.nrows();

    let linesearch  = MoreThuenteLineSearch::new();
    let mut solver = 
        LBFGS::new(linesearch, 10)
            .with_tolerance_grad(EPSILON.sqrt().max(tol)).unwrap();

    if let Some(l1) = l1_reg {
        solver = solver.with_l1_regularization(l1.max(EPSILON)).unwrap();
    }

    let result = Executor::new(problem, solver)
            .configure(
                |state| 
                state
                    .param(w)
                    .max_iters(max_iters as u64)
            )
            .run();
    
    match result {
        Ok(opt) => {
            match opt.state.best_param {
                Some(best) => best,
                None => Mat::full(nrows, 1, f64::NAN),
            }
        },
        Err(_) => Mat::full(nrows,1, f64::NAN),
    }
}
