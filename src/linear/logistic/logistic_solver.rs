use argmin::core::{CostFunction, Error, Gradient, Executor};
use argmin::solver::gradientdescent::SteepestDescent;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use faer::{Mat, MatRef, Scale};

#[inline]
pub fn stable_sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// y: actual
/// z: predicted
#[inline]
pub fn stable_log_loss(y: f64, z: f64) -> f64 {
    z.max(0.0) - y * z + (1.0 + (-z.abs()).exp()).ln()
}

struct LogRegProblem<'a> {
    x: MatRef<'a, f64>, // Features. n x m
    y: &'a[f64], // Labels (0 or 1). m x 1
}

impl <'a> CostFunction for LogRegProblem<'a> {
    type Param = MatRef<'a, f64>; // Weights/coefficients
    type Output = f64;

    fn cost(&self, w: &Self::Param) -> Result<Self::Output, Error> {
        let m = self.y.len() as f64;
        let out = self.x * w; // predicted value before sigmoid
        let total_loss = self.y.iter().enumerate().map(
            |(i, y)| stable_log_loss(*y, *out.get(i, 0))
        ).sum::<f64>();
        Ok(total_loss / m)
    }
}

impl <'a> Gradient for LogRegProblem<'a> {
    type Param = MatRef<'a, f64>;
    type Gradient = Mat<f64>;

    fn gradient(&self, w: &Self::Param) -> Result<Self::Gradient, Error> {
        let m = self.y.len() as f64;
        let mut diff_m = self.x * w; 
        for (i, v) in diff_m.col_as_slice_mut(0).iter_mut().enumerate() {
            *v = (stable_sigmoid(*v) - self.y[i]) / m;
        } // (y_hat - y) / m
        // Gradient of log loss: X^T * (y_hat - y) / m
        Ok(self.x.transpose() * diff_m)
    }
}

// mod test {
//     use super::*;

//     fn test_1() -> Result<(), Error> {
//         // 1. Prepare data (Add a column of 1s to X for the intercept/bias)
//         let x = MatRef::from_row_major_slice(
//             &[1.0, 2.0, 1.0, 3.0, 1.0, 4.0], 
//             3, 2)
//         ;
//         let y = [0.0, 0.0, 1.0];

//         let problem = LogRegProblem { x: x, y: &y };

//         let init_param = Mat::full(2, 1, 0f64);

//         // 3. Choose a solver (Steepest Descent with a Line Search)
//         let linesearch = MoreThuenteLineSearch::new();
//         let solver = SteepestDescent::new(linesearch);

//         // 4. Run the optimization
//         let res = Executor::new(problem, solver)
//             .configure(|state| state.param(init_param).max_iters(100))
//             .run()?;

//         println!("Optimal weights: {}", res.state().get_best_param().unwrap());
//         Ok(())
//     }
// }