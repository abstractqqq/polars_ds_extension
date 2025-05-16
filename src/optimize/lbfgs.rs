
use std::collections::VecDeque;

use faer::{Col, Row, ColRef, Mat, MatRef, Scale};
use faer_traits::RealField;
use crate::linear::utils::Elementwise;
use num::Float;
use cfavml::safe_trait_distance_ops::DistanceOps;

// DO NOT USE FAER. USE CFAVML

/// Runs the lbfgs algorithm to find the 
/// optimized value. This modifies x0 in place
/// and by the end x0 should contain the value that
/// achieves the optimum. This will also return the value of 
/// the objective function at x0;
pub fn lbfgs<T: RealField + Float + DistanceOps>(
    obj: fn(x: &[T]) -> T 
    , deriv: fn(x: T) -> T
    , x0: &mut [T]
    , m: usize
    , tol: T
    , max_iter: usize
) -> T {

    let mut loss: T = obj(x0);
    let half = T::one() / (T::one() + T::one());

    let mut s_list: VecDeque<Vec<T>> = VecDeque::new();
    let mut y_list: VecDeque<Vec<T>> = VecDeque::new();
    let mut rho_list: VecDeque<T> = VecDeque::new();
    
    for _ in 0..max_iter {
        let x = x0.as_ref();
        let mut g = x.iter().map(|z| deriv(*z)).collect::<Vec<_>>();

        if cfavml::squared_norm(&g) < tol {
            break 
        }

        let mut alpha: Vec<T> = Vec::new();
        // Loop backwards
        for i in (0..s_list.len()).rev() {
            let rho = rho_list[i];
            let s = s_list[i].as_ref();
            let y = y_list[i].as_ref();
            let alpha_i = rho * *(s.transpose() * &g).get(0, 0);
            alpha.push(alpha_i);
            g = g - Scale(alpha_i) * y;
        }

        let gamma = if s_list.len() > 0 {
            // Unwraps are safe here because we checked
            let last_s = s_list.back().unwrap();
            let last_y = y_list.back().unwrap();
            let denom = last_y.squared_norm_l2();
            let numerator = *(last_s.transpose() * last_y).get(0, 0);
            numerator / denom 
        } else {
            T::one()
        };

        let mut r = Scale(gamma) * &g;
        // # Loop forward
        for i in 0..s_list.len() {
            let s = s_list[i].as_ref();
            let y = y_list[i].as_ref();
            let rho = rho_list[i];
            let beta = rho * *(r.transpose() * y).get(0, 0);
            let a = alpha[alpha.len() - i] - beta;
            r = r - Scale(a) * s;
        }
        // # Search direction
        let dir = Scale(T::one().neg()) * r;
        // # Line search for optimal step size (simple backtracking)
        let mut step_size = T::one();
        let small = T::from(1e-4).unwrap();
        let mut x_new = &x + Scale(step_size) * &dir;
        x_new.row(0).
        loss = obj(x_new.col(0));
        while loss > obj(x.as_ref()) + small * *(dir.transpose() * &g).get(0, 0) {
            step_size = step_size * half;
            x_new = x_new + Scale(step_size) * &dir;
            loss = obj(x_new.as_ref());
        }

        // x_new is update to date
        let s = &x_new - x;
        let y = x_new.as_ref().map_elementwise(deriv) - g;

        let rho = (y.transpose() * &s).get(0, 0).recip();
        if s_list.len() == m {
            s_list.pop_front();
            y_list.pop_front();
            rho_list.pop_front();
        }
        s_list.push_back(s);
        y_list.push_back(y);
        rho_list.push_back(rho);
        x = x_new;

    }

    (x, loss)

}