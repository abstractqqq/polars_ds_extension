use std::ops::{AddAssign, DivAssign};

use faer::prelude::*;

use faer_ext::{IntoFaer, IntoNdarray};
use ndarray::{prelude::*, AssignElem, IndexLonger};
use ndarray::{s, Array1, Array3, ArrayView1};
use polars::error::PolarsResult;
use polars::prelude::*;

#[derive(PartialEq)]
enum BoundaryConditionType {
    NotAKnot,
    Periodic,
    Clamped,
    Natural,
}

fn solve_thomas(
    a: ArrayView1<f64>,
    b: ArrayView1<f64>,
    c: ArrayView1<f64>,
    d: ArrayView1<f64>,
) -> Array1<f64> {
    // https://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
    let n = d.len();

    let mut w = Array1::<f64>::zeros(n - 1);
    let mut g = Array1::<f64>::zeros(n);

    w[0].assign_elem(c[0] / b[0]);
    g[0].assign_elem(d[0] / b[0]);

    for i in 1..n - 1 {
        let m = b[i] - a[i - 1] * w[i - 1];
        w[i].assign_elem(c[i] / m);
        let val = (d[i] - a[i - 1] * g[i - 1]) / m;
        g[i].assign_elem(val);
    }

    let val = (d[n - 1] - a[n - 2] * g[n - 2]) / (b[n - 1] - a[n - 2] * w[n - 2]);
    g[n - 1].assign_elem(val);

    for i in (0..n - 1).rev() {
        let val = g[i] - w[i] * g[i + 1];
        g[i].assign_elem(val);
    }

    g
}

fn solve_cyclic_thomas(
    a: ArrayView1<f64>,
    b: ArrayView1<f64>,
    c: ArrayView1<f64>,
    d: ArrayView1<f64>,
) -> Array1<f64> {
    /*
    solves Ax = v, where A is a cyclic tridiagonal matrix consisting of vectors a, b, c
    X = number of equations
    x[] = initially contains the input v, and returns x. indexed from [0, ..., X - 1]
    a[] = subdiagonal, regularly indexed from [1, ..., X - 1], a[0] is lower left corner
    b[] = main diagonal, indexed from [0, ..., X - 1]
    c[] = superdiagonal, regularly indexed from [0, ..., X - 2], c[X - 1] is upper right
    cmod[], u[] = scratch vectors each of length X
    */

    let n = d.len();
    let mut cmod = Array1::<f64>::zeros(n);
    let mut u = Array1::<f64>::zeros(n);

    // Lower left and upper right corners of the cyclic tridiagonal system respectively
    let alpha = a[0];
    let beta = c[n - 1];

    // Arbitrary, but chosen such that division by zero is avoided
    let gamma = -b[0];

    let mut d = d.to_owned();

    cmod[0] = alpha / (b[0] - gamma);
    u[0] = gamma / (b[0] - gamma);
    let tmp = d[0];
    d[0] = tmp / (b[0] - gamma);

    // Loop from 1 to X - 2 inclusive
    for ix in 1..n - 1 {
        let m = 1.0 / (b[ix] - a[ix] * cmod[ix - 1]);
        cmod[ix] = c[ix] * m;
        u[ix] = (0.0 - a[ix] * u[ix - 1]) * m;
        d[ix] = (d[ix] - a[ix] * d[ix - 1]) * m;
    }

    // Handle X - 1
    let m = 1.0 / (b[n - 1] - alpha * beta / gamma - beta * cmod[n - 2]);
    u[n - 1] = (alpha - a[n - 1] * u[n - 2]) * m;
    d[n - 1] = (d[n - 1] - a[n - 1] * d[n - 2]) * m;

    // Loop from X - 2 to 0 inclusive
    for ix in (0..=n - 2).rev() {
        u[ix] -= cmod[ix] * u[ix + 1];
        d[ix] -= cmod[ix] * d[ix + 1];
    }

    let fact = (d[0] + d[n - 1] * beta / gamma) / (1.0 + u[0] + u[n - 1] * beta / gamma);

    // Loop from 0 to X - 1 inclusive
    for ix in 0..n {
        d[ix] -= fact * u[ix];
    }

    d
}

#[allow(non_snake_case)]
fn cubic_spline(
    x: ArrayView1<f64>,
    y: ArrayView1<f64>,
    bc_type: (BoundaryConditionType, BoundaryConditionType),
) -> PolarsResult<Series> {
    if x.len() < 2 {
        return Err(PolarsError::ComputeError(
            "`x` must contain at least 2 elements.".into(),
        ));
    }

    let dx = (&x.slice(s![1..]) - &x.slice(s![..-1])).slice(s![1..]);
    let dx_len = dx.len();

    if dx.iter().any(|dx| dx <= &0.0) {
        return Err(PolarsError::ComputeError(
            "`x` must be strictly increasing".into(),
        ));
    }

    let n = x.len();

    let dy = (&y.slice(s![1..]) - &y.slice(s![..-1])).slice(s![1..]);
    let slope = &dy / &dx;

    // In Scipy there's a check here for y.size == 0. I am not sure how that is
    // possible. I think we want to just error and enforce y > 0 on the Python side?

    let bc = (
        match bc_type.0 {
            Clamped => (1, 0.0),
            Natural => (2, 0.0),
            _ => (0, 0.0),
        },
        match bc_type.0 {
            Clamped => (1, 0.0),
            Natural => (2, 0.0),
            _ => (0, 0.0),
        },
    );

    if n == 2 {
        if [
            BoundaryConditionType::NotAKnot,
            BoundaryConditionType::Periodic,
        ]
        .contains(&bc_type.0)
        {
            bc.0 = (1, slope[0]);
        }

        if [
            BoundaryConditionType::NotAKnot,
            BoundaryConditionType::Periodic,
        ]
        .contains(&bc_type.1)
        {
            bc.1 = (1, slope[0]);
        }
    }

    let dydx = if n == 3
        && bc_type.0 == BoundaryConditionType::NotAKnot
        && bc_type.1 == BoundaryConditionType::NotAKnot
    {
        let A = faer::mat![
            [1.0, 1.0, 0.0],
            [dx[1], 2.0 * (dx[0] + dx[1]), dx[0]],
            [0.0, 1.0, 1.0],
        ];

        // Note that in scipy's implementation dx is dxr. We don't create dxr because
        // we are assuming a 1D array for y.
        // dxr = dx.reshape([dx.shape[0]] + [1] * (y.ndim - 1))
        let b = faer::mat![
            [2.0 * slope[0]],
            [3.0 * (dx[0] * slope[1] + dx[1] * slope[0])],
            [2.0 * slope[1]],
        ];

        A.partial_piv_lu()
            .solve(&b)
            .as_ref()
            .into_ndarray()
            .into_shape(3)
            .unwrap()
    } else if n == 3 && bc_type.0 == BoundaryConditionType::Periodic {
        let num = (slope / dx).sum();
        let denom: f64 = (Array1::ones(dx.len()) / dx).sum();
        let t = num / denom;

        Array1::from_elem(dx.len(), t).view()
    } else {
        let mut A = Array2::zeros((3, n)); // Banded matrix
        let mut b = Array1::zeros(n);

        let diag = 2.0 * (&dx.slice(s![..-1]) + &dx.slice(s![1..]));
        A.slice_mut(s![1, 1..-1]).assign(&diag);

        A.slice_mut(s![0, 2..]).assign(&dx.slice(s![..-1])); // upper diagonal

        A.slice_mut(s![-1, ..-2]).assign(&dx.slice(s![1..]));

        let bvals = 3.0
            * (&dx.slice(s![1..]) * &slope.slice(s![..-1])
                + &dx.slice(s![..-1]) * &slope.slice(s![1..]));
        b.slice_mut(s![1..-1]).assign(&bvals);

        let s = if bc_type.0 == BoundaryConditionType::Periodic {
            let mut A = A.slice(s![.., 0..-1]).to_owned();
            A.row_mut(1)[0].assign_elem(2.0 * (dx[dx.len() - 1] + dx[0]));
            A.row_mut(0)[1].assign_elem(dx[dx.len() - 1]);

            let b = b.slice(s![..-1]);

            let a_m1_0 = dx[dx_len - 2]; // lower left corner value: A[-1, 0]
            let a_m1_m2 = dx[dx_len - 1];
            let a_m1_m1 = 2.0 * (dx[dx_len - 1] + dx[dx_len - 2]);
            let a_m2_m1 = dx[dx_len - 3];
            let a_0_m1 = dx[0];

            b[0] = 3.0 * (dx[0] * slope[slope.len() - 1] + dx[dx_len - 1] * slope[0]);
            b[b.len() - 1] = 3.0
                * (dx[dx_len - 1] * slope[slope.len() - 2]
                    + dx[dx_len - 2] * slope[slope.len() - 1]);

            let Ac = A.slice(s![.., ..-1]);
            let b1 = b.slice(s![..-1]);
            let mut b2 = Array1::<f64>::zeros(b1.len());
            b2[0] = -a_0_m1;
            b2[b2.len() - 1] = -a_m2_m1;

            let du = Ac.slice(s![0, 1..]);
            let d = Ac.slice(s![1, ..]);
            let d1 = Ac.slice(s![2, ..-1]);

            let s1 = solve_cyclic_thomas(d1, d, du, b1);
            let s2 = solve_cyclic_thomas(d1, d, du, b2.view());

            // computing the s[n-2] solution:
            let s_m1 = (b[b.len() - 1] - a_m1_0 * s1[0] - a_m1_m2 * s1[s1.len() - 1])
                / (a_m1_m1 + a_m1_0 * s2[0] + a_m1_m2 * s2[s2.len() - 1]);

            // # s is the solution of the (n, n) system:
            let mut s = Array1::<f64>::zeros(n);
            s.slice_mut(s![..-2]).assign(&(s1 + s_m1 * s2));
            s[s.len() - 2] = s_m1;
            s[s.len() - 1] = s[0];

            s

            // du = a1[0, 1:]
            // d = a1[1, :]
            // dl = a1[2, :-1]
            // du2, d, du, x, info = gtsv(dl, d, du, b1, overwrite_ab, overwrite_ab,
            //                            overwrite_ab, overwrite_b)
            //             The array dl contains the (n - 1) subdiagonal elements of A.

            // The array d contains the diagonal elements of A.

            // The array du contains the (n - 1) superdiagonal elements of A.

            // The array b contains the matrix B whose columns are the right-hand sides for the systems of equations. The second dimension of b must be at least max(1,nrhs).
        } else {
            if bc_type.0 == BoundaryConditionType::NotAKnot {
                A.row_mut(1)[0] = dx[1];
                A.row_mut(0)[1] = x[2] - x[0];
                let d = x[2] - x[0];
                b[0] = ((dx[0] + 2.0 * d) * dx[1] * slope[0] + dx[0].powi(2) * slope[1]) / d;
            } else if bc.0 .0 == 1 {
                A.row_mut(1)[0] = 1.0;
                A.row_mut(0)[1] = 0.0;
                b[0] = bc.0 .1;
            } else if bc.0 .0 == 2 {
                A.row_mut(1)[0] = 2.0 * dx[0];
                A.row_mut(0)[1] = dx[0];
                b[0] = -0.5 * bc.0 .1 * dx[0].powi(2) + 3.0 * (y[1] - y[0]);
            }

            if bc_type.1 == BoundaryConditionType::NotAKnot {
                A.row_mut(1)[A.row(1).len() - 1] = dx[dx_len - 2];
                A.slice_mut(s![-1, -2])[0] = x[x.len() - 1] - x[x.len() - 3];
                let d = x[x.len() - 1] - x[x.len() - 3];
                b[b.len() - 1] = (dx[dx_len - 1].powi(2) * slope[slope.len() - 2]
                    + (2.0 * d + dx[dx_len - 1]) * dx[dx_len - 2] * slope[slope.len() - 1])
                    / d;
            } else if bc.1 .0 == 1 {
                A.slice_mut(s![1, -1])[0] = 1.0;
                A.slice_mut(s![-1, -2])[0] = 0.0;
                b[b.len() - 1] = bc.1 .1;
            } else if bc.1 .0 == 2 {
                A.slice_mut(s![1, -1])[0] = 2.0 * dx[dx_len - 1];
                A.slice(s![-1, -2])[0] = dx[dx_len - 1];
                b[b.len() - 1] =
                    0.5 * bc.1 .1 * dx[dx_len - 1].powi(2) + 3.0 * (y[y.len() - 1] - y[y.len() - 2])
            }

            let du = A.slice(s![0, 1..]);
            let d = A.slice(s![1, ..]);
            let d1 = A.slice(s![2, ..-1]);

            solve_cyclic_thomas(d1, d, du, b.view())
        };

        s.view()
    };

    // s refers to dydx

    let t = (&dydx.slice(s![..-1]) + &dydx.slice(s![1..]) - 2.0 * slope) / dx;
    let mut c = Array2::<f64>::zeros((4, x.len() - 1));
    c.row_mut(0).assign(&(t / dx));
    c.row_mut(1)
        .assign(&((slope - dydx.slice(s![..-1])) / dx - t));
    c.row_mut(2).assign(&dydx.slice(s![..-1]));
    c.row_mut(3).assign(&y.slice(s![..-1]));
}
