/// Least Squares using Faer and ndarray.

use crate::utils::to_frame;
use faer::prelude::*;
use faer_ext::IntoFaer;
use itertools::Itertools;
use ndarray::{s, Array2, ArrayView, ArrayView1, ArrayView2, ArrayViewMut2, Axis};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

enum LRMethods {
    Normal, // Normal. Normal Equation
    L2,     // Ridge, L2 regularized
    L1, // Lasso, L1 regularized
}

impl From<String> for LRMethods {
    fn from(value: String) -> Self {
        match value.as_str() {
            "l1" | "lasso" => LRMethods::L1,
            "l2" | "ridge" => LRMethods::L2,
            _ => LRMethods::Normal,
        }
    }
}

#[derive(Deserialize, Debug)]
pub(crate) struct LstsqKwargs {
    pub(crate) bias: bool,
    pub(crate) skip_null: bool,
    pub(crate) method: String,
    pub(crate) lambda: f64,
}

fn report_output(_: &[Field]) -> PolarsResult<Field> {
    let index = Field::new("idx", DataType::UInt16); // index of feature
    let coeff = Field::new("coeff", DataType::Float64); // estimated value for this coefficient
    let stderr = Field::new("std_err", DataType::Float64); // Std Err for this coefficient
    let t = Field::new("t", DataType::Float64); // t value for this coefficient
    let p = Field::new("p>|t|", DataType::Float64); // p value for this coefficient
    let v: Vec<Field> = vec![index, coeff, stderr, t, p];
    Ok(Field::new("lstsq_report", DataType::Struct(v)))
}

fn pred_residue_output(_: &[Field]) -> PolarsResult<Field> {
    let pred = Field::new("pred", DataType::Float64);
    let residue = Field::new("resid", DataType::Float64);
    let v = vec![pred, residue];
    Ok(Field::new("pred", DataType::Struct(v)))
}

fn coeff_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "coeffs",
        DataType::List(Box::new(DataType::Float64)),
    ))
}

/// Returns the coefficients for lstsq as a nrows x 1 matrix
/// The uses QR (column pivot) decomposition as default method to compute inverse,
/// Column Pivot QR is chosen to deal with rank deficient cases.
#[inline(always)]
fn faer_qr_lstsq(x: ArrayView2<f64>, y: ArrayView2<f64>) -> Mat<f64> {
    let x = x.into_faer();
    let y = y.into_faer();
    let qr = x.col_piv_qr();
    qr.solve_lstsq(y)
}

/// Returns the coefficients for lstsq with l2 regularization as a nrows x 1 matrix
/// This is currently slow. Using SVD might be faster.
#[inline(always)]
fn faer_qr_lstsq_l2(x: ArrayView2<f64>, y: ArrayView2<f64>, lambda: f64) -> Mat<f64> {

    let x = x.into_faer();
    let y = y.into_faer();

    let xt = x.transpose();
    let matrix = xt * x
        + Mat::from_fn(x.ncols(), x.ncols(), |i, j| {
            lambda * ((i == j) as u64) as f64
        });
    let qr = matrix.qr();
    let inv = qr.inverse();
    inv * xt * y
}

fn soft_threshold(rho:f64, lambda:f64) -> f64 {

    if rho < -lambda {
        rho + lambda
    } else if rho > lambda {
        rho - lambda
    } else {
        0f64
    }
    
    // lambda >= 0 always
    // So it is equivalent to ???
    // if |rho| > lambda, then |rho| - lambda, else 0

    // To make it branchless, we can 
    // let r = rho.abs();
    // let diff = r - lambda;
    // let flag = ((diff > 0.) as u64) as f64;
    // flag * diff + (1.0 - flag)
}

/// Computes Lasso Regression coefficients by the use of Coordinate Descent
/// It is easier to use ndarray to write the implementation. The output, however, is still a Faer matrix
/// for consistency with other functions.
/// 
/// Reference:
/// https://xavierbourretsicotte.github.io/lasso_implementation.html
#[inline(always)]
fn lasso_regression(x: ArrayView2<f64>, y: ArrayView2<f64>, lambda:f64, has_bias:bool, num_iter:usize) -> Mat<f64> {

    let n = x.ncols();
    let n1 = n.abs_diff(1);

    let mut beta = vec![1f64; n];

    for _ in 0..num_iter {

        for j in 0..n {

            let x_j = x.column(j);
            let beta_view = ArrayView2::from_shape((n, 1), &beta).unwrap();
            let y_pred = x.dot(&beta_view);
            let temp = &y - &y_pred + x_j.mapv(|x| x * beta[j]);
            let temp = temp.column(0); // temp has shape m by 1.
            let rho = x_j.dot(&temp);

            if has_bias {
                if j == n1 {
                    beta[j] = rho;
                } else {
                    beta[j] = soft_threshold(rho, lambda);
                }
            } else {
                beta[j] = soft_threshold(rho, lambda);
            }
        }
    }
    // Copy theta into a Faer Mat
    Mat::<f64>::from_fn(n, 1, |i, _| beta[i])


}



/// Returns a Array2 ready for linear regression, and a mask, indicating valid rows
#[inline(always)]
fn series_to_mat_for_lstsq(
    inputs: &[Series],
    add_bias: bool,
    skip_null: bool,
) -> PolarsResult<(Array2<f64>, BooleanChunked)> {
    // minus 1 because target is also in inputs. Target is at position 0.
    let n_features = inputs.len().abs_diff(1);
    let has_null = inputs.iter().fold(false, |acc, s| acc | s.has_validity());
    if has_null && !skip_null {
        Err(PolarsError::ComputeError(
            "Lstsq: Data must not contain nulls when skip_null is False.".into(),
        ))
    } else {
        let mut df = to_frame(inputs)?;
        // Return a mask where true is kept (true means not null).
        let mask = if has_null && skip_null {
            let mask = inputs[0].is_not_null(); //0 always exist
            let mask = inputs[1..]
                .iter()
                .fold(mask, |acc, s| acc & (s.is_not_null()));
            df = df.filter(&mask).unwrap();
            mask.clone()
        } else {
            BooleanChunked::from_iter(std::iter::repeat(true).take(df.height()))
        };

        if add_bias {
            df = df.lazy().with_column(lit(1f64)).collect().unwrap();
        }
        if df.height() < n_features {
            Err(PolarsError::ComputeError(
                "Lstsq: #Data < #features. No conclusive result.".into(),
            ))
        } else {

            // let y = df.drop_in_place(target_name).unwrap();
            // let y = y.f64().unwrap();

            let mat = df.to_ndarray::<Float64Type>(IndexOrder::Fortran)?;
            Ok((mat, mask))
        }
    }
}

#[polars_expr(output_type_func=coeff_output)]
fn pl_lstsq(inputs: &[Series], kwargs: LstsqKwargs) -> PolarsResult<Series> {
    let add_bias = kwargs.bias;
    let skip_null = kwargs.skip_null;
    let method = LRMethods::from(kwargs.method);
    // Target y is at index 0
    match series_to_mat_for_lstsq(inputs, add_bias, skip_null) {
        Ok((mut mat, _)) => {
            // Solving Least Square
            let coeffs = match method {
                LRMethods::Normal => {
                    let y = mat.slice(s![.., 0..1]);
                    let x = mat.slice(s![.., 1..]);
                    faer_qr_lstsq(x, y)
                },
                LRMethods::L2 => {
                    let y = mat.slice(s![.., 0..1]);
                    let x = mat.slice(s![.., 1..]);
                    faer_qr_lstsq_l2(x, y, kwargs.lambda)
                },
                LRMethods::L1 => {
                    let (y, mut x) = mat.multi_slice_mut((s![.., 0..1], s![.., 1..]));
                    x.rows_mut().into_iter().for_each(
                        |mut row| row /= row.dot(&row)
                    );
                    lasso_regression(x.view(), y.view(), kwargs.lambda, add_bias, 100)
                }
            };
            let mut builder: ListPrimitiveChunkedBuilder<Float64Type> =
                ListPrimitiveChunkedBuilder::new("betas", 1, coeffs.nrows(), DataType::Float64);

            builder.append_slice(&coeffs.col_as_slice(0));
            let out = builder.finish();
            Ok(out.into_series())
        }
        Err(e) => Err(e),
    }
}

#[polars_expr(output_type_func=pred_residue_output)]
fn pl_lstsq_pred(inputs: &[Series], kwargs: LstsqKwargs) -> PolarsResult<Series> {
    let add_bias = kwargs.bias;
    let skip_null = kwargs.skip_null;
    let method = LRMethods::from(kwargs.method);
    // Copy data
    // Target y is at index 0
    match series_to_mat_for_lstsq(inputs, add_bias, skip_null) {
        Ok((mat, mask)) => {
            // Mask = True indicates the the nulls that we skipped.
            let coeffs = match method {
                LRMethods::Normal => {
                    let y = mat.slice(s![.., 0..1]);
                    let x = mat.slice(s![.., 1..]);
                    faer_qr_lstsq(x, y)
                },
                LRMethods::L2 => {
                    let y = mat.slice(s![.., 0..1]);
                    let x = mat.slice(s![.., 1..]);
                    faer_qr_lstsq_l2(x, y, kwargs.lambda)
                },
                LRMethods::L1 => {
                    // Have to copy because we need to use Mat later.
                    let y = mat.slice(s![.., 0..1]);
                    let mut x_normalized = mat.slice(s![.., 1..]).to_owned();
                    // Normalize x 
                    x_normalized.rows_mut().into_iter().for_each(
                        |mut row| row /= row.dot(&row)
                    );
                    lasso_regression(x_normalized.view(), y, kwargs.lambda, add_bias, 100)
                }
            };

            let y = mat.slice(s![.., 0..1]);
            let x = mat.slice(s![.., 1..]);
            let pred = x.into_faer() * &coeffs;
            let resid = y.into_faer() - &pred;
            let pred = pred.col_as_slice(0);
            let resid = resid.col_as_slice(0);
            // Need extra work when skip_null is true and there are nulls
            let (p, r) = if skip_null && mask.any() {
                let mut p_builder: PrimitiveChunkedBuilder<Float64Type> =
                    PrimitiveChunkedBuilder::new("pred", mask.len());
                let mut r_builder: PrimitiveChunkedBuilder<Float64Type> =
                    PrimitiveChunkedBuilder::new("resid", mask.len());
                let mut i: usize = 0;
                for mm in mask.into_no_null_iter() {
                    // mask is always non-null, mm = true means is not null
                    if mm {
                        p_builder.append_value(pred[i]);
                        r_builder.append_value(resid[i]);
                        i += 1;
                    } else {
                        p_builder.append_value(f64::NAN);
                        r_builder.append_value(f64::NAN);
                    }
                }
                (p_builder.finish(), r_builder.finish())
            } else {
                let pred = Float64Chunked::from_slice("pred", pred);
                let residue = Float64Chunked::from_slice("resid", resid);
                (pred, residue)
            };
            let p = p.into_series();
            let r = r.into_series();
            let out = StructChunked::new("", &[p, r])?;
            Ok(out.into_series())
        }
        Err(e) => Err(e),
    }
}

#[polars_expr(output_type_func=report_output)]
fn pl_lstsq_report(inputs: &[Series], kwargs: LstsqKwargs) -> PolarsResult<Series> {
    let add_bias = kwargs.bias;
    let skip_null = kwargs.skip_null;
    // Copy data
    // Target y is at index 0
    match series_to_mat_for_lstsq(inputs, add_bias, skip_null) {
        Ok((mat, _)) => {
            let ncols = mat.ncols() - 1;
            let nrows = mat.nrows();

            let y = mat.slice(s![0..nrows, 0..1]);
            let y = y.view().into_faer();
            let x = mat.slice(s![0..nrows, 1..]);
            let xt = x.t().into_faer();
            let x = x.view().into_faer();
            // Solving Least Square
            let xtx = xt * &x;
            let xtx_qr = xtx.col_piv_qr();
            let xtx_inv = xtx_qr.inverse();
            let coeffs = &xtx_inv * xt * y;
            let betas = coeffs.col_as_slice(0);
            // Degree of Freedom
            let dof = nrows as f64 - ncols as f64;
            // Residue
            let res = y - x * &coeffs;
            let res2 = res.transpose() * &res; // total residue, sum of squares
            let res2 = res2.read(0, 0) / dof;
            // std err
            let std_err = (0..ncols)
                .map(|i| (res2 * xtx_inv.read(i, i)).sqrt())
                .collect_vec();
            // T values
            let t_values = betas
                .iter()
                .zip(std_err.iter())
                .map(|(b, se)| b / se)
                .collect_vec();
            // P values
            let p_values = t_values
                .iter()
                .map(
                    |t| match crate::stats_utils::beta::student_t_sf(t.abs(), dof) {
                        Ok(p) => 2.0 * p,
                        Err(_) => f64::NAN,
                    },
                )
                .collect_vec();
            // Finalize
            let idx_series = UInt16Chunked::from_iter((0..ncols).map(|i| Some(i as u16)));
            let idx_series = idx_series.with_name("idx").into_series();
            let coeffs_series = Float64Chunked::from_slice("coeff", betas);
            let coeffs_series = coeffs_series.into_series();
            let stderr_series = Float64Chunked::from_vec("std_err", std_err);
            let stderr_series = stderr_series.into_series();
            let t_series = Float64Chunked::from_vec("t", t_values);
            let t_series = t_series.into_series();
            let p_series = Float64Chunked::from_vec("p>|t|", p_values);
            let p_series = p_series.into_series();
            let out = StructChunked::new(
                "lstsq_report",
                &[idx_series, coeffs_series, stderr_series, t_series, p_series],
            )?;
            Ok(out.into_series())
        }
        Err(e) => Err(e),
    }
}
