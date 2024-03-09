/// OLS using Faer.
use faer::solvers::SolverCore;
use faer::{prelude::*, MatRef, Side};
use faer::{IntoFaer, Mat};
// use faer_ext::IntoFaer;
use crate::utils::rechunk_to_frame;
use itertools::Itertools;
use ndarray::{s, Array2};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub(crate) struct LstsqKwargs {
    pub(crate) bias: bool,
}

fn report_output(_: &[Field]) -> PolarsResult<Field> {
    let index = Field::new("feat_idx", DataType::UInt16); // index of feature
    let coeff = Field::new("coeff", DataType::Float64); // estimated value for this coefficient
    let stderr = Field::new("std_err", DataType::Float64); // Std Err for this coefficient
    let t = Field::new("t", DataType::Float64); // t value for this coefficient
    let p = Field::new("p>|t|", DataType::Float64); // p value for this coefficient
    let v: Vec<Field> = vec![index, coeff, stderr, t, p];
    Ok(Field::new("", DataType::Struct(v)))
}

fn pred_residue_output(_: &[Field]) -> PolarsResult<Field> {
    let pred = Field::new("pred", DataType::Float64);
    let residue = Field::new("resid", DataType::Float64);
    let v = vec![pred, residue];
    Ok(Field::new("pred", DataType::Struct(v)))
}

fn coeff_output(_: &[Field]) -> PolarsResult<Field> {
    // Update to array ???
    // DataType::Array(DataType::Float64, fields.len());
    // fields.len() + 1 when an input is true
    Ok(Field::new(
        "coeffs",
        DataType::List(Box::new(DataType::Float64)),
    ))
}

/// Returns the coefficients for lstsq as a nrows x 1 matrix
#[inline]
fn faer_lstsq_qr(x: MatRef<f64>, y: MatRef<f64>) -> Mat<f64> {
    let qr = x.qr();
    qr.solve_lstsq(y)
}

#[inline(always)]
fn series_to_mat_lstsq(inputs: &[Series], add_bias: bool) -> PolarsResult<Array2<f64>> {
    let nrows = inputs[0].len();
    let mut ncols = inputs.len();
    // Series at index 0 is target
    let df_x = if add_bias {
        ncols += 1;
        let mut series_vec = inputs.to_vec(); // cheap copy
        series_vec.push(Series::from_vec("const", vec![1_f64; nrows]));
        rechunk_to_frame(&series_vec)
    } else {
        rechunk_to_frame(&inputs)
    }?;

    if df_x.height() <= ncols {
        return Err(PolarsError::ComputeError(
            "Lstsq: Data must have more rows than columns.".into(),
        ));
    }
    df_x.to_ndarray::<Float64Type>(IndexOrder::Fortran)
}

#[polars_expr(output_type_func=coeff_output)]
fn pl_lstsq(inputs: &[Series], kwargs: LstsqKwargs) -> PolarsResult<Series> {
    let add_bias = kwargs.bias;
    // Copy data
    // Target y is at index 0
    match series_to_mat_lstsq(inputs, add_bias) {
        Ok(mat) => {
            let nrows = mat.nrows();

            let y = mat.slice(s![0..nrows, 0..1]);
            let y = y.view().into_faer();
            let x = mat.slice(s![0..nrows, 1..]);
            let x = x.view().into_faer();
            // Solving Least Square
            let coeffs = faer_lstsq_qr(x, y);
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
    // Copy data
    // Target y is at index 0
    match series_to_mat_lstsq(inputs, add_bias) {
        Ok(mat) => {
            let nrows = mat.nrows();

            let y = mat.slice(s![0..nrows, 0..1]);
            let y = y.view().into_faer();
            let x = mat.slice(s![0..nrows, 1..]);
            let x = x.view().into_faer();
            // Solving Least Square
            let coeffs = faer_lstsq_qr(x, y);
            let y_hat = x * coeffs;
            let pred = y_hat.col_as_slice(0).to_vec();
            let pred = Float64Chunked::from_slice("pred", &pred);
            let predictions = pred.into_series();
            let actuals = inputs[0].clone(); // ref counted
            let residue = (actuals - predictions.clone()).with_name("resid"); // ref counted
            let out = StructChunked::new("", &[predictions, residue])?;
            Ok(out.into_series())
        }
        Err(e) => Err(e),
    }
}

#[polars_expr(output_type_func=report_output)]
fn pl_lstsq_report(inputs: &[Series], kwargs: LstsqKwargs) -> PolarsResult<Series> {
    let add_bias = kwargs.bias;
    // Copy data
    // Target y is at index 0
    match series_to_mat_lstsq(inputs, add_bias) {
        Ok(mat) => {
            let ncols = mat.ncols() - 1;
            let nrows = mat.nrows();

            let y = mat.slice(s![0..nrows, 0..1]);
            let y = y.view().into_faer();
            let x = mat.slice(s![0..nrows, 1..]);
            let xt = x.t().into_faer();
            let x = x.view().into_faer();
            // Solving Least Square
            let xtx = xt * &x;
            let cholesky = xtx.cholesky(Side::Lower).map_err(|_| {
                PolarsError::ComputeError(
                    "Lstsq: X^t X is not numerically positive definite.".into(),
                )
            })?;

            let xtx_inv = cholesky.inverse();
            let coeffs = &xtx_inv * xt * y;
            let betas = coeffs.col_as_slice(0).to_vec();
            // Degree of Freedom
            let dof = nrows as f64 - ncols as f64;
            // Residue
            let res = y - x * coeffs;
            let res2 = res.transpose() * &res; // total residue
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
            let idx_series = idx_series.with_name("feat_idx").into_series();
            let coeffs_series = Float64Chunked::from_vec("coeff", betas);
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
