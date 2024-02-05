/// OLS using Faer.
use faer::{prelude::*, MatRef, Side};
use faer::{IntoFaer, Mat};
use itertools::Itertools;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

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
    // Update to array
    // DataType::Array(DataType::Float64, fields.len());
    // fields.len() + 1 when an input is true
    Ok(Field::new(
        "coeffs",
        DataType::List(Box::new(DataType::Float64)),
    ))
}

#[inline]
fn faer_lstsq_qr(x: MatRef<f64>, y: MatRef<f64>) -> Mat<f64> {
    let qr = x.qr();
    qr.solve_lstsq(y)
}

#[polars_expr(output_type_func=coeff_output)]
fn pl_lstsq(inputs: &[Series]) -> PolarsResult<Series> {
    let last_idx = inputs.len().abs_diff(1);
    let add_bias = inputs[last_idx].bool()?;
    let add_bias: bool = add_bias.get(0).unwrap();

    let nrows = inputs[0].len();
    // 0 is target
    // let ncols = inputs[1..last_idx].len() + add_bias as usize;
    let mut vs: Vec<Series> = Vec::with_capacity(inputs.len() - 1);
    for (i, s) in inputs[0..last_idx].into_iter().enumerate() {
        if s.null_count() > 0 || s.len() <= 1 {
            return Err(PolarsError::ComputeError(
                "Lstsq: Input must not contain nulls and must have length > 1".into(),
            ));
        } else if i >= 1 {
            let news = s
                .rechunk()
                .cast(&DataType::Float64)?
                .with_name(&i.to_string());
            vs.push(news);
        }
    }
    // Constant term
    if add_bias {
        let one = Series::from_vec("const", vec![1_f64; nrows]);
        vs.push(one);
    }
    // target
    let y = inputs[0].f64()?;
    let y = y.rechunk();
    let y = y.to_ndarray()?.into_shape((nrows, 1)).unwrap();
    let y = y.view().into_faer();

    let df_x = DataFrame::new(vs)?;
    // Copy data
    match df_x.to_ndarray::<Float64Type>(IndexOrder::Fortran) {
        Ok(x) => {
            // Solving Least Square
            let x = x.view().into_faer();
            let coeffs = faer_lstsq_qr(x, y);
            let mut betas: Vec<f64> = Vec::with_capacity(coeffs.nrows());
            for i in 0..coeffs.nrows() {
                betas.push(coeffs.read(i, 0));
            }
            let mut builder: ListPrimitiveChunkedBuilder<Float64Type> =
                ListPrimitiveChunkedBuilder::new("betas", 1, betas.len(), DataType::Float64);

            builder.append_slice(&betas);
            let out = builder.finish();
            Ok(out.into_series())
        }
        Err(e) => Err(e),
    }
}

#[polars_expr(output_type_func=pred_residue_output)]
fn pl_lstsq_pred(inputs: &[Series]) -> PolarsResult<Series> {
    let last_idx = inputs.len().abs_diff(1);
    let add_bias = inputs[last_idx].bool()?;
    let add_bias: bool = add_bias.get(0).unwrap();

    let nrows = inputs[0].len();
    // 0 is target
    // let ncols = inputs[1..last_idx].len() + add_bias as usize;
    let mut vs: Vec<Series> = Vec::with_capacity(inputs.len() - 1);
    for (i, s) in inputs[0..last_idx].into_iter().enumerate() {
        if s.null_count() > 0 || s.len() <= 1 {
            return Err(PolarsError::ComputeError(
                "Lstsq: Input must not contain nulls and must have length > 1".into(),
            ));
        } else if i >= 1 {
            let news = s
                .rechunk()
                .cast(&DataType::Float64)?
                .with_name(&i.to_string());
            vs.push(news);
        }
    }
    // Constant term
    if add_bias {
        let one = Series::from_vec("const", vec![1_f64; nrows]);
        vs.push(one);
    }
    // target
    let y = inputs[0].f64()?;
    let y = y.rechunk();
    let y = y.to_ndarray()?.into_shape((nrows, 1)).unwrap();
    let y = y.view().into_faer();

    let df_x = DataFrame::new(vs)?;
    // Copy data
    match df_x.to_ndarray::<Float64Type>(IndexOrder::Fortran) {
        Ok(x) => {
            // Solving Least Square
            let x = x.view().into_faer();
            let coeffs = faer_lstsq_qr(x, y);
            let y_hat = x * coeffs;
            let rows = y_hat.nrows();
            let mut pred: Vec<f64> = Vec::with_capacity(rows);
            for i in 0..rows {
                pred.push(y_hat.read(i, 0));
            }
            let predictions = Series::from_vec("pred", pred);
            let actuals = inputs[0].clone(); // ref counted
            let residue = (actuals - predictions.clone()).with_name("resid"); // ref counted
            let out = StructChunked::new("", &[predictions, residue])?;
            Ok(out.into_series())
        }
        Err(e) => Err(e),
    }
}

#[polars_expr(output_type_func=report_output)]
fn pl_lstsq_report(inputs: &[Series]) -> PolarsResult<Series> {
    let last_idx = inputs.len().abs_diff(1);
    let add_bias = inputs[last_idx].bool()?;
    let add_bias: bool = add_bias.get(0).unwrap();

    let nrows = inputs[0].len();
    // 0 is target
    let ncols = inputs[1..last_idx].len() + add_bias as usize;
    let mut vs: Vec<Series> = Vec::with_capacity(inputs.len() - 1);
    for (i, s) in inputs[0..last_idx].into_iter().enumerate() {
        if s.null_count() > 0 || s.len() <= ncols {
            // If there is null input, return NAN
            return Err(PolarsError::ComputeError(
                "Lstsq Report: Input must not contain nulls and must have length > # of features."
                    .into(),
            ));
        } else if i >= 1 {
            let news = s
                .rechunk()
                .cast(&DataType::Float64)?
                .with_name(&i.to_string());
            vs.push(news);
        }
    }
    // Constant term
    if add_bias {
        let one = Series::from_vec("const", vec![1_f64; nrows]);
        vs.push(one);
    }
    // target
    let y = inputs[0].f64()?;
    let y = y.rechunk();
    let y = y.to_ndarray()?.into_shape((nrows, 1)).unwrap();
    let y = y.view().into_faer();

    let df_x = DataFrame::new(vs)?;
    // Copy data
    match df_x.to_ndarray::<Float64Type>(IndexOrder::Fortran) {
        Ok(x) => {
            // Solving Least Square
            let xt = x.t();
            let xt = xt.view().into_faer();
            let x = x.view().into_faer();

            let xtx = xt * &x;
            let cholesky = xtx.cholesky(Side::Lower).map_err(|_| {
                PolarsError::ComputeError(
                    "Lstsq: X^t X is not numerically positive definite.".into(),
                )
            })?;
            let xtx_inv = cholesky.inverse();
            let coeffs = &xtx_inv * xt * y;
            let mut betas: Vec<f64> = Vec::with_capacity(coeffs.nrows());
            for i in 0..coeffs.nrows() {
                betas.push(coeffs.read(i, 0));
            }
            // Degree of Freedom
            let dof = nrows as f64 - ncols as f64;
            // Residue
            let res = y - x * coeffs;
            let res2 = res.transpose() * &res;
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
