use super::linear_regression::{coeff_output, pred_residue_output, series_to_mat_for_lr};
use crate::linear::{quantile::quantile_regression, NullPolicy};
use faer::MatRef;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

#[derive(Deserialize)]
struct QuantileKwargs {
    quantile: f64,
    bias: bool,
    null_policy: Box<str>,
    tol: f64,
    max_iter: usize,
}

struct QuantileFit {
    data: Vec<f64>,
    nrows: usize,
    nfeatures: usize,
    mask: BooleanChunked,
    coefficients: Vec<f64>,
}

fn fit(inputs: &[Series], kwargs: QuantileKwargs) -> PolarsResult<QuantileFit> {
    polars_ensure!(!inputs.is_empty(), ComputeError: "quantile regression needs a target");
    polars_ensure!(inputs.iter().all(|s| !s.is_empty()), ComputeError: "quantile regression input data is empty");
    let nrows = inputs.iter().map(|s| s.len()).max().unwrap_or(0);
    let mut aligned = Vec::with_capacity(inputs.len());
    for (i, s) in inputs.iter().enumerate() {
        polars_ensure!(s.len() == nrows || s.len() == 1, ShapeMismatch: "quantile regression inputs have mismatched lengths");
        let mut s = s.cast(&DataType::Float64)?;
        if s.len() == 1 && nrows != 1 {
            s = s.new_from_index(0, nrows);
        }
        s.rename(format!("quantile_input_{i}").into());
        aligned.push(s);
    }
    let null_policy = NullPolicy::try_from(kwargs.null_policy)
        .map_err(|e| PolarsError::ComputeError(e.into()))?;
    polars_ensure!(null_policy != NullPolicy::SKIP_WINDOW, ComputeError: "invalid null_policy for non-rolling quantile regression");
    if null_policy == NullPolicy::IGNORE {
        polars_ensure!(!aligned.iter().any(|s| s.has_nulls()), ComputeError: "null_policy='ignore' requires inputs without null values");
    }
    let (data, nrows, nfeatures, mask) = series_to_mat_for_lr(&aligned, kwargs.bias, null_policy)?;
    let y = MatRef::from_column_major_slice(&data[..nrows], nrows, 1);
    let x = MatRef::from_column_major_slice(&data[nrows..], nrows, nfeatures);
    let coefficients = quantile_regression(x, y, kwargs.quantile, kwargs.tol, kwargs.max_iter)
        .map_err(|e| PolarsError::ComputeError(e.into()))?;
    Ok(QuantileFit {
        data,
        nrows,
        nfeatures,
        mask,
        coefficients,
    })
}

#[polars_expr(output_type_func=coeff_output)]
fn pl_quantile_coeffs(inputs: &[Series], kwargs: QuantileKwargs) -> PolarsResult<Series> {
    let result = fit(inputs, kwargs)?;
    let mut builder: ListPrimitiveChunkedBuilder<Float64Type> =
        ListPrimitiveChunkedBuilder::new("coeffs".into(), 1, result.nfeatures, DataType::Float64);
    builder.append_slice(&result.coefficients);
    Ok(builder.finish().into_series())
}

#[polars_expr(output_type_func=pred_residue_output)]
fn pl_quantile_pred(inputs: &[Series], kwargs: QuantileKwargs) -> PolarsResult<Series> {
    let result = fit(inputs, kwargs)?;
    let mut predictions = Vec::with_capacity(result.nrows);
    let mut residuals = Vec::with_capacity(result.nrows);
    for i in 0..result.nrows {
        let pred: f64 = (0..result.nfeatures)
            .map(|j| result.data[(j + 1) * result.nrows + i] * result.coefficients[j])
            .sum();
        let resid = result.data[i] - pred;
        polars_ensure!(pred.is_finite() && resid.is_finite(), ComputeError: "quantile regression prediction exceeds the finite Float64 range");
        predictions.push(pred);
        residuals.push(resid);
    }
    let restore = |name: &str, values: &[f64]| -> Series {
        if (!&result.mask).any() {
            let mut rows = values.iter();
            let ca: Float64Chunked = result
                .mask
                .iter()
                .map(|keep| {
                    if keep == Some(true) {
                        rows.next().copied()
                    } else {
                        None
                    }
                })
                .collect();
            ca.with_name(name.into()).into_series()
        } else {
            Float64Chunked::from_slice(name.into(), values).into_series()
        }
    };
    let pred = restore("pred", &predictions);
    let resid = restore("resid", &residuals);
    Ok(
        StructChunked::from_series("".into(), pred.len(), [&pred, &resid].into_iter())?
            .into_series(),
    )
}
