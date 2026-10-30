//! One-predictor EWLS with finite-window statistics and scalar Cholesky solves.

use super::linear_regression::SWWLRKwargs;
use crate::linear::online_lr::ewls_1d::for_each_rolling_ewls_1d;
use crate::utils::{series_to_slice, IndexOrder};
use faer_traits::{math_utils::from_f64, RealField};
use num::Float;
use polars::prelude::*;

pub(super) fn rolling_ewls_1d<N>(inputs: &[Series], kwargs: SWWLRKwargs) -> PolarsResult<Series>
where
    N: PolarsNumericType,
    N::Native: RealField + Float,
{
    polars_ensure!(inputs.len() == 2, ComputeError:
        "rolling 1d EWLS requires one target and one predictor");
    polars_ensure!(kwargs.bias, ComputeError: "rolling 1d EWLS requires an intercept");
    polars_ensure!(kwargs.lambda == 0.0, ComputeError:
        "rolling 1d EWLS does not support regularization");
    polars_ensure!(kwargs.null_policy.eq_ignore_ascii_case("skip"), ComputeError:
        "rolling 1d EWLS supports only null_policy='skip'");
    polars_ensure!(kwargs.n >= 2 && kwargs.min_size > 0 && kwargs.min_size <= kwargs.n,
        ComputeError: "rolling 1d EWLS requires window_size >= 2 and a valid min_valid_rows");
    let half_life = kwargs
        .half_life
        .filter(|v| v.is_finite() && *v > 0.0)
        .ok_or_else(|| PolarsError::ComputeError("half_life must be positive and finite".into()))?;
    let nrows = inputs[0].len();
    polars_ensure!(inputs[1].len() == nrows, ShapeMismatch: "target and predictor lengths differ");
    let dtype = N::get_static_dtype();
    let columns = inputs
        .iter()
        .map(|s| s.cast(&dtype))
        .collect::<PolarsResult<Vec<_>>>()?;
    let data = series_to_slice::<N>(&columns, IndexOrder::Fortran)?;
    let (y, x) = data.split_at(nrows);
    let mut coeffs = ListPrimitiveChunkedBuilder::<N>::new(
        "coeffs".into(),
        nrows,
        nrows.saturating_sub(kwargs.n - 1) * 2,
        dtype.clone(),
    );
    let mut pred = PrimitiveChunkedBuilder::<N>::new("pred".into(), nrows);
    let rank_tol = from_f64(if dtype == DataType::Float32 {
        1e-6
    } else {
        1e-12
    });
    for_each_rolling_ewls_1d(
        x,
        y,
        kwargs.n,
        kwargs.min_size,
        half_life,
        rank_tol,
        |i, fit| {
            if let Some(beta) = fit {
                coeffs.append_slice(&beta);
                let value = x[i] * beta[0] + beta[1];
                pred.append_option(value.is_finite().then_some(value));
            } else {
                coeffs.append_null();
                pred.append_null();
            }
        },
    );
    let coefficients = coeffs.finish().into_series();
    let predictions = pred.finish().into_series();
    Ok(
        StructChunked::from_series("".into(), nrows, [&coefficients, &predictions].into_iter())?
            .into_series(),
    )
}
