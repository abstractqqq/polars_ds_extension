//! Shared typed adapter for the EWLS branches of the existing rolling plugins.

use super::linear_regression::SWWLRKwargs;
use crate::linear::online_lr::ewls::faer_rolling_ewls;
use crate::utils::{series_to_slice_with_extra_cap, IndexOrder};
use faer::MatRef;
use faer_traits::RealField;
use num::Float;
use polars::prelude::*;

pub(super) fn rolling_ewls<N>(
    inputs: &[Series],
    kwargs: &SWWLRKwargs,
    half_life: f64,
) -> PolarsResult<Series>
where
    N: PolarsNumericType,
    N::Native: RealField + Float + Into<f64>,
{
    polars_ensure!(half_life.is_finite() && half_life > 0.0,
        ComputeError: "`half_life` must be positive and finite.");
    polars_ensure!(kwargs.lambda == 0.0,
        ComputeError: "Exponentially weighted rolling regression requires `l2_reg=0`.");
    let skip = kwargs.null_policy.eq_ignore_ascii_case("skip");
    polars_ensure!(skip || kwargs.null_policy.eq_ignore_ascii_case("raise"),
        ComputeError: "EWLS supports only `null_policy='skip'` or 'raise'.");
    polars_ensure!(kwargs.n >= 2 && kwargs.min_size > 0 && kwargs.min_size <= kwargs.n,
        ComputeError: "EWLS requires window_size >= 2 and 1 <= min_valid_rows <= window_size.");
    let nrows = inputs
        .first()
        .ok_or_else(|| PolarsError::NoData("Empty inputs".into()))?
        .len();
    let nfeats = inputs.len() - 1 + usize::from(kwargs.bias);
    polars_ensure!(nfeats > 0, ComputeError: "EWLS requires a predictor or an intercept.");

    // Cast before conversion so all-null (Null dtype) predictors are accepted.
    // The shared converter validates lengths, preserves chunks/slice offsets,
    // and represents nulls as NaNs without dropping any rows.
    let dtype = N::get_static_dtype();
    let columns: Vec<Series> = inputs
        .iter()
        .map(|s| s.cast(&dtype))
        .collect::<PolarsResult<_>>()?;
    let extra = if kwargs.bias { nrows } else { 0 };
    let mut data = series_to_slice_with_extra_cap::<N>(&columns, IndexOrder::Fortran, extra)?;
    data.extend(std::iter::repeat(num::one::<N::Native>()).take(extra));
    let y = MatRef::from_column_major_slice(&data[..nrows], nrows, 1);
    let x = MatRef::from_column_major_slice(&data[nrows..], nrows, nfeats);
    let valid: Vec<bool> = (0..nrows)
        .map(|i| x.get(i, ..).is_all_finite() && y.get(i, ..).is_all_finite())
        .collect();
    polars_ensure!(skip || valid.iter().all(|v| *v), ComputeError:
        "EWLS with null_policy='raise' requires finite, non-null predictors and target.");
    let rank_tol = if dtype == DataType::Float32 {
        1e-6
    } else {
        1e-12
    };
    let fits = faer_rolling_ewls(x, y, &valid, kwargs.n, kwargs.min_size, half_life, rank_tol);

    let capacity = nrows.saturating_sub(kwargs.n - 1) * nfeats;
    let mut coeffs = ListPrimitiveChunkedBuilder::<N>::new("coeffs".into(), nrows, capacity, dtype);
    let mut pred = PrimitiveChunkedBuilder::<N>::new("pred".into(), nrows);
    for (i, fit) in fits.into_iter().enumerate() {
        let Some(beta) = fit else {
            coeffs.append_null();
            pred.append_null();
            continue;
        };
        let converted: Option<Vec<N::Native>> = beta
            .iter()
            .map(|&v| num::cast::<f64, N::Native>(v).filter(|c| c.is_finite()))
            .collect();
        let Some(converted) = converted else {
            // A finite f64 solution may overflow the requested f32 output.
            coeffs.append_null();
            pred.append_null();
            continue;
        };
        coeffs.append_slice(&converted);
        let prediction = if x.get(i, ..).is_all_finite() {
            let value: f64 = beta
                .iter()
                .enumerate()
                .map(|(j, b)| (*x.get(i, j)).into() * b)
                .sum();
            num::cast::<f64, N::Native>(value).filter(|v| v.is_finite())
        } else {
            None
        };
        pred.append_option(prediction);
    }
    let coefficients = coeffs.finish().into_series();
    let predictions = pred.finish().into_series();
    Ok(
        StructChunked::from_series("".into(), nrows, [&coefficients, &predictions].into_iter())?
            .into_series(),
    )
}
