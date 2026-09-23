//! Shared typed adapter for the EWLS branches of the existing rolling plugins.

use super::linear_regression::SWWLRKwargs;
use crate::linear::online_lr::ewls::faer_rolling_ewls;
use crate::linear::NullPolicy;
use crate::utils::{series_to_slice_with_extra_cap, IndexOrder};
use faer::MatRef;
use faer_traits::{math_utils::from_f64, RealField};
use num::Float;
use polars::prelude::*;

pub(super) fn rolling_ewls<N>(
    inputs: &[Series],
    kwargs: SWWLRKwargs,
    half_life: f64,
    one_predictor: bool,
) -> PolarsResult<Series>
where
    N: PolarsNumericType,
    N::Native: RealField + Float,
{
    polars_ensure!(half_life.is_finite() && half_life > 0.0,
        ComputeError: "`half_life` must be positive and finite.");
    polars_ensure!(kwargs.lambda == 0.0,
        ComputeError: "Exponentially weighted rolling regression requires `l2_reg=0`.");
    let null_policy = NullPolicy::<f64>::try_from(kwargs.null_policy)
        .map_err(|e| PolarsError::ComputeError(e.into()))?;
    let skip = null_policy == NullPolicy::SKIP;
    polars_ensure!(skip || null_policy == NullPolicy::RAISE,
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
    data.extend(std::iter::repeat_n(num::one::<N::Native>(), extra));
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
    let fits = faer_rolling_ewls(
        x,
        y,
        &valid,
        kwargs.n,
        kwargs.min_size,
        half_life,
        from_f64(rank_tol),
        one_predictor,
    );

    let capacity = nrows.saturating_sub(kwargs.n - 1) * nfeats;
    let mut coeffs = ListPrimitiveChunkedBuilder::<N>::new("coeffs".into(), nrows, capacity, dtype);
    let mut pred = PrimitiveChunkedBuilder::<N>::new("pred".into(), nrows);
    for (i, fit) in fits.into_iter().enumerate() {
        let Some(beta) = fit else {
            coeffs.append_null();
            pred.append_null();
            continue;
        };
        coeffs.append_slice(beta.col_as_slice(0));
        let prediction = if x.get(i, ..).is_all_finite() {
            let value = if one_predictor {
                *x.get(i, 0) * *beta.get(0, 0) + *beta.get(1, 0)
            } else {
                *(x.get(i..i + 1, ..) * &beta).get(0, 0)
            };
            value.is_finite().then_some(value)
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
