use crate::utils::{float_output, haversine_elementwise};
use num::Float;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

fn naive_haversine<T>(
    x_lat: &ChunkedArray<T>,
    x_long: &ChunkedArray<T>,
    y_lat: &ChunkedArray<T>,
    y_long: &ChunkedArray<T>,
) -> PolarsResult<ChunkedArray<T>>
where
    T: PolarsFloatType,
    T::Native: Float,
{
    if (y_lat.len() == y_long.len()) && (y_lat.len() == 1) {
        polars_ensure!(x_lat.len() == x_long.len(), ShapeMismatch: "inputs must have the same length");
        let (Some(e_lat), Some(e_long)) = (y_lat.get(0), y_long.get(0)) else {
            return Ok(ChunkedArray::<T>::full_null(
                x_lat.name().clone(),
                x_lat.len(),
            ));
        };
        let out: ChunkedArray<T> = x_lat
            .iter()
            .zip(x_long.iter())
            .map(|(x_lat, x_long)| {
                let x_lat = x_lat?;
                let x_long = x_long?;
                Some(haversine_elementwise(x_lat, x_long, e_lat, e_long))
            })
            .collect();
        Ok(out)
    } else if x_lat.len() == x_long.len()
        && x_long.len() == y_lat.len()
        && y_lat.len() == y_long.len()
    {
        let out: ChunkedArray<T> = x_lat
            .iter()
            .zip(x_long.iter())
            .zip(y_lat.iter())
            .zip(y_long.iter())
            .map(|(((x_lat, x_long), y_lat), y_long)| {
                let x_lat = x_lat?;
                let x_long = x_long?;
                let y_lat = y_lat?;
                let y_long = y_long?;
                Some(haversine_elementwise(x_lat, x_long, y_lat, y_long))
            })
            .collect();

        Ok(out)
    } else {
        Err(PolarsError::ShapeMismatch(
            "inputs must have the same length or one of them must be a scalar".into(),
        ))
    }
}

#[polars_expr(output_type_func=float_output)]
fn pl_haversine(inputs: &[Series]) -> PolarsResult<Series> {
    // Mixed signed integers/UInt64 promote to Int128 in Polars 2. The old
    // Float64 promotion was accepted here; keep the same floating computation.
    let casted: Vec<Series>;
    let inputs = if inputs[0].dtype() == &DataType::Int128 {
        casted = inputs
            .iter()
            .map(|s| s.cast(&DataType::Float64))
            .collect::<PolarsResult<_>>()?;
        casted.as_slice()
    } else {
        inputs
    };
    let out = match inputs[0].dtype() {
        DataType::Float32 => {
            let x_lat = inputs[0].f32().unwrap();
            let x_long = inputs[1].f32().unwrap();
            let y_lat = inputs[2].f32().unwrap();
            let y_long = inputs[3].f32().unwrap();
            let out = naive_haversine(x_lat, x_long, y_lat, y_long)?;
            out.into_series()
        }
        DataType::Float64 => {
            let x_lat = inputs[0].f64().unwrap();
            let x_long = inputs[1].f64().unwrap();
            let y_lat = inputs[2].f64().unwrap();
            let y_long = inputs[3].f64().unwrap();
            let out = naive_haversine(x_lat, x_long, y_lat, y_long)?;
            out.into_series()
        }
        _ => return Err(PolarsError::ComputeError("Data type not supported.".into())),
    };
    Ok(out)
}
