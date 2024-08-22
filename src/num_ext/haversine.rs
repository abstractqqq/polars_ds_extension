use num::Float;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

fn haversine_output(input_fields: &[Field]) -> PolarsResult<Field> {
    FieldsMapper::new(input_fields).map_to_float_dtype()
}

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
        let e_lat = y_lat.get(0).unwrap();
        let e_long = y_long.get(0).unwrap();
        let out: ChunkedArray<T> = x_lat
            .into_iter()
            .zip(x_long.into_iter())
            .map(|(x_lat, x_long)| {
                let x_lat = x_lat?;
                let x_long = x_long?;
                Some(super::haversine_elementwise(x_lat, x_long, e_lat, e_long))
            })
            .collect();
        Ok(out.with_name(x_lat.name()))
    } else if x_lat.len() == x_long.len()
        && x_long.len() == y_lat.len()
        && y_lat.len() == y_long.len()
    {
        let out: ChunkedArray<T> = x_lat
            .into_iter()
            .zip(x_long.into_iter())
            .zip(y_lat.into_iter())
            .zip(y_long.into_iter())
            .map(|(((x_lat, x_long), y_lat), y_long)| {
                let x_lat = x_lat?;
                let x_long = x_long?;
                let y_lat = y_lat?;
                let y_long = y_long?;
                Some(super::haversine_elementwise(x_lat, x_long, y_lat, y_long))
            })
            .collect();

        Ok(out.with_name(x_lat.name()))
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length or one of them must be a scalar.".into(),
        ))
    }
}

#[polars_expr(output_type_func=haversine_output)]
fn pl_haversine(inputs: &[Series]) -> PolarsResult<Series> {
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
