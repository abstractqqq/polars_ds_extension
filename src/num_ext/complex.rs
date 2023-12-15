use polars::prelude::*;
// use pyo3_polars::derive::polars_expr;
// use num::Complex;

pub fn complex_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "complex",
        DataType::List(Box::new(DataType::Float64)),
    ))
}
