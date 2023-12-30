use polars::prelude::*;

pub fn complex_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "complex",
        DataType::List(Box::new(DataType::Float64)),
    ))
}
