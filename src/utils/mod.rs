use ndarray::Array2;
use polars::{
    datatypes::{DataType, Field, Float32Type, Float64Type},
    error::{PolarsError, PolarsResult},
    frame::DataFrame,
    lazy::dsl::FieldsMapper,
    prelude::IndexOrder,
    series::Series,
};

// -------------------------------------------------------------------------------
// Common, Resuable Functions
// -------------------------------------------------------------------------------

// Rechunk series, rename then by the order, and return a PolarsResult<DataFrame>
#[inline(always)]
pub fn rechunk_to_frame(inputs: &[Series]) -> PolarsResult<DataFrame> {
    let mut df = DataFrame::new(inputs.to_vec())?;
    df = df.align_chunks().clone(); // ref count, cheap clone
    Ok(df)
}

// Rechunk series, rename then by the order, and return a PolarsResult<DataFrame>
#[inline(always)]
pub fn to_frame(inputs: &[Series]) -> PolarsResult<DataFrame> {
    DataFrame::new(inputs.to_vec())
}

#[inline(always)]
pub fn series_to_ndarray(inputs: &[Series], order: IndexOrder) -> PolarsResult<Array2<f64>> {
    let df = DataFrame::new(inputs.to_vec())?;
    df.to_ndarray::<Float64Type>(order)
}

#[inline(always)]
pub fn series_to_ndarray_f32(inputs: &[Series], order: IndexOrder) -> PolarsResult<Array2<f32>> {
    let df = DataFrame::new(inputs.to_vec())?;
    df.to_ndarray::<Float32Type>(order)
}

// Shared splitting method
pub fn split_offsets(len: usize, n: usize) -> Vec<(usize, usize)> {
    if n == 1 {
        vec![(0, len)]
    } else {
        let chunk_size = len / n;
        (0..n)
            .map(|partition| {
                let offset = partition * chunk_size;
                let len = if partition == (n - 1) {
                    len - offset
                } else {
                    chunk_size
                };
                (partition * chunk_size, len)
            })
            .collect()
    }
}

// pub fn get_common_float_dtype(inputs: &[Series]) -> DataType {
//     inputs
//         .into_iter()
//         .fold(DataType::Null, |_, s| match s.dtype() {
//             DataType::UInt8
//             | DataType::UInt16
//             | DataType::UInt32
//             | DataType::Int8
//             | DataType::Int16
//             | DataType::Int32
//             | DataType::Float32 => DataType::Float32,

//             DataType::Float64 | DataType::UInt64 | DataType::Int64 => DataType::Float64,

//             _ => DataType::Null,
//         })
// }

// -------------------------------------------------------------------------------
// Common Output Types
// -------------------------------------------------------------------------------

pub fn first_field_output(fields: &[Field]) -> PolarsResult<Field> {
    Ok(fields[0].clone())
}

// pub fn list_u64_output(_: &[Field]) -> PolarsResult<Field> {
//     Ok(Field::new(
//         "nodes",
//         DataType::List(Box::new(DataType::UInt64)),
//     ))
// }

pub fn list_u32_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "nodes",
        DataType::List(Box::new(DataType::UInt32)),
    ))
}

pub fn list_f64_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "floats",
        DataType::List(Box::new(DataType::Float64)),
    ))
}

pub fn complex_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "complex",
        DataType::Array(Box::new(DataType::Float64), 2),
    ))
}

pub fn list_str_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "list_str",
        DataType::List(Box::new(DataType::String)),
    ))
}

pub fn float_output(fields: &[Field]) -> PolarsResult<Field> {
    FieldsMapper::new(fields).map_to_float_dtype()
}
