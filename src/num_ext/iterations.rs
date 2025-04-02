use itertools::Itertools;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub(crate) struct CombinationKwargs {
    pub(crate) k: usize,
}

fn itertools_output(fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "".into(),
        DataType::List(Box::new(fields[0].dtype().clone())),
    ))
}

fn count_combinations(n: usize, k: usize) -> usize {
    if k > n {
        0
    } else {
        (1..=k.min(n - k)).fold(1, |acc, val| acc * (n - val + 1) / val)
    }
}

fn get_combinations<T>(ca: &ChunkedArray<T>, k: usize) -> Series
where
    T: PolarsNumericType,
{
    let mut builder: ListPrimitiveChunkedBuilder<T> = ListPrimitiveChunkedBuilder::new(
        "".into(),
        count_combinations(ca.len(), k),
        k,
        T::get_dtype(),
    );

    for comb in ca.into_no_null_iter().combinations(k) {
        builder.append_slice(&comb);
    }

    let ca = builder.finish();
    ca.into_series()
}

fn get_product<T>(ca1: &ChunkedArray<T>, ca2: &ChunkedArray<T>) -> Series
where
    T: PolarsNumericType,
{
    let mut builder: ListPrimitiveChunkedBuilder<T> =
        ListPrimitiveChunkedBuilder::new("".into(), ca1.len() * ca2.len(), 2, T::get_dtype());

    for a in ca1.into_no_null_iter() {
        for b in ca2.into_no_null_iter() {
            builder.append_slice(&[a, b]);
        }
    }

    let ca = builder.finish();
    ca.into_series()
}

fn get_combinations_str(ca: &StringChunked, k: usize) -> Series {
    let mut builder: ListStringChunkedBuilder =
        ListStringChunkedBuilder::new("".into(), count_combinations(ca.len(), k), k);

    for comb in ca.into_no_null_iter().combinations(k) {
        builder.append_values_iter(comb.into_iter());
    }

    let ca = builder.finish();
    ca.into_series()
}

fn get_product_str(ca1: &StringChunked, ca2: &StringChunked) -> Series {
    let mut builder: ListStringChunkedBuilder =
        ListStringChunkedBuilder::new("".into(), ca1.len() * ca2.len(), 2);

    for a in ca1.into_no_null_iter() {
        for b in ca2.into_no_null_iter() {
            builder.append_values_iter([a, b].into_iter());
        }
    }

    let ca = builder.finish();
    ca.into_series()
}

#[polars_expr(output_type_func=itertools_output)]
fn pl_combinations(inputs: &[Series], kwargs: CombinationKwargs) -> PolarsResult<Series> {
    let s = &inputs[0];
    let k = kwargs.k;

    if s.len() < k {
        return Err(PolarsError::ComputeError(
            "Source has < k (unique) values.".into(),
        ));
    }

    match s.dtype() {
        DataType::UInt8 => Ok(get_combinations(s.u8().unwrap(), k)),
        DataType::UInt16 => Ok(get_combinations(s.u16().unwrap(), k)),
        DataType::UInt32 => Ok(get_combinations(s.u32().unwrap(), k)),
        DataType::UInt64 => Ok(get_combinations(s.u64().unwrap(), k)),
        DataType::Int8 => Ok(get_combinations(s.i8().unwrap(), k)),
        DataType::Int16 => Ok(get_combinations(s.i16().unwrap(), k)),
        DataType::Int32 => Ok(get_combinations(s.i32().unwrap(), k)),
        DataType::Int64 => Ok(get_combinations(s.i64().unwrap(), k)),
        DataType::Int128 => Ok(get_combinations(s.i128().unwrap(), k)),
        DataType::Float32 => Ok(get_combinations(s.f32().unwrap(), k)),
        DataType::Float64 => Ok(get_combinations(s.f64().unwrap(), k)),
        DataType::String => Ok(get_combinations_str(s.str().unwrap(), k)),
        _ => Err(PolarsError::ComputeError("Unsupported data type.".into())),
    }
}

#[polars_expr(output_type_func=itertools_output)]
fn pl_product(inputs: &[Series]) -> PolarsResult<Series> {
    let s1 = &inputs[0];
    let s2 = &inputs[1];

    if s1.dtype() != s2.dtype() {
        return Err(PolarsError::ComputeError(
            "Dtype of first input series is not the same as the second.".into(),
        ));
    }

    match s1.dtype() {
        DataType::UInt8 => Ok(get_product(s1.u8().unwrap(), s2.u8().unwrap())),
        DataType::UInt16 => Ok(get_product(s1.u16().unwrap(), s2.u16().unwrap())),
        DataType::UInt32 => Ok(get_product(s1.u32().unwrap(), s2.u32().unwrap())),
        DataType::UInt64 => Ok(get_product(s1.u64().unwrap(), s2.u64().unwrap())),
        DataType::Int8 => Ok(get_product(s1.i8().unwrap(), s2.i8().unwrap())),
        DataType::Int16 => Ok(get_product(s1.i16().unwrap(), s2.i16().unwrap())),
        DataType::Int32 => Ok(get_product(s1.i32().unwrap(), s2.i32().unwrap())),
        DataType::Int64 => Ok(get_product(s1.i64().unwrap(), s2.i64().unwrap())),
        DataType::Int128 => Ok(get_product(s1.i128().unwrap(), s2.i128().unwrap())),
        DataType::Float32 => Ok(get_product(s1.f32().unwrap(), s2.f32().unwrap())),
        DataType::Float64 => Ok(get_product(s1.f64().unwrap(), s2.f64().unwrap())),
        DataType::String => Ok(get_product_str(s1.str().unwrap(), s2.str().unwrap())),
        _ => Err(PolarsError::ComputeError("Unsupported data type.".into())),
    }
}
