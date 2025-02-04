use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use itertools::Itertools;
use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub(crate) struct CombinationKwargs {
    pub(crate) unique: bool,
    pub(crate) k: usize,
}

fn combination_output(fields: &[Field]) -> PolarsResult<Field> {
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


fn get_combinations<T>(
    ca: &ChunkedArray<T>,
    k: usize,
) -> Series 
where
    T: PolarsNumericType,
{

    let mut builder: ListPrimitiveChunkedBuilder<T> = ListPrimitiveChunkedBuilder::new(
        "".into(), 
        count_combinations(ca.len(), k), 
        k, 
        T::get_dtype()
    );

    for comb in ca.into_no_null_iter().combinations(k) {
        builder.append_slice(&comb);
    }

    let ca = builder.finish();
    ca.into_series()
}

fn get_combinations_str(
    ca: &StringChunked,
    k: usize,
) -> Series {

    let mut builder: ListStringChunkedBuilder = ListStringChunkedBuilder::new(
        "".into(), 
        count_combinations(ca.len(), k),
        k,
    );

    for comb in ca.into_no_null_iter().combinations(k) {
        builder.append_values_iter(comb.into_iter());
    }

    let ca = builder.finish();
    ca.into_series()
}

#[polars_expr(output_type_func=combination_output)]
fn pl_combinations(inputs: &[Series], kwargs: CombinationKwargs) -> PolarsResult<Series> {

    let s = if kwargs.unique {
        inputs[0].unique()?
    } else {
        inputs[0].clone()
    };
    let k = kwargs.k;

    if s.len() < k {
        return Err(PolarsError::ComputeError("Source has < k (unique) values.".into()))
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
        _ => Err(PolarsError::ComputeError("Unsupported data type.".into()))
    }
}
