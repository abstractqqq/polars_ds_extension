use cfavml::safe_trait_distance_ops::DistanceOps;
use ndarray::Array2;
use num::Float;
use polars::{
    datatypes::{DataType, Field, Float64Type},
    error::{PolarsError, PolarsResult},
    frame::DataFrame,
    lazy::dsl::FieldsMapper,
    prelude::IndexOrder,
    series::Series,
};

// -------------------------------------------------------------------------------
// Common, Resuable Functions
// -------------------------------------------------------------------------------

#[inline(always)]
pub fn rechunk_to_frame(inputs: &[Series]) -> PolarsResult<DataFrame> {
    let mut df = DataFrame::new(inputs.to_vec())?;
    df = df.align_chunks().clone(); // ref count, cheap clone
    Ok(df)
}

#[inline(always)]
pub fn to_frame(inputs: &[Series]) -> PolarsResult<DataFrame> {
    DataFrame::new(inputs.to_vec())
}

#[inline(always)]
pub fn series_to_ndarray(inputs: &[Series], order: IndexOrder) -> PolarsResult<Array2<f64>> {
    let df = DataFrame::new(inputs.to_vec())?;
    if df.is_empty() {
        Err(PolarsError::ComputeError("Empty data.".into()))
    } else {
        df.to_ndarray::<Float64Type>(order)
    }
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

// -------------------------------------------------------------------------------
// Common, Structures
// -------------------------------------------------------------------------------
#[derive(PartialEq, Clone, Copy)]
pub enum NullPolicy {
    RAISE,
    SKIP,
    SKIP_WINDOW, // `SKIP` in rolling. Skip, but doesn't really drop data. A specialized algorithm will handle this.
    IGNORE,
    FILL(f64),
    FILL_WINDOW(f64), // `FILL`` rolling. Doesn't really drop data. A specialized algorithm will handle this.
}

impl TryFrom<String> for NullPolicy {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        let binding = value.to_lowercase();
        let test = binding.as_ref();
        match test {
            "raise" => Ok(Self::RAISE),
            "skip" => Ok(Self::SKIP),
            "zero" => Ok(Self::FILL(0.)),
            "one" => Ok(Self::FILL(1.)),
            "ignore" => Ok(Self::IGNORE),
            "skip_window" => Ok(Self::SKIP_WINDOW),
            _ => match test.parse::<f64>() {
                Ok(x) => Ok(Self::FILL(x)),
                Err(_) => Err("Invalid NullPolicy.".into()),
            },
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DIST<T: Float + 'static> {
    L1,
    L2,
    SQL2, // Squared L2
    LINF,
    ANY(fn(&[T], &[T]) -> T),
}

impl<T: Float + DistanceOps + 'static> DIST<T> {
    #[inline(always)]
    pub fn dist(&self, a1: &[T], a2: &[T]) -> T {
        match self {
            DIST::L1 => a1
                .iter()
                .copied()
                .zip(a2.iter().copied())
                .fold(T::zero(), |acc, (x, y)| acc + ((x - y).abs())),

            DIST::L2 => {
                if a1.len() < 16 {
                    a1.iter()
                        .copied()
                        .zip(a2.iter().copied())
                        .fold(T::zero(), |acc, (x, y)| acc + (x - y) * (x - y))
                        .sqrt()
                } else {
                    cfavml::squared_euclidean(a1, a2).sqrt()
                }
            }
            DIST::SQL2 => {
                // Small penalty to use cfavml when len is small
                if a1.len() < 16 {
                    a1.iter()
                        .copied()
                        .zip(a2.iter().copied())
                        .fold(T::zero(), |acc, (x, y)| acc + (x - y) * (x - y))
                } else {
                    cfavml::squared_euclidean(a1, a2)
                }
            }
            DIST::LINF => a1
                .iter()
                .copied()
                .zip(a2.iter().copied())
                .fold(T::zero(), |acc, (x, y)| acc.max((x - y).abs())),

            DIST::ANY(func) => func(a1, a2),
        }
    }
}
