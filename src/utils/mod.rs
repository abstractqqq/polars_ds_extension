use cfavml::safe_trait_distance_ops::DistanceOps;
use num::Float;
use polars::{
    datatypes::{DataType, Field},
    error::{polars_ensure, PolarsError, PolarsResult},
    frame::DataFrame,
    lazy::dsl::FieldsMapper,
    prelude::*,
    series::Series,
};
use pyo3_polars::export::polars_core::POOL;
use rayon::prelude::*;

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

// #[inline(always)]
// pub fn series_to_ndarray(inputs: &[Series], order: IndexOrder) -> PolarsResult<Array2<f64>> {
//     let df = DataFrame::new(inputs.to_vec())?;
//     if df.is_empty() {
//         Err(PolarsError::ComputeError("Empty data.".into()))
//     } else {
//         df.to_ndarray::<Float64Type>(order)
//     }
// }

/// Organizes the series data into a `matrix`, and return the underlying slice
/// as a row-major slice. This code here is taken from polars dataframe.to_ndarray()
#[inline(always)]
pub fn series_to_row_major_slice<N>(
    series: &[Series],
) -> PolarsResult<Vec<<N as PolarsNumericType>::Native>>
where
    N: PolarsNumericType,
{
    if series.is_empty() {
        return Err(PolarsError::NoData("Data is empty".into()));
    }
    // Safe because series is not empty
    let height: usize = series[0].len();
    for s in &series[1..] {
        if s.len() != height {
            return Err(PolarsError::ShapeMismatch(
                "Seires don't have the same length.".into(),
            ));
        }
    }
    let m = series.len();
    let mut membuf = Vec::with_capacity(height * m);
    let ptr = membuf.as_ptr() as usize;
    // let columns = self.get_columns();
    POOL.install(|| {
        series.par_iter().enumerate().try_for_each(|(col_idx, s)| {
            let s = s.cast(&N::get_dtype())?;
            let s = match s.dtype() {
                DataType::Float32 => {
                    let ca = s.f32().unwrap();
                    ca.none_to_nan().into_series()
                }
                DataType::Float64 => {
                    let ca = s.f64().unwrap();
                    ca.none_to_nan().into_series()
                }
                _ => s,
            };
            polars_ensure!(
                s.null_count() == 0,
                ComputeError: "creation of ndarray with null values is not supported"
            );
            let ca = s.unpack::<N>()?;

            let mut chunk_offset = 0;
            for arr in ca.downcast_iter() {
                let vals = arr.values();
                unsafe {
                    let num_cols = m;
                    let mut offset = (ptr as *mut N::Native).add(col_idx + chunk_offset * num_cols);
                    for v in vals.iter() {
                        *offset = *v;
                        offset = offset.add(num_cols);
                    }
                }
                chunk_offset += vals.len();
            }

            Ok(())
        })
    })?;

    // SAFETY:
    // we have written all data, so we can now safely set length
    unsafe {
        membuf.set_len(height * m);
    }
    Ok(membuf)
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
    L2SIMD,
    SQL2, // Squared L2
    SQL2SIMD,
    LINF,
    ANY(fn(&[T], &[T]) -> T),
}

impl<T: Float + DistanceOps + 'static> DIST<T> {
    /// New DIST from the string and informed by the dimension
    pub fn new_from_str_informed(dist_str: String, dim: usize) -> Result<Self, String> {
        match dist_str.as_ref() {
            "l1" => Ok(DIST::L1),
            "l2" => {
                if dim < 16 {
                    Ok(DIST::L2)
                } else {
                    Ok(DIST::L2SIMD)
                }
            }
            "sql2" => {
                if dim < 16 {
                    Ok(DIST::SQL2)
                } else {
                    Ok(DIST::SQL2SIMD)
                }
            }
            "linf" | "inf" => Ok(DIST::LINF),
            "cosine" => Ok(DIST::ANY(cfavml::cosine)),
            _ => Err("Unknown distance metric.".into()),
        }
    }

    #[inline(always)]
    pub fn dist(&self, a1: &[T], a2: &[T]) -> T {
        match self {
            DIST::L1 => a1
                .iter()
                .copied()
                .zip(a2.iter().copied())
                .fold(T::zero(), |acc, (x, y)| acc + ((x - y).abs())),

            DIST::L2 => a1
                .iter()
                .copied()
                .zip(a2.iter().copied())
                .fold(T::zero(), |acc, (x, y)| acc + (x - y) * (x - y))
                .sqrt(),

            DIST::L2SIMD => cfavml::squared_euclidean(a1, a2).sqrt(),

            DIST::SQL2 => a1
                .iter()
                .copied()
                .zip(a2.iter().copied())
                .fold(T::zero(), |acc, (x, y)| acc + (x - y) * (x - y)),

            DIST::SQL2SIMD => cfavml::squared_euclidean(a1, a2),

            DIST::LINF => a1
                .iter()
                .copied()
                .zip(a2.iter().copied())
                .fold(T::zero(), |acc, (x, y)| acc.max((x - y).abs())),

            DIST::ANY(func) => func(a1, a2),
        }
    }
}
