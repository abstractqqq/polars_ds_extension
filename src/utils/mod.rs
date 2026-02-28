use itertools::Itertools;
// use cfavml::safe_trait_distance_ops::DistanceOps;
use num::Float;
use polars::{
    datatypes::{DataType, Field},
    error::{polars_ensure, PolarsError, PolarsResult},
    frame::DataFrame,
    prelude::*,
    series::Series,
};
use pyo3_polars::export::{
    polars_core::{
        utils::rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator},
        POOL,
    },
    polars_plan::plans::FieldsMapper,
};

pub enum IndexOrder {
    C,
    Fortran,
}

// -------------------------------------------------------------------------------
// Common, Resuable Functions
// -------------------------------------------------------------------------------

#[inline(always)]
pub fn to_frame(inputs: &[Series]) -> PolarsResult<DataFrame> {
    DataFrame::new(inputs.iter().map(|s| s.clone().into_column()).collect())
}

/// Organizes the series data into a `vec`, which is either C(row major) or Fortran(column major).
/// This code here is taken from polars dataframe.to_ndarray()
#[inline(always)]
pub fn series_to_slice<N>(series: &[Series], ordering: IndexOrder) -> PolarsResult<Vec<<N>::Native>>
where
    N: PolarsNumericType,
{
    if series.is_empty() {
        return Err(PolarsError::NoData("Data is empty".into()));
    }
    if series.iter().any(|s| !s.dtype().is_numeric()) {
        return Err(PolarsError::ComputeError(
            "All columns need to be numeric.".into(),
        ));
    }
    if !series.iter().map(|s| s.len()).all_equal() {
        return Err(PolarsError::ShapeMismatch(
            "Seires don't have the same length.".into(),
        ));
    }
    // Safe because series is not empty
    let height: usize = series[0].len();
    let m = series.len();
    let mut membuf = Vec::with_capacity(height * m);
    let ptr = membuf.as_ptr() as usize;

    POOL.install(|| {
        series.par_iter().enumerate().try_for_each(|(col_idx, s)| {
            let s = s.cast(&N::get_static_dtype())?;
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
                // Depending on the desired order, we add items to the buffer.
                // SAFETY:
                // We get parallel access to the vector by offsetting index access accordingly.
                // For C-order, we only operate on every num-col-th element, starting from the
                // column index. For Fortran-order we only operate on n contiguous elements,
                // offset by n * the column index.
                match ordering {
                    IndexOrder::C => unsafe {
                        let num_cols = series.len();
                        let mut offset =
                            (ptr as *mut N::Native).add(col_idx + chunk_offset * num_cols);
                        for v in vals.iter() {
                            *offset = *v;
                            offset = offset.add(num_cols);
                        }
                    },
                    IndexOrder::Fortran => unsafe {
                        let offset_ptr =
                            (ptr as *mut N::Native).add(col_idx * height + chunk_offset);
                        // SAFETY:
                        // this is uninitialized memory, so we must never read from this data
                        // copy_from_slice does not read
                        let buf = std::slice::from_raw_parts_mut(offset_ptr, vals.len());
                        buf.copy_from_slice(vals)
                    },
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

// &[Column] -> Slice
pub fn columns_to_vec<N>(
    columns: Vec<Column>,
    ordering: IndexOrder,
) -> PolarsResult<Vec<<N>::Native>>
where
    N: PolarsNumericType,
{
    let v = columns
        .into_iter()
        .map(|s| s.as_materialized_series().clone())
        .collect::<Vec<_>>();

    series_to_slice::<N>(&v, ordering)
}

#[inline(always)]
pub fn to_f64_vec_without_nulls(inputs: &[Series], ordering: IndexOrder) -> PolarsResult<Vec<f64>> {
    let df = DataFrame::from_iter(inputs.iter().map(|s| Column::Series(s.clone().into())));
    let df = df.drop_nulls::<String>(None)?;

    columns_to_vec::<Float64Type>(df.take_columns(), ordering)
}

#[inline(always)]
pub fn to_f64_vec_fail_on_nulls(inputs: &[Series], ordering: IndexOrder) -> PolarsResult<Vec<f64>> {
    if inputs.iter().any(|s| s.has_nulls()) {
        Err(PolarsError::ComputeError(
            "Nulls are found in data and this method doesn't allow nulls.".into(),
        ))
    } else {
        let columns = inputs
            .iter()
            .map(|s| s.clone().into_column())
            .collect::<Vec<_>>();
        columns_to_vec::<Float64Type>(columns, ordering)
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

pub fn list_u32_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "nodes".into(),
        DataType::List(Box::new(DataType::UInt32)),
    ))
}

pub fn complex_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "complex".into(),
        DataType::Array(Box::new(DataType::Float64), 2),
    ))
}

pub fn float_output(fields: &[Field]) -> PolarsResult<Field> {
    FieldsMapper::new(fields).map_to_float_dtype()
}


// --- Distances and Distance Related Abstractions ---

pub fn haversine_elementwise<T: Float>(start_lat: T, start_long: T, end_lat: T, end_long: T) -> T {
    let r_in_km = T::from(6371.0).unwrap();
    let two = T::from(2.0).unwrap();
    let one = T::one();

    let d_lat = (end_lat - start_lat).to_radians();
    let d_lon = (end_long - start_long).to_radians();
    let lat1 = (start_lat).to_radians();
    let lat2 = (end_lat).to_radians();

    let a = ((d_lat / two).sin()) * ((d_lat / two).sin())
        + ((d_lon / two).sin()) * ((d_lon / two).sin()) * (lat1.cos()) * (lat2.cos());
    let c = two * ((a.sqrt()).atan2((one - a).sqrt()));
    r_in_km * c
}

pub fn haversine<T: Float + 'static>(first: &[T], second: &[T]) -> T {
    haversine_elementwise(first[0], first[1], second[0], second[1])
}


pub fn squared_l2_distance(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len(), "Slices must have the same length");

    let mut sum0 = 0.0;
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum3 = 0.0;

    let mut a_chunks = a.chunks_exact(4);
    let mut b_chunks = b.chunks_exact(4);

    for (chunk_a, chunk_b) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
        let diff0 = chunk_a[0] - chunk_b[0];
        let diff1 = chunk_a[1] - chunk_b[1];
        let diff2 = chunk_a[2] - chunk_b[2];
        let diff3 = chunk_a[3] - chunk_b[3];

        sum0 += diff0 * diff0;
        sum1 += diff1 * diff1;
        sum2 += diff2 * diff2;
        sum3 += diff3 * diff3;
    }

    let mut total_sq_dist = sum0 + sum1 + sum2 + sum3;

    // Leftover
    for (&x, &y) in a_chunks.remainder().iter().zip(b_chunks.remainder()) {
        let diff = x - y;
        total_sq_dist += diff * diff;
    }

    total_sq_dist
}

pub fn l1_distance(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len(), "Slices must have the same length");

    let mut sum0 = 0.0;
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum3 = 0.0;

    let mut a_chunks = a.chunks_exact(4);
    let mut b_chunks = b.chunks_exact(4);

    for (chunk_a, chunk_b) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
        sum0 += (chunk_a[0] - chunk_b[0]).abs();
        sum1 += (chunk_a[1] - chunk_b[1]).abs();
        sum2 += (chunk_a[2] - chunk_b[2]).abs();
        sum3 += (chunk_a[3] - chunk_b[3]).abs();
    }

    let mut total_dist = sum0 + sum1 + sum2 + sum3;

    // Handle any leftover
    for (&x, &y) in a_chunks.remainder().iter().zip(b_chunks.remainder()) {
        total_dist += (x - y).abs();
    }

    total_dist
}

pub fn linf_distance(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len(), "Slices must have the same length");

    let mut max0 = 0.0f64;
    let mut max1 = 0.0f64;
    let mut max2 = 0.0f64;
    let mut max3 = 0.0f64;

    let mut a_chunks = a.chunks_exact(4);
    let mut b_chunks = b.chunks_exact(4);

    for (chunk_a, chunk_b) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
        max0 = max0.max((chunk_a[0] - chunk_b[0]).abs());
        max1 = max1.max((chunk_a[1] - chunk_b[1]).abs());
        max2 = max2.max((chunk_a[2] - chunk_b[2]).abs());
        max3 = max3.max((chunk_a[3] - chunk_b[3]).abs());
    }

    let mut total_max = max0.max(max1).max(max2).max(max3);

    // leftover
    for (&x, &y) in a_chunks.remainder().iter().zip(b_chunks.remainder()) {
        total_max = total_max.max((x - y).abs());
    }

    total_max
}

pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len(), "Slices must have the same length");

    let mut sum0 = 0.0;
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum3 = 0.0;

    let mut a_chunks = a.chunks_exact(4);
    let mut b_chunks = b.chunks_exact(4);

    for (chunk_a, chunk_b) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
        sum0 += chunk_a[0] * chunk_b[0];
        sum1 += chunk_a[1] * chunk_b[1];
        sum2 += chunk_a[2] * chunk_b[2];
        sum3 += chunk_a[3] * chunk_b[3];
    }

    let mut total = sum0 + sum1 + sum2 + sum3;

    // leftover
    for (&x, &y) in a_chunks.remainder().iter().zip(b_chunks.remainder()) {
        total += x * y;
    }

    total
}


