/// Jaccard similarity for two columns
/// and Jaccard similarity for two columns of lists with compatible inner types.
use core::hash::Hash;
use ordered_float::OrderedFloat;
use polars::prelude::*;
use pyo3_polars::{
    derive::polars_expr,
    export::polars_core::{
        utils::arrow::{array::PrimitiveArray, types::NativeType},
        with_match_physical_integer_type,
    },
};
use DataType::*;

#[inline]
fn jaccard_str(c1: &StringChunked, c2: &StringChunked) -> f64 {
    // Jaccard similarity for strings
    let hs1 = c1.into_iter().collect::<PlHashSet<_>>();
    let hs2 = c2.into_iter().collect::<PlHashSet<_>>();
    let s3_len = hs1.intersection(&hs2).count();
    s3_len as f64 / (hs1.len() + hs2.len() - s3_len) as f64
    // Float64Chunked::from_iter([Some(out)])
}

#[inline]
fn jaccard_int<T: NativeType + Hash + Eq>(a: &PrimitiveArray<T>, b: &PrimitiveArray<T>) -> f64 {
    // jaccard similarity for all int types
    // convert to hashsets over Option<T>
    let s1 = a.into_iter().collect::<PlHashSet<_>>();
    let s2 = b.into_iter().collect::<PlHashSet<_>>();

    // count the number of intersections
    let s3_len = s1.intersection(&s2).count();
    // return similarity
    s3_len as f64 / (s1.len() + s2.len() - s3_len) as f64
}

#[inline]
fn _list_jaccard(a: &ListChunked, b: &ListChunked) -> Float64Chunked {
    // Using this avoids casting
    // This is copied from
    // https://github.com/pola-rs/pyo3-polars/blob/main/example/derive_expression/expression_lib/src/distances.rs)
    with_match_physical_integer_type!(a.inner_dtype(), |$T| {
        polars::prelude::arity::binary_elementwise(a, b, |a, b| {
            match (a, b) {
                (Some(a), Some(b)) => {
                    let a = a.as_any().downcast_ref::<PrimitiveArray<$T>>().unwrap();
                    let b = b.as_any().downcast_ref::<PrimitiveArray<$T>>().unwrap();
                    Some(jaccard_int(a, b))
                },
                _ => None
            }
        })
    })
}

#[polars_expr(output_type=Float64)]
fn pl_list_jaccard(inputs: &[Series]) -> PolarsResult<Series> {
    let s1 = inputs[0].list()?;
    let s2 = inputs[1].list()?;

    polars_ensure!(
        s1.inner_dtype() == s2.inner_dtype(),
        ComputeError: "Inner data types don't match."
    );

    if s1.inner_dtype().is_integer() {
        let out = _list_jaccard(s1, s2);
        Ok(out.into_series())
    } else if s1.inner_dtype() == DataType::String {
        Ok(s1
            .into_iter()
            .zip(s2.into_iter())
            .map(|(c1, c2)| match (c1, c2) {
                (Some(c1), Some(c2)) => {
                    let c1 = c1.str().unwrap();
                    let c2 = c2.str().unwrap();
                    Some(jaccard_str(c1, c2))
                }
                _ => None,
            })
            .collect())
    } else {
        Err(PolarsError::ComputeError(
            "List Jaccard similarity currently only supports Str or Int inner data type.".into(),
        ))
    }
}

#[polars_expr(output_type=Float64)]
fn pl_jaccard(inputs: &[Series]) -> PolarsResult<Series> {
    let count_null = inputs[2].bool()?;
    let count_null = count_null.get(0).unwrap_or(false);

    let (s1, s2) = (inputs[0].clone(), inputs[1].clone());

    let adj = if count_null {
        (s1.null_count() > 0 && s2.null_count() > 0) as usize // adjust for null
    } else {
        0
    };

    // All computation below assumes the input has no nulls, that is why we have the adj

    // God, help me with this unholy mess,
    let (len1, len2, intersection_size) = if s1.dtype().is_integer() && s1.dtype() != s2.dtype() {
        let ca1 = s1.cast(&DataType::Int64)?;
        let ca1 = ca1.i64().unwrap();
        let ca2 = s2.cast(&DataType::Int64)?;
        let ca2 = ca2.i64().unwrap();
        let hs1 = ca1.into_no_null_iter().collect::<PlHashSet<_>>();
        let hs2 = ca2.into_no_null_iter().collect::<PlHashSet<_>>();
        (hs1.len(), hs2.len(), hs1.intersection(&hs2).count())
    } else if s1.dtype() == s2.dtype() {
        match s1.dtype() {
            Int8 => {
                let ca1 = s1.i8().unwrap();
                let ca2 = s2.i8().unwrap();
                let hs1 = ca1.into_no_null_iter().collect::<PlHashSet<_>>();
                let hs2 = ca2.into_no_null_iter().collect::<PlHashSet<_>>();
                (hs1.len(), hs2.len(), hs1.intersection(&hs2).count())
            }
            Int16 => {
                let ca1 = s1.i16().unwrap();
                let ca2 = s2.i16().unwrap();
                let hs1 = ca1.into_no_null_iter().collect::<PlHashSet<_>>();
                let hs2 = ca2.into_no_null_iter().collect::<PlHashSet<_>>();
                (hs1.len(), hs2.len(), hs1.intersection(&hs2).count())
            }
            Int32 => {
                let ca1 = s1.i32().unwrap();
                let ca2 = s2.i32().unwrap();
                let hs1 = ca1.into_no_null_iter().collect::<PlHashSet<_>>();
                let hs2 = ca2.into_no_null_iter().collect::<PlHashSet<_>>();
                (hs1.len(), hs2.len(), hs1.intersection(&hs2).count())
            }
            Int64 => {
                let ca1 = s1.i64().unwrap();
                let ca2 = s2.i64().unwrap();
                let hs1 = ca1.into_no_null_iter().collect::<PlHashSet<_>>();
                let hs2 = ca2.into_no_null_iter().collect::<PlHashSet<_>>();
                (hs1.len(), hs2.len(), hs1.intersection(&hs2).count())
            }
            UInt8 => {
                let ca1 = s1.u8().unwrap();
                let ca2 = s2.u8().unwrap();
                let hs1 = ca1.into_no_null_iter().collect::<PlHashSet<_>>();
                let hs2 = ca2.into_no_null_iter().collect::<PlHashSet<_>>();
                (hs1.len(), hs2.len(), hs1.intersection(&hs2).count())
            }
            UInt16 => {
                let ca1 = s1.u16().unwrap();
                let ca2 = s2.u16().unwrap();
                let hs1 = ca1.into_no_null_iter().collect::<PlHashSet<_>>();
                let hs2 = ca2.into_no_null_iter().collect::<PlHashSet<_>>();
                (hs1.len(), hs2.len(), hs1.intersection(&hs2).count())
            }
            UInt32 => {
                let ca1 = s1.u32().unwrap();
                let ca2 = s2.u32().unwrap();
                let hs1 = ca1.into_no_null_iter().collect::<PlHashSet<_>>();
                let hs2 = ca2.into_no_null_iter().collect::<PlHashSet<_>>();
                (hs1.len(), hs2.len(), hs1.intersection(&hs2).count())
            }
            UInt64 => {
                let ca1 = s1.u64().unwrap();
                let ca2 = s2.u64().unwrap();
                let hs1 = ca1.into_no_null_iter().collect::<PlHashSet<_>>();
                let hs2 = ca2.into_no_null_iter().collect::<PlHashSet<_>>();
                (hs1.len(), hs2.len(), hs1.intersection(&hs2).count())
            }
            String => {
                let ca1 = s1.str().unwrap();
                let ca2 = s2.str().unwrap();
                let hs1 = ca1.into_no_null_iter().collect::<PlHashSet<_>>();
                let hs2 = ca2.into_no_null_iter().collect::<PlHashSet<_>>();
                (hs1.len(), hs2.len(), hs1.intersection(&hs2).count())
            }
            Float32 => {
                let ca1 = s1.f32().unwrap();
                let ca2 = s2.f32().unwrap();
                let hs1 = ca1
                    .into_no_null_iter()
                    .map(|x| OrderedFloat::from(x))
                    .collect::<PlHashSet<_>>();
                let hs2 = ca2
                    .into_no_null_iter()
                    .map(|x| OrderedFloat::from(x))
                    .collect::<PlHashSet<_>>();
                (hs1.len(), hs2.len(), hs1.intersection(&hs2).count())
            }
            Float64 => {
                let ca1 = s1.f64().unwrap();
                let ca2 = s2.f64().unwrap();
                let hs1 = ca1
                    .into_no_null_iter()
                    .map(|x| OrderedFloat::from(x))
                    .collect::<PlHashSet<_>>();
                let hs2 = ca2
                    .into_no_null_iter()
                    .map(|x| OrderedFloat::from(x))
                    .collect::<PlHashSet<_>>();
                (hs1.len(), hs2.len(), hs1.intersection(&hs2).count())
            }
            _ => {
                return Err(PolarsError::ComputeError(
                    "Jaccard similarity currently does not support the input data type.".into(),
                ))
            }
        }
    } else {
        return Err(PolarsError::ComputeError(
            "Inputs do not have compatible datatype.".into(),
        ));
    };

    let adjusted = intersection_size + adj;
    let out = adjusted as f64 / (len1 + len2).abs_diff(adjusted) as f64;
    Ok(Series::from_iter([out]))
}
