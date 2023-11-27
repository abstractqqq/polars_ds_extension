/// Jaccard similarity for two columns
/// + Jaccard similarity for two columns of lists
///
use core::hash::Hash;
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
fn jaccard_str(c1: &Utf8Chunked, c2: &Utf8Chunked) -> f64 {
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
    // let include_null = inputs[2].bool()?;
    // let include_null = include_null.get(0).unwrap();

    let s1 = inputs[0].list()?;
    let s2 = inputs[1].list()?;

    polars_ensure!(
        s1.inner_dtype() == s2.inner_dtype(),
        ComputeError: "Inner data types don't match."
    );

    if s1.inner_dtype().is_integer() {
        let out = _list_jaccard(s1, s2);
        Ok(out.into_series())
    } else if s1.inner_dtype() == DataType::Utf8 {
        // Not sure how to get binary elementwise to work here.
        Ok(s1
            .into_iter()
            .zip(s2.into_iter())
            .map(|(c1, c2)| match (c1, c2) {
                (Some(c1), Some(c2)) => {
                    let c1 = c1.utf8().unwrap();
                    let c2 = c2.utf8().unwrap();
                    Some(jaccard_str(c1, c2))
                }
                _ => None,
            })
            .collect())
    } else {
        Err(PolarsError::ComputeError(
            "List Jaccard similarity currently only supports utf8 or Int inner data type.".into(),
        ))
    }
}

#[polars_expr(output_type=Float64)]
fn pl_jaccard(inputs: &[Series]) -> PolarsResult<Series> {
    let include_null = inputs[2].bool()?;
    let include_null = include_null.get(0).unwrap();

    let (s1, s2) = if include_null {
        (inputs[0].clone(), inputs[1].clone())
    } else {
        let t1 = &inputs[0];
        let t2 = &inputs[1];
        (t1.drop_nulls(), t2.drop_nulls())
    };

    if s1.dtype() == s2.dtype() {
        // God, help me with this unholy mess,
        // All for the sake of not casting...
        // I wasn't able to do it using the same method above.
        // Nor was I able to figure out some macro to help me do this...
        // It is fine like this, because this is what the compiler would do anyways,
        // But it is just plain ugly...
        match s1.dtype() {
            Int8 => {
                let ca1 = s1.i8()?;
                let ca2 = s2.i8()?;
                let hs1 = ca1.into_iter().collect::<PlHashSet<_>>();
                let hs2 = ca2.into_iter().collect::<PlHashSet<_>>();
                let s3_len = hs1.intersection(&hs2).count();
                let out = s3_len as f64 / (hs1.len() + hs2.len() - s3_len) as f64;
                let out = Float64Chunked::from_iter([Some(out)]);
                Ok(out.into_series())
            }
            Int16 => {
                let ca1 = s1.i16()?;
                let ca2 = s2.i16()?;
                let hs1 = ca1.into_iter().collect::<PlHashSet<_>>();
                let hs2 = ca2.into_iter().collect::<PlHashSet<_>>();
                let s3_len = hs1.intersection(&hs2).count();
                let out = s3_len as f64 / (hs1.len() + hs2.len() - s3_len) as f64;
                let out = Float64Chunked::from_iter([Some(out)]);
                Ok(out.into_series())
            }
            Int32 => {
                let ca1 = s1.i32()?;
                let ca2 = s2.i32()?;
                let hs1 = ca1.into_iter().collect::<PlHashSet<_>>();
                let hs2 = ca2.into_iter().collect::<PlHashSet<_>>();
                let s3_len = hs1.intersection(&hs2).count();
                let out = s3_len as f64 / (hs1.len() + hs2.len() - s3_len) as f64;
                let out = Float64Chunked::from_iter([Some(out)]);
                Ok(out.into_series())
            }
            Int64 => {
                let ca1 = s1.i64()?;
                let ca2 = s2.i64()?;
                let hs1 = ca1.into_iter().collect::<PlHashSet<_>>();
                let hs2 = ca2.into_iter().collect::<PlHashSet<_>>();
                let s3_len = hs1.intersection(&hs2).count();
                let out = s3_len as f64 / (hs1.len() + hs2.len() - s3_len) as f64;
                let out = Float64Chunked::from_iter([Some(out)]);
                Ok(out.into_series())
            }
            UInt8 => {
                let ca1 = s1.u8()?;
                let ca2 = s2.u8()?;
                let hs1 = ca1.into_iter().collect::<PlHashSet<_>>();
                let hs2 = ca2.into_iter().collect::<PlHashSet<_>>();
                let s3_len = hs1.intersection(&hs2).count();
                let out = s3_len as f64 / (hs1.len() + hs2.len() - s3_len) as f64;
                let out = Float64Chunked::from_iter([Some(out)]);
                Ok(out.into_series())
            }
            UInt16 => {
                let ca1 = s1.u16()?;
                let ca2 = s2.u16()?;
                let hs1 = ca1.into_iter().collect::<PlHashSet<_>>();
                let hs2 = ca2.into_iter().collect::<PlHashSet<_>>();
                let s3_len = hs1.intersection(&hs2).count();
                let out = s3_len as f64 / (hs1.len() + hs2.len() - s3_len) as f64;
                let out = Float64Chunked::from_iter([Some(out)]);
                Ok(out.into_series())
            }
            UInt32 => {
                let ca1 = s1.u32()?;
                let ca2 = s2.u32()?;
                let hs1 = ca1.into_iter().collect::<PlHashSet<_>>();
                let hs2 = ca2.into_iter().collect::<PlHashSet<_>>();
                let s3_len = hs1.intersection(&hs2).count();
                let out = s3_len as f64 / (hs1.len() + hs2.len() - s3_len) as f64;
                let out = Float64Chunked::from_iter([Some(out)]);
                Ok(out.into_series())
            }
            UInt64 => {
                let ca1 = s1.u64()?;
                let ca2 = s2.u64()?;
                let hs1 = ca1.into_iter().collect::<PlHashSet<_>>();
                let hs2 = ca2.into_iter().collect::<PlHashSet<_>>();
                let s3_len = hs1.intersection(&hs2).count();
                let out = s3_len as f64 / (hs1.len() + hs2.len() - s3_len) as f64;
                let out = Float64Chunked::from_iter([Some(out)]);
                Ok(out.into_series())
            }
            Utf8 => {
                let ca1 = s1.utf8()?;
                let ca2 = s2.utf8()?;
                let out = jaccard_str(ca1, ca2);
                let out = Float64Chunked::from_iter([Some(out)]);
                Ok(out.into_series())
            }
            _ => Err(PolarsError::ComputeError(
                "Jaccard similarity currently does not support the input data type.".into(),
            )),
        }
    } else {
        Err(PolarsError::ComputeError(
            "Input column must have the same type.".into(),
        ))
    }
}
