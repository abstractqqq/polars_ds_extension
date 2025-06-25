/// Jaccard similarity for two columns
/// and Jaccard similarity for two columns of lists with compatible inner types.
use ordered_float::OrderedFloat;
use polars::prelude::*;
use pyo3_polars::{
    derive::polars_expr,
};
use DataType::*;


#[polars_expr(output_type=Float64)]
fn pl_jaccard(inputs: &[Series]) -> PolarsResult<Series> {
    let count_null = inputs[2].bool()?;
    let count_null = count_null.get(0).unwrap_or(false);

    let (s1, s2) = (inputs[0].clone(), inputs[1].clone());

    let adj = if count_null {
        (s1.has_nulls() && s2.has_nulls()) as usize // adjust for null
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
            Int128 => {
                let ca1 = s1.i128().unwrap();
                let ca2 = s2.i128().unwrap();
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
