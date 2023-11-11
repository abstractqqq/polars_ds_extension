use hashbrown::HashSet;
use pyo3_polars::derive::polars_expr;
use polars::prelude::*;


#[polars_expr(output_type=Float64)]
fn pl_jaccard(inputs: &[Series]) -> PolarsResult<Series> {

    let include_null = inputs[2].bool()?;
    let include_null = include_null.get(0).unwrap();
    
    let (s1, s2) = if include_null {
        (inputs[0].clone(), inputs[1].clone())
    } else {
        let t1 = inputs[0].clone();
        let t2 = inputs[1].clone();
        (t1.drop_nulls(), t2.drop_nulls())
    };

    // let parallel = inputs[3].bool()?;
    // let parallel = parallel.get(0).unwrap();

    if s1.dtype() != s2.dtype() {
        return Err(PolarsError::ComputeError(
            "Input column must have the same type.".into(),
        ))
    }

    let (n1, n2, intersection) = 
    if s1.dtype().is_integer() {
        let ca1 = s1.cast(&DataType::Int64)?;
        let ca2 = s2.cast(&DataType::Int64)?;
        let ca1 = ca1.i64()?;
        let ca2 = ca2.i64()?;

        let hs1: HashSet<Option<i64>> = HashSet::from_iter(ca1);
        let hs2: HashSet<Option<i64>> = HashSet::from_iter(ca2);
        let n1 = hs1.len();
        let n2 = hs2.len();

        let intersection = hs1.intersection(&hs2);

        (n1, n2, intersection.count())

    } else if s1.dtype() == &DataType::Utf8 {
        let ca1 = s1.utf8()?;
        let ca2 = s2.utf8()?;

        let hs1: HashSet<Option<&str>> = HashSet::from_iter(ca1);
        let hs2: HashSet<Option<&str>> = HashSet::from_iter(ca2);
        let n1 = hs1.len();
        let n2 = hs2.len();

        let intersection = hs1.intersection(&hs2);

        (n1, n2, intersection.count())
        
    } else {
        return Err(PolarsError::ComputeError(
            "Jaccard similarity can only be computed for integer/str columns.".into(),
        ))
    };

    let out: Series = Series::from_iter([
        intersection as f64 / (n1 + n2 - intersection) as f64
    ]);

    Ok(out)
    
}