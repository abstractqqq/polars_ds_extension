use hashbrown::HashSet;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

#[polars_expr(output_type=UInt32)]
fn pl_lempel_ziv_complexity(inputs: &[Series]) -> PolarsResult<Series> {
    let bools = inputs[0].bool()?;
    let bits: Vec<bool> = bools
        .into_iter()
        .map(|op_b| op_b.unwrap_or_default())
        .collect();

    let mut ind: usize = 0;
    let mut inc: usize = 1;
    let mut sub_strings: HashSet<&[bool]> = HashSet::new();
    while ind + inc <= bits.len() {
        let subseq: &[bool] = &bits[ind..ind + inc];
        if sub_strings.contains(subseq) {
            inc += 1;
        } else {
            sub_strings.insert(subseq);
            ind += inc;
            inc = 1;
        }
    }
    let c = sub_strings.len();
    Ok(Series::from_iter([c as u32]))
}
