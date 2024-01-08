use itertools::Itertools;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

#[polars_expr(output_type=Float64)]
fn pl_psi(inputs: &[Series]) -> PolarsResult<Series> {
    let data = inputs[0].f64()?;
    // breaks according to reference, already sorted
    let brk = inputs[1].f64()?;
    // cnts for each brk in reference
    let cnt_ref = inputs[2].u32()?;
    // Vec to do binary search with.
    let brk_vec: Vec<f64> = brk.into_no_null_iter().collect_vec();
    // Create the correct cnt (of values that are inside the bins defined by ref) for data
    let mut cnt_data = vec![0_u32; brk_vec.len()];
    for d in data.into_no_null_iter() {
        let res = brk_vec.binary_search_by(|x| x.partial_cmp(&d).unwrap());
        let idx = match res {
            Ok(i) => i,
            Err(j) => j,
        };
        cnt_data[idx] += 1;
    }
    // Total cnt in ref
    let ref_total = cnt_ref.sum().unwrap() as f64;
    // Total cnt in data
    let data_total = cnt_data.iter().sum::<u32>() as f64;
    // PSI
    let psi = cnt_ref
        .into_no_null_iter()
        .zip(cnt_data.into_iter())
        .fold(0., |acc, (a, b)| {
            let aa = (a as f64).clamp(0.0001, f64::MAX) / ref_total;
            let bb = (b as f64).clamp(0.0001, f64::MAX) / data_total;
            acc + (aa - bb) * (aa / bb).ln()
        });
    let out = Float64Chunked::from_iter([Some(psi)]);
    Ok(out.into_series())
}
