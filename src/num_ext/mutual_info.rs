use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

#[derive(Deserialize)]
pub(crate) struct KthNBKwargs {
    pub(crate) k: usize,
}

fn dist_from_kth_nb(data: &[f64], x:f64, k:usize) -> f64 {
    // Distance from the kth Neighbor
    // Not the most efficient
    let index = match data.binary_search_by(|y| y.partial_cmp(&x).unwrap()) {
        Ok(i) => i,
        Err(j) => j
    };
    let min_i = index.saturating_sub(k);
    let max_i = (index + k).min(data.len());
    let mut rank = (min_i..max_i).map(|i| (x - data[i]).abs()).collect::<Vec<_>>();
    rank.sort_unstable_by(|x, y| x.partial_cmp(y).unwrap());
    *(rank.get(k).unwrap_or(&f64::NAN))
}

#[polars_expr(output_type=Float64)]
fn pl_dist_from_kth_nb(inputs: &[Series], kwargs: KthNBKwargs) -> PolarsResult<Series> {
    // k-th nearest neighbor (1d) is a needed quantity during the computation of Mutual Info Score
    // X: NaN filled with Null.
    let x = inputs[0].f64()?;
    let data = x.drop_nulls().sort(false);
    let data = data.cont_slice()?;
    let k = kwargs.k;
    let output = x.apply_values(|y| dist_from_kth_nb(data, y, k));
    Ok(output.into_series())
}