use polars::prelude::*;
use polars_core::utils::rayon::iter::{ParallelBridge, ParallelIterator};
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

#[derive(Deserialize)]
pub(crate) struct KthNBKwargs {
    pub(crate) k: usize,
    parallel: bool,
}

fn dist_from_kth_nb(data: &[f64], x:f64, k:usize) -> f64 {
    // Distance from the kth Neighbor
    // Not the most efficient
    if k >= data.len() {
        f64::NAN
    } else {
        let index = match data.binary_search_by(|y| y.partial_cmp(&x).unwrap()) {
            Ok(i) => i,
            Err(j) => j
        };
        let min_i = index.saturating_sub(k);
        let max_i = (index + k + 1).min(data.len());
        let mut rank = (min_i..max_i).map(|i| (x - data[i]).abs()).collect::<Vec<_>>();
        rank.sort_unstable_by(|x, y| x.partial_cmp(y).unwrap());
        rank[k]
        // match rank.get(k) {
        //     Some(x) => *x,
        //     None => {
        //         println!("Rank {:?}", rank);
        //         println!("Min I {:?}", min_i);
        //         println!("Max I {:?}", max_i);
        //         f64::NAN
        //     },
        // }
    }
    // *(rank.get(k).unwrap_or(&f64::NAN))
}

#[polars_expr(output_type=Float64)]
fn _pl_dist_from_kth_nb_with_next_down(inputs: &[Series], kwargs: KthNBKwargs) -> PolarsResult<Series> {
    // k-th nearest neighbor (1d) is a quantity needed during the computation of Mutual Info Score
    // X: NaN filled with Null in Python
    // This is a special helper function only used in mutual_info_score
    let x = inputs[0].f64()?;
    let data = x.drop_nulls().sort(false);
    let data = data.cont_slice()?;
    let k = kwargs.k;

    let output = if kwargs.parallel {
        x
            .into_iter()
            .par_bridge()
            .map(|op_y| 
                op_y.map(|y| dist_from_kth_nb(data, y, k).next_down())
            ).collect::<Float64Chunked>()
    } else {
        x.apply_values(|y| dist_from_kth_nb(data, y, k).next_down())
    };

    Ok(output.into_series())
}