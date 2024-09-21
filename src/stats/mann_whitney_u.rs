/// Mann-Whitney U Statistics
use super::{simple_stats_output, Alternative};
use crate::stats_utils::normal;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

fn mann_whitney_tie_sum(ranks: &Float64Chunked) -> f64 {
    // NaN won't exist in ranks.
    let mut rank_number = f64::NAN;
    let mut rank_cnt: f64 = 0f64;
    let mut accumulant = 0f64;
    for v in ranks.into_no_null_iter() {
        if v == rank_number {
            rank_cnt += 1.;
        } else {
            accumulant += rank_cnt * (rank_cnt + 1.0) * (rank_cnt - 1.0);
            rank_number = v;
            rank_cnt = 1.0;
        }
    }
    accumulant
}

#[polars_expr(output_type_func=simple_stats_output)]
fn pl_mann_whitney_u(inputs: &[Series]) -> PolarsResult<Series> {
    // Reference: https://github.com/scipy/scipy/blob/v1.13.1/scipy/stats/_mannwhitneyu.py#L177

    let u1 = inputs[0].f64().unwrap();
    let u1 = u1.get(0).unwrap();

    let u2 = inputs[1].f64().unwrap();
    let u2 = u2.get(0).unwrap();

    let mean = inputs[2].f64().unwrap();
    let mean = mean.get(0).unwrap();

    // Custom RLE
    let sorted_ranks = inputs[3].f64().unwrap();
    let n = sorted_ranks.len() as f64;
    let tie_term_sum = mann_whitney_tie_sum(sorted_ranks);
    let std_ = ((mean / 6.0) * ((n + 1.0) - tie_term_sum / (n * (n - 1.0)))).sqrt();

    let alt = inputs[4].str()?;
    let alt = alt.get(0).unwrap();
    let alt = Alternative::from(alt);

    let (u, factor) = match alt {
        // if I use min here, always wrong p value. But wikipedia says it is min. I wonder wtf..
        Alternative::TwoSided => (u1.max(u2), 2.0),
        Alternative::Less => (u2, 1.0),
        Alternative::Greater => (u1, 1.0),
    };

    let p = if std_ == 0. {
        0.
    } else {
        // -0.5 is some continuity adjustment. See Scipy's impl
        (factor * normal::sf_unchecked(u, mean + 0.5, std_)).clamp(0., 1.)
    };

    let s = Series::from_vec("statistic", vec![u1]);
    let p = Series::from_vec("pvalue", vec![p]);
    let out = StructChunked::new("", &[s, p])?;
    Ok(out.into_series())
}
