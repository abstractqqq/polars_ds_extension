use crate::num::knn::{build_standard_kdtree, query_nb_cnt, KdtreeKwargs};
use ndarray::s;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

// https://en.wikipedia.org/wiki/Sample_entropy
// https://en.wikipedia.org/wiki/Approximate_entropy

// Could be made faster once https://github.com/mrhooray/kdtree-rs/pull/52 is merged

#[polars_expr(output_type=Float64)]
fn pl_approximate_entropy(inputs: &[Series], kwargs: KdtreeKwargs) -> PolarsResult<Series> {
    // inputs[0] is radius, the rest are the shifted columns
    // Set up radius. r is a scalar and set up at Python side.
    let radius = inputs[0].f64()?;
    let name = inputs[1].name();
    if radius.get(0).is_none() {
        return Ok(Series::from_vec(name, vec![f64::NAN]));
    }
    let r = radius.get(0).unwrap();
    // Set up params
    let dim = inputs[1..].len();
    let mut vs: Vec<Series> = Vec::with_capacity(dim);
    for (i, s) in inputs[1..].into_iter().enumerate() {
        let news = s.rechunk().with_name(&i.to_string());
        vs.push(news)
    }
    let data = DataFrame::new(vs)?;
    let n1 = data.height(); // This is equal to original length - m + 1
    let data = data.to_ndarray::<Float64Type>(IndexOrder::C)?;
    // Here, dim equals to run_length + 1, or m + 1
    // + 1 because I am intentionally generating one more, so that we do to_ndarray only once.
    if (n1 < dim) || (r <= 0.) || (!r.is_finite()) {
        return Ok(Series::from_vec(name, vec![f64::NAN]));
    }
    let parallel = kwargs.parallel;
    let leaf_size = kwargs.leaf_size;

    // Step 3, 4, 5 in wiki
    let data_1_view = data.slice(s![..n1, ..dim.abs_diff(1)]);
    let tree = build_standard_kdtree(dim.abs_diff(1), leaf_size, &data_1_view)?;
    let nb_in_radius = query_nb_cnt(&tree, data_1_view, &super::l_inf_dist, r, parallel);
    let phi_m: f64 = nb_in_radius
        .into_no_null_iter()
        .fold(0_f64, |acc, x| acc + (x as f64 / n1 as f64).ln())
        / n1 as f64;

    // Step 3, 4, 5 for m + 1 in wiki
    let n2 = n1.abs_diff(1);
    let data_2_view = data.slice(s![..n2, ..]);
    let tree = build_standard_kdtree(dim, leaf_size, &data_2_view)?;
    let nb_in_radius = query_nb_cnt(&tree, data_2_view, &super::l_inf_dist, r, parallel);
    let phi_m1: f64 = nb_in_radius
        .into_no_null_iter()
        .fold(0_f64, |acc, x| acc + (x as f64 / n2 as f64).ln())
        / n2 as f64;

    // Output
    Ok(Series::from_vec(name, vec![(phi_m1 - phi_m).abs()]))
}

#[polars_expr(output_type=Float64)]
fn pl_sample_entropy(inputs: &[Series], kwargs: KdtreeKwargs) -> PolarsResult<Series> {
    // inputs[0] is radius, the rest are the shifted columns
    // Set up radius. r is a scalar and set up at Python side.
    let radius = inputs[0].f64()?;
    let name = inputs[1].name();
    if radius.get(0).is_none() {
        return Ok(Series::from_vec(name, vec![f64::NAN]));
    }
    let r = radius.get(0).unwrap();
    // Set up params
    let dim = inputs[1..].len();
    let mut vs: Vec<Series> = Vec::with_capacity(dim);
    for (i, s) in inputs[1..].into_iter().enumerate() {
        let news = s.rechunk().with_name(&i.to_string());
        vs.push(news)
    }
    let data = DataFrame::new(vs)?;
    let n1 = data.height(); // This is equal to original length - m + 1
    let data = data.to_ndarray::<Float64Type>(IndexOrder::C)?;
    // Here, dim equals to run_length + 1, or m + 1
    // + 1 because I am intentionally generating one more, so that we do to_ndarray only once.
    if (n1 < dim) || (r <= 0.) || (!r.is_finite()) {
        return Ok(Series::from_vec(name, vec![f64::NAN]));
    }
    let parallel = kwargs.parallel;
    let leaf_size = kwargs.leaf_size;

    let data_1_view = data.slice(s![..n1, ..dim.abs_diff(1)]);
    let tree = build_standard_kdtree(dim.abs_diff(1), leaf_size, &data_1_view)?;
    let nb_in_radius = query_nb_cnt(&tree, data_1_view, &super::l_inf_dist, r, parallel);
    let b = (nb_in_radius.sum().unwrap_or(0) as f64) - (n1 as f64);

    let n2 = n1.abs_diff(1);
    let data_2_view = data.slice(s![..n2, ..]);
    let tree = build_standard_kdtree(dim, leaf_size, &data_2_view)?;
    let nb_in_radius = query_nb_cnt(&tree, data_2_view, &super::l_inf_dist, r, parallel);
    let a = (nb_in_radius.sum().unwrap_or(0) as f64) - (n2 as f64);

    // Output
    Ok(Series::from_vec(name, vec![(b / a).ln()]))
}
