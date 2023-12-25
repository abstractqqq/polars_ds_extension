use crate::{
    no_null_in_inputs,
    num_ext::{
        knn::{build_standard_kdtree, query_nb_cnt, KdtreeKwargs},
        which_distance,
    },
};
use ndarray::s;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

#[polars_expr(output_type=Float64)]
fn pl_approximate_entropy(inputs: &[Series], kwargs: KdtreeKwargs) -> PolarsResult<Series> {
    // inputs[0] is radius, the rest are the shifted columns
    let n_inputs = inputs.len();
    let last_idx = n_inputs.abs_diff(1);

    // Set up radius. r is a scalar and set up at Python side.
    let radius = inputs[0].f64()?;
    if radius.get(0).is_none() {
        return Ok(Series::from_vec("", vec![f64::NAN]));
    }
    let r = radius.get(0).unwrap();
    // Set up params
    let data = DataFrame::new(inputs[1..].to_vec())?.agg_chunks();
    let n1 = data.height(); // This is equal to original length - m + 1
    let data = data.to_ndarray::<Float64Type>(IndexOrder::C)?;
    // Here, dim equals to run_length + 1, or m + 1
    // + 1 because I am intentionally generating one more, so that we do to_ndarray only once.
    let dim = inputs[1..].len();
    if (n1 < dim) || (r == 0.) || (!r.is_finite()) {
        return Ok(Series::from_vec("", vec![f64::NAN]));
    }
    let parallel = kwargs.parallel;
    let leaf_size = kwargs.leaf_size;
    let dist_func = which_distance("inf", dim)?;

    // Check null for all but the last column. Check null here because we want to let the above
    // shortcircuit happen.
    let _ = no_null_in_inputs(&inputs[..last_idx], "KNN: Input contains null.".into())?;
    // Check the last column individually, because it is gauranteed to have a null at the last position
    // because the last column is only used in generating data_2, and it involves an extra shift.
    // I am doing this extra shift upfront to avoid doing to_ndarray twice.
    if inputs[last_idx].null_count() >= 2 {
        return Err(PolarsError::ComputeError(
            "KNN: Input contains null.".into(),
        ));
    }

    // Step 3, 4, 5 in wiki
    let data_1_view = data.slice(s![..n1, ..dim.abs_diff(1)]);
    let tree = build_standard_kdtree(dim.abs_diff(1), leaf_size, &data_1_view)?;
    let nb_in_radius = query_nb_cnt(&tree, data_1_view, &dist_func, &r, parallel);
    let temp: Float64Chunked = nb_in_radius
        .apply_values_generic(|x| ((x + 1) as f64 / n1 as f64).log(std::f64::consts::E)); // add 1 because query_nb_cnt does not consider the point itself as its neighbor.
    let phi_m = temp.mean().unwrap_or(0.);

    // Step 3, 4, 5 for m + 1 in wiki
    let n2 = n1.abs_diff(1);
    let data_2_view = data.slice(s![..n2, ..]);
    let tree = build_standard_kdtree(dim, leaf_size, &data_2_view)?;
    let nb2_in_radius = query_nb_cnt(&tree, data_2_view, &dist_func, &r, parallel);
    let temp: Float64Chunked = nb2_in_radius
        .apply_values_generic(|x| ((x + 1) as f64 / n2 as f64).log(std::f64::consts::E));
    let phi_m1 = temp.mean().unwrap_or(0.);
    // Output
    let out = Series::from_vec("", vec![(phi_m1 - phi_m).abs()]);
    Ok(out)
}
