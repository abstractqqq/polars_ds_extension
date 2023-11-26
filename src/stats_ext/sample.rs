// use polars::prelude::*;
// use pyo3_polars::derive::polars_expr;

// Random sample
// Get a column of random integers, with min and max from user input,
// (with a few random distributions options), while respecting the nulls
// in the reference column

// use rand::distributions::Standard;
// use polars::prelude::*;
// use pyo3_polars::derive::polars_expr;

// #[polars_expr(output_type=Float64)]
// fn pl_sample_normal(inputs: &[Series]) -> PolarsResult<Series> {

//     let reference = &inputs[0];
//     let mean = inputs[1].f64()?;
//     let mean = mean.get(0).unwrap();
//     let std = inputs[2].f64()?;
//     let std = std.get(0).unwrap();
//     let respect_null = inputs[3].bool()?;
//     let respect_null = respect_null.get(0).unwrap();

//     if respect_null {
//         todo!()
//     } else {
//         let n = reference.len();

//         todo!()
//     }

// }
