// use num::{complex::Complex64, Zero};
// use polars::prelude::*;
// use pyo3_polars::derive::polars_expr;
// use rustfft::FftPlanner;
// use serde::Deserialize;

// #[derive(Deserialize, Debug)]
// pub(crate) struct WelchKwargs {
//     pub(crate) window_size: usize,
//     pub(crate) overlap_size: usize,
// }

// fn welch(s:&[f64], window_length:usize, overlap_size:usize) -> f64 {

//     let mut left_idx: usize = 0;
//     let mut planner = FftPlanner::<f64>::new();
//     let r2c = planner.plan_fft_forward(window_length);
//     let mut scratch = vec![Complex64::zero(); r2c.get_inplace_scratch_len()];
//     let mut sums = 0f64;
//     let mut count = 0u32;
//     while left_idx + window_length <= s.len() {

//         let right_idx = left_idx + window_length;
//         let mut vec = s[left_idx..right_idx]
//             .iter()
//             .map(|x| x.into())
//             .collect::<Vec<Complex64>>();

//         let _ = r2c.process_with_scratch(&mut vec, &mut scratch);
//         sums += vec
//             .into_iter()
//             .fold(0f64, |acc, z| acc + z.re * z.re + z.im * z.im);

//         count += 1;
//         left_idx += overlap_size;
//     }

//     sums / count as f64
// }

// #[polars_expr(output_type=Float64)]
// fn pl_psd(
//     inputs: &[Series],
//     kwargs: WelchKwargs,
// ) -> PolarsResult<Series> {

//     let s = inputs[0].f64()?;
//     let s = s.cont_slice()?;
//     let result = welch(s, kwargs.window_size, kwargs.overlap_size);
//     Ok(Series::from_iter([result]))
// }
