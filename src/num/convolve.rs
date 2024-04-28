use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use realfft::RealFftPlanner;

// Optimization ideas: small size, e.g. <= 2048, always allocate a fixed sized slice?
// 2^n padding in the general case

enum ConvMode {
    FULL,
    SAME,
    LEFT,
    RIGHT,
    VALID,
}

impl From<&str> for ConvMode {
    fn from(value: &str) -> Self {
        match value.to_lowercase().as_ref() {
            "full" => Self::FULL,
            "same" => Self::SAME,
            "left" => Self::LEFT,
            "right" => Self::RIGHT,
            "valid" => Self::VALID,
            _ => Self::FULL,
        }
    }
}

// fn next_pow_2(n:usize) -> usize {
//     let mut m:usize = 2;
//     while m < n {
//         m <<= 1;
//     }
//     m
// }

fn valid_fft_convolve(input: &[f64], filter: &[f64]) -> PolarsResult<Vec<f64>> {
    let in_shape = input.len();
    // let good_size = next_pow_2(in_shape);
    // Prepare
    let mut output_vec = vec![0.; in_shape];
    output_vec[..in_shape].copy_from_slice(input);

    let mut oth = vec![0.; in_shape];
    oth[..filter.len()].copy_from_slice(filter);

    // let n = output_vec.len() as f64;
    let mut planner: RealFftPlanner<f64> = RealFftPlanner::new();
    let r2c = planner.plan_fft_forward(in_shape);
    let c2r = planner.plan_fft_inverse(in_shape);
    let mut spec_p = r2c.make_output_vec();
    let mut spec_q = r2c.make_output_vec();
    // Forward FFT on the inputs
    let _ = r2c.process(&mut output_vec, &mut spec_p);
    // .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    let _ = r2c.process(&mut oth, &mut spec_q);
    // .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;

    // After forward FFT, multiply in place in spec_p.
    for (z1, z2) in spec_p.iter_mut().zip(spec_q.into_iter()) {
        *z1 = *z1 * z2;
    }
    // Inverse FFT
    let _ = c2r.process(&mut spec_p, &mut output_vec);
    // .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;

    // output_vec.truncate(in_shape);
    Ok(output_vec)
}

fn valid_fft_convolve_direct(input: &[f64], filter: &[f64]) -> PolarsResult<Vec<f64>> {
    Ok(input
        .windows(filter.len())
        .map(|sl| {
            sl.iter()
                .zip(filter.iter())
                .fold(0., |acc, (x, y)| acc + x * y)
        })
        .collect())
}

fn fft_convolve(input: &[f64], filter: &[f64], mode: ConvMode) -> PolarsResult<Vec<f64>> {
    match mode {
        ConvMode::FULL => {
            let t = filter.len() - 1;
            let mut padded_input = vec![0.; input.len() + 2 * t];
            let from_to = t..(t + input.len());
            padded_input[from_to].copy_from_slice(input);
            fft_convolve(&padded_input, filter, ConvMode::VALID)
        }
        ConvMode::SAME => {
            let skip = (filter.len() - 1) / 2;
            let out = fft_convolve(input, filter, ConvMode::FULL)?;
            Ok(out.into_iter().skip(skip).take(input.len()).collect())
        }
        ConvMode::LEFT => {
            let n = input.len();
            let mut out = fft_convolve(input, filter, ConvMode::FULL)?;
            out.truncate(n);
            Ok(out)
        }
        ConvMode::RIGHT => {
            let out = fft_convolve(input, filter, ConvMode::FULL)?;
            Ok(out.into_iter().skip(filter.len() - 1).collect())
        }
        ConvMode::VALID => {
            let out = valid_fft_convolve(input, filter)?;
            let n = out.len() as f64;
            Ok(out
                .into_iter()
                .skip(filter.len() - 1)
                .map(|x| x / n)
                .collect())
        }
    }
}

fn fft_convolve2(input: &[f64], filter: &[f64], mode: ConvMode) -> PolarsResult<Vec<f64>> {
    match mode {
        ConvMode::FULL => {
            let t = filter.len() - 1;
            let mut padded_input = vec![0.; input.len() + 2 * t];
            let from_to = t..(t + input.len());
            padded_input[from_to].copy_from_slice(input);
            fft_convolve2(&padded_input, filter, ConvMode::VALID)
        }
        ConvMode::SAME => {
            let skip = (filter.len() - 1) / 2;
            let out = fft_convolve2(input, filter, ConvMode::FULL)?;
            Ok(out.into_iter().skip(skip).take(input.len()).collect())
        }
        ConvMode::LEFT => {
            let n = input.len();
            let mut out = fft_convolve2(input, filter, ConvMode::FULL)?;
            out.truncate(n);
            Ok(out)
        }
        ConvMode::RIGHT => {
            let out = fft_convolve2(input, filter, ConvMode::FULL)?;
            Ok(out.into_iter().skip(filter.len() - 1).collect())
        }
        ConvMode::VALID => {
            let out = valid_fft_convolve_direct(input, filter)?;
            Ok(out)
        }
    }
}

#[polars_expr(output_type=Float64)]
fn pl_fft_convolve(inputs: &[Series]) -> PolarsResult<Series> {
    let s1 = inputs[0].f64()?;
    let s2 = inputs[1].f64()?;
    let mode = inputs[2].str()?;
    let mode = mode.get(0).unwrap_or("full");
    let mode: ConvMode = mode.into();

    if s1.len() < s2.len() || s2.len() < 2 {
        return Err(PolarsError::ComputeError(
            "Convolution: The filter should have smaller length than the input column, and filter should have length >= 2.".into(),
        ));
    }
    if s2.has_validity() {
        return Err(PolarsError::ComputeError(
            "Convolution: The filter should not have nulls.".into(),
        ));
    }

    let input = s1.cont_slice().unwrap();
    let filter = s2.cont_slice().unwrap();

    let out = fft_convolve(input, filter, mode)?;

    let ca = Float64Chunked::from_slice(s1.name(), &out);
    Ok(ca.into_series())
}

#[polars_expr(output_type=Float64)]
fn pl_fft_convolve2(inputs: &[Series]) -> PolarsResult<Series> {
    let s1 = inputs[0].f64()?;
    let s2 = inputs[1].f64()?;
    let mode = inputs[2].str()?;
    let mode = mode.get(0).unwrap_or("full");
    let mode: ConvMode = mode.into();

    if s1.len() < s2.len() || s2.len() < 2 {
        return Err(PolarsError::ComputeError(
            "Convolution: The filter should have smaller length than the input column, and filter should have length >= 2.".into(),
        ));
    }
    if s2.has_validity() {
        return Err(PolarsError::ComputeError(
            "Convolution: The filter should not have nulls.".into(),
        ));
    }

    let input = s1.cont_slice().unwrap();
    let filter = s2.cont_slice().unwrap(); // already reversed in Python

    let out = fft_convolve2(input, filter, mode)?;

    let ca = Float64Chunked::from_slice(s1.name(), &out);
    Ok(ca.into_series())
}
