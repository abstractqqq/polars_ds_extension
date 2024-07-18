use crate::utils::float_output;
/// Generates random sample from distributions for Polars DataFrame
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distributions::Uniform;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand::{distributions::DistString, Rng};
use rand_distr::{Alphanumeric, Binomial, Distribution, Exp, Exp1, Normal, StandardNormal};

#[polars_expr(output_type=Int32)]
fn pl_rand_int(inputs: &[Series]) -> PolarsResult<Series> {
    let n = inputs[0].u32()?;
    let n = n.get(0).unwrap() as usize;
    let low = inputs[1].i32()?;
    let mut low = low.get(0).unwrap();
    let high = inputs[2].i32()?;
    let mut high = high.get(0).unwrap();
    if low == high {
        let out = Int32Chunked::from_vec("", vec![low; n]);
        return Ok(out.into_series());
    } else if high < low {
        std::mem::swap(&mut low, &mut high);
    }
    let seed = inputs[3].u64()?;
    let seed = seed.get(0);
    let dist = Uniform::new(low, high);
    let mut rng = if let Some(s) = seed {
        StdRng::seed_from_u64(s)
    } else {
        StdRng::from_entropy()
    };
    let out = Int32Chunked::from_iter_values("", (&mut rng).sample_iter(dist).take(n));
    Ok(out.into_series())
}

#[polars_expr(output_type=UInt64)]
fn pl_rand_binomial(inputs: &[Series]) -> PolarsResult<Series> {
    let len = inputs[0].u32()?;
    let len = len.get(0).unwrap() as usize;
    let n = inputs[1].u32()?;
    let n = n.get(0).unwrap();
    let p = inputs[2].f64()?;
    let p = p.get(0).unwrap();
    let seed = inputs[3].u64()?;
    let seed = seed.get(0);
    let dist =
        Binomial::new(n.into(), p).map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    let mut rng = if let Some(s) = seed {
        StdRng::seed_from_u64(s)
    } else {
        StdRng::from_entropy()
    };
    let out = UInt64Chunked::from_iter_values("", (&mut rng).sample_iter(dist).take(len));
    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn pl_rand_exp(inputs: &[Series]) -> PolarsResult<Series> {
    let len = inputs[0].u32()?;
    let len = len.get(0).unwrap() as usize;
    let lambda = inputs[1].f64()?;
    let lambda = lambda.get(0).unwrap();

    let seed = inputs[2].u64()?;
    let seed = seed.get(0);
    let mut rng = if let Some(s) = seed {
        StdRng::seed_from_u64(s)
    } else {
        StdRng::from_entropy()
    };
    if lambda == 1.0 {
        let out = Float64Chunked::from_iter_values("", (&mut rng).sample_iter(Exp1).take(len));
        Ok(out.into_series())
    } else {
        let dist = Exp::new(lambda).map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
        let out = Float64Chunked::from_iter_values("", (&mut rng).sample_iter(dist).take(len));
        Ok(out.into_series())
    }
}

#[polars_expr(output_type=Float64)]
fn pl_random(inputs: &[Series]) -> PolarsResult<Series> {
    let len: &ChunkedArray<UInt32Type> = inputs[0].u32()?;
    let len = len.get(0).unwrap() as usize;
    let low = inputs[1].f64()?;
    let low = low.get(0).unwrap();
    let high = inputs[2].f64()?;
    let high = high.get(0).unwrap();
    let seed = inputs[3].u64()?;
    let seed = seed.get(0);
    let dist = Uniform::new(low, high);
    let mut rng = if let Some(s) = seed {
        StdRng::seed_from_u64(s)
    } else {
        StdRng::from_entropy()
    };
    let out = Float64Chunked::from_iter_values("", (&mut rng).sample_iter(dist).take(len));
    Ok(out.into_series())
}

// Perturb and Jitter respect float type. Unlike others which default to f64

#[polars_expr(output_type_func=float_output)]
fn pl_perturb(inputs: &[Series]) -> PolarsResult<Series> {
    let reference = &inputs[0];
    let low = inputs[1].f64()?;
    let low = low.get(0).unwrap();
    let high = inputs[2].f64()?;
    let high = high.get(0).unwrap();
    match reference.dtype() {
        DataType::Float32 => {
            let dist: Uniform<f32> = Uniform::new(low as f32, high as f32);
            let mut rng = rand::thread_rng();
            let ca = reference.f32().unwrap();
            // Have to use _generic here to avoid the Copy trait
            let out: Float32Chunked = ca.apply_values_generic(|x| x + dist.sample(&mut rng));
            Ok(out.into_series())
        }
        DataType::Float64 => {
            let dist: Uniform<f64> = Uniform::new(low, high);
            let mut rng = rand::thread_rng();
            let ca = reference.f64().unwrap();
            // Have to use _generic here to avoid the Copy trait
            let out: Float64Chunked = ca.apply_values_generic(|x| x + dist.sample(&mut rng));
            Ok(out.into_series())
        }
        _ => Err(PolarsError::ComputeError(
            "Input column must be floats.".into(),
        )),
    }
}

#[polars_expr(output_type_func=float_output)]
fn pl_jitter(inputs: &[Series]) -> PolarsResult<Series> {
    let reference = &inputs[0];
    let std_ = inputs[1].f64()?;
    let std_ = std_.get(0).unwrap();

    match reference.dtype() {
        DataType::Float32 => {
            let std_ = std_ as f32;
            let mut rng = rand::thread_rng();
            let ca = reference.f32().unwrap();
            let out: Float32Chunked = if std_ == 1.0 {
                // Avoid extra multiplication
                ca.apply_values_generic(|x| x + rng.sample::<f32, _>(StandardNormal))
            } else {
                ca.apply_values_generic(|x| x + std_ * rng.sample::<f32, _>(StandardNormal))
            };
            Ok(out.into_series())
        }
        DataType::Float64 => {
            let mut rng = rand::thread_rng();
            let ca = reference.f64().unwrap();
            let out: Float64Chunked = if std_ == 1.0 {
                ca.apply_values_generic(|x| x + rng.sample::<f64, _>(StandardNormal))
            } else {
                ca.apply_values_generic(|x| x + std_ * rng.sample::<f64, _>(StandardNormal))
            };
            Ok(out.into_series())
        }
        _ => Err(PolarsError::ComputeError(
            "Input column must be floats.".into(),
        )),
    }
}

#[polars_expr(output_type=Float64)]
fn pl_rand_normal(inputs: &[Series]) -> PolarsResult<Series> {
    let len = inputs[0].u32()?;
    let len = len.get(0).unwrap() as usize;
    let mean = inputs[1].f64()?;
    let mean = mean.get(0).unwrap_or(0.);
    let std_ = inputs[2].f64()?;
    let std_ = std_.get(0).unwrap_or(1.);

    let seed = inputs[3].u64()?;
    let seed = seed.get(0);
    let mut rng = if let Some(s) = seed {
        StdRng::seed_from_u64(s)
    } else {
        StdRng::from_entropy()
    };
    if mean == 0. && std_ == 1.0 {
        let out =
            Float64Chunked::from_iter_values("", (&mut rng).sample_iter(StandardNormal).take(len));
        Ok(out.into_series())
    } else {
        let dist = Normal::new(mean, std_).unwrap();
        let out = Float64Chunked::from_iter_values("", (&mut rng).sample_iter(dist).take(len));
        Ok(out.into_series())
    }
}

#[polars_expr(output_type=Float64)]
fn pl_rand_str(inputs: &[Series]) -> PolarsResult<Series> {
    let len = inputs[0].u32()?;
    let len = len.get(0).unwrap() as usize;
    let min_size = inputs[1].u32()?;
    let min_size = min_size.get(0).unwrap() as usize;
    let max_size = inputs[2].u32()?;
    let max_size = max_size.get(0).unwrap() as usize;
    let seed = inputs[3].u64()?;
    let seed = seed.get(0);

    let dist = Uniform::new_inclusive(min_size, max_size);
    let mut rng = if let Some(s) = seed {
        StdRng::seed_from_u64(s)
    } else {
        StdRng::from_entropy()
    };

    let sample = (0..len).map(|_| {
        let length = rng.sample(dist);
        Alphanumeric.sample_string(&mut rng, length)
    });
    let out = StringChunked::from_iter_values("", sample);
    Ok(out.into_series())
}
