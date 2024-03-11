use crate::utils::float_output;
/// Generates random sample from distributions for Polars DataFrame
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distributions::Uniform;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand::{distributions::DistString, Rng};
use rand_distr::{Alphanumeric, Binomial, Distribution, Exp, Exp1, Normal, StandardNormal};
use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub(crate) struct SampleKwargs {
    pub(crate) seed: Option<u64>,
    pub(crate) respect_null: bool,
}

#[polars_expr(output_type=Int32)]
fn pl_rand_int(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
    let reference = &inputs[0];
    let low = inputs[1].i32()?;
    let high = inputs[2].i32()?;

    let respect_null = kwargs.respect_null;
    let seed = kwargs.seed;

    let (mut low, mut high) = (low.get(0).unwrap_or(0), high.get(0).unwrap_or(10));
    if high == low {
        return Err(PolarsError::ComputeError(
            "Sample: Low and high must be different values.".into(),
        ));
    } else if high < low {
        std::mem::swap(&mut low, &mut high);
    }

    let n = reference.len();
    let dist = Uniform::new(low, high);
    let mut rng = if let Some(s) = seed {
        StdRng::seed_from_u64(s)
    } else {
        StdRng::from_entropy()
    };
    if respect_null && reference.has_validity() {
        let ca = reference.is_null();
        let out: Int32Chunked = ca.apply_generic(|x| {
            if x.unwrap_or(true) {
                None
            } else {
                Some(rng.sample(dist))
            }
        });
        Ok(out.into_series())
    } else {
        let out = Int32Chunked::from_iter_values("", (&mut rng).sample_iter(dist).take(n));
        Ok(out.into_series())
    }
}

#[polars_expr(output_type=UInt64)]
fn pl_sample_binomial(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
    let reference = &inputs[0];
    let n = inputs[1].u64()?;
    let n = n.get(0).unwrap();
    let p = inputs[2].f64()?;
    let p = p.get(0).unwrap();

    let m = reference.len();
    let respect_null = kwargs.respect_null;
    let seed = kwargs.seed;

    let dist = Binomial::new(n, p).map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    let mut rng = if let Some(s) = seed {
        StdRng::seed_from_u64(s)
    } else {
        StdRng::from_entropy()
    };
    if respect_null && reference.has_validity() {
        let ca = reference.is_null();
        let out: UInt64Chunked = ca.apply_generic(|x| {
            if x.unwrap_or(true) {
                None
            } else {
                Some(rng.sample(dist))
            }
        });
        Ok(out.into_series())
    } else {
        let out = UInt64Chunked::from_iter_values("", (&mut rng).sample_iter(dist).take(m));
        Ok(out.into_series())
    }
}

#[polars_expr(output_type=Float64)]
fn pl_sample_exp(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
    let reference = &inputs[0];
    let lambda = inputs[1].f64()?;
    let lambda = lambda.get(0).unwrap();
    let m = reference.len();

    let respect_null = kwargs.respect_null;
    let seed = kwargs.seed;

    let mut rng = if let Some(s) = seed {
        StdRng::seed_from_u64(s)
    } else {
        StdRng::from_entropy()
    };
    if lambda == 1.0 {
        if respect_null && reference.has_validity() {
            let ca = reference.is_null();
            let out: Float64Chunked = ca.apply_generic(|x| {
                if x.unwrap_or(true) {
                    None
                } else {
                    Some(rng.sample(Exp1))
                }
            });
            Ok(out.into_series())
        } else {
            let out = Float64Chunked::from_iter_values("", (&mut rng).sample_iter(Exp1).take(m));
            Ok(out.into_series())
        }
    } else {
        let dist = Exp::new(lambda).map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
        if respect_null && reference.has_validity() {
            let ca = reference.is_null();
            let out: Float64Chunked = ca.apply_generic(|x| {
                if x.unwrap_or(true) {
                    None
                } else {
                    Some(rng.sample(dist))
                }
            });
            Ok(out.into_series())
        } else {
            let out = Float64Chunked::from_iter_values("", (&mut rng).sample_iter(dist).take(m));
            Ok(out.into_series())
        }
    }
}

#[polars_expr(output_type=Float64)]
fn pl_sample_uniform(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
    let reference = &inputs[0];
    let low = inputs[1].f64()?;
    let mut low = low.get(0).unwrap();
    let high = inputs[2].f64()?;
    let mut high = high.get(0).unwrap();

    let respect_null = kwargs.respect_null;
    let seed = kwargs.seed;

    let n = reference.len();
    let valid = low.is_finite() && high.is_finite();
    if !valid {
        return Err(PolarsError::ComputeError(
            "Sample: Low and high must be finite.".into(),
        ));
    }
    if high == low {
        return Err(PolarsError::ComputeError(
            "Sample: Low and high must be different values.".into(),
        ));
    } else if high < low {
        std::mem::swap(&mut low, &mut high);
    }

    let dist = Uniform::new(low, high);
    let mut rng = if let Some(s) = seed {
        StdRng::seed_from_u64(s)
    } else {
        StdRng::from_entropy()
    };
    if respect_null && reference.has_validity() {
        // Create a dummy. I just need to access the apply_nonnull_values_generic method
        // on chunked arrays.
        let ca = reference.is_null();
        let out: Float64Chunked = ca.apply_generic(|x| {
            if x.unwrap_or(true) {
                None
            } else {
                Some(dist.sample(&mut rng))
            }
        });
        Ok(out.into_series())
    } else {
        let out = Float64Chunked::from_iter_values("", dist.sample_iter(&mut rng).take(n));
        Ok(out.into_series())
    }
}

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

#[polars_expr(output_type=Float64)]
fn pl_sample_normal(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
    let reference = &inputs[0];
    let mean = inputs[1].f64()?;
    let mean = mean.get(0).unwrap_or(0.);
    let std_ = inputs[2].f64()?;
    let std_ = std_.get(0).unwrap_or(1.);

    let n = reference.len();
    let respect_null = kwargs.respect_null;
    let seed = kwargs.seed;

    let valid = mean.is_finite() && std_.is_finite();
    if !valid {
        return Err(PolarsError::ComputeError(
            "Sample: Mean and std must be finite.".into(),
        ));
    }
    let mut rng = if let Some(s) = seed {
        StdRng::seed_from_u64(s)
    } else {
        StdRng::from_entropy()
    };
    // Standard Normal, use the more optimized version
    if mean == 0. && std_ == 1. {
        let dist = StandardNormal;
        if respect_null && reference.has_validity() {
            let ca = reference.is_null();
            let out: Float64Chunked = ca.apply_generic(|x| {
                if x.unwrap_or(true) {
                    None
                } else {
                    Some(rng.sample(dist))
                }
            });
            Ok(out.into_series())
        } else {
            let out = Float64Chunked::from_iter_values("", (&mut rng).sample_iter(dist).take(n));
            Ok(out.into_series())
        }
    } else {
        // Guaranteed that std is finite
        let dist = Normal::new(mean, std_).unwrap();
        if respect_null && reference.has_validity() {
            let ca = reference.is_null();
            let out: Float64Chunked = ca.apply_generic(|x| {
                if x.unwrap_or(true) {
                    None
                } else {
                    Some(rng.sample(dist))
                }
            });
            Ok(out.into_series())
        } else {
            let out = Float64Chunked::from_iter_values("", (&mut rng).sample_iter(dist).take(n));
            Ok(out.into_series())
        }
    }
}

#[polars_expr(output_type=String)]
fn pl_sample_alphanumeric(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
    let reference = &inputs[0];
    let min_size = inputs[1].u32()?;
    let mut min_size = min_size.get(0).unwrap() as usize;
    let max_size = inputs[2].u32()?;
    let mut max_size = max_size.get(0).unwrap() as usize;

    let n = reference.len();
    let respect_null = kwargs.respect_null;
    let seed = kwargs.seed;

    if min_size > max_size {
        std::mem::swap(&mut min_size, &mut max_size);
    }
    let uniform = Uniform::new_inclusive(min_size, max_size);
    let mut rng = if let Some(s) = seed {
        StdRng::seed_from_u64(s)
    } else {
        StdRng::from_entropy()
    };
    if respect_null && reference.has_validity() {
        let ca = reference.is_null();
        let out: StringChunked = ca.apply_generic(|x| {
            if x.unwrap_or(true) {
                None
            } else {
                let length = rng.sample(uniform);
                Some(Alphanumeric.sample_string(&mut rng, length))
            }
        });
        Ok(out.into_series())
    } else {
        let sample = (0..n).map(|_| {
            let length = rng.sample(uniform);
            Alphanumeric.sample_string(&mut rng, length)
        });
        let out = StringChunked::from_iter_values("", sample);
        Ok(out.into_series())
    }
}
