/// Generates random sample from distributions for Polars DataFrame
///
/// We are sacrificing speed and memory usage a little bit here by using CSPRNSGs. See
/// details here: https://rust-random.github.io/book/guide-rngs.html
///
/// I think it is ok to use CSPRNGS because it is fast enough and we generally do not
/// want output to be easily guessable.
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distributions::Uniform;
use rand::{distributions::DistString, Rng};
use rand_distr::{Alphanumeric, Binomial, Exp, Exp1, Normal, StandardNormal};

#[polars_expr(output_type=Int32)]
fn pl_rand_int(inputs: &[Series]) -> PolarsResult<Series> {
    let reference = &inputs[0];
    let low = inputs[1].i32()?;
    let high = inputs[2].i32()?;

    let respect_null = inputs[3].bool()?;
    let respect_null = respect_null.get(0).unwrap();

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
    let mut rng = rand::thread_rng();
    if respect_null && (reference.null_count() > 0) {
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
fn pl_sample_binomial(inputs: &[Series]) -> PolarsResult<Series> {
    let reference = &inputs[0];
    let n = inputs[1].u64()?;
    let n = n.get(0).unwrap();
    let p = inputs[2].f64()?;
    let p = p.get(0).unwrap();
    let respect_null = inputs[3].bool()?;
    let respect_null = respect_null.get(0).unwrap();
    let m = reference.len();

    let dist = Binomial::new(n, p).map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    let mut rng = rand::thread_rng();
    if respect_null && (reference.null_count() > 0) {
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
fn pl_sample_exp(inputs: &[Series]) -> PolarsResult<Series> {
    let reference = &inputs[0];
    let lambda = inputs[1].f64()?;
    let lambda = lambda.get(0).unwrap();
    let respect_null = inputs[2].bool()?;
    let respect_null = respect_null.get(0).unwrap();
    let m = reference.len();

    let mut rng = rand::thread_rng();
    if lambda == 1.0 {
        if respect_null && (reference.null_count() > 0) {
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
        if respect_null && (reference.null_count() > 0) {
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
fn pl_sample_uniform(inputs: &[Series]) -> PolarsResult<Series> {
    let reference = &inputs[0];
    let low = inputs[1].f64()?;
    let mut low = low.get(0).unwrap();
    let high = inputs[2].f64()?;
    let mut high = high.get(0).unwrap();
    let respect_null = inputs[3].bool()?;
    let respect_null = respect_null.get(0).unwrap();
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
    let mut rng = rand::thread_rng();
    if respect_null && (reference.null_count() > 0) {
        // Create a dummy. I just need to access the apply_nonnull_values_generic method
        // on chunked arrays.
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

#[polars_expr(output_type=Float64)]
fn pl_sample_normal(inputs: &[Series]) -> PolarsResult<Series> {
    let reference = &inputs[0];
    let mean = inputs[1].f64()?;
    let mean = mean.get(0).unwrap_or(0.);
    let std_ = inputs[2].f64()?;
    let std_ = std_.get(0).unwrap_or(1.);
    let respect_null = inputs[3].bool()?;
    let respect_null = respect_null.get(0).unwrap();

    let n = reference.len();

    let valid = mean.is_finite() && std_.is_finite();
    if !valid {
        return Err(PolarsError::ComputeError(
            "Sample: Mean and std must be finite.".into(),
        ));
    }
    // Standard Normal, use the more optimized version
    if mean == 0. && std_ == 1. {
        let dist = StandardNormal;
        let mut rng = rand::thread_rng();
        if respect_null && (reference.null_count() > 0) {
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
        let mut rng = rand::thread_rng();
        if respect_null && (reference.null_count() > 0) {
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
fn pl_sample_alphanumeric(inputs: &[Series]) -> PolarsResult<Series> {
    let reference = &inputs[0];
    let min_size = inputs[1].u32()?;
    let mut min_size = min_size.get(0).unwrap() as usize;
    let max_size = inputs[2].u32()?;
    let mut max_size = max_size.get(0).unwrap() as usize;
    let respect_null = inputs[3].bool()?;
    let respect_null = respect_null.get(0).unwrap();

    let n = reference.len();

    if min_size > max_size {
        std::mem::swap(&mut min_size, &mut max_size);
    }
    let uniform = Uniform::new_inclusive(min_size, max_size);

    let mut rng = rand::thread_rng();
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
