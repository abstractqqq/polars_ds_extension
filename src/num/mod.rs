use kdtree::distance::squared_euclidean;
use num::Float;
use polars::error::{PolarsError, PolarsResult};

mod cond_entropy;
mod entrophies;
mod fft;
mod float_extras;
mod gcd_lcm;
mod haversine;
mod jaccard;
mod knn;
mod lempel_ziv;
mod ols;
mod psi;
mod tp_fp;
mod trapz;
mod woe_iv;

// Collection of distances, most will be used as function pointers in kd tree related queries,
// which may be bad for perf.

// Are these fast?

#[inline]
pub fn l_inf_dist<T: Float>(a: &[T], b: &[T]) -> T {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .fold(T::zero(), |acc, (x, y)| acc.max((*x - *y).abs()))
}

#[inline]
pub fn l1_dist<T: Float>(a: &[T], b: &[T]) -> T {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .fold(T::zero(), |acc, (x, y)| acc + (*x - *y).abs())
}

#[inline]
pub fn cosine_dist<T: Float>(a: &[T], b: &[T]) -> T {
    debug_assert_eq!(a.len(), b.len());

    let a_norm = a.iter().fold(T::zero(), |acc, x| acc + *x * *x);
    let b_norm = b.iter().fold(T::zero(), |acc, x| acc + *x * *x);
    let out = a
        .iter()
        .zip(b.iter())
        .fold(T::zero(), |acc, (x, y)| acc + *x * *y);
    T::one() - out / (a_norm * b_norm).sqrt()
}

#[inline]
fn haversine_elementwise<T: Float>(start_lat: T, start_long: T, end_lat: T, end_long: T) -> T {
    let r_in_km = T::from(6371.0).unwrap();
    let two = T::from(2.0).unwrap();
    let one = T::one();

    let d_lat = (end_lat - start_lat).to_radians();
    let d_lon = (end_long - start_long).to_radians();
    let lat1 = (start_lat).to_radians();
    let lat2 = (end_lat).to_radians();

    let a = ((d_lat / two).sin()) * ((d_lat / two).sin())
        + ((d_lon / two).sin()) * ((d_lon / two).sin()) * (lat1.cos()) * (lat2.cos());
    let c = two * ((a.sqrt()).atan2((one - a).sqrt()));
    r_in_km * c
}

#[inline]
pub fn haversine<T: Float>(start: &[T], end: &[T]) -> T {
    haversine_elementwise(start[0], start[1], end[0], end[1])
}

#[inline(always)]
pub fn which_distance(metric: &str, dim: usize) -> PolarsResult<fn(&[f64], &[f64]) -> f64> {
    match metric {
        "l1" => Ok(l1_dist),
        "inf" => Ok(l_inf_dist),
        "h" | "haversine" => {
            if dim == 2 {
                Ok(haversine)
            } else {
                Err(
                    PolarsError::ComputeError(
                        "KNN: Haversine distance must take 2 columns as features, one for lat and one for long.".into()
                    )
                )
            }
        }
        "cosine" => Ok(cosine_dist),
        _ => Ok(squared_euclidean),
    }
}
