use ndarray::ArrayView1;
use num::Float;
use polars::error::{PolarsError, PolarsResult};

mod benford;
mod cond_entropy;
mod convolve;
mod entrophies;
mod fft;
mod float_extras;
mod gcd_lcm;
mod haversine;
mod jaccard;
mod knn;
mod lempel_ziv;
mod linear_regression;
mod pca;
mod psi;
mod target_encode;
mod tp_fp;
mod trapz;
mod woe_iv;

// Collection of other distances

#[inline]
pub fn cosine_dist<T: Float + 'static>(a: &[T], b: &[T]) -> T {
    debug_assert_eq!(a.len(), b.len());

    let a = ArrayView1::from(a);
    let b = ArrayView1::from(b);

    let a_norm = a.dot(&a);
    let b_norm = b.dot(&b);
    T::one() - (a.dot(&b)) / (a_norm * b_norm).sqrt()
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
