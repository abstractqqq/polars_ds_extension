/// Linear Algebra, matrix related utilities

use faer::{Mat, MatRef};
use faer_traits::RealField;
use num::Float;

pub trait Elementwise<T: RealField> {
    fn map_elementwise(&self, f: impl Fn(T) -> T) -> Mat<T>;
}

impl <T: RealField + Float> Elementwise<T> for MatRef<'_, T> {
    fn map_elementwise(&self, f: impl Fn(T) -> T) -> Mat<T> {
        unsafe {
            Mat::from_fn(
                self.nrows(), 
                self.ncols(), 
                |i, j| f(*self.get_unchecked(i, j))
            )
        }
    }
}