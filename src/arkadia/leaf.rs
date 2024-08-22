use ndarray::ArrayView1;
use num::Float;

#[derive(Clone, Copy)]
pub struct Leaf<'a, T: Float, A> {
    pub item: A,
    pub row_vec: &'a [T],
}

impl<'a, T: Float, A> From<(A, ArrayView1<'a, T>)> for Leaf<'a, T, A> {
    fn from(value: (A, ArrayView1<'a, T>)) -> Self {
        Leaf {
            item: value.0,
            row_vec: value.1.to_slice().unwrap(),
        }
    }
}

pub trait KdLeaf<'a, T: Float> {
    fn dim(&self) -> usize;

    fn value_at(&self, idx: usize) -> T;

    fn vec(&self) -> &'a [T];
}

impl<'a, T: Float, A> KdLeaf<'a, T> for Leaf<'a, T, A> {
    fn dim(&self) -> usize {
        self.row_vec.len()
    }

    fn value_at(&self, idx: usize) -> T {
        self.row_vec[idx]
    }

    fn vec(&self) -> &'a [T] {
        self.row_vec
    }
}
