use num::Float;

#[derive(Clone, Copy, Debug)]
pub struct Leaf<'a, T: Float, A> {
    pub item: A,
    pub row_vec: &'a [T],
}

impl<'a, T: Float, A: Copy> From<(A, &'a [T])> for Leaf<'a, T, A> {
    fn from(value: (A, &'a [T])) -> Self {
        Leaf {
            item: value.0,
            row_vec: value.1,
        }
    }
}
pub trait KdLeaf<T: Float, A> {
    fn dim(&self) -> usize;

    fn value_at(&self, idx: usize) -> T;

    fn vec(&self) -> &[T];

    fn get_item(&self) -> A;
}

impl<'a, T: Float, A: Copy> KdLeaf<T, A> for Leaf<'a, T, A> {
    fn dim(&self) -> usize {
        self.row_vec.len()
    }

    fn value_at(&self, idx: usize) -> T {
        self.row_vec[idx]
    }

    fn vec(&self) -> &'a [T] {
        self.row_vec
    }

    fn get_item(&self) -> A {
        self.item
    }
}
