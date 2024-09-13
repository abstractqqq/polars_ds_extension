use num::Float;

#[derive(Clone, Copy, Debug)]
pub struct Leaf<'a, T: Float, A> {
    pub item: A,
    pub row_vec: &'a [T],
}

#[derive(Clone)]
pub struct OwnedLeaf<T: Float, A> {
    pub item: A,
    pub row_vec: Vec<T>,
}

impl<'a, T: Float, A: Copy> From<(A, &'a [T])> for Leaf<'a, T, A> {
    fn from(value: (A, &'a [T])) -> Self {
        Leaf {
            item: value.0,
            row_vec: value.1,
        }
    }
}

impl<T: Float, A: Copy> From<(A, &[T])> for OwnedLeaf<T, A> {
    fn from(value: (A, &[T])) -> Self {
        OwnedLeaf {
            item: value.0,
            row_vec: value.1.to_vec(),
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

impl<'a, T: Float, A: Copy> KdLeaf<T, A> for OwnedLeaf<T, A> {
    fn dim(&self) -> usize {
        self.row_vec.len()
    }

    fn value_at(&self, idx: usize) -> T {
        self.row_vec[idx]
    }

    fn vec(&self) -> &[T] {
        self.row_vec.as_slice()
    }

    fn get_item(&self) -> A {
        self.item.clone()
    }
}
