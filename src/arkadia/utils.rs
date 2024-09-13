use crate::arkadia::leaf::{Leaf, OwnedLeaf};
use num::Float;

#[derive(Clone, Default)]
pub enum SplitMethod {
    MIDPOINT, // min + (max - min) / 2
    #[default]
    MEDIAN,
}

impl From<bool> for SplitMethod {
    fn from(balanced: bool) -> Self {
        if balanced {
            Self::MEDIAN
        } else {
            Self::MIDPOINT
        }
    }
}

pub fn suggest_capacity(dim: usize) -> usize {
    if dim < 5 {
        10
    } else if dim < 10 {
        20
    } else if dim < 15 {
        40
    } else if dim < 20 {
        100
    } else {
        4098
    }
}

pub fn slice_to_leaves<'a, T: Float + 'static, A: Copy>(
    slice: &'a [T],
    row_len: usize,
    values: &'a [A],
) -> Vec<Leaf<'a, T, A>> {
    values
        .iter()
        .copied()
        .zip(slice.chunks_exact(row_len))
        .map(|pair| pair.into())
        .collect()
}

pub fn slice_to_owned_leaves<T: Float + 'static, A: Copy>(
    slice: &[T],
    row_len: usize,
    values: &[A],
) -> Vec<OwnedLeaf<T, A>> {
    values
        .iter()
        .copied()
        .zip(slice.chunks_exact(row_len))
        .map(|pair| pair.into())
        .collect()
}

pub fn slice_to_empty_leaves<'a, T: Float + 'static>(
    slice: &'a [T],
    row_len: usize,
) -> Vec<Leaf<'a, T, ()>> {
    slice
        .chunks_exact(row_len)
        .map(|row| ((), row).into())
        .collect()
}

// pub fn matrix_to_leaves<'a, T: Float + 'static, A: Copy>(
//     matrix: &'a ArrayView2<'a, T>,
//     values: &'a [A],
// ) -> Vec<Leaf<'a, T, A>> {
//     values
//         .iter()
//         .copied()
//         .zip(matrix.rows())
//         .map(|pair| pair.into())
//         .collect::<Vec<_>>()
// }

// pub fn matrix_to_leaves_w_row_num<'a, T: Float + 'static>(
//     matrix: &'a ArrayView2<'a, T>,
// ) -> Vec<Leaf<'a, T, usize>> {
//     matrix
//         .rows()
//         .into_iter()
//         .enumerate()
//         .map(|pair| pair.into())
//         .collect::<Vec<_>>()
// }

// pub fn matrix_to_empty_leaves<'a, T: Float + 'static>(
//     matrix: &'a ArrayView2<'a, T>,
// ) -> Vec<Leaf<'a, T, ()>> {
//     matrix
//         .rows()
//         .into_iter()
//         .map(|row| ((), row).into())
//         .collect::<Vec<_>>()
// }
