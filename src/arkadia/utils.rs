use crate::arkadia::leaf::Leaf;
use ndarray::ArrayView2;
use num::Float;

// ---

// #[derive(Clone, Default)]
// pub enum SplitMethod {
//     #[default]
//     MIDPOINT, // min + (max - min) / 2
//     MEAN,
//     MEDIAN,
// }

// ---

pub fn suggest_capacity(dim: usize) -> usize {
    if dim < 5 {
        8
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

pub fn matrix_to_leaves<'a, T: Float + 'static, A: Copy>(
    matrix: &'a ArrayView2<'a, T>,
    values: &'a [A],
) -> Vec<Leaf<'a, T, A>> {
    values
        .iter()
        .copied()
        .zip(matrix.rows())
        .map(|pair| pair.into())
        .collect::<Vec<_>>()
}

pub fn matrix_to_leaves_w_row_num<'a, T: Float + 'static>(
    matrix: &'a ArrayView2<'a, T>,
) -> Vec<Leaf<'a, T, usize>> {
    matrix
        .rows()
        .into_iter()
        .enumerate()
        .map(|pair| pair.into())
        .collect::<Vec<_>>()
}

pub fn matrix_to_empty_leaves<'a, T: Float + 'static>(
    matrix: &'a ArrayView2<'a, T>,
) -> Vec<Leaf<'a, T, ()>> {
    matrix
        .rows()
        .into_iter()
        .map(|row| ((), row).into())
        .collect::<Vec<_>>()
}
