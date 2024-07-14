/// KDT with any distance metric. L1, L2, SQL2 and LINF are included already.
use crate::arkadia::{leaf::KdLeaf, suggest_capacity, Leaf, SplitMethod, KDTQ, NB};
use num::Float;

// Although this implements L2 and SQL2 distances, the fastest way is still to use arkadia.rs if multiple queries
// are needed. This is because norm-caching benefits reduces a lot of computation when running lots of queries on the same tree

#[derive(Clone, PartialEq, Eq)]
pub enum DIST<T: Float + 'static> {
    L1,
    L2,
    SQL2, // Squared LP, but without using cached norms. Good for one-time queries
    LINF,
    ANY(fn(&[T], &[T]) -> T),
}

impl<T: Float + 'static> DIST<T> {
    #[inline(always)]
    fn dist(&self, a1: &[T], a2: &[T]) -> T {
        match self {
            DIST::L1 => a1
                .iter()
                .copied()
                .zip(a2.iter().copied())
                .fold(T::zero(), |acc, (x, y)| acc + ((x - y).abs())),

            DIST::L2 => a1
                .iter()
                .copied()
                .zip(a2.iter().copied())
                .fold(T::zero(), |acc, (x, y)| acc + ((x - y).powi(2)))
                .sqrt(),

            DIST::SQL2 => a1
                .iter()
                .copied()
                .zip(a2.iter().copied())
                .fold(T::zero(), |acc, (x, y)| acc + ((x - y).powi(2))),

            DIST::LINF => a1
                .iter()
                .copied()
                .zip(a2.iter().copied())
                .fold(T::zero(), |acc, (x, y)| acc.max((x - y).abs())),

            DIST::ANY(func) => func(a1, a2),
        }
    }
}

pub struct AnyKDT<'a, T: Float + 'static, A> {
    dim: usize,
    // Nodes
    left: Option<Box<AnyKDT<'a, T, A>>>,
    right: Option<Box<AnyKDT<'a, T, A>>>,
    // Is a leaf node if this has values
    split_axis: Option<usize>,
    split_axis_value: Option<T>,
    min_bounds: Vec<T>,
    max_bounds: Vec<T>,
    // Data
    data: Option<&'a [Leaf<'a, T, A>]>, // Not none when this is a leaf
    //
    lp: DIST<T>,
}

impl<'a, T: Float + 'static, A: Copy> AnyKDT<'a, T, A> {
    // Add method to create the tree by adding leaf elements one by one

    pub fn from_leaves(
        data: &'a mut [Leaf<'a, T, A>],
        how: SplitMethod,
        lp: DIST<T>,
    ) -> Result<Self, String> {
        if data.is_empty() {
            return Err("Empty data.".into());
        }
        let dim = data[0].dim();
        Ok(Self::from_leaves_unchecked(
            data,
            dim,
            suggest_capacity(dim),
            0,
            how,
            lp,
        ))
    }

    pub fn with_capacity(
        data: &'a mut [Leaf<'a, T, A>],
        capacity: usize,
        how: SplitMethod,
        lp: DIST<T>,
    ) -> Result<Self, String> {
        if data.is_empty() {
            return Err("Empty data.".into());
        }
        let dim = data.last().unwrap().dim();
        if capacity == 0 {
            return Err("Zero capacity.".into());
        }
        Ok(Self::from_leaves_unchecked(data, dim, capacity, 0, how, lp))
    }

    fn from_leaves_unchecked(
        data: &'a mut [Leaf<'a, T, A>],
        dim: usize,
        capacity: usize,
        depth: usize,
        how: SplitMethod,
        lp: DIST<T>,
    ) -> Self {
        let n = data.len();
        let (min_bounds, max_bounds) = Self::find_bounds(data, dim);
        if n <= capacity {
            AnyKDT {
                dim: dim,
                left: None,
                right: None,
                split_axis: None,
                split_axis_value: None,
                min_bounds: min_bounds,
                max_bounds: max_bounds,
                data: Some(data),
                lp: lp,
            }
        } else {
            let axis = depth % dim;
            let (split_axis_value, split_idx) = match how {
                SplitMethod::MIDPOINT => {
                    let midpoint = min_bounds[axis]
                        + (max_bounds[axis] - min_bounds[axis]) / (T::one() + T::one());

                    // Partition point basically uses the same value as the key function. Maybe I can cache some stuff?
                    // True will go right, false go left
                    data.sort_unstable_by_key(|leaf| leaf.value_at(axis) >= midpoint);
                    let split_idx = data.partition_point(|elem| elem.value_at(axis) < midpoint); // first index of True. If it doesn't exist, all points goes into left
                    (midpoint, split_idx)
                }
                SplitMethod::MEAN => {
                    let mut sum = T::zero();
                    for row in data.iter() {
                        sum = sum + row.value_at(axis);
                    }
                    let mean = sum / T::from(n).unwrap();
                    data.sort_unstable_by_key(|leaf| leaf.value_at(axis) >= mean);
                    let split_idx = data.partition_point(|elem| elem.value_at(axis) < mean); // first index of True. If it doesn't exist, all points goes into left
                    (mean, split_idx)
                }
                SplitMethod::MEDIAN => {
                    data.sort_unstable_by(|l1, l2| {
                        l1.value_at(axis).partial_cmp(&l2.value_at(axis)).unwrap()
                    });
                    let half = n >> 1;
                    let split_value = data[half].value_at(axis);
                    (split_value, half)
                }
            };

            let (left, right) = data.split_at_mut(split_idx);

            if left.is_empty() {
                // Left is size 0, right is all, is a very rare case, which happens when all the values at this
                // dimension are the same. In this case we proceed by (maybe) breaking the capacity rule and create
                // a leaf tree.
                // There are two cases that may ensue:
                // 1. We let the recursion keep going with left being an empty tree. In the next dimension, right
                // will split into 2 and everything works.
                // 2. In the next dimension, right also has exactly the same problem. And all remaining dimensions have
                // the same problem. Stack overflow. Game over.
                // Although 2 is rare, we can't predict which situation may arise. We opt for a safer approach by always
                // creating a leaf tree to end the recursion. We know 2 happens when the remaining leaves are identical in
                // each dimension.
                // So the solution makes sense. We also note that 2 may happen in perfectly periodic data generated
                // with sin/cos functions, which is common in time series, which is also how this error came to be known...
                AnyKDT {
                    dim: dim,
                    left: None,
                    right: None,
                    split_axis: None,
                    split_axis_value: None,
                    min_bounds: min_bounds,
                    max_bounds: max_bounds,
                    data: Some(right),
                    lp: lp,
                }
            } else {
                AnyKDT {
                    dim: dim,
                    left: Some(Box::new(Self::from_leaves_unchecked(
                        left,
                        dim,
                        capacity,
                        depth + 1,
                        how.clone(),
                        lp.clone(),
                    ))),
                    right: Some(Box::new(Self::from_leaves_unchecked(
                        right,
                        dim,
                        capacity,
                        depth + 1,
                        how,
                        lp.clone(),
                    ))),
                    split_axis: Some(axis),
                    split_axis_value: Some(split_axis_value),
                    min_bounds: min_bounds,
                    max_bounds: max_bounds,
                    data: None,
                    lp: lp,
                }
            }
        }
    }

    fn is_leaf(&self) -> bool {
        self.data.is_some()
    }

    #[inline(always)]
    fn closest_dist_to_box(&self, min_bounds: &[T], max_bounds: &[T], point: &[T]) -> T {
        let mut dist = T::zero();
        match self.lp {
            DIST::L1 => {
                for i in 0..point.len() {
                    if point[i] > max_bounds[i] {
                        dist = dist + (point[i] - max_bounds[i]).abs();
                    } else if point[i] < min_bounds[i] {
                        dist = dist + (point[i] - min_bounds[i]).abs();
                    }
                }
                dist
            }

            DIST::L2 => {
                for i in 0..point.len() {
                    if point[i] > max_bounds[i] {
                        dist = dist + (point[i] - max_bounds[i]).powi(2);
                    } else if point[i] < min_bounds[i] {
                        dist = dist + (point[i] - min_bounds[i]).powi(2);
                    }
                }
                dist.sqrt()
            }

            DIST::SQL2 => {
                for i in 0..point.len() {
                    if point[i] > max_bounds[i] {
                        dist = dist + (point[i] - max_bounds[i]).powi(2);
                    } else if point[i] < min_bounds[i] {
                        dist = dist + (point[i] - min_bounds[i]).powi(2);
                    }
                }
                dist
            }

            DIST::LINF => {
                for i in 0..point.len() {
                    if point[i] > max_bounds[i] {
                        dist = dist.max((point[i] - max_bounds[i]).abs());
                    } else if point[i] < min_bounds[i] {
                        dist = dist.max((point[i] - min_bounds[i]).abs());
                    }
                }
                dist
            }

            DIST::ANY(func) => {
                let mut new_point = point.to_vec();
                for i in 0..point.len() {
                    if point[i] > max_bounds[i] {
                        new_point[i] = max_bounds[i];
                    } else if point[i] < min_bounds[i] {
                        new_point[i] = min_bounds[i];
                    }
                }
                func(point, &new_point)
            }
        }
    }

    #[inline(always)]
    fn update_top_k(&self, top_k: &mut Vec<NB<T, A>>, k: usize, point: &[T], max_dist_bound: T) {
        let max_permissible_dist = T::max_value().min(max_dist_bound);
        // This is only called if is_leaf. Safe to unwrap.
        for element in self.data.unwrap().iter() {
            let cur_max_dist = top_k.last().map(|nb| nb.dist).unwrap_or(max_dist_bound);
            let y = element.row_vec;
            let dist = self.lp.dist(y, point);
            if dist < cur_max_dist || (top_k.len() < k && dist <= max_permissible_dist) {
                let nb = NB {
                    dist: dist,
                    item: element.item,
                };
                let idx: usize = top_k.partition_point(|s| s <= &nb);
                if idx < top_k.len() {
                    if top_k.len() + 1 > k {
                        top_k.pop();
                    }
                    top_k.insert(idx, nb);
                } else if top_k.len() < k {
                    top_k.push(nb);
                }
            }
        }
        // You can find code comments in arkadia.rs
    }

    #[inline(always)]
    fn update_nb_within(&self, neighbors: &mut Vec<NB<T, A>>, point: &[T], radius: T) {
        // This is only called if is_leaf. Safe to unwrap.
        for element in self.data.unwrap().iter() {
            let y = element.row_vec;
            let dist = self.lp.dist(y, point);
            if dist <= radius {
                neighbors.push(NB {
                    dist: dist,
                    item: element.item,
                });
            }
        }
    }
}

impl<'a, T: Float + 'static, A: Copy> KDTQ<'a, T, A> for AnyKDT<'a, T, A> {
    fn dim(&self) -> usize {
        self.dim
    }

    #[inline(always)]
    fn knn_one_step(
        &self,
        pending: &mut Vec<(T, &AnyKDT<'a, T, A>)>,
        top_k: &mut Vec<NB<T, A>>,
        k: usize,
        point: &[T],
        _: T,
        max_dist_bound: T,
        epsilon: T,
    ) {
        let current_max = if top_k.len() < k {
            max_dist_bound
        } else {
            top_k.last().unwrap().dist
        };
        let (dist_to_box, tree) = pending.pop().unwrap(); // safe
        if dist_to_box > current_max {
            return;
        }
        let mut current = tree;
        while !current.is_leaf() {
            let split_axis = current.split_axis.unwrap();
            let axis_value = current.split_axis_value.unwrap();
            let next = if point[split_axis] < axis_value {
                let next = current.right.as_ref().unwrap().as_ref();
                current = current.left.as_ref().unwrap().as_ref();
                next
            } else {
                let next = current.left.as_ref().unwrap().as_ref();
                current = current.right.as_ref().unwrap().as_ref();
                next
            };

            let dist_to_box =
                self.closest_dist_to_box(next.min_bounds.as_ref(), next.max_bounds.as_ref(), point); // (min dist from the box to point, the next Tree)
            if dist_to_box + epsilon < current_max {
                pending.push((dist_to_box, next));
            }
        }
        current.update_top_k(top_k, k, point, max_dist_bound);
    }

    #[inline(always)]
    fn within_one_step(
        &self,
        pending: &mut Vec<(T, &AnyKDT<'a, T, A>)>,
        neighbors: &mut Vec<NB<T, A>>,
        point: &[T],
        _: T,
        radius: T,
    ) {
        let (dist_to_box, tree) = pending.pop().unwrap(); // safe
        if dist_to_box > radius {
            return;
        }
        let mut current = tree;
        while !current.is_leaf() {
            let split_axis = current.split_axis.unwrap();
            let axis_value = current.split_axis_value.unwrap();
            let next = if point[split_axis] < axis_value {
                let next = current.right.as_ref().unwrap().as_ref();
                current = current.left.as_ref().unwrap().as_ref();
                next
            } else {
                let next = current.left.as_ref().unwrap().as_ref();
                current = current.right.as_ref().unwrap().as_ref();
                next
            };
            let dist_to_box =
                self.closest_dist_to_box(next.min_bounds.as_ref(), next.max_bounds.as_ref(), point); // (min dist from the box to point, the next Tree)
            if dist_to_box <= radius {
                pending.push((dist_to_box, next));
            }
        }
        current.update_nb_within(neighbors, point, radius);
    }

    #[inline(always)]
    fn within_count_one_step(
        &self,
        pending: &mut Vec<(T, &AnyKDT<'a, T, A>)>,
        point: &[T],
        _: T,
        radius: T,
    ) -> u32 {
        let (dist_to_box, tree) = pending.pop().unwrap(); // safe
        if dist_to_box > radius {
            0
        } else {
            let mut current = tree;
            while !current.is_leaf() {
                let split_axis = current.split_axis.unwrap();
                let axis_value = current.split_axis_value.unwrap();
                let next = if point[split_axis] < axis_value {
                    let next = current.right.as_ref().unwrap().as_ref();
                    current = current.left.as_ref().unwrap().as_ref();
                    next
                } else {
                    let next = current.left.as_ref().unwrap().as_ref();
                    current = current.right.as_ref().unwrap().as_ref();
                    next
                };

                let dist_to_box = self.closest_dist_to_box(
                    next.min_bounds.as_ref(),
                    next.max_bounds.as_ref(),
                    point,
                ); // (min dist from the box to point, the next Tree)
                if dist_to_box <= radius {
                    pending.push((dist_to_box, next));
                }
            }
            // Return the count in current
            current.data.unwrap().iter().fold(0u32, |acc, element| {
                let y = element.vec();
                let dist = self.lp.dist(y, point);
                acc + (dist <= radius) as u32
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::matrix_to_leaves;
    use super::*;
    use ndarray::{arr1, Array2, ArrayView1, ArrayView2};

    fn l1_dist_slice(a1: &[f64], a2: &[f64]) -> f64 {
        a1.iter()
            .zip(a2.iter())
            .fold(0., |acc, (x, y)| acc + (x - y).abs())
    }

    fn linf_dist_slice(a1: &[f64], a2: &[f64]) -> f64 {
        a1.iter()
            .zip(a2.iter())
            .fold(0., |acc, (x, y)| acc.max((x - y).abs()))
    }

    fn random_10d_rows() -> [f64; 10] {
        rand::random()
    }

    fn generate_test_answer(
        mat: ArrayView2<f64>,
        point: ArrayView1<f64>,
        dist_func: fn(&[f64], &[f64]) -> f64,
    ) -> (Vec<usize>, Vec<f64>) {
        let mut ans_distances = mat
            .rows()
            .into_iter()
            .map(|v| dist_func(v.to_slice().unwrap(), &point.to_vec()))
            .collect::<Vec<_>>();

        let mut ans_argmins = (0..mat.nrows()).collect::<Vec<_>>();
        ans_argmins.sort_by(|&i, &j| ans_distances[i].partial_cmp(&ans_distances[j]).unwrap());
        ans_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        (ans_argmins, ans_distances)
    }

    #[test]
    fn test_10d_knn_linf_dist_midpoint() {
        // 10 nearest neighbors, matrix of size 1000 x 10
        let k = 10usize;
        let mut v = Vec::new();
        let rows = 5_000usize;
        for _ in 0..rows {
            v.extend_from_slice(&random_10d_rows());
        }

        let mat = Array2::from_shape_vec((rows, 10), v).unwrap();
        let mat = mat.as_standard_layout().to_owned();
        let point = arr1(&[0.5; 10]);
        // brute force test
        let (ans_argmins, ans_distances) =
            generate_test_answer(mat.view(), point.view(), linf_dist_slice);

        let values = (0..rows).collect::<Vec<_>>();
        let binding = mat.view();
        let mut leaves = matrix_to_leaves(&binding, &values);

        let tree = AnyKDT::from_leaves(&mut leaves, SplitMethod::MIDPOINT, DIST::LINF).unwrap();

        let output = tree.knn(k, point.as_slice().unwrap(), 0f64);

        assert!(output.is_some());
        let output = output.unwrap();
        let indices = output.iter().map(|nb| nb.item).collect::<Vec<_>>();
        let distances = output.iter().map(|nb| nb.dist).collect::<Vec<_>>();

        assert_eq!(&ans_argmins[..k], &indices);
        for (d1, d2) in ans_distances[..k].iter().zip(distances.into_iter()) {
            assert!((d1 - d2).abs() < 1e-10);
        }
    }

    #[test]
    fn test_10d_knn_linf_dist_mean() {
        // 10 nearest neighbors, matrix of size 1000 x 10
        let k = 10usize;
        let mut v = Vec::new();
        let rows = 1_000usize;
        for _ in 0..rows {
            v.extend_from_slice(&random_10d_rows());
        }

        let mat = Array2::from_shape_vec((rows, 10), v).unwrap();
        let mat = mat.as_standard_layout().to_owned();
        let point = arr1(&[0.5; 10]);
        // brute force test
        let (ans_argmins, ans_distances) =
            generate_test_answer(mat.view(), point.view(), linf_dist_slice);

        let values = (0..rows).collect::<Vec<_>>();
        let binding = mat.view();
        let mut leaves = matrix_to_leaves(&binding, &values);

        let tree = AnyKDT::from_leaves(&mut leaves, SplitMethod::MEAN, DIST::LINF).unwrap();

        let output = tree.knn(k, point.as_slice().unwrap(), 0f64);

        assert!(output.is_some());
        let output = output.unwrap();
        let indices = output.iter().map(|nb| nb.item).collect::<Vec<_>>();
        let distances = output.iter().map(|nb| nb.dist).collect::<Vec<_>>();

        assert_eq!(&ans_argmins[..k], &indices);
        for (d1, d2) in ans_distances[..k].iter().zip(distances.into_iter()) {
            assert!((d1 - d2).abs() < 1e-10);
        }
    }

    #[test]
    fn test_10d_knn_linf_dist_median() {
        // 10 nearest neighbors, matrix of size 1000 x 10
        let k = 10usize;
        let mut v = Vec::new();
        let rows = 1_000usize;
        for _ in 0..rows {
            v.extend_from_slice(&random_10d_rows());
        }

        let mat = Array2::from_shape_vec((rows, 10), v).unwrap();
        let mat = mat.as_standard_layout().to_owned();
        let point = arr1(&[0.5; 10]);
        // brute force test
        let (ans_argmins, ans_distances) =
            generate_test_answer(mat.view(), point.view(), linf_dist_slice);

        let values = (0..rows).collect::<Vec<_>>();
        let binding = mat.view();
        let mut leaves = matrix_to_leaves(&binding, &values);

        let tree = AnyKDT::from_leaves(&mut leaves, SplitMethod::MEDIAN, DIST::LINF).unwrap();

        let output = tree.knn(k, point.as_slice().unwrap(), 0f64);

        assert!(output.is_some());
        let output = output.unwrap();
        let indices = output.iter().map(|nb| nb.item).collect::<Vec<_>>();
        let distances = output.iter().map(|nb| nb.dist).collect::<Vec<_>>();

        assert_eq!(&ans_argmins[..k], &indices);
        for (d1, d2) in ans_distances[..k].iter().zip(distances.into_iter()) {
            assert!((d1 - d2).abs() < 1e-10);
        }
    }

    #[test]
    fn test_10d_knn_l1_dist_midpoint() {
        // 10 nearest neighbors, matrix of size 1000 x 10
        let k = 10usize;
        let mut v = Vec::new();
        let rows = 1_000usize;
        for _ in 0..rows {
            v.extend_from_slice(&random_10d_rows());
        }

        let mat = Array2::from_shape_vec((rows, 10), v).unwrap();
        let mat = mat.as_standard_layout().to_owned();
        let point = arr1(&[0.5; 10]);
        // brute force test
        let (ans_argmins, ans_distances) =
            generate_test_answer(mat.view(), point.view(), l1_dist_slice);

        let values = (0..rows).collect::<Vec<_>>();
        let binding = mat.view();
        let mut leaves = matrix_to_leaves(&binding, &values);

        let tree = AnyKDT::from_leaves(&mut leaves, SplitMethod::MIDPOINT, DIST::L1).unwrap();

        let output = tree.knn(k, point.as_slice().unwrap(), 0f64);

        assert!(output.is_some());
        let output = output.unwrap();
        let indices = output.iter().map(|nb| nb.item).collect::<Vec<_>>();
        let distances = output.iter().map(|nb| nb.dist).collect::<Vec<_>>();

        assert_eq!(&ans_argmins[..k], &indices);
        for (d1, d2) in ans_distances[..k].iter().zip(distances.into_iter()) {
            assert!((d1 - d2).abs() < 1e-10);
        }
    }

    #[test]
    fn test_10d_knn_l1_dist_mean() {
        // 10 nearest neighbors, matrix of size 1000 x 10
        let k = 10usize;
        let mut v = Vec::new();
        let rows = 1_000usize;
        for _ in 0..rows {
            v.extend_from_slice(&random_10d_rows());
        }

        let mat = Array2::from_shape_vec((rows, 10), v).unwrap();
        let mat = mat.as_standard_layout().to_owned();
        let point = arr1(&[0.5; 10]);
        // brute force test
        let (ans_argmins, ans_distances) =
            generate_test_answer(mat.view(), point.view(), l1_dist_slice);

        let values = (0..rows).collect::<Vec<_>>();
        let binding = mat.view();
        let mut leaves = matrix_to_leaves(&binding, &values);

        let tree = AnyKDT::from_leaves(&mut leaves, SplitMethod::MEAN, DIST::L1).unwrap();

        let output = tree.knn(k, point.as_slice().unwrap(), 0f64);

        assert!(output.is_some());
        let output = output.unwrap();
        let indices = output.iter().map(|nb| nb.item).collect::<Vec<_>>();
        let distances = output.iter().map(|nb| nb.dist).collect::<Vec<_>>();

        assert_eq!(&ans_argmins[..k], &indices);
        for (d1, d2) in ans_distances[..k].iter().zip(distances.into_iter()) {
            assert!((d1 - d2).abs() < 1e-10);
        }
    }

    #[test]
    fn test_10d_knn_l1_dist_median() {
        // 10 nearest neighbors, matrix of size 1000 x 10
        let k = 10usize;
        let mut v = Vec::new();
        let rows = 1_000usize;
        for _ in 0..rows {
            v.extend_from_slice(&random_10d_rows());
        }

        let mat = Array2::from_shape_vec((rows, 10), v).unwrap();
        let mat = mat.as_standard_layout().to_owned();
        let point = arr1(&[0.5; 10]);
        // brute force test
        let (ans_argmins, ans_distances) =
            generate_test_answer(mat.view(), point.view(), l1_dist_slice);

        let values = (0..rows).collect::<Vec<_>>();
        let binding = mat.view();
        let mut leaves = matrix_to_leaves(&binding, &values);

        let tree = AnyKDT::from_leaves(&mut leaves, SplitMethod::MEDIAN, DIST::L1).unwrap();

        let output = tree.knn(k, point.as_slice().unwrap(), 0f64);

        assert!(output.is_some());
        let output = output.unwrap();
        let indices = output.iter().map(|nb| nb.item).collect::<Vec<_>>();
        let distances = output.iter().map(|nb| nb.dist).collect::<Vec<_>>();

        assert_eq!(&ans_argmins[..k], &indices);
        for (d1, d2) in ans_distances[..k].iter().zip(distances.into_iter()) {
            assert!((d1 - d2).abs() < 1e-10);
        }
    }
}
