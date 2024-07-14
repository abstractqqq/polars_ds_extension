/// Norm cached KDT
use crate::arkadia::{
    leaf::{KdLeaf, LeafWithNorm},
    suggest_capacity, SplitMethod, KDTQ, NB,
};
use ndarray::ArrayView1;
use num::Float;

// We can also include cosine distance here because the norm trick works too

#[inline(always)]
pub fn squared_l2<T: Float + 'static>(a: &[T], b: &[T], a_norm: T, b_norm: T) -> T {
    let aa = ArrayView1::from(a);
    let bb = ArrayView1::from(b);
    let dot = aa.dot(&bb);
    a_norm + b_norm - dot - dot
}

pub struct KDT<'a, T: Float + 'static, A> {
    dim: usize,
    // Nodes
    left: Option<Box<KDT<'a, T, A>>>,
    right: Option<Box<KDT<'a, T, A>>>,
    // Is a leaf node if this has values
    split_axis: Option<usize>,
    split_axis_value: Option<T>,
    min_bounds: Vec<T>,
    max_bounds: Vec<T>,
    // Data
    data: Option<&'a [LeafWithNorm<'a, T, A>]>, // Not none when this is a leaf
}

impl<'a, T: Float + 'static, A: Copy> KDT<'a, T, A> {
    // Add method to create the tree by adding leaf elements one by one

    pub fn from_leaves(
        data: &'a mut [LeafWithNorm<'a, T, A>],
        how: SplitMethod,
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
        ))
    }

    pub fn with_capacity(
        data: &'a mut [LeafWithNorm<'a, T, A>],
        capacity: usize,
        how: SplitMethod,
    ) -> Result<Self, String> {
        if data.is_empty() {
            return Err("Empty data.".into());
        }
        let dim = data.last().unwrap().dim();
        if capacity == 0 {
            return Err("Zero capacity.".into());
        }

        Ok(Self::from_leaves_unchecked(data, dim, capacity, 0, how))
    }

    fn from_leaves_unchecked(
        data: &'a mut [LeafWithNorm<'a, T, A>],
        dim: usize,
        capacity: usize,
        depth: usize,
        how: SplitMethod,
    ) -> Self {
        let n = data.len();
        let (min_bounds, max_bounds) = Self::find_bounds(data, dim);
        if n <= capacity {
            KDT {
                dim: dim,
                left: None,
                right: None,
                split_axis: None,
                split_axis_value: None,
                min_bounds: min_bounds,
                max_bounds: max_bounds,
                data: Some(data),
            }
        } else {
            let axis = depth % dim;
            let (split_axis_value, split_idx) = match how {
                SplitMethod::MIDPOINT => {
                    let midpoint = min_bounds[axis]
                        + (max_bounds[axis] - min_bounds[axis]) / (T::one() + T::one());
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
                    }); // False <<< True. Now split by the first True location
                    let half = n >> 1;
                    let split_value = data[half].value_at(axis);
                    (split_value, half)
                }
            };

            let (left, right) = data.split_at_mut(split_idx);
            if left.is_empty() {
                // See comment in arkadia_lp.rs
                KDT {
                    dim: dim,
                    left: None,
                    right: None,
                    split_axis: None,
                    split_axis_value: None,
                    min_bounds: min_bounds,
                    max_bounds: max_bounds,
                    data: Some(right),
                }
            } else {
                KDT {
                    dim: dim,
                    left: Some(Box::new(Self::from_leaves_unchecked(
                        left,
                        dim,
                        capacity,
                        depth + 1,
                        how.clone(),
                    ))),
                    right: Some(Box::new(Self::from_leaves_unchecked(
                        right,
                        dim,
                        capacity,
                        depth + 1,
                        how,
                    ))),
                    split_axis: Some(axis),
                    split_axis_value: Some(split_axis_value),
                    min_bounds: min_bounds,
                    max_bounds: max_bounds,
                    data: None,
                }
            }
        }
    }

    fn is_leaf(&self) -> bool {
        self.data.is_some()
    }

    // Computes the distance from the closest potential point in box to P
    #[inline(always)]
    fn closest_dist_to_box(min_bounds: &[T], max_bounds: &[T], point: &[T]) -> T {
        let mut dist = T::zero();
        for i in 0..point.len() {
            if point[i] > max_bounds[i] {
                dist = dist + (point[i] - max_bounds[i]).powi(2);
            } else if point[i] < min_bounds[i] {
                dist = dist + (point[i] - min_bounds[i]).powi(2);
            }
        }
        dist
    }

    #[inline(always)]
    fn update_top_k(
        &self,
        top_k: &mut Vec<NB<T, A>>,
        k: usize,
        point: &[T],
        point_norm_cache: T,
        max_dist_bound: T,
    ) {
        let max_permissible_dist = T::max_value().min(max_dist_bound);
        // This is only called if is_leaf. Safe to unwrap.
        for element in self.data.unwrap().iter() {
            let cur_max_dist = top_k.last().map(|nb| nb.dist).unwrap_or(max_dist_bound);
            let y = element.vec();
            // This hack actually produces faster Euclidean dist calculation (because .dot is unrolled in ndarray)
            let dist = squared_l2(y, point, element.norm(), point_norm_cache);
            if dist < cur_max_dist || (top_k.len() < k && dist <= max_permissible_dist) {
                let nb = NB {
                    dist: dist,
                    item: element.item,
                };
                let idx: usize = top_k.partition_point(|s| s <= &nb);
                if idx < top_k.len() {
                    if top_k.len() + 1 > k {
                        top_k.pop();
                    } // This ensures top_k has k elements and no need to allocate
                    top_k.insert(idx, nb);
                } else if top_k.len() < k {
                    // Note if idx < top_k.len() is false, then index == top_k.len() for free!
                    // by the end of the push, len is at most k.
                    top_k.push(nb);
                } // Do nothing if idx >= k, because that means no partition point was found within existing values
            }
            // No need to check if we already have k elements and the dist is >= current max dist
        }
    }

    #[inline(always)]
    fn update_nb_within(
        &self,
        neighbors: &mut Vec<NB<T, A>>,
        point: &[T],
        point_norm_cache: T,
        radius: T,
    ) {
        // This is only called if is_leaf. Safe to unwrap.
        for element in self.data.unwrap().iter() {
            let y = element.vec();
            let dist = squared_l2(y, point, element.norm(), point_norm_cache);
            if dist <= radius {
                neighbors.push(NB {
                    dist: dist,
                    item: element.item,
                });
            }
        }
    }
}

impl<'a, T: Float + 'static, A: Copy> KDTQ<'a, T, A> for KDT<'a, T, A> {
    fn dim(&self) -> usize {
        self.dim
    }

    #[inline(always)]
    fn knn_one_step(
        &self,
        pending: &mut Vec<(T, &KDT<'a, T, A>)>,
        top_k: &mut Vec<NB<T, A>>,
        k: usize,
        point: &[T],
        point_norm_cache: T,
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
            let dist_to_box = Self::closest_dist_to_box(
                next.min_bounds.as_ref(),
                next.max_bounds.as_ref(),
                point,
            ); // (min dist from the box to point, the next Tree)
            if dist_to_box + epsilon < current_max {
                pending.push((dist_to_box, next));
            }
        }
        current.update_top_k(top_k, k, point, point_norm_cache, max_dist_bound);
    }

    #[inline(always)]
    fn within_count_one_step(
        &self,
        pending: &mut Vec<(T, &KDT<'a, T, A>)>,
        point: &[T],
        point_norm_cache: T,
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
                let dist_to_box = Self::closest_dist_to_box(
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
                let dist = squared_l2(y, point, element.norm(), point_norm_cache);
                acc + (dist <= radius) as u32
            })
        }
    }

    #[inline(always)]
    fn within_one_step(
        &self,
        pending: &mut Vec<(T, &KDT<'a, T, A>)>,
        neighbors: &mut Vec<NB<T, A>>,
        point: &[T],
        point_norm_cache: T,
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
            let dist_to_box = Self::closest_dist_to_box(
                next.min_bounds.as_ref(),
                next.max_bounds.as_ref(),
                point,
            ); // (min dist from the box to point, the next Tree)
            if dist_to_box <= radius {
                pending.push((dist_to_box, next));
            }
        }
        current.update_nb_within(neighbors, point, point_norm_cache, radius);
    }

    fn knn(&self, k: usize, point: &[T], epsilon: T) -> Option<Vec<NB<T, A>>> {
        if k == 0 || (point.len() != self.dim()) || (point.iter().any(|x| !x.is_finite())) {
            None
        } else {
            // Always allocate 1 more.
            let mut top_k = Vec::with_capacity(k + 1);
            let mut pending = Vec::with_capacity(k + 1);
            pending.push((T::min_value(), self));
            let p_arr = ArrayView1::from(point);
            let point_norm = p_arr.dot(&p_arr);
            while !pending.is_empty() {
                self.knn_one_step(
                    &mut pending,
                    &mut top_k,
                    k,
                    point,
                    point_norm,
                    T::max_value(),
                    epsilon,
                );
            }
            Some(top_k)
        }
    }

    // For bounded, epsilon is 0
    fn knn_bounded(&self, k: usize, point: &[T], max_dist_bound: T) -> Option<Vec<NB<T, A>>> {
        if k == 0
            || (point.len() != self.dim())
            || (point.iter().any(|x| !x.is_finite()))
            || max_dist_bound <= T::zero() + T::epsilon()
        {
            None
        } else {
            // Always allocate 1 more.
            let mut top_k = Vec::with_capacity(k + 1);
            let mut pending = Vec::with_capacity(k + 1);
            pending.push((T::min_value(), self));
            let p_arr = ArrayView1::from(point);
            let point_norm = p_arr.dot(&p_arr);
            while !pending.is_empty() {
                self.knn_one_step(
                    &mut pending,
                    &mut top_k,
                    k,
                    point,
                    point_norm,
                    max_dist_bound,
                    T::zero(),
                );
            }
            Some(top_k)
        }
    }

    fn within(&self, point: &[T], radius: T, sort: bool) -> Option<Vec<NB<T, A>>> {
        // radius is actually squared radius
        if radius <= T::zero() + T::epsilon() || (point.iter().any(|x| !x.is_finite())) {
            None
        } else {
            // Always allocate some.
            let mut neighbors = Vec::with_capacity(32);
            let mut pending = Vec::with_capacity(32);
            let p_arr = ArrayView1::from(point);
            let point_norm = p_arr.dot(&p_arr);
            pending.push((T::min_value(), self));
            while !pending.is_empty() {
                self.within_one_step(&mut pending, &mut neighbors, point, point_norm, radius);
            }
            if sort {
                neighbors.sort_unstable();
            }
            neighbors.shrink_to_fit();
            Some(neighbors)
        }
    }

    fn within_count(&self, point: &[T], radius: T) -> Option<u32> {
        // radius is actually squared radius
        if radius <= T::zero() + T::epsilon() || (point.iter().any(|x| !x.is_finite())) {
            None
        } else {
            // Always allocate some.
            let mut cnt = 0u32;
            let mut pending = Vec::with_capacity(32);
            let p_arr = ArrayView1::from(point);
            let point_norm = p_arr.dot(&p_arr);
            pending.push((T::min_value(), self));
            while !pending.is_empty() {
                cnt += self.within_count_one_step(&mut pending, point, point_norm, radius);
            }
            Some(cnt)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::matrix_to_leaves_w_norm;
    use super::*;
    use ndarray::{arr1, Array2, ArrayView2};

    pub fn squared_l2<T: Float + 'static>(a: &[T], b: &[T]) -> T {
        a.iter()
            .zip(b.iter())
            .fold(T::zero(), |acc, (&a, &b)| acc + (a - b) * (a - b))
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
    fn test_10d_knn_l2_dist_midpoint() {
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
            generate_test_answer(mat.view(), point.view(), squared_l2::<f64>);

        let values = (0..rows).collect::<Vec<_>>();
        let binding = mat.view();
        let mut leaf_elements = matrix_to_leaves_w_norm(&binding, &values);

        let tree = KDT::from_leaves(&mut leaf_elements, SplitMethod::MIDPOINT).unwrap();

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
    fn test_10d_knn_l2_dist_bounded() {
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
            generate_test_answer(mat.view(), point.view(), squared_l2::<f64>);

        let bound: f64 = 0.28; // usually random data will give min dist ~ 0.24, 0.25
        let idx = ans_distances.partition_point(|d| d < &bound);
        let (mut ans_argmins, _) = ans_argmins.split_at(idx);
        let (mut ans_distances, _) = ans_distances.split_at(idx);
        ans_argmins = &ans_argmins[..(k).min(ans_argmins.len())];
        ans_distances = &ans_distances[..(k).min(ans_distances.len())];

        let values = (0..rows).collect::<Vec<_>>();
        let binding = mat.view();
        let mut leaf_elements = matrix_to_leaves_w_norm(&binding, &values);

        let tree = KDT::from_leaves(&mut leaf_elements, SplitMethod::MIDPOINT).unwrap();

        let output = tree.knn_bounded(k, point.as_slice().unwrap(), bound);

        assert!(output.is_some());
        let output = output.unwrap();
        let indices = output.iter().map(|nb| nb.item).collect::<Vec<_>>();
        let distances = output.iter().map(|nb| nb.dist).collect::<Vec<_>>();

        // May have <= k elements
        assert_eq!(ans_argmins.len(), indices.len());
        assert_eq!(ans_distances.len(), distances.len());
        assert_eq!(&ans_argmins, &indices);
        for (d1, d2) in ans_distances.iter().zip(distances.into_iter()) {
            assert!((d1 - d2).abs() < 1e-10);
        }
    }

    #[test]
    fn test_10d_knn_l2_dist_within() {
        // 10 nearest neighbors, matrix of size 1000 x 10
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
            generate_test_answer(mat.view(), point.view(), squared_l2::<f64>);

        let bound: f64 = 0.29; // usually random data will give min dist ~ 0.24, 0.25
        let idx = ans_distances.partition_point(|d| d < &bound);
        let (ans_argmins, _) = ans_argmins.split_at(idx);
        let (ans_distances, _) = ans_distances.split_at(idx);

        let values = (0..rows).collect::<Vec<_>>();
        let binding = mat.view();
        let mut leaf_elements = matrix_to_leaves_w_norm(&binding, &values);

        let tree = KDT::from_leaves(&mut leaf_elements, SplitMethod::MIDPOINT).unwrap();

        let output = tree.within(point.as_slice().unwrap(), bound, true);

        assert!(output.is_some());
        let output = output.unwrap();
        let indices = output.iter().map(|nb| nb.item).collect::<Vec<_>>();
        let distances = output.iter().map(|nb| nb.dist).collect::<Vec<_>>();

        // May have <= k elements
        assert_eq!(ans_argmins.len(), indices.len());
        assert_eq!(ans_distances.len(), distances.len());
        assert_eq!(&ans_argmins, &indices);
        for (d1, d2) in ans_distances.iter().zip(distances.into_iter()) {
            assert!((d1 - d2).abs() < 1e-10);
        }
    }

    #[test]
    fn test_10d_knn_l2_dist_within_count() {
        // 10 nearest neighbors, matrix of size 1000 x 10
        let mut v = Vec::new();
        let rows = 1_000usize;
        for _ in 0..rows {
            v.extend_from_slice(&random_10d_rows());
        }

        let mat = Array2::from_shape_vec((rows, 10), v).unwrap();
        let mat = mat.as_standard_layout().to_owned();
        let point = arr1(&[0.5; 10]);
        // brute force test
        let (_, ans_distances) = generate_test_answer(mat.view(), point.view(), squared_l2::<f64>);

        let bound: f64 = 0.29; // usually random data will give min dist ~ 0.24, 0.25
        let idx = ans_distances.partition_point(|d| d < &bound);

        let values = (0..rows).collect::<Vec<_>>();
        let binding = mat.view();
        let mut leaf_elements = matrix_to_leaves_w_norm(&binding, &values);

        let tree = KDT::from_leaves(&mut leaf_elements, SplitMethod::MIDPOINT).unwrap();

        let output = tree.within_count(point.as_slice().unwrap(), bound);

        // May have <= k elements
        assert!(output.is_some());
        assert_eq!(idx as u32, output.unwrap());
    }

    #[test]
    fn test_10d_knn_l2_dist_mean() {
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
            generate_test_answer(mat.view(), point.view(), squared_l2::<f64>);

        let values = (0..rows).collect::<Vec<_>>();
        let binding = mat.view();
        let mut leaf_elements = matrix_to_leaves_w_norm(&binding, &values);

        let tree = KDT::from_leaves(&mut leaf_elements, SplitMethod::MEAN).unwrap();

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
    fn test_10d_knn_l2_dist_median() {
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
            generate_test_answer(mat.view(), point.view(), squared_l2::<f64>);

        let values = (0..rows).collect::<Vec<_>>();
        let binding = mat.view();
        let mut leaf_elements = matrix_to_leaves_w_norm(&binding, &values);

        let tree = KDT::from_leaves(&mut leaf_elements, SplitMethod::MEDIAN).unwrap();

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
