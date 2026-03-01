use crate::arkadia::KNNRegressor;

/// A Kdtree
use super::{
    leaf::KdLeaf,
    KNNDist,
    suggest_capacity, Leaf, SpatialQueries, SplitMethod, NB,
};
use num::Float;
use std::usize;

/// This checks the closest distance from point to the boundaries of the box (subtree),
/// which can help us skip entire boxes.
#[inline]
fn _closest_dist_to_box(bounds: &[f64], point: &[f64], dim: usize, d: KNNDist) -> f64 {
    let mut dist = 0f64;
    match d {
        KNNDist::L1 => {
            for i in 0..point.len() {
                if point[i] > bounds[i + dim] {
                    dist += (point[i] - bounds[i + dim]).abs();
                } else if point[i] < bounds[i] {
                    dist += dist + (point[i] - bounds[i]).abs();
                }
            }
            dist
        }

        KNNDist::L2 => {
            for i in 0..point.len() {
                if point[i] > bounds[i + dim] {
                    dist += (point[i] - bounds[i + dim]).powi(2);
                } else if point[i] < bounds[i] {
                    dist += (point[i] - bounds[i]).powi(2);
                }
            }
            dist.sqrt()
        }

        KNNDist::SQL2 => {
            for i in 0..point.len() {
                if point[i] > bounds[i + dim] {
                    dist += (point[i] - bounds[i + dim]).powi(2);
                } else if point[i] < bounds[i] {
                    dist += (point[i] - bounds[i]).powi(2);
                }
            }
            dist
        }

        KNNDist::LINF => {
            for i in 0..point.len() {
                if point[i] > bounds[i + dim] {
                    dist = dist.max((point[i] - bounds[i + dim]).abs());
                } else if point[i] < bounds[i] {
                    dist = dist.max((point[i] - bounds[i]).abs());
                }
            }
            dist
        }
    }
}

pub struct KDT<'a, A> {
    pub dim: usize,
    pub capacity: usize,
    // Nodes
    left: Option<Box<KDT<'a, A>>>,
    right: Option<Box<KDT<'a, A>>>,
    // Is a leaf node if this has valid values
    split_axis: usize,
    split_axis_value: f64,
    // vec of len 2 * dim. First dim values are mins in each dim, second dim values are maxs
    bounds: Vec<f64>,
    // Data
    data: Vec<Leaf<'a, f64, A>>, // Not empty when this is a leaf
    //
    d: KNNDist,
}

impl<'a, A: Copy> KDT<'a, A> {
    // Helper function that finds the bounding box for each (sub)kdtree
    // Vec of length 2 * dim. First dim values are the mins, and the rest are maxes
    fn find_bounds(data: &[Leaf<'a, f64, A>], dim: usize) -> Vec<f64> {
        let mut bounds = vec![f64::max_value(); dim];
        bounds.extend(std::iter::repeat(f64::min_value()).take(dim));
        for elem in data.iter() {
            for i in 0..dim {
                bounds[i] = bounds[i].min(elem.value_at(i));
                bounds[dim + i] = bounds[dim + i].max(elem.value_at(i));
            }
        }
        bounds
    }

    pub fn from_leaves(data: &'a mut [Leaf<'a, f64, A>], d: KNNDist) -> Result<Self, String> {
        if data.is_empty() {
            return Err("Empty data.".into());
        }
        let dim = data[0].dim();
        let mut tree = KDT::new_empty(dim, suggest_capacity(dim), d);
        for leaf in data.iter().copied() {
            if leaf.dim() != dim {
                return Err("Dimension isn't consistent.".into());
            } else {
                tree.add(leaf)?;
            }
        }
        Ok(tree)
    }

    pub fn from_leaves_unchecked(data: &'a mut [Leaf<'a, f64, A>], d: KNNDist) -> Self {
        let dim = data[0].dim();
        let mut tree = KDT::new_empty(dim, suggest_capacity(dim), d);
        for leaf in data.iter().copied() {
            tree.add_unchecked(leaf, 0);
        }
        tree
    }

    pub fn new_empty(dim: usize, capacity: usize, d: KNNDist) -> Self {
        let mut bounds = vec![f64::MAX; dim];
        bounds.extend(std::iter::repeat(f64::min_value()).take(dim));
        KDT {
            dim: dim,
            capacity: capacity,
            left: None,
            right: None,
            split_axis: usize::MAX,
            split_axis_value: f64::nan(),
            bounds: bounds,
            data: vec![],
            d: d,
        }
    }

    /// Creates a new leaf node out of the data.
    pub fn grow_new_leaf(&self, data: Vec<Leaf<'a, f64, A>>) -> Self {
        let bounds = Self::find_bounds(&data, self.dim);
        KDT {
            dim: self.dim,
            capacity: self.capacity,
            left: None,
            right: None,
            split_axis: usize::MAX,
            split_axis_value: f64::nan(),
            bounds: bounds,
            data: data,
            d: self.d,
        }
    }

    fn is_leaf(&self) -> bool {
        !(self.left.is_some() || self.right.is_some())
    }

    /// Updates the bounds according to the new leaf
    fn update_bounds(&mut self, leaf: &Leaf<'a, f64, A>) {
        for i in 0..self.dim {
            self.bounds[i] = self.bounds[i].min(leaf.value_at(i));
            self.bounds[i + self.dim] = self.bounds[i + self.dim].max(leaf.value_at(i));
        }
    }

    /// Updates the bounds and push to new leaf to the data vec.
    fn update_and_push(&mut self, leaf: Leaf<'a, f64, A>) {
        for i in 0..self.dim {
            self.bounds[i] = self.bounds[i].min(leaf.value_at(i));
            self.bounds[i + self.dim] = self.bounds[i + self.dim].max(leaf.value_at(i));
        }
        self.data.push(leaf);
    }

    /// Attach a new point (leaf) to the Kdtree. This disregards capacity and
    /// will not further split the leaf if capacity is reached. It is recommended
    /// to use this if most the data have been already ingested in bulk and you
    /// are only adding a few more points. Attaching too much can be
    /// bad for performance in lower dimensions.
    pub fn attach(&mut self, leaf: Leaf<'a, f64, A>) -> Result<(), String> {
        if leaf.dim() != self.dim {
            Err("Dimension does not match.".into())
        } else {
            Ok(self.attach_unchecked(leaf))
        }
    }

    #[inline(always)]
    pub fn attach_unchecked(&mut self, leaf: Leaf<'a, f64, A>) {
        if self.is_leaf() {
            self.update_and_push(leaf);
        } else {
            if leaf.value_at(self.split_axis) < self.split_axis_value {
                self.left.as_mut().unwrap().attach_unchecked(leaf)
            } else {
                self.right.as_mut().unwrap().attach_unchecked(leaf)
            }
        }
    }

    /// Add a new point (leaf) to the Kdtree. This will further split the kdtree
    /// if capacity is reached in the leaf node.
    pub fn add(&mut self, leaf: Leaf<'a, f64, A>) -> Result<(), String> {
        if leaf.dim() != self.dim {
            Err("Dimension does not match.".into())
        } else {
            Ok(self.add_unchecked(leaf, 0))
        }
    }

    #[inline(always)]
    pub fn add_unchecked(&mut self, leaf: Leaf<'a, f64, A>, depth: usize) {
        if self.is_leaf() {
            // Always update
            self.update_and_push(leaf);
            if self.data.len() > self.capacity {
                // The bounds are updated. New leaf is pushed to self.data
                // this is a copy of all content in self.data, which, by now, should contain the new leaf
                let mut new_data = self.data.split_off(0);

                let axis = depth % self.dim;
                let midpoint = self.bounds[axis]
                    + (self.bounds[axis + self.dim] - self.bounds[axis]) / (2f64);

                // True will go right, false go left
                new_data.sort_unstable_by_key(|leaf| leaf.value_at(axis) >= midpoint);
                let split_idx = new_data.partition_point(|elem| elem.value_at(axis) < midpoint);

                let (left, right) = new_data.split_at(split_idx);

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
                    self.data = right.to_vec();
                } else {
                    // Actually split this leaf into two. This leaf will become a non-leaf node.
                    // update self.split_axis_value
                    self.split_axis_value = midpoint;
                    // update self.split_axis
                    self.split_axis = axis;
                    // empty out self.data's capacity. This should have len 0 and capacity 0 by now.
                    self.data.shrink_to_fit();
                    self.left = Some(Box::new(self.grow_new_leaf(left.to_vec())));
                    self.right = Some(Box::new(self.grow_new_leaf(right.to_vec())));
                }
            }
        } else {
            self.update_bounds(&leaf);
            if leaf.value_at(self.split_axis) < self.split_axis_value {
                self.left.as_mut().unwrap().add_unchecked(leaf, depth + 1);
            } else {
                self.right.as_mut().unwrap().add_unchecked(leaf, depth + 1);
            }
        }
    }

    fn update_top_k(
        &self,
        top_k: &mut Vec<NB<f64, A>>,
        k: usize,
        point: &[f64],
        current_max: f64,
        max_dist_bound: f64,
    ) {
        let mut cur_max = current_max;
        for element in self.data.iter() {
            let dist = self.d.dist(element.row_vec, point);
            if dist <= max_dist_bound && (dist < cur_max || top_k.len() < k) {
                let idx = top_k.partition_point(|s| s.dist <= dist);
                top_k.insert(
                    idx,
                    NB {
                        dist: dist,
                        item: element.item,
                    },
                );
                if top_k.len() > k {
                    top_k.pop();
                }
                cur_max = cur_max.max(dist);
            }
        }
    }

    // #[inline(always)]
    fn update_nb_within(&self, neighbors: &mut Vec<NB<f64, A>>, point: &[f64], radius: f64) {
        for element in self.data.iter() {
            let y = element.row_vec;
            let dist = self.d.dist(y, point);
            if dist <= radius {
                neighbors.push(NB {
                    dist: dist,
                    item: element.item,
                });
            }
        }
    }
}

impl<'a, A: Copy> SpatialQueries<'a, A>
    for KDT<'a, A>
{
    fn dim(&self) -> usize {
        self.dim
    }

    // #[inline(always)]
    fn knn_one_step(
        &self,
        pending: &mut Vec<(f64, &KDT<'a, A>)>,
        top_k: &mut Vec<NB<f64, A>>,
        k: usize,
        point: &[f64],
        max_dist_bound: f64,
        epsilon: f64,
    ) {
        // k > 0 is guaranteed.
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
            let next = if point[current.split_axis] < current.split_axis_value {
                let next = current.right.as_ref().unwrap().as_ref();
                current = current.left.as_ref().unwrap().as_ref();
                next
            } else {
                let next = current.left.as_ref().unwrap().as_ref();
                current = current.right.as_ref().unwrap().as_ref();
                next
            };

            let dist_to_box = _closest_dist_to_box(&next.bounds, point, self.dim, self.d); // (min dist from the box to point, the next Tree)
            if dist_to_box + epsilon < current_max {
                pending.push((dist_to_box, next));
            }
        }
        current.update_top_k(top_k, k, point, current_max, max_dist_bound);
    }

    #[inline(always)]
    fn within_one_step(
        &self,
        pending: &mut Vec<(f64, &KDT<'a, A>)>,
        neighbors: &mut Vec<NB<f64, A>>,
        point: &[f64],
        radius: f64,
    ) {
        let (dist_to_box, tree) = pending.pop().unwrap(); // safe
        if dist_to_box > radius {
            return;
        }
        let mut current = tree;
        while !current.is_leaf() {
            let next = if point[current.split_axis] < current.split_axis_value {
                let next = current.right.as_ref().unwrap().as_ref();
                current = current.left.as_ref().unwrap().as_ref();
                next
            } else {
                let next = current.left.as_ref().unwrap().as_ref();
                current = current.right.as_ref().unwrap().as_ref();
                next
            };
            let dist_to_box = _closest_dist_to_box(&next.bounds, point, self.dim, self.d); // (min dist from the box to point, the next Tree)
            if dist_to_box <= radius {
                pending.push((dist_to_box, next));
            }
        }
        current.update_nb_within(neighbors, point, radius);
    }

    #[inline(always)]
    fn within_count_one_step(
        &self,
        pending: &mut Vec<(f64, &KDT<'a, A>)>,
        point: &[f64],
        radius: f64,
    ) -> u32 {
        let (dist_to_box, tree) = pending.pop().unwrap(); // safe
        if dist_to_box > radius {
            0
        } else {
            let mut current = tree;
            while !current.is_leaf() {
                let split_axis = current.split_axis;
                let axis_value = current.split_axis_value;
                let next = if point[split_axis] < axis_value {
                    let next = current.right.as_ref().unwrap().as_ref();
                    current = current.left.as_ref().unwrap().as_ref();
                    next
                } else {
                    let next = current.left.as_ref().unwrap().as_ref();
                    current = current.right.as_ref().unwrap().as_ref();
                    next
                };

                let dist_to_box = _closest_dist_to_box(&next.bounds, point, self.dim, self.d); // (min dist from the box to point, the next Tree)
                if dist_to_box <= radius {
                    pending.push((dist_to_box, next));
                }
            }
            // Return the count in current
            current.data.iter().fold(0u32, |acc, element| {
                let y = element.vec();
                let dist = self.d.dist(y, point);
                acc + (dist <= radius) as u32
            })
        }
    }
}

impl <'a, A: Float + Into<f64>> KNNRegressor<'a, A> for KDT<'a, A> {}

#[cfg(test)]
mod tests {
    use crate::arkadia::slice_to_leaves;

    use super::*;

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

    pub fn squared_l2(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .fold(0f64, |acc, (&a, &b)| acc + (a - b) * (a - b))
    }

    fn random_10d_rows() -> [f64; 10] {
        rand::random()
    }

    fn generate_test_answer(
        data: &[f64],
        row_size: usize,
        point: &[f64],
        dist_func: fn(&[f64], &[f64]) -> f64,
    ) -> (Vec<usize>, Vec<f64>) {
        let mut ans_distances = data
            .chunks_exact(row_size)
            .map(|v| dist_func(v, point))
            .collect::<Vec<_>>();

        let nrows = data.len() / row_size;
        let mut ans_argmins = (0..nrows).collect::<Vec<_>>();
        ans_argmins.sort_by(|&i, &j| ans_distances[i].partial_cmp(&ans_distances[j]).unwrap());
        ans_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        (ans_argmins, ans_distances)
    }

    #[test]
    fn test_10d_knn_linf_dist() {
        // 10 nearest neighbors, matrix of size 1000 x 10
        let k = 10usize;
        let mut v = Vec::new();
        let rows = 5_000usize;
        for _ in 0..rows {
            v.extend_from_slice(&random_10d_rows());
        }

        let point = [0.5; 10];
        // brute force test
        let (ans_argmins, ans_distances) = generate_test_answer(&v, 10, &point, linf_dist_slice);

        let values = (0..rows).collect::<Vec<_>>();
        let mut leaves = slice_to_leaves(&v, 10, &values);

        let tree = KDT::from_leaves(&mut leaves, KNNDist::LINF).unwrap();

        let output = tree.knn(k, &point, 0f64);

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
    fn test_10d_knn_l1_dist() {
        // 10 nearest neighbors, matrix of size 1000 x 10
        let k = 10usize;
        let mut v = Vec::new();
        let rows = 5_000usize;
        for _ in 0..rows {
            v.extend_from_slice(&random_10d_rows());
        }

        let point = [0.5; 10];
        // brute force test
        let (ans_argmins, ans_distances) = generate_test_answer(&v, 10, &point, l1_dist_slice);

        let values = (0..rows).collect::<Vec<_>>();
        let mut leaves = slice_to_leaves(&v, 10, &values);

        let tree = KDT::from_leaves(&mut leaves, KNNDist::L1).unwrap();

        let output = tree.knn(k, &point, 0f64);

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
    fn test_10d_knn_l2_dist() {
        // 10 nearest neighbors, matrix of size 1000 x 10
        let k = 10usize;
        let mut v = Vec::new();
        let rows = 5_000usize;
        for _ in 0..rows {
            v.extend_from_slice(&random_10d_rows());
        }

        let point = [0.5; 10];
        // brute force test
        let (ans_argmins, ans_distances) = generate_test_answer(&v, 10, &point, squared_l2);

        let values = (0..rows).collect::<Vec<_>>();
        let mut leaves = slice_to_leaves(&v, 10, &values);

        let tree = KDT::from_leaves(&mut leaves, KNNDist::SQL2).unwrap();

        let output = tree.knn(k, &point, 0f64);

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
