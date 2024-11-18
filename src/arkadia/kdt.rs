/// A Kdtree
use super::{
    leaf::{KdLeaf, OwnedLeaf},
    suggest_capacity, KNNRegressor, Leaf, SpatialQueries, SplitMethod, NB,
};
use crate::utils::DIST;
use cfavml::safe_trait_distance_ops::DistanceOps;
use num::Float;
use std::{fmt::Debug, usize};

/// This checks the closest distance from point to the boundaries of the box (subtree),
/// which can help us skip entire boxes.
#[inline]
fn _closest_dist_to_box<T: Float>(bounds: &[T], point: &[T], dim: usize, d: DIST<T>) -> T {
    let mut dist = T::zero();
    match d {
        DIST::L1 => {
            for i in 0..point.len() {
                if point[i] > bounds[i + dim] {
                    dist = dist + (point[i] - bounds[i + dim]).abs();
                } else if point[i] < bounds[i] {
                    dist = dist + (point[i] - bounds[i]).abs();
                }
            }
            dist
        }

        DIST::L2 | DIST::L2SIMD => {
            for i in 0..point.len() {
                if point[i] > bounds[i + dim] {
                    dist = dist + (point[i] - bounds[i + dim]).powi(2);
                } else if point[i] < bounds[i] {
                    dist = dist + (point[i] - bounds[i]).powi(2);
                }
            }
            dist.sqrt()
        }

        DIST::SQL2 | DIST::SQL2SIMD => {
            for i in 0..point.len() {
                if point[i] > bounds[i + dim] {
                    dist = dist + (point[i] - bounds[i + dim]).powi(2);
                } else if point[i] < bounds[i] {
                    dist = dist + (point[i] - bounds[i]).powi(2);
                }
            }
            dist
        }

        DIST::LINF => {
            for i in 0..point.len() {
                if point[i] > bounds[i + dim] {
                    dist = dist.max((point[i] - bounds[i + dim]).abs());
                } else if point[i] < bounds[i] {
                    dist = dist.max((point[i] - bounds[i]).abs());
                }
            }
            dist
        }

        DIST::ANY(func) => {
            let mut new_point = point.to_vec();
            for i in 0..point.len() {
                if point[i] > bounds[i + dim] {
                    new_point[i] = bounds[i + dim];
                } else if point[i] < bounds[i] {
                    new_point[i] = bounds[i];
                }
            }
            func(point, &new_point)
        }
    }
}

pub struct KDT<'a, T: Float + DistanceOps + 'static + Debug, A> {
    pub dim: usize,
    pub capacity: usize,
    // Nodes
    left: Option<Box<KDT<'a, T, A>>>,
    right: Option<Box<KDT<'a, T, A>>>,
    // Is a leaf node if this has valid values
    split_axis: usize,
    split_axis_value: T,
    // vec of len 2 * dim. First dim values are mins in each dim, second dim values are maxs
    bounds: Vec<T>,
    // Data
    data: Vec<Leaf<'a, T, A>>, // Not empty when this is a leaf
    //
    d: DIST<T>,
}

impl<'a, T: Float + DistanceOps + 'static + Debug, A: Copy> KDT<'a, T, A> {
    // Helper function that finds the bounding box for each (sub)kdtree
    // Vec of length 2 * dim. First dim values are the mins, and the rest are maxes
    fn find_bounds(data: &[Leaf<'a, T, A>], dim: usize) -> Vec<T> {
        let mut bounds = vec![T::max_value(); dim];
        bounds.extend(std::iter::repeat(T::min_value()).take(dim));
        for elem in data.iter() {
            for i in 0..dim {
                bounds[i] = bounds[i].min(elem.value_at(i));
                bounds[dim + i] = bounds[dim + i].max(elem.value_at(i));
            }
        }
        bounds
    }

    pub fn from_leaves(data: &'a mut [Leaf<'a, T, A>], d: DIST<T>) -> Result<Self, String> {
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

    pub fn from_leaves_unchecked(data: &'a mut [Leaf<'a, T, A>], d: DIST<T>) -> Self {
        let dim = data[0].dim();
        let mut tree = KDT::new_empty(dim, suggest_capacity(dim), d);
        for leaf in data.iter().copied() {
            tree.add_unchecked(leaf, 0);
        }
        tree
    }

    pub fn new_empty(dim: usize, capacity: usize, d: DIST<T>) -> Self {
        let mut bounds = vec![T::max_value(); dim];
        bounds.extend(std::iter::repeat(T::min_value()).take(dim));
        KDT {
            dim: dim,
            capacity: capacity,
            left: None,
            right: None,
            split_axis: usize::MAX,
            split_axis_value: T::nan(),
            bounds: bounds,
            data: vec![],
            d: d,
        }
    }

    /// Creates a new leaf node out of the data.
    pub fn grow_new_leaf(&self, data: Vec<Leaf<'a, T, A>>) -> Self {
        let bounds = Self::find_bounds(&data, self.dim);
        KDT {
            dim: self.dim,
            capacity: self.capacity,
            left: None,
            right: None,
            split_axis: usize::MAX,
            split_axis_value: T::nan(),
            bounds: bounds,
            data: data,
            d: self.d,
        }
    }

    fn is_leaf(&self) -> bool {
        !(self.left.is_some() || self.right.is_some())
    }

    /// Updates the bounds according to the new leaf
    fn update_bounds(&mut self, leaf: &Leaf<'a, T, A>) {
        for i in 0..self.dim {
            self.bounds[i] = self.bounds[i].min(leaf.value_at(i));
            self.bounds[i + self.dim] = self.bounds[i + self.dim].max(leaf.value_at(i));
        }
    }

    /// Updates the bounds and push to new leaf to the data vec.
    fn update_and_push(&mut self, leaf: Leaf<'a, T, A>) {
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
    pub fn attach(&mut self, leaf: Leaf<'a, T, A>) -> Result<(), String> {
        if leaf.dim() != self.dim {
            Err("Dimension does not match.".into())
        } else {
            Ok(self.attach_unchecked(leaf))
        }
    }

    #[inline(always)]
    pub fn attach_unchecked(&mut self, leaf: Leaf<'a, T, A>) {
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
    pub fn add(&mut self, leaf: Leaf<'a, T, A>) -> Result<(), String> {
        if leaf.dim() != self.dim {
            Err("Dimension does not match.".into())
        } else {
            Ok(self.add_unchecked(leaf, 0))
        }
    }

    #[inline(always)]
    pub fn add_unchecked(&mut self, leaf: Leaf<'a, T, A>, depth: usize) {
        if self.is_leaf() {
            // Always update
            self.update_and_push(leaf);
            if self.data.len() > self.capacity {
                // The bounds are updated. New leaf is pushed to self.data
                // this is a copy of all content in self.data, which, by now, should contain the new leaf
                let mut new_data = self.data.split_off(0);

                let axis = depth % self.dim;
                let midpoint = self.bounds[axis]
                    + (self.bounds[axis + self.dim] - self.bounds[axis]) / (T::one() + T::one());

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
        top_k: &mut Vec<NB<T, A>>,
        k: usize,
        point: &[T],
        current_max: T,
        max_dist_bound: T,
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
    fn update_nb_within(&self, neighbors: &mut Vec<NB<T, A>>, point: &[T], radius: T) {
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

impl<'a, T: Float + DistanceOps + 'static + Debug, A: Copy> SpatialQueries<'a, T, A>
    for KDT<'a, T, A>
{
    fn dim(&self) -> usize {
        self.dim
    }

    // #[inline(always)]
    fn knn_one_step(
        &self,
        pending: &mut Vec<(T, &KDT<'a, T, A>)>,
        top_k: &mut Vec<NB<T, A>>,
        k: usize,
        point: &[T],
        max_dist_bound: T,
        epsilon: T,
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
        pending: &mut Vec<(T, &KDT<'a, T, A>)>,
        neighbors: &mut Vec<NB<T, A>>,
        point: &[T],
        radius: T,
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
        pending: &mut Vec<(T, &KDT<'a, T, A>)>,
        point: &[T],
        radius: T,
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

impl<'a, T: Float + DistanceOps + 'static + Debug + Into<f64>, A: Float + Into<f64>>
    KNNRegressor<'a, T, A> for KDT<'a, T, A>
{
}

// ---------- Owned KDT (for Python Interop) ----------

pub struct OwnedKDT<T: Float + DistanceOps + 'static + Debug, A: Copy> {
    pub dim: usize,
    pub capacity: usize,
    // Nodes
    left: Option<Box<OwnedKDT<T, A>>>,
    right: Option<Box<OwnedKDT<T, A>>>,
    // Is a leaf node if this has valid values
    split_axis: usize,
    split_axis_value: T,
    // vec of len 2 * dim. First dim values are mins in each dim, second dim values are maxs
    bounds: Vec<T>,
    // Data
    data: Vec<OwnedLeaf<T, A>>, // Not empty when this is a leaf
    //
    d: DIST<T>,
    // Split Method
    split_method: SplitMethod,
}

impl<T: Float + DistanceOps + 'static + Debug, A: Copy> OwnedKDT<T, A> {
    // Helper function that finds the bounding box for each (sub)kdtree
    // Vec of length 2 * dim. First dim values are the mins, and the rest are maxes
    fn find_bounds(data: &[OwnedLeaf<T, A>], dim: usize) -> Vec<T> {
        let mut bounds = vec![T::max_value(); dim];
        bounds.extend(std::iter::repeat(T::min_value()).take(dim));
        for elem in data.iter() {
            for i in 0..dim {
                bounds[i] = bounds[i].min(elem.value_at(i));
                bounds[dim + i] = bounds[dim + i].max(elem.value_at(i));
            }
        }
        bounds
    }

    pub fn from_leaves(
        data: Vec<OwnedLeaf<T, A>>,
        d: DIST<T>,
        split_method: SplitMethod,
    ) -> Result<Self, String> {
        if data.is_empty() {
            return Err("Empty data.".into());
        }
        let dim = data[0].dim();
        let mut tree = OwnedKDT::new_empty(dim, suggest_capacity(dim), d, split_method);
        for leaf in data.into_iter() {
            if leaf.dim() != dim {
                return Err("Dimension isn't consistent.".into());
            } else {
                tree.add(leaf)?;
            }
        }
        Ok(tree)
    }

    pub fn from_leaves_unchecked(
        data: Vec<OwnedLeaf<T, A>>,
        d: DIST<T>,
        split_method: SplitMethod,
    ) -> Self {
        let dim = data[0].dim();
        let mut tree = OwnedKDT::new_empty(dim, suggest_capacity(dim), d, split_method);
        for leaf in data.into_iter() {
            tree.add_unchecked(leaf, 0);
        }
        tree
    }

    // pub fn from_leaves_bulk_load(
    //     data: Vec<OwnedLeaf<T, A>>,
    //     dim: usize,
    //     capacity: usize,
    //     depth: usize,
    //     d: DIST<T>,
    //     split_method: SplitMethod
    // ) -> Self {
    //     let n = data.len();

    //     let bounds = Self::find_bounds(&data, dim);
    //     // let (min_bounds, max_bounds) = Self::find_bounds(data, dim);
    //     if n <= capacity {
    //         OwnedKDT {
    //             dim: dim,
    //             capacity: capacity,
    //             left: None,
    //             right: None,
    //             split_axis: usize::MAX, // This should never be used, because this is a leaf
    //             split_axis_value: T::nan(), // This should never be used, because this is a leaf
    //             bounds: bounds,
    //             data: data.to_vec(),
    //             d: d,
    //             split_method: split_method
    //         }
    //     } else {
    //         let axis = depth % dim;
    //         let (split_value, split_idx) = match split_method {
    //             SplitMethod::MIDPOINT => {
    //                 let midpoint = bounds[axis]
    //                     + (bounds[axis + dim] - bounds[axis]) / (T::one() + T::one());
    //                 // True will go right, false go left
    //                 data.sort_unstable_by_key(|leaf| leaf.value_at(axis) >= midpoint);
    //                 let split_idx = data.partition_point(|elem| elem.value_at(axis) < midpoint);
    //                 (midpoint, split_idx)
    //             },
    //             SplitMethod::MEDIAN => {
    //                 data.sort_unstable_by(|l1, l2| {
    //                     l1.value_at(axis).partial_cmp(&l2.value_at(axis)).unwrap()
    //                 });
    //                 let split_idx = data.len() >> 1;
    //                 let split_value = data[split_idx].value_at(axis);
    //                 (split_value, split_idx)
    //             },
    //         };
    //         let (left, right) = data.split_at_mut(split_idx);

    //         if left.is_empty() {
    //             OwnedKDT {
    //                 dim: dim,
    //                 capacity: capacity,
    //                 left: None,
    //                 right: None,
    //                 split_axis: usize::MAX, // This should never be used, because this is a leaf
    //                 split_axis_value: T::nan(), // This should never be used, because this is a leaf
    //                 bounds: bounds,
    //                 data: right.to_vec(),
    //                 d: d,
    //                 split_method: split_method
    //             }
    //         } else {
    //             OwnedKDT {
    //                 dim: dim,
    //                 capacity: capacity,
    //                 left: Some(Box::new(Self::from_leaves_bulk_load(
    //                     left,
    //                     dim,
    //                     capacity,
    //                     depth + 1,
    //                     d.clone(),
    //                     split_method.clone()
    //                 ))),
    //                 right: Some(Box::new(Self::from_leaves_bulk_load(
    //                     right,
    //                     dim,
    //                     capacity,
    //                     depth + 1,
    //                     d.clone(),
    //                     split_method.clone()
    //                 ))),
    //                 split_axis: axis,
    //                 split_axis_value: split_value,
    //                 bounds: bounds,
    //                 data: vec![],
    //                 d: d,
    //                 split_method: split_method
    //             }
    //         }
    //     }
    // }

    pub fn new_empty(dim: usize, capacity: usize, d: DIST<T>, split_method: SplitMethod) -> Self {
        let mut bounds = vec![T::max_value(); dim];
        bounds.extend(std::iter::repeat(T::min_value()).take(dim));
        OwnedKDT {
            dim: dim,
            capacity: capacity,
            left: None,
            right: None,
            split_axis: usize::MAX,
            split_axis_value: T::nan(),
            bounds: bounds,
            data: vec![],
            d: d,
            split_method: split_method,
        }
    }

    /// Creates a new leaf node out of the data.
    pub fn grow_new_leaf(&self, data: Vec<OwnedLeaf<T, A>>) -> Self {
        let bounds = Self::find_bounds(&data, self.dim);
        OwnedKDT {
            dim: self.dim,
            capacity: self.capacity,
            left: None,
            right: None,
            split_axis: usize::MAX,
            split_axis_value: T::nan(),
            bounds: bounds,
            data: data,
            d: self.d,
            split_method: self.split_method.clone(),
        }
    }

    fn is_leaf(&self) -> bool {
        !(self.left.is_some() || self.right.is_some())
    }

    /// Count of leaves in the current node
    pub fn count(&self) -> usize {
        self.data.len()
    }

    /// Count of leaves in the entire subree
    pub fn count_all(&self) -> usize {
        let mut stack = vec![self];
        let mut answer = 0usize;
        while !stack.is_empty() {
            let top = stack.pop().unwrap();
            if top.is_leaf() {
                answer += top.count();
            } else {
                stack.push(top.left.as_ref().unwrap());
                stack.push(top.right.as_ref().unwrap());
            }
        }
        answer
    }

    /// Updates the bounds according to the new leaf
    fn update_bounds(&mut self, leaf: &OwnedLeaf<T, A>) {
        for i in 0..self.dim {
            self.bounds[i] = self.bounds[i].min(leaf.value_at(i));
            self.bounds[i + self.dim] = self.bounds[i + self.dim].max(leaf.value_at(i));
        }
    }

    /// Updates the bounds and push to new leaf to the data vec.
    fn update_and_push(&mut self, leaf: OwnedLeaf<T, A>) {
        for i in 0..self.dim {
            self.bounds[i] = self.bounds[i].min(leaf.value_at(i));
            self.bounds[i + self.dim] = self.bounds[i + self.dim].max(leaf.value_at(i));
        }
        self.data.push(leaf);
    }

    /// Add a new point (leaf) to the Kdtree. This will further split the kdtree
    /// if capacity is reached in the leaf node.
    pub fn add(&mut self, leaf: OwnedLeaf<T, A>) -> Result<(), String> {
        if leaf.dim() != self.dim {
            Err("Dimension does not match.".into())
        } else {
            Ok(self.add_unchecked(leaf, 0))
        }
    }

    #[inline(always)]
    pub fn add_unchecked(&mut self, leaf: OwnedLeaf<T, A>, depth: usize) {
        if self.is_leaf() {
            // Always update
            self.update_and_push(leaf);
            if self.data.len() > self.capacity {
                // The bounds are updated. New leaf is pushed to self.data
                // this is a copy of all content in self.data, which, by now, should contain the new leaf
                let mut new_data = self.data.split_off(0);

                let axis = depth % self.dim;

                let (split_value, split_idx) = match self.split_method {
                    SplitMethod::MIDPOINT => {
                        let midpoint = self.bounds[axis]
                            + (self.bounds[axis + self.dim] - self.bounds[axis])
                                / (T::one() + T::one());
                        // True will go right, false go left
                        new_data.sort_unstable_by_key(|leaf| leaf.value_at(axis) >= midpoint);
                        let split_idx =
                            new_data.partition_point(|elem| elem.value_at(axis) < midpoint);
                        (midpoint, split_idx)
                    }
                    SplitMethod::MEDIAN => {
                        new_data.sort_unstable_by(|l1, l2| {
                            l1.value_at(axis).partial_cmp(&l2.value_at(axis)).unwrap()
                        });
                        let split_idx = new_data.len() >> 1;
                        let split_value = new_data[split_idx].value_at(axis);
                        (split_value, split_idx)
                    }
                };

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
                    self.split_axis_value = split_value;
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
        top_k: &mut Vec<NB<T, A>>,
        k: usize,
        point: &[T],
        current_max: T,
        max_dist_bound: T,
    ) {
        // This is only called if is_leaf. Safe to unwrap.
        let mut cur_max = current_max;
        for element in self.data.iter() {
            let dist = self.d.dist(element.vec(), point);
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
    fn update_nb_within(&self, neighbors: &mut Vec<NB<T, A>>, point: &[T], radius: T) {
        // This is only called if is_leaf. Safe to unwrap.
        for element in self.data.iter() {
            let dist = self.d.dist(element.vec(), point);
            if dist <= radius {
                neighbors.push(NB {
                    dist: dist,
                    item: element.item,
                });
            }
        }
    }
}

impl<'a, T: Float + DistanceOps + 'static + Debug, A: Copy> SpatialQueries<'a, T, A>
    for OwnedKDT<T, A>
{
    fn dim(&self) -> usize {
        self.dim
    }

    // #[inline(always)]
    fn knn_one_step(
        &self,
        pending: &mut Vec<(T, &OwnedKDT<T, A>)>,
        top_k: &mut Vec<NB<T, A>>,
        k: usize,
        point: &[T],
        max_dist_bound: T,
        epsilon: T,
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
        pending: &mut Vec<(T, &OwnedKDT<T, A>)>,
        neighbors: &mut Vec<NB<T, A>>,
        point: &[T],
        radius: T,
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
        pending: &mut Vec<(T, &OwnedKDT<T, A>)>,
        point: &[T],
        radius: T,
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

impl<'a, T: Float + DistanceOps + 'static + Debug + Into<f64>, A: Float + Into<f64>>
    KNNRegressor<'a, T, A> for OwnedKDT<T, A>
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arkadia::{slice_to_leaves, slice_to_owned_leaves};

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

    pub fn squared_l2<T: Float + 'static>(a: &[T], b: &[T]) -> T {
        a.iter()
            .zip(b.iter())
            .fold(T::zero(), |acc, (&a, &b)| acc + (a - b) * (a - b))
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

        let tree = KDT::from_leaves(&mut leaves, DIST::LINF).unwrap();

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

        let tree = KDT::from_leaves(&mut leaves, DIST::L1).unwrap();

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

        let tree = KDT::from_leaves(&mut leaves, DIST::SQL2).unwrap();

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
    fn test_10d_knn_l2_dist_owned() {
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
        let leaves = slice_to_owned_leaves(&v, 10, &values);

        let tree = OwnedKDT::from_leaves(leaves, DIST::SQL2, SplitMethod::MIDPOINT).unwrap();
        assert!(tree.count_all() == rows);

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
