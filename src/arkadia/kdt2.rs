use super::{suggest_capacity, KNNDist, KNNMethod, Leaf, Metric, NB};
use std::usize;

const NULL_IDX: u32 = u32::MAX;

// KD-Tree is immutable after bulk loading.
// Bounding boxes are stored flatly in KDT.bounds (single contiguous Box<[f64]>),
// eliminating O(nodes) individual heap allocations and reducing Node enum size.
enum Node<'a, A> {
    Internal {
        split_axis: usize,
        split_value: f64,
        left: u32,
        right: u32,
    },
    Leaf {
        data: Box<[Leaf<'a, f64, A>]>,
    },
}

impl<'a, A> Node<'a, A> {
    fn is_leaf(&self) -> bool {
        matches!(self, Node::Leaf { .. })
    }

    fn data_mut(&mut self) -> Option<&mut [Leaf<'a, f64, A>]> {
        if let Node::Leaf { data, .. } = self {
            Some(data)
        } else {
            None
        }
    }
}

pub struct KDT<'a, A, M: Metric = KNNDist> {
    pub dim: usize,
    pub capacity: usize,
    // Boxed slice ensures fixed allocation with no unused vector capacity once built
    nodes: Box<[Node<'a, A>]>,
    // Contiguous bounds for all nodes: node `i` bounds at [i * 2 * dim .. (i + 1) * 2 * dim]
    bounds: Box<[f64]>,
    root: u32,
    pub d: M,
}

impl<'a, A: Copy, M: Metric> KDT<'a, A, M> {
    #[inline(always)]
    pub fn dim(&self) -> usize {
        self.dim
    }

    #[inline(always)]
    fn node_bounds(&self, node_idx: usize) -> &[f64] {
        let stride = 2 * self.dim;
        let start = node_idx * stride;
        &self.bounds[start..start + stride]
    }

    pub fn new_empty(dim: usize, capacity: usize, d: M) -> Self {
        let mut bounds = vec![f64::INFINITY; dim];
        bounds.extend(std::iter::repeat(f64::NEG_INFINITY).take(dim));

        let root_node = Node::Leaf {
            data: Vec::with_capacity(capacity).into_boxed_slice(),
        };

        KDT {
            dim,
            capacity,
            nodes: vec![root_node].into_boxed_slice(),
            bounds: bounds.into_boxed_slice(),
            root: 0,
            d,
        }
    }

    fn write_bounds(data: &[Leaf<'a, f64, A>], dim: usize, bounds_buf: &mut Vec<f64>) {
        let start = bounds_buf.len();
        bounds_buf.resize(start + 2 * dim, f64::INFINITY);
        for i in 0..dim {
            bounds_buf[start + dim + i] = f64::NEG_INFINITY;
        }
        for elem in data {
            for i in 0..dim {
                let val = elem.row_vec[i];
                if val < bounds_buf[start + i] {
                    bounds_buf[start + i] = val;
                }
                if val > bounds_buf[start + dim + i] {
                    bounds_buf[start + dim + i] = val;
                }
            }
        }
    }

    pub fn from_leaves(data: &'a mut [Leaf<'a, f64, A>], d: M) -> Result<Self, String> {
        if data.is_empty() {
            return Err("input data is empty".into());
        }
        let dim = data[0].row_vec.len();
        let capacity = suggest_capacity(dim);
        let est_nodes = (data.len() / capacity * 2).max(1);
        let mut nodes = Vec::with_capacity(est_nodes);
        let mut bounds_buf = Vec::with_capacity(est_nodes * 2 * dim);
        let root = Self::build_recursive(&mut nodes, &mut bounds_buf, dim, capacity, data, 0);
        Ok(KDT {
            dim,
            capacity,
            nodes: nodes.into_boxed_slice(),
            bounds: bounds_buf.into_boxed_slice(),
            root,
            d,
        })
    }

    fn build_recursive(
        nodes: &mut Vec<Node<'a, A>>,
        bounds_buf: &mut Vec<f64>,
        dim: usize,
        capacity: usize,
        data: &mut [Leaf<'a, f64, A>],
        depth: usize,
    ) -> u32 {
        let node_idx = nodes.len() as u32;
        Self::write_bounds(data, dim, bounds_buf);

        if data.len() <= capacity {
            nodes.push(Node::Leaf {
                data: Box::from(&*data),
            });
            return node_idx;
        }

        let axis = depth % dim;
        let mid = data.len() / 2;
        // O(N) partitioning to find the median
        data.select_nth_unstable_by(mid, |a, b| {
            a.row_vec[axis].partial_cmp(&b.row_vec[axis]).unwrap()
        });
        let split_value = data[mid].row_vec[axis];

        // Placeholder node to maintain index order
        nodes.push(Node::Internal {
            split_axis: axis,
            split_value,
            left: NULL_IDX,
            right: NULL_IDX,
        });

        let left =
            Self::build_recursive(nodes, bounds_buf, dim, capacity, &mut data[..mid], depth + 1);
        let right =
            Self::build_recursive(nodes, bounds_buf, dim, capacity, &mut data[mid..], depth + 1);

        // Update placeholder with actual child indices
        if let Node::Internal {
            left: l, right: r, ..
        } = &mut nodes[node_idx as usize]
        {
            *l = left;
            *r = right;
        }
        node_idx
    }

    // pub fn add_unchecked(&mut self, leaf: Leaf<'a, f64, A>, _depth: usize) {
    //     self.recursive_add(self.root, leaf, 0);
    // }

    // fn recursive_add(&mut self, node_idx: u32, leaf: Leaf<'a, f64, A>, depth: usize) {
    //     let mut node = std::mem::replace(&mut self.nodes[node_idx as usize], Node::Leaf { data: Vec::new(), bounds: Vec::new() });

    //     match &mut node {
    //         Node::Leaf { data, bounds } => {
    //             // Update bounds
    //             for i in 0..self.dim {
    //                 let v = leaf.row_vec[i];
    //                 if v < bounds[i] { bounds[i] = v; }
    //                 if v > bounds[i + self.dim] { bounds[i + self.dim] = v; }
    //             }
    //             data.push(leaf);

    //             if data.len() > self.capacity {
    //                 let axis = depth % self.dim;
    //                 // MIDPOINT split
    //                 let midpoint = bounds[axis] + (bounds[axis + self.dim] - bounds[axis]) * 0.5;

    //                 let mut left_v = Vec::new();
    //                 let mut right_v = Vec::new();

    //                 for item in data.drain(..) {
    //                     if item.row_vec[axis] < midpoint { left_v.push(item); }
    //                     else { right_v.push(item); }
    //                 }

    //                 if left_v.is_empty() || right_v.is_empty() {
    //                     *data = if left_v.is_empty() { right_v } else { left_v };
    //                     self.nodes[node_idx as usize] = node;
    //                 } else {
    //                     let l_bounds = Self::find_bounds(&left_v, self.dim);
    //                     let r_bounds = Self::find_bounds(&right_v, self.dim);

    //                     let left_idx = self.nodes.len() as u32;
    //                     self.nodes.push(Node::Leaf { data: left_v, bounds: l_bounds });

    //                     let right_idx = self.nodes.len() as u32;
    //                     self.nodes.push(Node::Leaf { data: right_v, bounds: r_bounds });

    //                     self.nodes[node_idx as usize] = Node::Internal {
    //                         split_axis: axis,
    //                         split_value: midpoint,
    //                         left: left_idx,
    //                         right: right_idx,
    //                         bounds: bounds.clone(),
    //                     };
    //                 }
    //             } else {
    //                 self.nodes[node_idx as usize] = node;
    //             }
    //         }
    //         Node::Internal { split_axis, split_value, left, right, bounds } => {
    //             let axis = *split_axis;
    //             let val = *split_value;
    //             let l_idx = *left;
    //             let r_idx = *right;

    //             for i in 0..self.dim {
    //                 let v = leaf.row_vec[i];
    //                 if v < bounds[i] { bounds[i] = v; }
    //                 if v > bounds[i + self.dim] { bounds[i + self.dim] = v; }
    //             }

    //             let target = if leaf.row_vec[axis] < val { l_idx } else { r_idx };
    //             self.nodes[node_idx as usize] = node;
    //             self.recursive_add(target, leaf, depth + 1);
    //         }
    //     }
    // }

    #[inline(always)]
    fn update_top_k(
        &self,
        data: &[Leaf<'a, f64, A>],
        top_k: &mut Vec<NB<f64, A>>,
        k: usize,
        point: &[f64],
        max_dist_bound: f64,
    ) {
        for element in data {
            let dist = self.d.dist(element.row_vec, point);
            let current_max = top_k
                .last()
                .map(|nb: &NB<f64, A>| nb.dist)
                .unwrap_or(max_dist_bound);

            if dist <= max_dist_bound && (dist < current_max || top_k.len() < k) {
                let idx = top_k.partition_point(|s| s.dist <= dist);
                top_k.insert(
                    idx,
                    NB {
                        dist,
                        item: element.item,
                    },
                );
                if top_k.len() > k {
                    top_k.pop();
                }
            }
        }
    }

    pub fn knn(&self, k: usize, point: &[f64], epsilon: f64) -> Option<Vec<NB<f64, A>>> {
        if k == 0 || point.len() != self.dim || point.iter().any(|x| !x.is_finite()) {
            return None;
        }

        let mut top_k = Vec::with_capacity(k + 1);
        let mut stack = Vec::with_capacity(32);
        let d_root = self
            .d
            .dist_to_box(self.node_bounds(self.root as usize), point);
        stack.push((d_root, self.root));

        while let Some((d_box, idx)) = stack.pop() {
            let current_max = top_k
                .last()
                .map(|nb: &NB<f64, A>| nb.dist)
                .unwrap_or(f64::MAX);
            if d_box > current_max {
                continue;
            }

            match &self.nodes[idx as usize] {
                Node::Internal {
                    split_axis,
                    split_value,
                    left,
                    right,
                    ..
                } => {
                    let (near, far) = if point[*split_axis] < *split_value {
                        (*left, *right)
                    } else {
                        (*right, *left)
                    };

                    let d_far = self.d.dist_to_box(self.node_bounds(far as usize), point);
                    if d_far + epsilon < current_max {
                        stack.push((d_far, far));
                    }
                    stack.push((d_box, near));
                }
                Node::Leaf { data, .. } => {
                    self.update_top_k(data, &mut top_k, k, point, f64::MAX);
                }
            }
        }
        Some(top_k)
    }

    pub fn knn_bounded(
        &self,
        k: usize,
        point: &[f64],
        max_dist_bound: f64,
        epsilon: f64,
    ) -> Option<Vec<NB<f64, A>>> {
        if k == 0
            || point.len() != self.dim
            || point.iter().any(|x| !x.is_finite())
            || max_dist_bound <= f64::EPSILON
        {
            return None;
        }

        let mut top_k = Vec::with_capacity(k + 1);
        let mut stack = Vec::with_capacity(32);
        let d_root = self
            .d
            .dist_to_box(self.node_bounds(self.root as usize), point);
        stack.push((d_root, self.root));

        while let Some((d_box, idx)) = stack.pop() {
            let current_max = top_k
                .last()
                .map(|nb: &NB<f64, A>| nb.dist)
                .unwrap_or(max_dist_bound);
            if d_box > current_max {
                continue;
            }

            match &self.nodes[idx as usize] {
                Node::Internal {
                    split_axis,
                    split_value,
                    left,
                    right,
                    ..
                } => {
                    let (near, far) = if point[*split_axis] < *split_value {
                        (*left, *right)
                    } else {
                        (*right, *left)
                    };

                    let d_far = self.d.dist_to_box(self.node_bounds(far as usize), point);
                    if d_far + epsilon < current_max {
                        stack.push((d_far, far));
                    }
                    stack.push((d_box, near));
                }
                Node::Leaf { data, .. } => {
                    self.update_top_k(data, &mut top_k, k, point, max_dist_bound);
                }
            }
        }
        Some(top_k)
    }

    pub fn knn_regress(
        &self,
        k: usize,
        point: &[f64],
        min_dist_bound: f64,
        max_dist_bound: f64,
        how: KNNMethod,
    ) -> Option<f64>
    where
        A: num::Float + Into<f64>,
    {
        let nn = self.knn_bounded(k, point, max_dist_bound, 0.0)?;

        let mut sum_vw = 0.0;
        let mut sum_w = 0.0;
        let mut count = 0;

        match how {
            KNNMethod::P1Weighted => {
                for nb in nn {
                    if nb.dist >= min_dist_bound {
                        let w = (1.0 + nb.dist).recip();
                        sum_w += w;
                        sum_vw += w * nb.item.into();
                        count += 1;
                    }
                }
            }
            KNNMethod::Weighted => {
                for nb in nn {
                    if nb.dist >= min_dist_bound {
                        let w = nb.dist.recip();
                        sum_w += w;
                        sum_vw += w * nb.item.into();
                        count += 1;
                    }
                }
            }
            KNNMethod::NotWeighted => {
                for nb in nn {
                    if nb.dist >= min_dist_bound {
                        sum_vw += nb.item.into();
                        count += 1;
                    }
                }
                if count > 0 {
                    return Some(sum_vw / count as f64);
                }
            }
        }

        if count > 0 {
            Some(sum_vw / sum_w)
        } else {
            None
        }
    }

    pub fn within(&self, point: &[f64], radius: f64, sort: bool) -> Option<Vec<NB<f64, A>>> {
        if radius <= f64::EPSILON || point.iter().any(|x| !x.is_finite()) {
            return None;
        }

        let mut neighbors = Vec::with_capacity(32);
        let mut stack = Vec::with_capacity(32);
        let d_root = self
            .d
            .dist_to_box(self.node_bounds(self.root as usize), point);
        stack.push((d_root, self.root));

        while let Some((d_box, idx)) = stack.pop() {
            if d_box > radius {
                continue;
            }

            match &self.nodes[idx as usize] {
                Node::Internal {
                    split_axis,
                    split_value,
                    left,
                    right,
                    ..
                } => {
                    let (near, far) = if point[*split_axis] < *split_value {
                        (*left, *right)
                    } else {
                        (*right, *left)
                    };

                    let d_far = self.d.dist_to_box(self.node_bounds(far as usize), point);
                    if d_far <= radius {
                        stack.push((d_far, far));
                    }
                    stack.push((d_box, near));
                }
                Node::Leaf { data, .. } => {
                    for element in data {
                        let dist = self.d.dist(element.row_vec, point);
                        if dist <= radius {
                            neighbors.push(NB {
                                dist,
                                item: element.item,
                            });
                        }
                    }
                }
            }
        }
        if sort {
            neighbors.sort_unstable();
        }
        Some(neighbors)
    }

    pub fn within_count(&self, point: &[f64], radius: f64) -> Option<u32> {
        if radius <= f64::EPSILON || point.iter().any(|x| !x.is_finite()) {
            return None;
        }

        let mut count = 0u32;
        let mut stack = Vec::with_capacity(32);
        let d_root = self
            .d
            .dist_to_box(self.node_bounds(self.root as usize), point);
        stack.push((d_root, self.root));

        while let Some((d_box, idx)) = stack.pop() {
            if d_box > radius {
                continue;
            }

            match &self.nodes[idx as usize] {
                Node::Internal {
                    split_axis,
                    split_value,
                    left,
                    right,
                    ..
                } => {
                    let (near, far) = if point[*split_axis] < *split_value {
                        (*left, *right)
                    } else {
                        (*right, *left)
                    };
                    let d_far = self.d.dist_to_box(self.node_bounds(far as usize), point);
                    if d_far <= radius {
                        stack.push((d_far, far));
                    }
                    stack.push((d_box, near));
                }
                Node::Leaf { data, .. } => {
                    for element in data {
                        if self.d.dist(element.row_vec, point) <= radius {
                            count += 1;
                        }
                    }
                }
            }
        }
        Some(count)
    }
}
