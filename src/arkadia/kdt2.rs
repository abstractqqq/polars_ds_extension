use super::{suggest_capacity, KNNDist, KNNMethod, Leaf, Metric, NB};
use std::usize;

const NULL_IDX: u32 = u32::MAX;

enum Node<'a, A> {
    Internal {
        split_axis: usize,
        split_value: f64,
        left: u32,
        right: u32,
        bounds: Vec<f64>,
    },
    Leaf {
        data: Vec<Leaf<'a, f64, A>>,
        bounds: Vec<f64>,
    },
}

impl<'a, A> Node<'a, A> {
    #[inline(always)]
    fn bounds(&self) -> &[f64] {
        match self {
            Node::Internal { bounds, .. } => bounds,
            Node::Leaf { bounds, .. } => bounds,
        }
    }

    fn is_leaf(&self) -> bool {
        matches!(self, Node::Leaf { .. })
    }

    fn data_mut(&mut self) -> Option<&mut Vec<Leaf<'a, f64, A>>> {
        if let Node::Leaf { data, .. } = self { Some(data) } else { None }
    }
}

pub struct KDT<'a, A, M: Metric = KNNDist> {
    pub dim: usize,
    pub capacity: usize,
    nodes: Vec<Node<'a, A>>,
    root: u32,
    pub d: M,
}

impl<'a, A: Copy, M: Metric> KDT<'a, A, M> {
    #[inline(always)]
    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn new_empty(dim: usize, capacity: usize, d: M) -> Self {
        let mut bounds = vec![f64::INFINITY; dim];
        bounds.extend(std::iter::repeat(f64::NEG_INFINITY).take(dim));
        
        let root_node = Node::Leaf {
            data: Vec::with_capacity(capacity),
            bounds,
        };

        KDT {
            dim,
            capacity,
            nodes: vec![root_node],
            root: 0,
            d,
        }
    }

    fn find_bounds(data: &[Leaf<'a, f64, A>], dim: usize) -> Vec<f64> {
        let mut bounds = vec![f64::INFINITY; dim];
        bounds.extend(std::iter::repeat(f64::NEG_INFINITY).take(dim));
        for elem in data {
            for i in 0..dim {
                let val = elem.row_vec[i];
                if val < bounds[i] { bounds[i] = val; }
                if val > bounds[i + dim] { bounds[i + dim] = val; }
            }
        }
        bounds
    }

    pub fn from_leaves(data: &'a mut [Leaf<'a, f64, A>], d: M) -> Result<Self, String> {
        if data.is_empty() { return Err("Empty data.".into()); }
        let dim = data[0].row_vec.len();
        let capacity = suggest_capacity(dim);
        let mut tree = KDT {
            dim,
            capacity,
            nodes: Vec::with_capacity(data.len() / capacity * 2),
            root: 0,
            d,
        };
        tree.root = tree.build_recursive(data, 0);
        Ok(tree)
    }

    fn build_recursive(&mut self, data: &mut [Leaf<'a, f64, A>], depth: usize) -> u32 {
        let dim = self.dim;
        let bounds = Self::find_bounds(data, dim);

        if data.len() <= self.capacity {
            let node_idx = self.nodes.len() as u32;
            self.nodes.push(Node::Leaf {
                data: data.to_vec(),
                bounds,
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

        let node_idx = self.nodes.len() as u32;
        // Placeholder node to maintain index order
        self.nodes.push(Node::Internal {
            split_axis: axis,
            split_value,
            left: NULL_IDX,
            right: NULL_IDX,
            bounds,
        });

        let left = self.build_recursive(&mut data[..mid], depth + 1);
        let right = self.build_recursive(&mut data[mid..], depth + 1);

        // Update placeholder with actual child indices
        if let Node::Internal { left: l, right: r, .. } = &mut self.nodes[node_idx as usize] {
            *l = left;
            *r = right;
        }
        node_idx
    }

    pub fn add_unchecked(&mut self, leaf: Leaf<'a, f64, A>, _depth: usize) {
        self.recursive_add(self.root, leaf, 0);
    }

    fn recursive_add(&mut self, node_idx: u32, leaf: Leaf<'a, f64, A>, depth: usize) {
        let mut node = std::mem::replace(&mut self.nodes[node_idx as usize], Node::Leaf { data: Vec::new(), bounds: Vec::new() });
        
        match &mut node {
            Node::Leaf { data, bounds } => {
                // Update bounds
                for i in 0..self.dim {
                    let v = leaf.row_vec[i];
                    if v < bounds[i] { bounds[i] = v; }
                    if v > bounds[i + self.dim] { bounds[i + self.dim] = v; }
                }
                data.push(leaf);

                if data.len() > self.capacity {
                    let axis = depth % self.dim;
                    // MIDPOINT split
                    let midpoint = bounds[axis] + (bounds[axis + self.dim] - bounds[axis]) * 0.5;
                    
                    let mut left_v = Vec::new();
                    let mut right_v = Vec::new();
                    
                    for item in data.drain(..) {
                        if item.row_vec[axis] < midpoint { left_v.push(item); }
                        else { right_v.push(item); }
                    }

                    if left_v.is_empty() || right_v.is_empty() {
                        *data = if left_v.is_empty() { right_v } else { left_v };
                        self.nodes[node_idx as usize] = node;
                    } else {
                        let l_bounds = Self::find_bounds(&left_v, self.dim);
                        let r_bounds = Self::find_bounds(&right_v, self.dim);
                        
                        let left_idx = self.nodes.len() as u32;
                        self.nodes.push(Node::Leaf { data: left_v, bounds: l_bounds });
                        
                        let right_idx = self.nodes.len() as u32;
                        self.nodes.push(Node::Leaf { data: right_v, bounds: r_bounds });

                        self.nodes[node_idx as usize] = Node::Internal {
                            split_axis: axis,
                            split_value: midpoint,
                            left: left_idx,
                            right: right_idx,
                            bounds: bounds.clone(),
                        };
                    }
                } else {
                    self.nodes[node_idx as usize] = node;
                }
            }
            Node::Internal { split_axis, split_value, left, right, bounds } => {
                let axis = *split_axis;
                let val = *split_value;
                let l_idx = *left;
                let r_idx = *right;

                for i in 0..self.dim {
                    let v = leaf.row_vec[i];
                    if v < bounds[i] { bounds[i] = v; }
                    if v > bounds[i + self.dim] { bounds[i + self.dim] = v; }
                }
                
                let target = if leaf.row_vec[axis] < val { l_idx } else { r_idx };
                self.nodes[node_idx as usize] = node;
                self.recursive_add(target, leaf, depth + 1);
            }
        }
    }

    #[inline(always)]
    fn update_top_k(&self, data: &[Leaf<'a, f64, A>], top_k: &mut Vec<NB<f64, A>>, k: usize, point: &[f64], max_dist_bound: f64) {
        for element in data {
            let dist = self.d.dist(element.row_vec, point);
            let current_max = top_k.last().map(|nb: &NB<f64, A>| nb.dist).unwrap_or(max_dist_bound);
            
            if dist <= max_dist_bound && (dist < current_max || top_k.len() < k) {
                let idx = top_k.partition_point(|s| s.dist <= dist);
                top_k.insert(idx, NB { dist, item: element.item });
                if top_k.len() > k { top_k.pop(); }
            }
        }
    }

    pub fn knn(&self, k: usize, point: &[f64], epsilon: f64) -> Option<Vec<NB<f64, A>>> {
        if k == 0 || point.len() != self.dim || point.iter().any(|x| !x.is_finite()) { return None; }
        
        let mut top_k = Vec::with_capacity(k + 1);
        let mut stack = Vec::with_capacity(32);
        let d_root = self.d.dist_to_box(self.nodes[self.root as usize].bounds(), point);
        stack.push((d_root, self.root));

        while let Some((d_box, idx)) = stack.pop() {
            let current_max = top_k.last().map(|nb: &NB<f64, A>| nb.dist).unwrap_or(f64::MAX);
            if d_box > current_max { continue; }

            match &self.nodes[idx as usize] {
                Node::Internal { split_axis, split_value, left, right, .. } => {
                    let (near, far) = if point[*split_axis] < *split_value {
                        (*left, *right)
                    } else {
                        (*right, *left)
                    };

                    let d_far = self.d.dist_to_box(self.nodes[far as usize].bounds(), point);
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
        let d_root = self.d.dist_to_box(self.nodes[self.root as usize].bounds(), point);
        stack.push((d_root, self.root));

        while let Some((d_box, idx)) = stack.pop() {
            let current_max = top_k.last().map(|nb: &NB<f64, A>| nb.dist).unwrap_or(max_dist_bound);
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

                    let d_far = self.d.dist_to_box(self.nodes[far as usize].bounds(), point);
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
        if radius <= f64::EPSILON || point.iter().any(|x| !x.is_finite()) { return None; }

        let mut neighbors = Vec::with_capacity(32);
        let mut stack = Vec::with_capacity(32);
        let d_root = self.d.dist_to_box(self.nodes[self.root as usize].bounds(), point);
        stack.push((d_root, self.root));

        while let Some((d_box, idx)) = stack.pop() {
            if d_box > radius { continue; }

            match &self.nodes[idx as usize] {
                Node::Internal { split_axis, split_value, left, right, .. } => {
                    let (near, far) = if point[*split_axis] < *split_value {
                        (*left, *right)
                    } else {
                        (*right, *left)
                    };

                    let d_far = self.d.dist_to_box(self.nodes[far as usize].bounds(), point);
                    if d_far <= radius { stack.push((d_far, far)); }
                    stack.push((d_box, near));
                }
                Node::Leaf { data, .. } => {
                    for element in data {
                        let dist = self.d.dist(element.row_vec, point);
                        if dist <= radius {
                            neighbors.push(NB { dist, item: element.item });
                        }
                    }
                }
            }
        }
        if sort { neighbors.sort_unstable(); }
        Some(neighbors)
    }

    pub fn within_count(&self, point: &[f64], radius: f64) -> Option<u32> {
        if radius <= f64::EPSILON || point.iter().any(|x| !x.is_finite()) { return None; }

        let mut count = 0u32;
        let mut stack = Vec::with_capacity(32);
        let d_root = self.d.dist_to_box(self.nodes[self.root as usize].bounds(), point);
        stack.push((d_root, self.root));

        while let Some((d_box, idx)) = stack.pop() {
            if d_box > radius { continue; }

            match &self.nodes[idx as usize] {
                Node::Internal { split_axis, split_value, left, right, .. } => {
                    let (near, far) = if point[*split_axis] < *split_value { (*left, *right) } else { (*right, *left) };
                    let d_far = self.d.dist_to_box(self.nodes[far as usize].bounds(), point);
                    if d_far <= radius { stack.push((d_far, far)); }
                    stack.push((d_box, near));
                }
                Node::Leaf { data, .. } => {
                    for element in data {
                        if self.d.dist(element.row_vec, point) <= radius { count += 1; }
                    }
                }
            }
        }
        Some(count)
    }
}