/// Performs KNN related search queries, classification and regression, and
/// other features/entropies that require KNN to be efficiently computed.
use super::leaf::{Leaf, KdLeaf, OwnedLeaf};
use crate::utils::DIST;
use cfavml::safe_trait_distance_ops::DistanceOps;
use num::Float;
use std::{collections::BinaryHeap, fmt::Debug, usize};

/// Midpoint between two given leaves.
fn midpoint<T, A>(
    a: &[T],
    b: &[T],
) -> Vec<T>
where
    T: Float + DistanceOps + 'static,
    A: Copy,
{ 
    a.iter()
        .copied()
        .zip(b.iter().copied())
        .map(|(a, b)| (a + b) / (T::one() + T::one()))
        .collect()
}

#[derive(Clone, PartialEq)]
struct Sphere<T: Float + DistanceOps + 'static> {
    /// A sphere is the "balls" this stores a center coordinates, a radius and a distance metric
    center: Vec<T>,
    radius: T,
    metric: DIST<T>
}

impl <'a, T: Float + DistanceOps + 'static> Sphere<T> {

    /// Nearest distance of any point in the ball to the point `point`
    fn nearest_distance(&self, point: &[T]) -> T {
        (self.metric.dist(self.center.as_ref(), point) - self.radius).max(T::zero())
    }

    /// Farthest distance of any point in the ball to the point `point`. This is gauranteed by 
    /// the triangular inequality of a mathematical metric
    fn farthest_distance(&self, point: &[T]) -> T {
        self.metric.dist(self.center.as_ref(), point) + self.radius
    }
}

// /// OrdF64 is a wrapper around f64 that implements Ord and Eq
// #[derive(Debug, Clone, PartialEq, PartialOrd)]
// struct OrdF64(f64);

// impl OrdF64 {
//     fn new(f: f64) -> Self {
//         assert!(!f.is_nan(), "We can not compare NaN values");
//         Self(f)
//     }
// }

// impl Eq for OrdF64 {}
// impl Ord for OrdF64 {
//     fn cmp(&self, other: &Self) -> std::cmp::Ordering {
//         self.partial_cmp(other).unwrap()
//     }
// }

/// Uses a Bouncing ball algorithm to determine a tight bounding sphere around the given leaves
fn bounding_sphere<T, A>(
    leaves: &mut [OwnedLeaf<T, A>],
    metric: &DIST<T>,
) -> Sphere<T>
where
    T: Float + DistanceOps + 'static,
    A: Copy,
{
    assert!(
        leaves.len() >= 2,
        "Bounding sphere requires at least 2 leaves"
    ); 
    // what to do with duplicates? 

    let pivot = leaves[0].vec();
    let mut point_a = leaves[0].vec();
    let mut max_dist = T::zero();
    for leaf in &leaves[1..] {
        let d = metric.dist(pivot, leaf.vec());
        if d > max_dist {
            max_dist = d;
            point_a = leaf.vec();
        }        
    }

    max_dist = T::zero();
    let mut point_b = leaves[0].vec();
    for leaf in leaves {
        let d = metric.dist(point_a, leaf.vec());
        if d > max_dist {
            max_dist = d;
            point_b = leaf.vec();
        }     
    }

    let mut center = midpoint(point_a, point_b);
    let mut radius = metric.dist(&center, point_b).max(T::epsilon());


    let mut not_covered = leaves
        .iter()
        .filter(|&p| metric.dist(&center, p.vec()) > radius)
        .collect::<Vec<_>>();

    loop {
        match not_covered.pop()
        {
            None => {
                break Sphere {
                    center,
                    radius,
                    metric: metric.clone(),
                }
            }
            Some(p) => {

                let c_to_p = metric.dist(&center, p.vec());
                if c_to_p > T::zero() {
                    let d = c_to_p - radius;
                    
                    todo!()
                    radius = radius * (T::from(1.01).unwrap());

                }
            }
        };
        not_covered = not_covered
            .into_iter()
            .filter(|&p| metric.dist(&center, p.vec()) > radius)
            .collect();
    }
}

/// Partition the leaves into two Left and Right
fn partition<'a, T, A>(
    mut leaves: Vec<Leaf<'a, T, A>>,
    metric: &DistanceMetrics<T>,
) -> (Vec<Leaf<'a, T, A>>, Vec<Leaf<'a, T, A>>)
where
    T: Float,
    A: Clone,
{
    assert!(leaves.len() >= 2, "Partition requires at least 2 leaves");

    let a_i = leaves
        .iter()
        .enumerate()
        .max_by_key(|(_, a)| OrdF64::new(leaves[0].distance(a, metric).to_f64().unwrap()))
        .unwrap()
        .0;

    let b_i = leaves
        .iter()
        .enumerate()
        .max_by_key(|(_, b)| OrdF64::new(leaves[a_i].distance(b, metric).to_f64().unwrap()))
        .unwrap()
        .0;

    let (a_i, b_i) = (a_i.max(b_i), a_i.min(b_i));

    let mut aps = vec![leaves.swap_remove(a_i)];
    let mut bps = vec![leaves.swap_remove(b_i)];

    for p in leaves.iter() {
        if aps[0].distance(p, &metric) < bps[0].distance(p, metric) {
            aps.push(p.clone());
        } else {
            bps.push(p.clone());
        }
    }
    (aps, bps)
}

/// Inner Ball Tree. This is the recursive structure of the Ball Tree
/// It can be a Leaf, Branch or Empty
/// The new function recursively calls itself till each input Leaf is a Leaf in the BallTree
#[derive(Clone, Debug)]
enum BallTreeInner<'a, T, A>
where
    T: Float,
{
    Empty,
    Leaf(Leaf<'a, T, A>),
    Branch {
        sphere: Sphere<Leaf<'a, T, A>, T>,
        count: usize,
        left: Box<BallTreeInner<'a, T, A>>,
        right: Box<BallTreeInner<'a, T, A>>,
    },
}

impl<'a, T, A> Default for BallTreeInner<'a, T, A>
where
    T: Float,
{
    fn default() -> Self {
        BallTreeInner::Empty
    }
}

impl<'a, T, A> BallTreeInner<'a, T, A>
where
    T: Float,
    A: Clone,
{
    /// Iterates over the Leaves and recursively calls itseld to create a BallTree
    /// Only the Branch nodes have children and recursively call BallTreeInner
    fn new(mut leaves: Vec<Leaf<'a, T, A>>, metric: &DistanceMetrics<T>) -> Self {
        if leaves.is_empty() {
            return BallTreeInner::Empty;
        } else if leaves.iter().all(|p| p.row_vec == leaves[0].row_vec) {
            return BallTreeInner::Leaf(leaves.pop().unwrap());
        } else {
            let count = leaves.len();
            let sphere = bounding_sphere(&leaves, &metric);
            let (left, right) = if leaves.len() > 2 {
                partition(leaves, metric)
            } else {
                (vec![leaves[0].clone()], vec![leaves[1].clone()])
            };
            BallTreeInner::Branch {
                sphere,
                count,
                left: Box::new(BallTreeInner::new(left, metric)),
                right: Box::new(BallTreeInner::new(right, metric)),
            }
        }
    }

    fn nearest_distance(&self, leaf: &Leaf<'a, T, A>, metric: &DistanceMetrics<T>) -> f64 {
        match self {
            BallTreeInner::Empty => std::f64::INFINITY,
            BallTreeInner::Leaf(l) => leaf.distance(l, metric).to_f64().unwrap(),
            BallTreeInner::Branch {
                sphere,
                left,
                right,
                ..
            } => {
                let d = sphere.nearest_distance(leaf);
                let d_left = left.as_ref().nearest_distance(leaf, metric);
                let d_right = right.as_ref().nearest_distance(leaf, metric);
                d.min(d_left).min(d_right)
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Item<T>(f64, T);
impl<T> PartialEq for Item<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T> Eq for Item<T> {}

impl<T> PartialOrd for Item<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0).map(|o| o.reverse())
    }
}

impl<T> Ord for Item<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

// Iterator over Nearest Neighbors
#[derive(Debug)]
pub struct NNIter<'tree, 'query, T, A>
where
    T: Float,
    A: Clone,
{
    leaf: &'query Leaf<'query, T, A>,
    balls: &'query mut BinaryHeap<Item<&'tree BallTreeInner<'tree, T, A>>>,
    i: usize,
    max_radius: f64,
    metric: DistanceMetrics<T>,
}

impl<'tree, 'query, T, A> Iterator for NNIter<'tree, 'query, T, A>
where
    T: Float,
    A: Clone,
{
    type Item = (&'tree Leaf<'tree, T, A>, f64);

    fn next(&mut self) -> Option<Self::Item> {
        while self.balls.len() > 0 {
            let _bb = self.balls.peek().unwrap();
            // convert self.balls into a vec of BallTreeInner. extract distance and BallTreeInner from Item
            let _balls = self.balls.clone().into_sorted_vec();
            if let Item(d, BallTreeInner::Leaf(l)) = self.balls.peek().unwrap() {
                if *d <= self.max_radius {
                    // Since the item has been consumed i wold like to remove it from self.balls
                    let (d, _lx) = {
                        let item = self.balls.pop().unwrap();
                        (item.0, item.1)
                    };
                    return Some((l, d));
                }
            }
            self.i = 0;
            // Extend Branch nodes
            if let Item(_, BallTreeInner::Branch { left, right, .. }) = self.balls.pop().unwrap() {
                let d_a = left.as_ref().nearest_distance(self.leaf, &self.metric);
                let d_b = right.as_ref().nearest_distance(self.leaf, &self.metric);
                if d_a < self.max_radius {
                    self.balls.push(Item(d_a, left));
                }
                if d_b < self.max_radius {
                    self.balls.push(Item(d_b, right));
                }
            }
        }
        None
    }
}

#[derive(Debug, Clone)]
pub struct BallTree<'a, T: Float, A: Clone>(BallTreeInner<'a, T, A>, DistanceMetrics<T>);

impl<'a, T, A> BallTree<'a, T, A>
where
    T: Float,
    A: Clone,
{
    pub fn new(leaves: Vec<Leaf<'a, T, A>>, distance_metric: DistanceMetrics<T>) -> Self {
        Self(
            BallTreeInner::new(leaves, &distance_metric),
            distance_metric.clone(),
        )
    }

    pub fn query(&self) -> Query<T, A> {
        Query {
            ball_tree: self,
            balls: Default::default(),
            metric: self.1.clone(),
        }
    }
}

/// The query struct provides access to the inner tree.
/// Since the BallTree is expensive to create but relatively cheap to query
/// We use the query struct to query the BallTree
#[derive(Debug, Clone)]
pub struct Query<'a, T: Float, A: Clone> {
    ball_tree: &'a BallTree<'a, T, A>,
    balls: BinaryHeap<Item<&'a BallTreeInner<'a, T, A>>>,
    metric: DistanceMetrics<T>,
}

/// The _lazy functions only return an iterator.
/// The code heavily relies on nn_within_lazy since this is the core function to traverse the tree
impl<'a, T, A> Query<'a, T, A>
where
    T: Float,
    A: Clone,
{
    pub fn nn_lazy<'query>(
        &'query mut self,
        leaf: &'query Leaf<'query, T, A>,
    ) -> NNIter<'a, 'query, T, A> {
        self.nn_within_lazy(leaf, f64::INFINITY)
    }

    pub fn nn_within_lazy<'query>(
        &'query mut self,
        leaf: &'query Leaf<'query, T, A>,
        max_radius: f64,
    ) -> NNIter<'a, 'query, T, A> {
        let balls = &mut self.balls;
        balls.clear();
        balls.push(Item(
            self.ball_tree.0.nearest_distance(leaf, &self.metric),
            &self.ball_tree.0,
        ));
        NNIter {
            leaf,
            balls,
            i: 0,
            max_radius,
            metric: self.metric.clone(),
        }
    }
    // use nn_within_lazy to implement nn_within
    pub fn nn_within<'query>(
        &'query mut self,
        leaf: &'query Leaf<'query, T, A>,
        k: usize,
        max_radius: f64,
    ) -> Vec<(&'query Leaf<'query, T, A>, f64)> {
        let iter = self.nn_within_lazy(leaf, max_radius);
        iter.take(k).collect()
    }

    // Return the k nearest neighbors with distances
    pub fn knn<'query>(
        &'query mut self,
        leaf: &'query Leaf<'query, T, A>,
        k: usize,
    ) -> Vec<(&'query Leaf<'query, T, A>, f64)> {
        let iter = self.nn_lazy(leaf);
        iter.take(k).collect()
    }

    pub fn is_knn<'query>(
        &'query mut self,
        leaf: &'query Leaf<'query, T, A>,
        other_point: &[T],
        k: usize,
        max_radius: Option<f64>,
        epsilon: Option<T>,
    ) -> bool {
        let max_radius = max_radius.unwrap_or(f64::INFINITY);
        let epsilon = epsilon.unwrap_or_else(T::epsilon);
        let iter = self.nn_within_lazy(leaf, max_radius);
        // iterate over iter upto k
        // And compare each neighbors row vec to other_point elementwise within an epsilon
        // Do not use take k since k can be large
        // but do ensure we never go over k
        let mut count = 0;
        for (l, _) in iter {
            if l.row_vec
                .iter()
                .zip(other_point.iter())
                .all(|(a, b)| (*a - *b).abs() < epsilon)
            {
                return true;
            }
            if count >= k {
                return false;
            }
            count += 1;
        }
        false
    }

    // Min radius to encompas k points
    pub fn min_radius<'query>(&'query mut self, leaf: &'query Leaf<'query, T, A>, k: usize) -> f64 {
        let mut total_count = 0;
        let balls = &mut self.balls;
        balls.clear();
        balls.push(Item(
            self.ball_tree.0.nearest_distance(leaf, &self.metric),
            &self.ball_tree.0,
        ));

        while let Some(Item(distance, node)) = balls.pop() {
            match node {
                BallTreeInner::Empty => {}
                BallTreeInner::Leaf(_) => {
                    total_count += 1;
                    if total_count >= k {
                        return distance;
                    }
                }
                BallTreeInner::Branch {
                    sphere,
                    left,
                    right,
                    count,
                } => {
                    let next_distance = balls.peek().map(|Item(d, _)| *d).unwrap_or(f64::INFINITY);
                    if total_count + count < k && sphere.farthest_distance(leaf) < next_distance {
                        total_count += count;
                    } else {
                        balls.push(Item(sphere.nearest_distance(leaf), left));
                        balls.push(Item(sphere.nearest_distance(leaf), right));
                    }
                }
            }
        }
        f64::INFINITY
    }

    /// Needs some reassessing and validation
    /// The tests do seem correct so far.
    pub fn count<'query>(
        &'query mut self,
        leaf: &'query Leaf<'query, T, A>,
        max_radius: f64,
    ) -> usize {
        let mut total_count = 0;
        let iter = self.nn_within_lazy(leaf, max_radius);
        for _ in iter {
            total_count += 1;
        }
        total_count
    }
    pub fn allocated_size(&self) -> usize {
        self.balls.capacity() * std::mem::size_of::<Item<&BallTreeInner<T, A>>>()
    }

    pub fn deallocate_memory(&mut self) {
        self.balls.clear();
        self.balls.shrink_to_fit();
    }
}

/// So far this is only available on f64 leaves but can be made generic over T
impl<'a> Query<'a, f64, u32> {
    pub fn knn_regress<'query>(
        &'query mut self,
        leaf: &'query Leaf<'query, f64, u32>,
        k: usize,
        max_radius: f64,
    ) -> Option<f64> {
        let neighbors: Vec<(&Leaf<f64, u32>, f64)> = self
            .nn_within_lazy(leaf, max_radius)
            .into_iter()
            .take(k)
            .collect();
        let weights = neighbors
            .iter()
            .map(|(_, d)| (1.0f64 + *d).recip().into())
            .collect::<Vec<f64>>();
        let sum = weights.iter().copied().sum::<f64>();
        Some(
            neighbors
                .into_iter()
                .zip(weights.into_iter())
                .fold(0f64, |acc, (nb, w)| acc + w * nb.0.item as f64)
                / sum,
        )
    }
}


#[cfg(test)]
mod tests {
    use core::f64;
    use std::collections::HashSet;

    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaChaRng;
    use rapidfuzz::distance;

    macro_rules! generate_leaves {
        ($size:expr, $rng:ident, $random_leaves:ident, $leaves:ident) => {
            let mut $leaves = vec![];
            let mut $random_leaves = vec![]; // Store the generated leaves here
            for i in 0..$size {
                let leaf_count: usize = if i < 100 {
                    $rng.gen_range(1..=3)
                } else if i < 500 {
                    $rng.gen_range(1..=10)
                } else {
                    $rng.gen_range(1..=100)
                };

                for _ in 0..leaf_count {
                    let random_leaf = random_3d_leaf!();
                    $random_leaves.push(random_leaf); // Store the generated leaf
                }
            }
            for (i, random_leaf) in $random_leaves.iter().enumerate() {
                let leaf: Leaf<f64, u32> = Leaf {
                    row_vec: random_leaf, // Reference the stored leaf
                    item: i as u32,
                };
                $leaves.push(leaf);
            }
        };
    }

    macro_rules! setup_rng_and_macros {
        ($rng:ident) => {
            let mut $rng: ChaChaRng = SeedableRng::seed_from_u64(0xcb42c94d23346e96);

            macro_rules! random_small_f64 {
                () => {
                    $rng.gen_range(-100.0..=100.0)
                };
            }

            macro_rules! random_3d_leaf {
                () => {
                    [random_small_f64!(), random_small_f64!()]
                };
            }
        };
    }

    macro_rules! generate_knn_inputs {
        ($id:ident, $input1:ident, $input2:ident, $size:expr, $rng:ident) => {
            let mut $id = vec![];
            let mut $input1 = vec![];
            let mut $input2 = vec![];

            for i in 0..$size {
                let random_value1 = $rng.gen_range(-100.0..=100.0);
                let random_value2 = $rng.gen_range(-100.0..=100.0);
                $id.push(i as u32);
                $input1.push(random_value1);
                $input2.push(random_value2);
            }
        };
    }

    #[test]
    fn test_knn_within() {
        setup_rng_and_macros!(rng);
        generate_leaves!(100, rng, random_leaves, leaves);
        let distance_metric = DistanceMetrics::Haversine(haversine_distance);
        let tree = BallTree::new(leaves.clone(), distance_metric);
        let mut binding = tree.query();
        let point = leaves[2].row_vec;
        let k = 399;
        let within = binding.is_knn(
            &leaves[2],
            point,
            k,
            Some(f64::INFINITY),
            Some(f64::EPSILON),
        );
        assert_eq!(within, true);
    }

    #[test]
    fn test_2d_leaves() {
        setup_rng_and_macros!(rng);
        generate_leaves!(10, rng, random_leaves, leaves);
        let distance_metric = DistanceMetrics::Haversine(haversine_distance);
        let tree = BallTree::new(leaves.clone(), distance_metric);
        let mut binding = tree.query();
        let mut nnw = binding.nn_within_lazy(&leaves[3], 399.99);

        println!("nnw {:?}", nnw);
        println!("tree {:?}", tree);

        println!("nnw_nest {:?}", nnw.next());
        let res = binding.nn_within(&leaves[6], 5, 399.99);
        assert!(res.len() <= 5);
    }

    #[test]
    fn test_bounding_leaves() {
        setup_rng_and_macros!(rng);
        generate_leaves!(10, rng, random_leaves, leaves);
        let distance_metric = DistanceMetrics::Haversine(haversine_distance);
        let tree = BallTree::new(leaves.clone(), distance_metric);
        let mut binding = tree.query();
        let nnw = binding.nn_within_lazy(&leaves[9], 999.99);
        let mut total = 0;
        nnw.for_each(|x| {
            total += 1;
        });
        assert!(total > 1);
    }

    #[test]
    fn test_knn_regress() {
        setup_rng_and_macros!(rng);
        generate_leaves!(100, rng, random_leaves, leaves);
        let distance_metric = DistanceMetrics::Haversine(haversine_distance);
        let tree = BallTree::new(leaves.clone(), distance_metric);
        let mut binding = tree.query();
        let res = binding.knn_regress(&leaves[9], 18, 99999.99);
        println!("{:?}", res);
    }

    #[test]
    fn test_nb_count() {
        setup_rng_and_macros!(rng);
        generate_leaves!(100, rng, random_leaves, leaves);
        let distance_metric = DistanceMetrics::Haversine(haversine_distance);
        let tree = BallTree::new(leaves.clone(), distance_metric);
        let mut binding = tree.query();
        let leaf = leaves[9].clone();
        let res = binding.count(&leaf, 9999999.0f64);
        println!("count for leaf: {:?} is {:?}", leaf, res);
    }

    #[test]
    fn test_overall_tree() {
        setup_rng_and_macros!(rng);
        generate_leaves!(100, rng, random_leaves, leaves);
        let distance_metric = DistanceMetrics::Haversine(euclidean_distance);
        let tree = BallTree::new(leaves.clone(), distance_metric.clone());
        let mut binding = tree.query();

        for _ in 0..100 {
            let point = random_3d_leaf!();
            let max_radius = rng.gen_range(0.0..=100.0);

            let expected_values = leaves
                .iter()
                .filter(|leaf| {
                    leaf.distance(
                        &Leaf {
                            row_vec: &point,
                            item: 0,
                        },
                        &distance_metric,
                    ) <= max_radius
                })
                .map(|leaf| leaf.item)
                .collect::<HashSet<_>>();
            let mut found_values = HashSet::new();

            let mut previous_d = 0.0;
            let leaf = Leaf {
                row_vec: &point,
                item: 0,
            };
            for (leaf, d) in binding.nn_within_lazy(&leaf, max_radius) {
                assert_eq!(
                    leaf.distance(
                        &Leaf {
                            row_vec: &point,
                            item: 0
                        },
                        &distance_metric
                    ),
                    d
                );

                assert!(d >= previous_d);

                assert!(d <= max_radius);
                previous_d = d;
                found_values.insert(leaf.item);
            }
            assert_eq!(expected_values, found_values);
            let binding_count = binding.count(&leaf, max_radius);
            assert_eq!(found_values.len(), binding_count);

            let radius = binding.min_radius(&leaf, expected_values.len());
            let should_be_fewer = binding.count(&leaf, radius * 0.3);
            assert!(
                expected_values.is_empty() || should_be_fewer < expected_values.len(),
                "{} < {}",
                should_be_fewer,
                expected_values.len()
            );
        }

        assert!(binding.allocated_size() > 0);

        assert!(binding.allocated_size() <= 2 * 8 * leaves.len().next_power_of_two().max(4));
        binding.deallocate_memory();
        assert_eq!(binding.allocated_size(), 0);
    }

    #[test]
    fn test_leaf_impls() {
        let row_vecs = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
        let leaf1 = Leaf {
            row_vec: &row_vecs[0],
            item: 0,
        };
        let leaf2 = Leaf {
            row_vec: &row_vecs[1],
            item: 0,
        };
        let distance_metric = DistanceMetrics::Euclidean(euclidean_distance);

        assert_eq!(leaf1.distance(&leaf2, &distance_metric), 1.4142135623730951);

        let leaf3 = Leaf {
            row_vec: &row_vecs[2],
            item: 1,
        };

        assert_eq!(leaf1.distance(&leaf3, &distance_metric), 2.8284271247461903);

        unsafe {
            let lv = leaf1.move_towards(&leaf3, 1.3, &distance_metric).row_vec;
            assert_eq!(lv, [1.3788582233137676, 1.8384776310850235]);
        }
    }
}
