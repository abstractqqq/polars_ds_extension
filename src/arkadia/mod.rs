/// IMPORTANT!
/// This crate is intentionally built to be imperfect.
/// E.g.
/// I am not checking whether all row_vecs have the same dimension in &[Leaf].
/// The reason for this is that I only intend to use this crate in my Python package polars_ds,
/// where it is always the case that that's true.
/// Since I do not plan to make this a general purpose Kdtree package for Rust yet, I do not
/// want to do those checks.
/// E.g.
/// I am not properly defining error types because
/// it will be casted to PolarsErrors when integrated with polars_ds, OR it will be used as Python errors
/// and it is more convenient to just use strings.
/// E.g.
/// within_count returns a u32 as opposed to usize because that can help me skip a type conversion when used with Polars.
pub mod arkadia;
pub mod arkadia_any;
pub mod leaf;
pub mod neighbor;
pub mod utils;

pub use arkadia::KDT;
pub use arkadia_any::{AnyKDT, DIST};
pub use leaf::{KdLeaf, Leaf, LeafWithNorm};
pub use neighbor::NB;
pub use utils::{
    matrix_to_empty_leaves, matrix_to_empty_leaves_w_norm, matrix_to_leaves,
    matrix_to_leaves_w_norm, matrix_to_leaves_w_row_num, suggest_capacity, SplitMethod,
};

// ---------------------------------------------------------------------------------------------------------
use num::Float;

#[derive(Clone, Default)]
pub enum KNNMethod {
    DInvW, // Distance Inversion Weighted. E.g. Use (1/(1+d)) to weight the regression / classification
    #[default]
    NoW, // No Weight
}

impl From<bool> for KNNMethod {
    fn from(weighted: bool) -> Self {
        if weighted {
            KNNMethod::DInvW
        } else {
            KNNMethod::NoW
        }
    }
}

/// KD Tree Queries
pub trait KDTQ<'a, T: Float + 'static, A> {
    fn dim(&self) -> usize;

    fn knn_one_step(
        &self,
        pending: &mut Vec<(T, &Self)>,
        top_k: &mut Vec<NB<T, A>>,
        k: usize,
        point: &[T],
        point_norm_cache: T,
        max_dist_bound: T,
        epsilon: T,
    );

    fn within_one_step(
        &self,
        pending: &mut Vec<(T, &Self)>,
        neighbors: &mut Vec<NB<T, A>>,
        point: &[T],
        point_norm_cache: T,
        radius: T,
    );

    fn within_count_one_step(
        &self,
        pending: &mut Vec<(T, &Self)>,
        point: &[T],
        point_norm_cache: T,
        radius: T,
    ) -> u32;

    fn knn(&self, k: usize, point: &[T], epsilon: T) -> Option<Vec<NB<T, A>>> {
        if k == 0 || (point.len() != self.dim()) || (point.iter().any(|x| !x.is_finite())) {
            None
        } else {
            // Always allocate 1 more.
            let mut top_k = Vec::with_capacity(k + 1);
            let mut pending = Vec::with_capacity(k + 1);
            pending.push((T::min_value(), self));
            while !pending.is_empty() {
                self.knn_one_step(
                    &mut pending,
                    &mut top_k,
                    k,
                    point,
                    T::zero(),
                    T::max_value(),
                    epsilon,
                );
            }
            Some(top_k)
        }
    }

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
            while !pending.is_empty() {
                self.knn_one_step(
                    &mut pending,
                    &mut top_k,
                    k,
                    point,
                    T::zero(),
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
            let mut pending = vec![(T::min_value(), self)];
            while !pending.is_empty() {
                self.within_one_step(&mut pending, &mut neighbors, point, T::zero(), radius);
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
            let mut pending = vec![(T::min_value(), self)];
            while !pending.is_empty() {
                cnt += self.within_count_one_step(&mut pending, point, T::zero(), radius);
            }
            Some(cnt)
        }
    }

    // Helper function that finds the bounding box for each (sub)kdtree
    fn find_bounds(data: &[impl KdLeaf<'a, T>], dim: usize) -> (Vec<T>, Vec<T>) {
        let mut min_bounds = vec![T::max_value(); dim];
        let mut max_bounds = vec![T::min_value(); dim];

        for elem in data.iter() {
            for i in 0..dim {
                min_bounds[i] = min_bounds[i].min(elem.value_at(i));
                max_bounds[i] = max_bounds[i].max(elem.value_at(i));
            }
        }
        (min_bounds, max_bounds)
    }
}

pub trait KNNRegressor<'a, T: Float + Into<f64> + 'static, A: Float + Into<f64>>:
    KDTQ<'a, T, A>
{
    fn knn_regress(&self, k: usize, point: &[T], max_dist_bound: T, how: KNNMethod) -> Option<f64> {
        let knn = self.knn_bounded(k, point, max_dist_bound);
        match knn {
            Some(nn) => match how {
                KNNMethod::DInvW => {
                    let weights = nn
                        .iter()
                        .map(|nb| (nb.dist + T::one()).recip().into())
                        .collect::<Vec<f64>>();
                    let sum = weights.iter().copied().sum::<f64>();
                    Some(
                        nn.into_iter()
                            .zip(weights.into_iter())
                            .fold(0f64, |acc, (nb, w)| acc + w * nb.to_item().into())
                            / sum,
                    )
                }
                KNNMethod::NoW => {
                    let n = nn.len() as f64;
                    Some(
                        nn.into_iter()
                            .fold(A::zero(), |acc, nb| acc + nb.to_item())
                            .into()
                            / n,
                    )
                }
            },
            None => None,
        }
    }
}

pub trait KNNClassifier<'a, T: Float + 'static>: KDTQ<'a, T, u32> {
    fn knn_classif(&self, k: usize, point: &[T], max_dist_bound: T, how: KNNMethod) -> Option<u32> {
        let knn = self.knn_bounded(k, point, max_dist_bound);
        match knn {
            Some(nn) => match how {
                KNNMethod::DInvW => {
                    todo!()
                }
                KNNMethod::NoW => {
                    todo!()
                }
            },
            None => None,
        }
    }
}

// pub enum KDT<'a, T:Float + 'static, A:Copy> {
//     L2(Kdtree<'a, T, A>),
//     LP(LpKdtree<'a, T, A>),
// }

// impl <'a, T: Float + 'static, A: Copy> KDTQ<'a, T, A> for KDT<'a, T, A> {
//     fn dim(&self) -> usize {
//         match self {
//             KDT::L2(t) => t.dim(),
//             KDT::LP(t) => t.dim(),
//         }
//     }

//     fn knn_one_step(
//         &self,
//         pending: &mut Vec<(T, &Self)>,
//         top_k: &mut Vec<NB<T, A>>,
//         k: usize,
//         point: &[T],
//         point_norm_cache: T,
//         max_dist_bound: T,
//         epsilon: T,
//     ) {
//         match self {
//             KDT::L2(t) => t.knn_one_step(pending, top_k, k, point, point_norm_cache, max_dist_bound, epsilon),
//             KDT::LP(t) => t.knn_one_step(pending, top_k, k, point, point_norm_cache, max_dist_bound, epsilon),
//         }
//     }

//     fn within_one_step(
//         &self,
//         pending: &mut Vec<(T, &Self)>,
//         neighbors: &mut Vec<NB<T, A>>,
//         point: &[T],
//         point_norm_cache: T,
//         radius: T,
//     ) {
//         todo!()
//     }

//     fn within_count_one_step(
//         &self,
//         pending: &mut Vec<(T, &Self)>,
//         point: &[T],
//         point_norm_cache: T,
//         radius: T,
//     ) -> u32 {
//         todo!()
//     }
// }
