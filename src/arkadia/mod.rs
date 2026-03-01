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
pub mod kdt;
// pub mod ball_tree;
pub mod leaf;
pub mod neighbor;
pub mod utils;

pub use kdt::KDT;
// pub use ball_tree::BallTree;
pub use leaf::{KdLeaf, Leaf};
pub use neighbor::NB;
use serde::Deserialize;
pub use utils::{
    slice_to_empty_leaves, slice_to_leaves, suggest_capacity, SplitMethod,
};
use num::Float;
use crate::utils::{
    l1_distance, squared_l2_distance, linf_distance
};
// ---------------------------------------------------------------------------------------------------------
#[derive(Clone, Copy)]
pub enum KNNDist {
    L1,
    L2,
    SQL2,
    LINF
}

impl TryFrom<String> for KNNDist {
    type Error = String;
    fn try_from(value: String) -> Result<Self, Self::Error> {
        match value.to_lowercase().as_ref() {
            "l1" => Ok(KNNDist::L1)
            , "sql2" => Ok(KNNDist::SQL2)
            , "l2" => Ok(KNNDist::L2)
            , "linf" | "inf" => Ok(KNNDist::LINF)
            , _ => Err(format!("Unknown distance indicator: {}", value))
        }
    }
}

impl KNNDist {
    pub fn dist(&self, v1: &[f64], v2: &[f64]) -> f64 {
        match self {
            KNNDist::L1 => l1_distance(v1, v2),
            KNNDist::L2 => squared_l2_distance(v1, v2).sqrt(),
            KNNDist::SQL2 => squared_l2_distance(v1, v2),
            KNNDist::LINF => linf_distance(v1, v2),
        }
    }
}


#[derive(Clone, Copy, Default, Deserialize)]
pub enum KNNMethod {
    P1Weighted, // Distance Inversion Weighted. E.g. Use (1/(1+d)) to weight the regression / classification
    Weighted, // Distance Inversion Weighted. E.g. Use (1/d) to weight the regression / classification
    #[default]
    NotWeighted, // No Weight
}

impl KNNMethod {
    pub fn new(weighted: bool, min_dist: f64) -> Self {
        if weighted {
            if min_dist <= f64::epsilon() {
                Self::P1Weighted
            } else {
                Self::Weighted
            }
        } else {
            Self::NotWeighted
        }
    }
}

/// K Dimensional Tree Queries. Should be the same for ball trees, etc.
pub trait SpatialQueries<'a, A> {
    fn dim(&self) -> usize;

    fn knn_one_step(
        &self,
        pending: &mut Vec<(f64, &Self)>,
        top_k: &mut Vec<NB<f64, A>>,
        k: usize,
        point: &[f64],
        max_dist_bound: f64,
        epsilon: f64,
    );

    fn within_one_step(
        &self,
        pending: &mut Vec<(f64, &Self)>,
        neighbors: &mut Vec<NB<f64, A>>,
        point: &[f64],
        radius: f64,
    );

    fn within_count_one_step(&self, pending: &mut Vec<(f64, &Self)>, point: &[f64], radius: f64) -> u32;

    fn knn(&self, k: usize, point: &[f64], epsilon: f64) -> Option<Vec<NB<f64, A>>> {
        if k == 0 || (point.len() != self.dim()) || (point.iter().any(|x| !x.is_finite())) {
            None
        } else {
            // Always allocate 1 more.
            let mut top_k = Vec::with_capacity(k + 1);
            let mut pending = Vec::with_capacity(k + 1);
            pending.push((f64::min_value(), self));
            while !pending.is_empty() {
                self.knn_one_step(&mut pending, &mut top_k, k, point, f64::max_value(), epsilon);
            }
            Some(top_k)
        }
    }

    fn knn_bounded(
        &self,
        k: usize,
        point: &[f64],
        max_dist_bound: f64,
        epsilon: f64,
    ) -> Option<Vec<NB<f64, A>>> {
        if k == 0
            || (point.len() != self.dim())
            || (point.iter().any(|x| !x.is_finite()))
            || max_dist_bound <= f64::epsilon()
        {
            None
        } else {
            // Always allocate 1 more.
            let mut top_k = Vec::with_capacity(k + 1);
            let mut pending = Vec::with_capacity(k + 1);
            pending.push((f64::min_value(), self));
            while !pending.is_empty() {
                self.knn_one_step(&mut pending, &mut top_k, k, point, max_dist_bound, epsilon);
            }
            Some(top_k)
        }
    }

    fn knn_bounded_unchecked(
        &self,
        k: usize,
        point: &[f64],
        max_dist_bound: f64,
        epsilon: f64,
    ) -> Vec<NB<f64, A>> {
        let mut top_k = Vec::with_capacity(k + 1);
        let mut pending = Vec::with_capacity(k + 1);
        pending.push((f64::min_value(), self));
        while !pending.is_empty() {
            self.knn_one_step(&mut pending, &mut top_k, k, point, max_dist_bound, epsilon);
        }
        top_k
    }

    fn within(&self, point: &[f64], radius: f64, sort: bool) -> Option<Vec<NB<f64, A>>> {
        // radius is actually squared radius
        if radius <= 0f64 + f64::epsilon() || (point.iter().any(|x| !x.is_finite())) {
            None
        } else {
            // Always allocate some.
            let mut neighbors = Vec::with_capacity(32);
            let mut pending = vec![(f64::min_value(), self)];
            while !pending.is_empty() {
                self.within_one_step(&mut pending, &mut neighbors, point, radius);
            }
            if sort {
                neighbors.sort_unstable();
            }
            neighbors.shrink_to_fit();
            Some(neighbors)
        }
    }

    fn within_count(&self, point: &[f64], radius: f64) -> Option<u32> {
        // radius is actually squared radius
        if radius <= 0f64 + f64::epsilon() || (point.iter().any(|x| !x.is_finite())) {
            None
        } else {
            // Always allocate some.
            let mut cnt = 0u32;
            let mut pending = vec![(f64::min_value(), self)];
            while !pending.is_empty() {
                cnt += self.within_count_one_step(&mut pending, point, radius);
            }
            Some(cnt)
        }
    }
}

pub trait KNNRegressor<'a, A: Float + Into<f64>>:
    SpatialQueries<'a, A>
{
    fn knn_regress(
        &self,
        k: usize,
        point: &[f64],
        min_dist_bound: f64,
        max_dist_bound: f64,
        how: KNNMethod,
    ) -> Option<f64> {
        let knn = self
            .knn_bounded(k, point, max_dist_bound, 0f64)
            .map(|nn| {
                nn.into_iter()
                    .filter(|nb| nb.dist >= min_dist_bound)
                    .collect::<Vec<_>>()
            });
        match knn {
            Some(nn) if !nn.is_empty() => match how {
                KNNMethod::P1Weighted => {
                    let weights = nn
                        .iter()
                        .map(|nb| (1.0 + nb.dist).recip().into())
                        .collect::<Vec<f64>>();
                    let sum = weights.iter().copied().sum::<f64>();
                    Some(
                        nn.into_iter()
                            .zip(weights.into_iter())
                            .fold(0f64, |acc, (nb, w)| acc + w * nb.to_item().into())
                            / sum,
                    )
                }
                KNNMethod::Weighted => {
                    let weights = nn
                        .iter()
                        .map(|nb| nb.dist.recip().into())
                        .collect::<Vec<f64>>();
                    let sum = weights.iter().copied().sum::<f64>();
                    Some(
                        nn.into_iter()
                            .zip(weights.into_iter())
                            .fold(0f64, |acc, (nb, w)| acc + w * nb.to_item().into())
                            / sum,
                    )
                }
                KNNMethod::NotWeighted => {
                    let n = nn.len() as f64;
                    Some(
                        nn.into_iter()
                            .fold(A::zero(), |acc, nb| acc + nb.to_item())
                            .into()
                            / n,
                    )
                }
            },
            _ => None,
        }
    }
}
