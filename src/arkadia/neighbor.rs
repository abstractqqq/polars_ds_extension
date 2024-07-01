/// NB: Neighbor, search result
/// (Data, and distance)
use num::Float;

pub struct NB<T: Float, A> {
    pub dist: T,
    pub item: A,
}

impl<T: Float, A> NB<T, A> {
    pub fn to_item(self) -> A {
        self.item
    }

    pub fn to_dist(self) -> T {
        self.dist
    }

    pub fn to_pair(self) -> (T, A) {
        (self.dist, self.item)
    }
    /// Is the neighbor almost equal to the point itself?
    pub fn identity(&self) -> bool {
        self.dist <= T::epsilon()
    }
}

impl<T: Float, A> PartialEq for NB<T, A> {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}

impl<T: Float, A> PartialOrd for NB<T, A> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.dist.partial_cmp(&other.dist)
    }
}

impl<T: Float, A> Eq for NB<T, A> {}

impl<T: Float, A> Ord for NB<T, A> {
    // Unwrap is safe because in all use cases, the data should not contain any non-finite values.
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist.partial_cmp(&other.dist).unwrap()
    }

    fn max(self, other: Self) -> Self
    where
        Self: Sized,
    {
        std::cmp::max_by(self, other, |a, b| a.dist.partial_cmp(&b.dist).unwrap())
    }

    fn min(self, other: Self) -> Self
    where
        Self: Sized,
    {
        std::cmp::min_by(self, other, |a, b| a.dist.partial_cmp(&b.dist).unwrap())
    }

    fn clamp(self, min: Self, max: Self) -> Self
    where
        Self: Sized,
        Self: PartialOrd,
    {
        assert!(min <= max);
        if self.dist < min.dist {
            min
        } else if self.dist > max.dist {
            max
        } else {
            self
        }
    }
}
