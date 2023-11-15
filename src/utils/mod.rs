use std::cmp::PartialOrd;

pub enum BinarySearchSide {
    Left,
    Right,
    Any
}

pub fn binary_search_w_side<T: PartialOrd>(arr: &[T], t:&T, side:&BinarySearchSide) -> Option<usize> {
    
    // This assumes arr is sorted
    // This extends Vec's binary search to floats, and adds a side to it.
    match side {
        BinarySearchSide::Any => binary_search_any(arr, t),
        BinarySearchSide::Left => binary_search_left(arr, t),
        BinarySearchSide::Right => binary_search_right(arr, t)
    }
}

// Faster by branchless?

#[inline]
fn binary_search_any<T: PartialOrd>(arr: &[T], t:&T) -> Option<usize> {

    let mut left = 0;
    let mut right = arr.len();
    
    while left != right {
        let mid = left + (right - left) / 2;
        if let Some(c) = arr[mid].partial_cmp(t) {
            match c {
                std::cmp::Ordering::Greater => right = mid - 1,
                _ => left = mid
            }
        } else {
            return None
        }
    }
    Some(left)
}

#[inline]
fn binary_search_left<T: PartialOrd>(arr: &[T], t:&T) -> Option<usize> {

    let mut left = 0;
    let mut right = arr.len();
    
    while left < right {
        let mid = left + (right - left) / 2;
        if let Some(c) = arr[mid].partial_cmp(t) {
            match c {
                std::cmp::Ordering::Less => left = mid + 1,
                _ => right = mid
            }
        } else {
            return None
        }
    }
    Some(left)
}

#[inline]
fn binary_search_right<T: PartialOrd>(arr: &[T], t:&T) -> Option<usize> {

    let mut left = 0;
    let mut right = arr.len();
    
    while left < right {
        let mid = left + (right - left) / 2;
        if let Some(c) = arr[mid].partial_cmp(t) {
            match c {
                std::cmp::Ordering::Greater => right = mid,
                _ => left = mid + 1
            }
        } else {
            return None
        }
    }
    Some(left)
}