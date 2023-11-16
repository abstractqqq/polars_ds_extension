use std::cmp::PartialOrd;

// Faster by branchless?

// pub fn binary_search_any<T: PartialOrd>(arr: &[T], t:&T) -> Option<usize> {

//     let mut left = 0;
//     let mut right = arr.len();

//     while left != right {
//         let mid = left + (right - left) / 2;
//         if let Some(c) = arr[mid].partial_cmp(t) {
//             match c {
//                 std::cmp::Ordering::Greater => right = mid - 1,
//                 _ => left = mid
//             }
//         } else {
//             return None
//         }
//     }
//     Some(left)
// }

// pub fn binary_search_left<T: PartialOrd>(arr: &[T], t:&T) -> Option<usize> {

//     let mut left = 0;
//     let mut right = arr.len();

//     while left < right {
//         let mid = left + (right - left) / 2;
//         if let Some(c) = arr[mid].partial_cmp(t) {
//             match c {
//                 std::cmp::Ordering::Less => left = mid + 1,
//                 _ => right = mid
//             }
//         } else {
//             return None
//         }
//     }
//     Some(left)
// }

pub fn binary_search_right<T: PartialOrd>(arr: &[T], t: &T) -> Option<usize> {
    let mut left = 0;
    let mut right = arr.len();

    while left < right {
        let mid = left + (right - left) / 2;
        if let Some(c) = arr[mid].partial_cmp(t) {
            match c {
                std::cmp::Ordering::Greater => right = mid,
                _ => left = mid + 1,
            }
        } else {
            return None;
        }
    }
    Some(left)
}
