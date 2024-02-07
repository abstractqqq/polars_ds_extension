mod consts;
mod fuzz;
mod hamming;
mod inflections;
mod is_stopword;
mod jaro;
mod knn_strs;
mod levenshtein;
mod osa;
mod overlap;
mod snowball;
mod snowball_stem;
mod sorensen_dice;
mod str_jaccard;

// Hashbrown has better perf than Rust's HashSet
use hashbrown::HashSet;

#[inline(always)]
pub fn str_set_sim_helper(w1: &str, w2: &str, n: usize) -> (usize, usize, usize) {
    // output: set 1 size, set 2 size, intersection size

    let w1_len = w1.len();
    let w2_len = w2.len();

    // as long as intersection size is 0, output will be correct
    if (w1_len == 0) || (w2_len == 0) {
        return (0, 0, 0);
    }

    // Both are nonempty
    // Another version that has slices of size <= n?
    let s1: HashSet<&[u8]> = if w1_len < n {
        HashSet::from_iter([w1.as_bytes()])
    } else {
        HashSet::from_iter(w1.as_bytes().windows(n))
    };

    let s2: HashSet<&[u8]> = if w2_len < n {
        HashSet::from_iter([w2.as_bytes()])
    } else {
        HashSet::from_iter(w2.as_bytes().windows(n))
    };

    let intersection = s1.intersection(&s2).count();
    (s1.len(), s2.len(), intersection)
}
