mod aho_corasick;
mod consts;
mod hamming;
mod jaro;
mod levenshtein;
mod overlap;
mod snowball;
mod snowball_stem;
mod sorensen_dice;
mod str_jaccard;

// Most str dist / similarity metrics are powered by strsim. They have good performance.
// However, I saw people saying in the github issue section that things can be improved.
// The fastest Levenshtein algorithm for large strings is Meyer's bitparallel algorithm.
// Though, I do not understand it well enough to implement it.
// Right now, for common, smaller strings, my Levenshtein seems to be doing fine.

// Hashbrown has better perf than Rust's HashSet
use hashbrown::HashSet;
use itertools::Itertools;

#[inline]
pub fn str_set_sim_helper(w1: &str, w2: &str, n: usize) -> (usize, usize, usize) {
    // output: set 1 size, set 2 size, intersection size

    let w1_len = w1.len();
    let w2_len = w2.len();

    // as long as intersection size is 0, output will be correct
    if (w1_len == 0) && (w2_len == 0) {
        return (0, 0, 0);
    } else if (w1_len == 0) | (w2_len == 0) {
        return (0, 0, 0);
    }

    // Both are nonempty
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

#[inline]
pub fn common_char_prefix(a: &[char], b: &[char]) -> usize {
    let (left, _) = a
        .into_iter()
        .zip(b.into_iter())
        .find_position(|(&c1, &c2)| c1 != c2)
        .unwrap_or((0, (&'a', &'a')));
    left
}

#[inline]
pub fn common_char_suffix(a: &[char], b: &[char]) -> usize {
    let (right, _) = a
        .into_iter()
        .rev()
        .zip(b.into_iter().rev())
        .find_position(|(&c1, &c2)| c1 != c2)
        .unwrap_or((0, (&'a', &'a')));
    right
}

#[inline]
/// Strip common prefix, suffix characters
pub fn strip_common<'a>(a: &'a [char], b: &'a [char]) -> (&'a [char], &'a [char]) {
    let left = common_char_prefix(a, b);
    let right = common_char_suffix(a, b);

    (&a[left..(a.len() - right)], &b[left..(b.len() - right)])
}
