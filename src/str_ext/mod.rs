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
// E.g. Levenshtein has 3x better performance than my own implementation.
// However, I saw people saying in the github issue section that things can be improved.
// The strsim project is no longer maintained. If there is a need to improve performance
// further, we can fork and develop it ourselves (currently just me).

// Hashbrown has better perf than Rust's HashSet
use hashbrown::HashSet;

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
