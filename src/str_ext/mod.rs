mod fuzz;
mod generic_str_distancer;
mod hamming;
mod inflections;
mod jaro;
mod lcs_seq;
mod lcs_str;
mod levenshtein;
mod nearest_str;
mod osa;
mod overlap;
mod sorensen_dice;
mod str_cleaning;
mod str_jaccard;
mod tversky;

// foldhash has better perf than Rust's HashSet for these operations
use foldhash::{HashSet, HashSetExt};

#[inline(always)]
pub fn str_to_hashset(s: &str, ngram: usize) -> HashSet<&[u8]> {
    let s_len = s.len();
    if s_len < ngram {
        HashSet::from_iter([s.as_bytes()])
    } else {
        let mut set = HashSet::with_capacity(s_len - ngram + 1);
        set.extend(s.as_bytes().windows(ngram));
        set
    }
}

#[inline(always)]
pub fn fill_str_hashset<'a>(s: &'a str, ngram: usize, set: &mut HashSet<&'a [u8]>) {
    set.clear();
    let s_len = s.len();
    if s_len < ngram {
        set.insert(s.as_bytes());
    } else {
        set.extend(s.as_bytes().windows(ngram));
    }
}

#[inline(always)]
pub fn str_set_sim_helper(w1: &str, w2: &str, ngram: usize) -> (usize, usize, usize) {
    if w1.is_empty() || w2.is_empty() {
        return (0, 0, 0);
    }

    let s1 = str_to_hashset(w1, ngram);
    let s2 = str_to_hashset(w2, ngram);

    let intersection = s1.intersection(&s2).count();
    (s1.len(), s2.len(), intersection)
}
