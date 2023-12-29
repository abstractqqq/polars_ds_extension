use super::consts::EN_STOPWORDS;
use super::snowball::{algorithms, SnowballEnv};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use pyo3_polars::export::polars_core::utils::rayon::prelude::ParallelIterator;

#[inline]
pub fn snowball_stem(word: Option<&str>, no_stopwords: bool) -> Option<String> {
    match word {
        Some(w) => {
            if (no_stopwords) & (EN_STOPWORDS.binary_search(&w).is_ok()) {
                None
            } else if w.parse::<f64>().is_ok() {
                None
            } else {
                let mut env: SnowballEnv<'_> = SnowballEnv::create(w);
                algorithms::english_stemmer::stem(&mut env);
                Some(env.get_current().to_string())
            }
        }
        _ => None,
    }
}

#[polars_expr(output_type=Utf8)]
fn pl_snowball_stem(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].utf8()?;
    let no_stop = inputs[1].bool()?;
    let no_stop = no_stop.get(0).unwrap();
    let parallel = inputs[2].bool()?;
    let parallel = parallel.get(0).unwrap();
    let out: Utf8Chunked = if parallel {
        ca.par_iter()
            .map(|op_s| snowball_stem(op_s, no_stop))
            .collect()
    } else {
        ca.apply_generic(|op_s| snowball_stem(op_s, no_stop))
    };
    Ok(out.into_series())
}
