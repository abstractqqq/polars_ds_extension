use super::consts::EN_STOPWORDS;
use super::snowball::{algorithms, SnowballEnv};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use std::fmt::Write;

fn snowball(s: &str, no_stopwords: bool, output: &mut String) {
    if s.parse::<f64>().is_ok() || (no_stopwords && EN_STOPWORDS.binary_search(&s).is_ok()) {
        write!(output, "{}", "").unwrap()
    } else {
        let mut env: SnowballEnv<'_> = SnowballEnv::create(s);
        algorithms::english_stemmer::stem(&mut env);
        write!(output, "{}", env.get_current()).unwrap()
    };
}

#[polars_expr(output_type=String)]
fn pl_snowball_stem(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let no_stop = inputs[1].bool()?;
    let no_stop = no_stop.get(0).unwrap();
    let out = ca.apply_to_buffer(|s, buf| snowball(s, no_stop, buf));
    Ok(out.into_series())
}
