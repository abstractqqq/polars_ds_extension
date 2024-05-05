use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use unicode_normalization::UnicodeNormalization;

fn _remove_non_ascii(value: &str, output: &mut String) {
    *output = value.chars().filter(char::is_ascii).collect()
}

#[polars_expr(output_type=String)]
fn remove_non_ascii(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out = ca.apply_to_buffer(_remove_non_ascii);

    Ok(out.into_series())
}

fn _remove_diacritics(value: &str, output: &mut String) {
    *output = value.nfd().filter(char::is_ascii).collect()
}

#[polars_expr(output_type=String)]
fn remove_diacritics(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out = ca.apply_to_buffer(_remove_diacritics);

    Ok(out.into_series())
}

#[derive(serde::Deserialize)]
struct NormalizeKwargs {
    form: String,
}

#[polars_expr(output_type=String)]
fn normalize_string(inputs: &[Series], kwargs: NormalizeKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let form = kwargs.form;

    let out = if form == "NFC" {
        ca.apply_to_buffer(|val, buf| *buf = val.nfc().collect())
    } else if form == "NFKC" {
        ca.apply_to_buffer(|val, buf| *buf = val.nfkc().collect())
    } else if form == "NFD" {
        ca.apply_to_buffer(|val, buf| *buf = val.nfd().collect())
    } else if form == "NFKD" {
        ca.apply_to_buffer(|val, buf| *buf = val.nfkd().collect())
    } else {
        panic!()
    };

    Ok(out.into_series())
}

#[derive(serde::Deserialize)]
struct MapWordsKwargs {
    mapping: ahash::HashMap<String, String>,
}

fn _map_words(value: &str, mapping: &ahash::HashMap<String, String>, output: &mut String) {
    let vec: Vec<&str> = value
        .split_whitespace()
        .map(|word| match mapping.get(word) {
            Some(val) => val,
            None => word,
        })
        .collect();

    output.push_str(vec.join(" ").as_str())
}

#[polars_expr(output_type=String)]
fn map_words(inputs: &[Series], kwargs: MapWordsKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out = ca.apply_to_buffer(|val, buf| _map_words(val, &kwargs.mapping, buf));

    Ok(out.into_series())
}
