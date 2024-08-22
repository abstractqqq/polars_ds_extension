use itertools::Itertools;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use unicode_normalization::UnicodeNormalization;

enum NormalForm {
    NFC,
    NFKC,
    NFD,
    NFKD,
}

impl TryFrom<String> for NormalForm {
    type Error = PolarsError;
    fn try_from(value: String) -> PolarsResult<Self> {
        match value.to_uppercase().as_ref() {
            "NFC" => Ok(Self::NFC),
            "NFKC" => Ok(Self::NFKC),
            "NFD" => Ok(Self::NFD),
            "NFKD" => Ok(Self::NFKD),
            _ => Err(PolarsError::ComputeError("Unknown NormalizeForm.".into())),
        }
    }
}

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
    let form: NormalForm = kwargs.form.try_into()?;
    let out = match form {
        NormalForm::NFC => ca.apply_to_buffer(|val, buf| *buf = val.nfc().collect()),
        NormalForm::NFKC => ca.apply_to_buffer(|val, buf| *buf = val.nfkc().collect()),
        NormalForm::NFD => ca.apply_to_buffer(|val, buf| *buf = val.nfd().collect()),
        NormalForm::NFKD => ca.apply_to_buffer(|val, buf| *buf = val.nfkd().collect()),
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

fn _normalize_whitespace(value: &str, output: &mut String) {
    *output = value.split_whitespace().join(" ")
}

#[polars_expr(output_type=String)]
fn normalize_whitespace(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out = ca.apply_to_buffer(_normalize_whitespace);

    Ok(out.into_series())
}
