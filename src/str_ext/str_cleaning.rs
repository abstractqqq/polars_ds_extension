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

#[polars_expr(output_type=String)]
fn remove_non_ascii(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out =
        ca.apply_into_string_amortized(|s, buf| *buf = s.chars().filter(char::is_ascii).collect());
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn remove_diacritics(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out =
        ca.apply_into_string_amortized(|s, buf| *buf = s.nfd().filter(char::is_ascii).collect());
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
        NormalForm::NFC => ca.apply_into_string_amortized(|val, buf| *buf = val.nfc().collect()),
        NormalForm::NFKC => ca.apply_into_string_amortized(|val, buf| *buf = val.nfkc().collect()),
        NormalForm::NFD => ca.apply_into_string_amortized(|val, buf| *buf = val.nfd().collect()),
        NormalForm::NFKD => ca.apply_into_string_amortized(|val, buf| *buf = val.nfkd().collect()),
    };
    Ok(out.into_series())
}

#[derive(serde::Deserialize)]
struct MapWordsKwargs {
    mapping: ahash::HashMap<String, String>,
}

#[polars_expr(output_type=String)]
fn map_words(inputs: &[Series], kwargs: MapWordsKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let mapping = kwargs.mapping;
    let out = ca.apply_into_string_amortized(|s, buf| {
        buf.push_str(
            s.split_whitespace()
                .map(|word| mapping.get(word).map_or(word, |v| v))
                .join(" ")
                .as_ref(),
        )
    });
    Ok(out.into_series())
}

#[polars_expr(output_type=String)]
fn normalize_whitespace(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out = ca.apply_into_string_amortized(|s, buf| *buf = s.split_whitespace().join(" "));
    Ok(out.into_series())
}
