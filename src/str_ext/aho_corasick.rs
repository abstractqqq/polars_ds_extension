use aho_corasick::{AhoCorasick, MatchKind};
use polars::prelude::*;
use pyo3_polars::{derive::polars_expr, export::polars_core::utils::rayon::iter::ParallelIterator};

fn list_u16_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "list_u16",
        DataType::List(Box::new(DataType::UInt16)),
    ))
}

fn list_str_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "list_str",
        DataType::List(Box::new(DataType::Utf8)),
    ))
}

fn str_to_matchkind(value: &str) -> MatchKind {
    match value {
        "left_most_longest" => MatchKind::LeftmostLongest,
        "left_most_first" => MatchKind::LeftmostFirst,
        _ => MatchKind::Standard,
    }
}

#[polars_expr(output_type_func=list_u16_output)]
fn pl_ac_match(inputs: &[Series]) -> PolarsResult<Series> {
    let str_col = inputs[0].utf8()?;
    let patterns = inputs[1].utf8()?;
    let patterns: Vec<&str> = patterns.into_iter().filter_map(|s| s).collect();
    let n_pat = patterns.len();
    if n_pat > u16::MAX as usize {
        return Err(PolarsError::ComputeError(
            "AC: Too many patterns to match.".into(),
        ));
    }

    let case_insensitive = inputs[2].bool()?;
    let case_insensitive = case_insensitive.get(0).unwrap();

    let mk = inputs[3].utf8()?;
    let mk = mk.get(0).unwrap();
    let mk: MatchKind = str_to_matchkind(mk);

    let ac_builder = AhoCorasick::builder()
        .ascii_case_insensitive(case_insensitive)
        .match_kind(mk)
        .build(patterns);

    match ac_builder {
        Ok(ac) => {
            // Is there a way to make this work?
            // let out:ChunkedArray<FixedSizeListType> = str_col.apply_values_generic(op);

            let mut builder = ListPrimitiveChunkedBuilder::<UInt16Type>::new(
                "match",
                str_col.len(),
                n_pat,
                DataType::UInt16,
            );
            // n_pat is just capacity to initialize. We can go beyond n_pat, with a performance penalty.
            // Right now this is not the best choice.
            // One thing we can do is to enforce the length to be <= values_capacity.

            for op_s in str_col.into_iter() {
                if let Some(s) = op_s {
                    let matches = ac
                        .find_iter(s)
                        .map(|m| m.pattern().as_u32() as u16)
                        .collect::<Vec<u16>>();
                    if matches.is_empty() {
                        builder.append_null();
                    } else {
                        builder.append_slice(&matches);
                    }
                } else {
                    builder.append_null();
                }
            }
            let out = builder.finish();
            Ok(out.into_series())
        }
        Err(e) => Err(PolarsError::ComputeError(e.to_string().into())),
    }
}

#[polars_expr(output_type_func=list_str_output)]
fn pl_ac_match_str(inputs: &[Series]) -> PolarsResult<Series> {
    let str_col = inputs[0].utf8()?;
    let patterns = inputs[1].utf8()?;
    let patterns: Vec<&str> = patterns.into_iter().filter_map(|s| s).collect();
    let n_pat = patterns.len();
    if n_pat > u16::MAX as usize {
        return Err(PolarsError::ComputeError(
            "AC: Too many patterns to match.".into(),
        ));
    }

    let case_insensitive = inputs[2].bool()?;
    let case_insensitive = case_insensitive.get(0).unwrap();

    let mk = inputs[3].utf8()?;
    let mk = mk.get(0).unwrap();
    let mk: MatchKind = str_to_matchkind(mk);

    let ac_builder = AhoCorasick::builder()
        .ascii_case_insensitive(case_insensitive)
        .match_kind(mk)
        .build(&patterns);

    match ac_builder {
        Ok(ac) => {
            // Is there a way to make this work?
            // let out:ChunkedArray<FixedSizeListType> = str_col.apply_values_generic(op);

            let mut builder = ListUtf8ChunkedBuilder::new("match", str_col.len(), 20);
            // n_pat is just capacity to initialize. We can go beyond n_pat, with a performance penalty.
            // Right now this is not the best choice.

            for op_s in str_col.into_iter() {
                if let Some(s) = op_s {
                    let matches = ac
                        .find_iter(s)
                        .map(|m| patterns[m.pattern().as_usize()])
                        .collect::<Utf8Chunked>();
                    if matches.is_empty() {
                        builder.append_null();
                    } else {
                        let s: Series = matches.into();
                        let _ = builder.append_series(&s)?;
                    }
                } else {
                    builder.append_null();
                }
            }
            let out = builder.finish();
            Ok(out.into_series())
        }
        Err(e) => Err(PolarsError::ComputeError(e.to_string().into())),
    }
}


#[polars_expr(output_type=Utf8)]
fn pl_ac_replace(inputs: &[Series]) -> PolarsResult<Series> {
    let str_col = inputs[0].utf8()?;
    let patterns = inputs[1].utf8()?;
    let patterns: Vec<&str> = patterns.into_iter().filter_map(|s| s).collect();

    let replace = inputs[2].utf8()?;
    let replace: Vec<&str> = replace.into_iter().filter_map(|s| s).collect();

    let parallel = inputs[3].bool()?;
    let parallel = parallel.get(0).unwrap();

    let ac_builder = AhoCorasick::builder().build(&patterns);

    match ac_builder {
        Ok(ac) => {
            let op = |op_s: Option<&str>| {
                let s = op_s?;
                let bytes = ac.replace_all_bytes(s.as_bytes(), &replace);
                match String::from_utf8(bytes) {
                    Ok(new_str) => Some(new_str),
                    _ => None,
                }
            };
            let out: Utf8Chunked = if parallel {
                str_col.par_iter().map(op).collect()
            } else {
                str_col.apply_generic(op)
            };
            Ok(out.into_series())
        }
        Err(e) => Err(PolarsError::ComputeError(e.to_string().into())),
    }
}
