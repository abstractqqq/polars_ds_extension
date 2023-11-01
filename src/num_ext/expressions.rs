use faer::prelude::*;
use faer::solvers::Qr;
use faer::{IntoFaer, IntoNdarray};
use ndarray::ArrayView2;
use num;
use polars::prelude::*;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

#[polars_expr(output_type=Int64)]
fn pl_gcd(inputs: &[Series]) -> PolarsResult<Series> {
    let ca1 = inputs[0].i64()?;
    let ca2 = inputs[1].i64()?;

    if ca2.len() == 1 {
        let b = ca2.get(0).unwrap();
        let out: Int64Chunked = ca1
            .into_iter()
            .map(|op_a| {
                if let Some(a) = op_a {
                    Some(num::integer::gcd(a, b))
                } else {
                    None
                }
            })
            .collect();
        Ok(out.into_series())
    } else if ca1.len() == ca2.len() {
        let out: Int64Chunked = ca1
            .into_iter()
            .zip(ca2.into_iter())
            .map(|(op_a, op_b)| {
                if let (Some(a), Some(b)) = (op_a, op_b) {
                    Some(num::integer::gcd(a, b))
                } else {
                    None
                }
            })
            .collect();
        Ok(out.into_series())
    } else {
        Err(PolarsError::ComputeError(
            "Inputs must have the same length.".into(),
        ))
    }
}

#[polars_expr(output_type=Int64)]
fn pl_lcm(inputs: &[Series]) -> PolarsResult<Series> {
    let ca1 = inputs[0].i64()?;
    let ca2 = inputs[1].i64()?;

    if ca2.len() == 1 {
        let b = ca2.get(0).unwrap();
        let out: Int64Chunked = ca1
            .into_iter()
            .map(|op_a| {
                if let Some(a) = op_a {
                    Some(num::integer::lcm(a, b))
                } else {
                    None
                }
            })
            .collect();
        Ok(out.into_series())
    } else if ca1.len() == ca2.len() {
        let out: Int64Chunked = ca1
            .into_iter()
            .zip(ca2.into_iter())
            .map(|(op_a, op_b)| {
                if let (Some(a), Some(b)) = (op_a, op_b) {
                    Some(num::integer::lcm(a, b))
                } else {
                    None
                }
            })
            .collect();
        Ok(out.into_series())
    } else {
        Err(PolarsError::ComputeError(
            "Inputs must have the same length.".into(),
        ))
    }
}

// I am not sure this is right. I still don't quite understand the purpose of this.
fn lstsq_output(input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "betas",
        DataType::Struct(
            input_fields[1..]
                .iter()
                .map(|f| Field::new(&format!("beta_{}", f.name()), DataType::Float64))
                .collect(),
        ),
    ))
}

/// This function returns a struct series with betas, y_pred, and residuals
#[polars_expr(output_type_func=lstsq_output)]
fn lstsq(inputs: &[Series]) -> PolarsResult<Series> {
    // Iterate over the inputs and name each one with .with_name() and collect them into a vector
    let mut series_vec = Vec::new();

    // Have to name each one because they don't have names if passed in via .over()
    for (i, series) in inputs[1..].iter().enumerate() {
        let series = series.clone().with_name(&format!("x{i}"));
        series_vec.push(series);
    }
    let beta_names: Vec<String> = series_vec.iter().map(|s| s.name().to_string()).collect();
    let y = &inputs[0]
        .f64()
        .unwrap()
        .to_ndarray()
        .unwrap()
        .to_owned()
        .into_shape((inputs[0].len(), 1))
        .unwrap();
    let y = y.view().into_faer();

    // Create a polars DataFrame from the input series
    let todf = DataFrame::new(series_vec);
    match todf {
        Ok(df) => {
            let x = df
                .to_ndarray::<Float64Type>(IndexOrder::Fortran)
                .unwrap()
                .to_owned();
            let x = x.view().into_faer();
            Qr::new(x);
            let betas = Qr::new(x).solve_lstsq(y);
            let preds = x * &betas;
            let preds_array = preds.as_ref().into_ndarray();
            let resids = y - &preds;
            let resid_array: ArrayView2<f64> = resids.as_ref().into_ndarray();
            let betas = betas.as_ref().into_ndarray();

            let mut out_series: Vec<Series> = betas
                .iter()
                .zip(beta_names.iter())
                .map(|(beta, name)| Series::new(name, vec![*beta; inputs[0].len()]))
                .collect();
            // Add a series of residuals and y_pred to the output
            let y_pred_series =
                Series::new("y_pred", preds_array.iter().copied().collect::<Vec<f64>>());
            let resid_series =
                Series::new("resid", resid_array.iter().copied().collect::<Vec<f64>>());
            out_series.push(y_pred_series);
            out_series.push(resid_series);
            let out = StructChunked::new("results", &out_series)?.into_series();
            Ok(out)
        }
        Err(e) => {
            println!("Error: {}", e);
            PolarsResult::Err(e)
        }
    }
}
