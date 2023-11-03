use faer::{prelude::*, MatRef, Side};
use faer::{IntoFaer, IntoNdarray};
// use faer::polars::{polars_to_faer_f64, Frame};
use ndarray::{Array2, Array1};
use num;
use polars::prelude::*;
use polars::chunked_array::ops::arity::binary_elementwise;
use pyo3_polars::derive::polars_expr;

fn optional_gcd(op_a:Option<i64>, op_b:Option<i64>) -> Option<i64> {
    if let (Some(a), Some(b)) = (op_a, op_b) {
        Some(num::integer::gcd(a, b))
    } else {
        None
    }
}

fn optional_lcm(op_a:Option<i64>, op_b:Option<i64>) -> Option<i64> {
    if let (Some(a), Some(b)) = (op_a, op_b) {
        Some(num::integer::lcm(a, b))
    } else {
        None
    }
}


#[polars_expr(output_type=Int64)]
fn pl_gcd(inputs: &[Series]) -> PolarsResult<Series> {

    let ca1 = inputs[0].i64()?;
    let ca2 = inputs[1].i64()?;
    if ca2.len() == 1 {
        let b = ca2.get(0).unwrap();
        let out:Int64Chunked = ca1.apply_generic(|op_a:Option<i64>| {
            if let Some(a) = op_a {
                Some(num::integer::gcd(a, b))
            } else {
                None
            }
        });
        Ok(out.into_series())
    } else if ca1.len() == ca2.len() {
        let out:Int64Chunked = binary_elementwise(ca1, ca2, optional_gcd);
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
        let out:Int64Chunked = ca1.apply_generic(|op_a:Option<i64>| {
            if let Some(a) = op_a {
                Some(num::integer::lcm(a, b))
            } else {
                None
            }
        });
        Ok(out.into_series())
    } else if ca1.len() == ca2.len() {
        let out:Int64Chunked = binary_elementwise(ca1, ca2, optional_lcm);
        Ok(out.into_series())
    } else {
        Err(PolarsError::ComputeError(
            "Inputs must have the same length.".into(),
        ))
    }
}


// Use QR to solve
fn faer_lstsq_qr(
    x: MatRef<f64>,
    y: MatRef<f64>
) -> Result<Array2<f64>, String> {

    let qr = x.qr();
    let betas = qr.solve_lstsq(y);
    Ok(betas.as_ref().into_ndarray().to_owned())

}

// Closed form.
fn faer_lstsq_cf(
    x: MatRef<f64>,
    y: MatRef<f64>
) -> Result<Array2<f64>, String> {

    let xt = x.transpose();
    let xtx = xt * x;
    let decomp = xtx.cholesky(Side::Lower); // .unwrap();
    if let Ok(cholesky) = decomp {
        let xtx_inv = cholesky.inverse();
        let betas = xtx_inv * xt * y;
        Ok(
            betas.as_ref().into_ndarray().to_owned()
        )
    } else {
        Err("Linear algebra error. Likely cause: column duplication or extremely high correlation.".to_owned())
    }

}

fn lstsq_beta_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new("betas", DataType::List(Box::new(DataType::Float64))))
}

#[polars_expr(output_type_func=lstsq_beta_output)]
fn pl_lstsq(inputs: &[Series]) -> PolarsResult<Series> {

    let nrows = inputs[0].len();
    let add_bias = inputs[1].bool()?;
    let add_bias:bool = add_bias.get(0).unwrap();  
    // y
    let y = inputs[0].f64()?;
    let y = y.to_ndarray()?.into_shape((nrows,1)).unwrap();
    let y = y.view().into_faer();

    // X, Series is ref counted, so cheap
    let mut vec_series: Vec<Series> = inputs[2..].iter().enumerate().map(
        |(i,s)| s.clone().with_name(&i.to_string())
    ).collect();
    if add_bias {
        let one = Series::new_empty("cst", &DataType::Float64);
        vec_series.push(
            one.extend_constant(polars::prelude::AnyValue::Float64(1.), nrows)?
        )
    }
    let df_x = DataFrame::new(vec_series)?;
    // Copy data
    match df_x.to_ndarray::<Float64Type>(IndexOrder::Fortran) {

        Ok(x) => {
            // Solving Least Square, without bias term
            // Change this after faer updates
            let x = x.view().into_faer();
            let betas = faer_lstsq_qr(x,y); // .unwrap();
            match betas {
                Ok(b) => {
                    let betas:Array1<f64> = Array1::from_iter(b);
                    let mut builder:ListPrimitiveChunkedBuilder<Float64Type> = 
                        ListPrimitiveChunkedBuilder::new("betas", 1, betas.len(), DataType::Float64);
        
                    builder.append_slice(betas.as_slice().unwrap());
                    let out = builder.finish();
                    Ok(out.into_series())
                },
                Err(e) => Err(PolarsError::ComputeError(e.into()))
            }
        }
        , Err(e) => Err(e)
    }
}


// -----------------------------------------------------------------------------------------

// // I am not sure this is right. I still don't quite understand the purpose of this.
// fn lstsq_output(input_fields: &[Field]) -> PolarsResult<Field> {
//     Ok(Field::new(
//         "betas",
//         DataType::Struct(
//             input_fields[1..]
//                 .iter()
//                 .map(|f| Field::new(&format!("beta_{}", f.name()), DataType::Float64))
//                 .collect(),
//         ),
//     ))
// }


// /// This function returns a struct series with betas, y_pred, and residuals
// #[polars_expr(output_type_func=lstsq_output)]
// fn pl_lstsq_old(inputs: &[Series]) -> PolarsResult<Series> {

//     // Iterate over the inputs and name each one with .with_name() and collect them into a vector
//     let mut series_vec = Vec::with_capacity(inputs.len());
//     // Have to name each one because they don't have names if passed in via .over()

//     for (i, series) in inputs[1..].iter().enumerate() {
//         let series = series.clone().with_name(&format!("x{i}"));
//         series_vec.push(series);
//     }
//     let beta_names: Vec<String> = series_vec.iter().map(|s| s.name().to_string()).collect();
//     let y = &inputs[0]
//         .f64()
//         .unwrap()
//         .to_ndarray()
//         .unwrap()
//         .to_owned()
//         .into_shape((inputs[0].len(), 1))
//         .unwrap();
    
//     let y = y.view().into_faer();

//     // Create a polars DataFrame from the input series
//     let todf = DataFrame::new(series_vec);
//     match todf {
//         Ok(df) => {
//             let x = df
//                 .to_ndarray::<Float64Type>(IndexOrder::Fortran)
//                 .unwrap()
//                 .to_owned();
//             let x = x.view().into_faer();

//             // Solving Least Square
//             Qr::new(x);
//             let betas = Qr::new(x).solve_lstsq(y);
//             let preds = x * &betas;
//             let preds_array = preds.as_ref().into_ndarray();
//             let resids = y - &preds;
//             let resid_array: ArrayView2<f64> = resids.as_ref().into_ndarray();
//             let betas = betas.as_ref().into_ndarray();

//             let mut out_series: Vec<Series> = betas
//                 .iter()
//                 .zip(beta_names.iter())
//                 .map(|(beta, name)| Series::new(name, vec![*beta; inputs[0].len()]))
//                 .collect();
//             // Add a series of residuals and y_pred to the output
//             let y_pred_series = Series::from_iter(preds_array).with_name("y_pred");

//             let resid_series = Series::from_iter(resid_array).with_name("resid");
            
//             out_series.push(y_pred_series);
//             out_series.push(resid_series);
//             let out = StructChunked::new("results", &out_series)?.into_series();
//             Ok(out)
//         }
//         Err(e) => {
//             println!("Error: {}", e);
//             PolarsResult::Err(e)
//         }
//     }
// }