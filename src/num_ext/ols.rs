use faer::{prelude::*, MatRef};
use faer::{IntoFaer, IntoNdarray};
use ndarray::{Array2, Array1};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

fn list_float_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "list_float",
        DataType::List(Box::new(DataType::Float64)),
    ))
}


// // Closed form. Likely don't need
// fn faer_lstsq_cf(x: MatRef<f64>, y: MatRef<f64>) -> Result<Array2<f64>, String> {
//     let xt = x.transpose();
//     let xtx = xt * &x;
//     let decomp = xtx.cholesky(Side::Lower);
//     if let Ok(cholesky) = decomp {
//         let xtx_inv = cholesky.inverse();
//         let betas = xtx_inv * xt * y;
//         Ok(betas.as_ref().into_ndarray().to_owned())
//     } else {
//         Err("Linear algebra error, likely caused by linear dependency".to_owned())
//     }
// }


fn faer_lstsq_qr(x: MatRef<f64>, y: MatRef<f64>) -> Result<Array2<f64>, String> {
    let qr = x.qr();
    let betas = qr.solve_lstsq(y);
    Ok(betas.as_ref().into_ndarray().to_owned())
}

#[polars_expr(output_type_func=list_float_output)]
fn pl_lstsq(inputs: &[Series]) -> PolarsResult<Series> {
    let nrows = inputs[0].len();
    let add_bias = inputs[1].bool()?;
    let add_bias: bool = add_bias.get(0).unwrap();
    // y
    let y = inputs[0].rechunk(); // if not contiguous, this will do work. Otherwise, just a clone
    let y = y.f64()?;
    let y = y.to_ndarray()?.into_shape((nrows, 1)).unwrap();
    let y = y.view().into_faer();

    // X, Series is ref counted, so cheap
    let mut vec_series: Vec<Series> = Vec::with_capacity(inputs[2..].len() + 1);
    for (i, s) in inputs[2..].iter().enumerate() {
        // Don't use iterator because we want ? to work.
        let t = s.rechunk().cast(&DataType::Float64)?;
        vec_series.push(
            t.with_name(&i.to_string())
        );
    }
    if add_bias {
        let one = Series::new_empty("cst", &DataType::Float64);
        vec_series.push(one.extend_constant(polars::prelude::AnyValue::Float64(1.), nrows)?)
    }

    let df_x = DataFrame::new(vec_series)?;
    // Copy data
    match df_x.to_ndarray::<Float64Type>(IndexOrder::Fortran) {
        Ok(x) => {
            // Solving Least Square, without bias term
            // Change this after faer updates
            let x = x.view().into_faer();
            let betas = faer_lstsq_qr(x, y); // .unwrap();
            match betas {
                Ok(b) => {
                    // b is 2d
                    let betas: Array1<f64> = Array1::from_iter(b);
                    let mut builder: ListPrimitiveChunkedBuilder<Float64Type> =
                        ListPrimitiveChunkedBuilder::new(
                            "betas",
                            1,
                            betas.len(),
                            DataType::Float64,
                        );
                    
                    builder.append_slice(betas.as_slice().unwrap());
                    let out = builder.finish();
                    Ok(out.into_series())
                }
                Err(e) => Err(PolarsError::ComputeError(e.into())),
            }
        }
        Err(e) => Err(e),
    }
}