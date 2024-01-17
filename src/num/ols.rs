use faer::{prelude::*, MatRef};
/// OLS using Faer.
use faer::{IntoFaer, IntoNdarray, Mat};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

fn pred_residue_output(_: &[Field]) -> PolarsResult<Field> {
    let pred = Field::new("pred", DataType::Float64);
    let residue = Field::new("resid", DataType::Float64);
    let v = vec![pred, residue];
    Ok(Field::new("pred", DataType::Struct(v)))
}

fn coeff_output(_: &[Field]) -> PolarsResult<Field> {
    // Update to array
    Ok(Field::new(
        "coeffs",
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

#[inline]
fn faer_lstsq_qr(x: MatRef<f64>, y: MatRef<f64>) -> Mat<f64> {
    let qr = x.qr();
    qr.solve_lstsq(y)
}

#[polars_expr(output_type_func=coeff_output)]
fn pl_lstsq(inputs: &[Series]) -> PolarsResult<Series> {
    let last_idx = inputs.len().abs_diff(1);
    let add_bias = inputs[last_idx].bool()?;
    let add_bias: bool = add_bias.get(0).unwrap();

    let nrows = inputs[0].len();
    // 0 is target
    let ncols = inputs[1..last_idx].len() + add_bias as usize;
    let mut vs: Vec<Series> = Vec::with_capacity(inputs.len() - 1);
    for (i, s) in inputs[0..last_idx].into_iter().enumerate() {
        if s.null_count() > 0 || s.len() <= 1 {
            // If there is null input, return NAN
            let mut builder: ListPrimitiveChunkedBuilder<Float64Type> =
                ListPrimitiveChunkedBuilder::new("betas", 1, ncols, DataType::Float64);
            let nan = vec![f64::NAN; ncols];
            builder.append_slice(&nan);
            let out = builder.finish();
            return Ok(out.into_series());
        } else if i >= 1 {
            let news = s
                .rechunk()
                .cast(&DataType::Float64)?
                .with_name(&i.to_string());
            vs.push(news);
        }
    }
    // Constant term
    if add_bias {
        let one = Series::from_vec("const", vec![1_f64; nrows]);
        vs.push(one);
    }
    // target
    let y = inputs[0].f64()?;
    let y = y.rechunk();
    let y = y.to_ndarray()?.into_shape((nrows, 1)).unwrap();
    let y = y.view().into_faer();

    let df_x = DataFrame::new(vs)?;
    // Copy data
    match df_x.to_ndarray::<Float64Type>(IndexOrder::Fortran) {
        Ok(x) => {
            // Solving Least Square
            let x = x.view().into_faer();
            let coeffs = faer_lstsq_qr(x, y);
            let mut betas: Vec<f64> = Vec::with_capacity(coeffs.nrows());
            for i in 0..coeffs.nrows() {
                betas.push(coeffs.read(i, 0));
            }
            let mut builder: ListPrimitiveChunkedBuilder<Float64Type> =
                ListPrimitiveChunkedBuilder::new("betas", 1, betas.len(), DataType::Float64);

            builder.append_slice(&betas);
            let out = builder.finish();
            Ok(out.into_series())
        }
        Err(e) => Err(e),
    }
}

#[polars_expr(output_type_func=pred_residue_output)]
fn pl_lstsq_pred(inputs: &[Series]) -> PolarsResult<Series> {
    let last_idx = inputs.len().abs_diff(1);
    let add_bias = inputs[last_idx].bool()?;
    let add_bias: bool = add_bias.get(0).unwrap();

    let nrows = inputs[0].len();
    // 0 is target
    let ncols = inputs[1..last_idx].len() + add_bias as usize;
    let mut vs: Vec<Series> = Vec::with_capacity(inputs.len() - 1);
    for (i, s) in inputs[0..last_idx].into_iter().enumerate() {
        if s.null_count() > 0 || s.len() <= 1 {
            // If there is null input, return NAN
            let mut builder: ListPrimitiveChunkedBuilder<Float64Type> =
                ListPrimitiveChunkedBuilder::new("betas", 1, ncols, DataType::Float64);
            let nan = vec![f64::NAN; ncols];
            builder.append_slice(&nan);
            let out = builder.finish();
            return Ok(out.into_series());
        } else if i >= 1 {
            let news = s
                .rechunk()
                .cast(&DataType::Float64)?
                .with_name(&i.to_string());
            vs.push(news);
        }
    }
    // Constant term
    if add_bias {
        let one = Series::from_vec("const", vec![1_f64; nrows]);
        vs.push(one);
    }
    // target
    let y = inputs[0].f64()?;
    let y = y.rechunk();
    let y = y.to_ndarray()?.into_shape((nrows, 1)).unwrap();
    let y = y.view().into_faer();

    let df_x = DataFrame::new(vs)?;
    // Copy data
    match df_x.to_ndarray::<Float64Type>(IndexOrder::Fortran) {
        Ok(x) => {
            // Solving Least Square
            let x = x.view().into_faer();
            let coeffs = faer_lstsq_qr(x, y);
            let y_hat = x * coeffs;
            let act = y.into_ndarray();
            let mut pred: Vec<f64> = Vec::with_capacity(y_hat.nrows());
            let mut resid: Vec<f64> = Vec::with_capacity(y_hat.nrows());

            for i in 0..y_hat.nrows() {
                let yh = y_hat.read(i, 0);
                let a = act[(i, 0)];
                pred.push(yh);
                resid.push(a - yh);
            }

            let predictions = Series::from_vec("pred", pred);
            let residue = Series::from_vec("resid", resid);
            let out = StructChunked::new("pred", &[predictions, residue])?;
            Ok(out.into_series())
        }
        Err(e) => Err(e),
    }
}
