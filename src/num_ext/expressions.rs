use faer::{prelude::*, MatRef};
use faer::{IntoFaer, IntoNdarray};
// use faer::polars::{polars_to_faer_f64, Frame};
use ndarray::{Array1, Array2};
use num;
use num::traits::Inv;
use polars::prelude::*;
use polars_core::prelude::arity::binary_elementwise_values;
use pyo3_polars::derive::polars_expr;
use rustfft::FftPlanner;

// fn numeric_output(input_fields: &[Field]) -> PolarsResult<Field> {
//     let field = input_fields[0].clone();
//     Ok(field)
// }

fn complex_output(_: &[Field]) -> PolarsResult<Field> {
    let real = Field::new("re", DataType::Float64);
    let complex = Field::new("im", DataType::Float64);
    let v: Vec<Field> = vec![real, complex];
    Ok(Field::new("complex", DataType::Struct(v)))
}

fn list_float_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "list_float",
        DataType::List(Box::new(DataType::Float64)),
    ))
}

#[polars_expr(output_type=Int64)]
fn pl_gcd(inputs: &[Series]) -> PolarsResult<Series> {
    let ca1 = inputs[0].i64()?;
    let ca2 = inputs[1].i64()?;
    if ca2.len() == 1 {
        let b = ca2.get(0).unwrap();
        let out: Int64Chunked = ca1.apply_values(|a| num::integer::gcd(a, b));
        Ok(out.into_series())
    } else if ca1.len() == ca2.len() {
        let out: Int64Chunked = binary_elementwise_values(ca1, ca2, num::integer::gcd);
        Ok(out.into_series())
    } else {
        Err(PolarsError::ShapeMismatch(
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
        let out: Int64Chunked = ca1.apply_values(|a| num::integer::lcm(a, b));
        Ok(out.into_series())
    } else if ca1.len() == ca2.len() {
        let out: Int64Chunked = binary_elementwise_values(ca1, ca2, num::integer::lcm);
        Ok(out.into_series())
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length.".into(),
        ))
    }
}


fn fast_exp_single(s:Series, n:i32) -> Series {

    if n == 0 {
        let ss = s.f64().unwrap();
        let out:Float64Chunked = ss.apply_values(|x| {
            if x == 0. {
                f64::NAN
            } else if x.is_infinite() | x.is_nan() {
                x
            } else {
                1.0
            }
        });
        return out.into_series()
    } else if n < 0 {
        return fast_exp_single(1.div(&s), -n)
    }

    let mut ss = s.clone();
    let mut m = n;
    let mut y = Series::from_vec("", vec![1_f64; s.len()]);
    while m > 0 {
        if m % 2 == 1 {
            y = &y * &ss;
        }
        ss = &ss * &ss;
        m >>= 1;
    }
    y

 }

 #[inline]
 fn _fast_exp_pairwise(x:f64, n:u32) -> f64 {

    let mut m = n;
    let mut x = x;
    let mut y:f64 = 1.0;
    while m > 0 {
        if m % 2 == 1 {
            y *= x;
        }
        x *= x;
        m >>= 1;
    } 
    y

}

#[inline]
fn fast_exp_pairwise(x:f64, n:i32) -> f64 {

    if n == 0 {
        if x == 0. { // 0^0 is NaN
            return f64::NAN
        } else {
            return 1.
        }
    } else if n < 0 {
        return _fast_exp_pairwise(x.inv(), (-n) as u32)
    }
    _fast_exp_pairwise(x, n as u32)

}


#[polars_expr(output_type=Float64)]
fn pl_fast_exp(inputs: &[Series]) -> PolarsResult<Series> {

    let s = inputs[0].clone();
    let exp = inputs[1].i32()?;

    if exp.len() == 1 {
        let n = exp.get(0).unwrap();
        if s.dtype().is_numeric() {
            let ss = s.cast(&DataType::Float64)?;
            Ok(fast_exp_single(ss, n))
        } else {  
            Err(PolarsError::ComputeError(
                "Input column type must be numeric.".into(),
            ))
        }
    } else if s.len() == exp.len() {
        if s.dtype().is_numeric() {
            if s.dtype() == &DataType::Float64 {
                let ca = s.f64()?;
                let out:Float64Chunked = binary_elementwise_values(ca, exp, fast_exp_pairwise);
                Ok(out.into_series())
            } else {
                let t = s.cast(&DataType::Float64)?;
                let ca = t.f64()?;
                let out:Float64Chunked = binary_elementwise_values(ca, exp, fast_exp_pairwise);
                Ok(out.into_series())
            }
        } else {
            Err(PolarsError::ComputeError(
                "Input column type must be numeric.".into(),
            ))
        }
    } else {
        Err(PolarsError::ShapeMismatch(
            "Inputs must have the same length.".into(),
        ))
    }

}

// Use QR to solve
fn faer_lstsq_qr(x: MatRef<f64>, y: MatRef<f64>) -> Result<Array2<f64>, String> {
    let qr = x.qr();
    let betas = qr.solve_lstsq(y);
    Ok(betas.as_ref().into_ndarray().to_owned())
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
        let t: Series = match s.dtype() {
            DataType::Float64 => s.rechunk().with_name(&i.to_string()),
            _ => {
                let t = s.rechunk().cast(&DataType::Float64)?;
                t.with_name(&i.to_string())
            }
        };
        vec_series.push(t);
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

#[polars_expr(output_type=Float64)]
fn pl_conditional_entropy(inputs: &[Series]) -> PolarsResult<Series> {
    let x = inputs[0].name();
    let y = inputs[1].name();
    let out_name = format!("H({x}|{y})");
    let out_name = out_name.as_str();

    let df = DataFrame::new(inputs.to_vec())?;
    let mut out = df
        .lazy()
        .group_by([col(x), col(y)])
        .agg([count()])
        .with_columns([
            (col("count").sum().cast(DataType::Float64).over([col(y)])
                / col("count").sum().cast(DataType::Float64))
            .alias("p(y)"),
            (col("count").cast(DataType::Float64) / col("count").sum().cast(DataType::Float64))
                .alias("p(x,y)"),
        ])
        .select([(lit(-1.0_f64)
            * ((col("p(x,y)") / col("p(y)"))
                .log(std::f64::consts::E)
                .dot(col("p(x,y)"))))
        .alias(out_name)])
        .collect()?;

    out.drop_in_place(out_name)
}

#[polars_expr(output_type_func=complex_output)]
fn pl_fft(inputs: &[Series]) -> PolarsResult<Series> {
    let s = inputs[0].clone();
    if s.null_count() > 0 {
        return Err(PolarsError::ComputeError(
            "FFT Input cannot have null values.".into(),
        ));
    }
    let mut buf: Vec<num::complex::Complex64> = match s.dtype() {
        DataType::Float64 => s.f64()?.into_no_null_iter().map(|x| x.into()).collect(),
        DataType::Float32 => {
            let temp = s.cast(&DataType::Float64)?;
            temp.f64()?.into_no_null_iter().map(|x| x.into()).collect()
        }
        _ => {
            return Err(PolarsError::ComputeError(
                "FFT Input must be floats.".into(),
            ))
        }
    };

    let name = s.name();
    let forward = inputs[1].bool()?;
    let forward = forward.get(0).unwrap();

    let mut planner: FftPlanner<f64> = FftPlanner::new();
    let fft = if forward {
        planner.plan_fft_forward(buf.len())
    } else {
        planner.plan_fft_inverse(buf.len())
    };
    fft.process(&mut buf);

    let mut re: Vec<f64> = Vec::with_capacity(buf.len());
    let mut im: Vec<f64> = Vec::with_capacity(buf.len());
    for c in buf {
        re.push(c.re);
        im.push(c.im);
    }

    let fft_struct = df!(
        "re" => re,
        "im" => im,
    )?
    .into_struct(name)
    .into_series();

    Ok(fft_struct)
}
