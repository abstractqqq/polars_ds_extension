use super::linear_regression::{
    LRKwargs
    , series_to_mat_for_lr
    , coeff_output
};
use crate::linear::{
    NullPolicy
    , logistic::logistic_solver::{
        faer_logistic_reg
        , stable_sigmoid
    }
};
use faer::MatRef;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

#[polars_expr(output_type_func=coeff_output)]
fn pl_logistic_coeffs(inputs: &[Series], kwargs: LRKwargs) -> PolarsResult<Series> {
    let add_bias = kwargs.bias;
    let null_policy = NullPolicy::try_from(kwargs.null_policy)
        .map_err(|e| PolarsError::ComputeError(e.into()))?;

    let l1_reg = if kwargs.l1_reg > 0.0 {
        Some(kwargs.l1_reg)
    } else {
        None
    };
    let l2_reg = if kwargs.l2_reg > 0.0 {
        Some(kwargs.l2_reg)
    } else {
        None
    };
    match series_to_mat_for_lr(inputs, add_bias, null_policy) {
        Ok((mat_slice, nrows, nfeats, _)) => {
            let y = MatRef::from_column_major_slice(&mat_slice[..nrows], nrows, 1);
            let x = MatRef::from_column_major_slice(&mat_slice[nrows..], nrows, nfeats);
            let coeffs = faer_logistic_reg(
                x, 
                y, 
                add_bias,
                l1_reg,
                l2_reg, 
                kwargs.tol,
                kwargs.max_iter, 
            );
            let mut builder: ListPrimitiveChunkedBuilder<Float64Type> =
                ListPrimitiveChunkedBuilder::new(
                    "coeffs".into(),
                    1,
                    coeffs.nrows(),
                    DataType::Float64,
                );

            builder.append_slice(coeffs.col_as_slice(0));
            let out = builder.finish();
            Ok(out.into_series())
        }
        Err(e) => Err(e),
    }
}


#[polars_expr(output_type=Float64)]
fn pl_logistic_pred(inputs: &[Series], kwargs: LRKwargs) -> PolarsResult<Series> {
    let add_bias = kwargs.bias;
    let null_policy = NullPolicy::try_from(kwargs.null_policy)
        .map_err(|e| PolarsError::ComputeError(e.into()))?;

    let l1_reg = if kwargs.l1_reg > 0.0 {
        Some(kwargs.l1_reg)
    } else {
        None
    };
    let l2_reg = if kwargs.l2_reg > 0.0 {
        Some(kwargs.l2_reg)
    } else {
        None
    };
    match series_to_mat_for_lr(inputs, add_bias, null_policy) {
        Ok((mat_slice, nrows, nfeats, mask)) => {
            let y = MatRef::from_column_major_slice(&mat_slice[..nrows], nrows, 1);
            let x = MatRef::from_column_major_slice(&mat_slice[nrows..], nrows, nfeats);
            let coeffs = faer_logistic_reg(
                x, 
                y, 
                add_bias,
                l1_reg,
                l2_reg, 
                kwargs.tol,
                kwargs.max_iter, 
            );

            let mut pred = x * coeffs;
            for p in pred.col_as_slice_mut(0) {
                *p = stable_sigmoid(*p);
            }
            let pred = pred.col_as_slice(0);
            let ca = if (!&mask).any() {
                let mut builder: PrimitiveChunkedBuilder<Float64Type> =
                    PrimitiveChunkedBuilder::new("pred".into(), mask.len());

                let mut i: usize = 0;
                for mm in mask.into_no_null_iter() {
                    // mask is always non-null, mm = true means it is not null
                    if mm {
                        builder.append_value(pred[i]);
                        i += 1;
                    } else {
                        builder.append_null();
                    }
                }
                builder.finish()
            } else {
                // No nulls.
                Float64Chunked::from_slice("pred".into(), pred)
            };
            Ok(ca.into_series())
        }
        Err(e) => Err(e),
    }
}

