use super::linear_regression::{LstsqKwargs, MultiLstsqKwargs, SWWLstsqKwargs, StandardError};
/// A copy of linear_regression, but with f32.
/// Unfortunately, it is not so easily to write generic functions. If I do so,
/// there would be too many functions that have purpose that is not obvious by first sight.
/// Polars Datatype is also not as easy to "generalize" as f32 and f64.
/// Any issue with linear regression should be solved first in f64, then copied into here.
/// Most dtype casting is implicitly handled by the series_to_mat_for_lstsq function,
/// but there are some exceptions, mainly from functions with weights. In those cases,
/// a manual cast need to be used before calling .f64() or .f32() on the series.
use crate::linalg::{
    lr_online_solvers::{faer_recursive_lstsq, faer_rolling_lstsq, faer_rolling_skipping_lstsq},
    lr_solvers::{
        faer_coordinate_descent, faer_solve_lstsq, faer_solve_lstsq_rcond, faer_weighted_lstsq,
    },
    IntoFaer, LRMethods,
};
use crate::utils::{to_frame, NullPolicy};
/// Least Squares using Faer and ndarray.
use core::f32;
use faer::{
    linalg::solvers::{DenseSolveCore, Solve},
    Col,
};
use itertools::Itertools;
use ndarray::{s, Array2};
use polars::prelude as pl;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

fn report_output(_: &[Field]) -> PolarsResult<Field> {
    let features = Field::new("features".into(), DataType::String); // index of feature
    let beta = Field::new("beta".into(), DataType::Float32); // estimated value for this coefficient
    let stderr = Field::new("std_err".into(), DataType::Float32); // Std Err for this coefficient
    let t = Field::new("t".into(), DataType::Float32); // t value for this coefficient
    let p = Field::new("p>|t|".into(), DataType::Float32); // p value for this coefficient
    let ci_lower = Field::new("0.025".into(), DataType::Float32); // CI lower bound at 0.025
    let ci_upper = Field::new("0.975".into(), DataType::Float32); // CI upper bound at 0.975
    let r2 = Field::new("r2".into(), DataType::Float32); // Coefficient of determination
    let adj_r2 = Field::new("adj_r2".into(), DataType::Float32); // adjusted
    let v: Vec<Field> = vec![features, beta, stderr, t, p, ci_lower, ci_upper, r2, adj_r2];
    Ok(Field::new("lin_reg_report".into(), DataType::Struct(v)))
}

fn pred_residue_output(_: &[Field]) -> PolarsResult<Field> {
    let pred = Field::new("pred".into(), DataType::Float32);
    let residue = Field::new("resid".into(), DataType::Float32);
    let v = vec![pred, residue];
    Ok(Field::new("pred".into(), DataType::Struct(v)))
}

fn coeff_pred_output(_: &[Field]) -> PolarsResult<Field> {
    let coeffs = Field::new("coeffs".into(), DataType::List(Box::new(DataType::Float32)));
    let pred = Field::new("prediction".into(), DataType::Float32);
    let v: Vec<Field> = vec![coeffs, pred];
    Ok(Field::new("".into(), DataType::Struct(v)))
}

fn coeff_singular_values_output(_: &[Field]) -> PolarsResult<Field> {
    let coeffs = Field::new("coeffs".into(), DataType::List(Box::new(DataType::Float32)));
    let singular_values = Field::new(
        "singular_values".into(),
        DataType::List(Box::new(DataType::Float32)),
    );
    let v: Vec<Field> = vec![coeffs, singular_values];
    Ok(Field::new("".into(), DataType::Struct(v)))
}

fn coeff_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "coeffs".into(),
        DataType::List(Box::new(DataType::Float32)),
    ))
}

// --------------------------------------------------------------------------------------------------

/// Returns a Array2 ready for linear regression, and a mask, where true means the row doesn't contain null
#[inline(always)]
fn series_to_mat_for_lstsq_f32(
    inputs: &[Series],
    has_bias: bool,
    null_policy: NullPolicy<f32>,
) -> PolarsResult<(Array2<f32>, BooleanChunked)> {
    let n_features = inputs.len().abs_diff(1);

    // minus 1 because target is also in inputs. Target is at position 0.
    let y_has_null = inputs[0].has_nulls();
    let has_null = inputs[1..].iter().any(|s| s.has_nulls()) | y_has_null;

    let mut df = to_frame(inputs)?;
    if df.is_empty() {
        return Err(PolarsError::ComputeError("Empty data".into()));
    }
    // Add a constant column if has_bias
    if has_bias {
        df = df.lazy().with_column(lit(1f32)).collect()?;
    }

    // In mask, true means not null.
    let y_name = inputs[0].name();
    let init_mask = inputs[0].is_not_null();
    let (df, mask) = if has_null {
        match null_policy {
            // Like ignore, skip_window takes the raw data. The actual skip is done in the underlying function in linalg.
            NullPolicy::IGNORE | NullPolicy::SKIP_WINDOW => {
                // false, because it has nulls
                Ok((df, BooleanChunked::from_slice("".into(), &[false])))
            }
            NullPolicy::RAISE => Err(PolarsError::ComputeError("Nulls found in data".into())),
            NullPolicy::SKIP => {
                let init_mask = inputs[0].is_not_null(); //0 always exist
                let mask = inputs[1..]
                    .iter()
                    .fold(init_mask, |acc, s| acc & s.is_not_null());

                df = df.filter(&mask).unwrap();
                Ok((df, mask))
            }
            NullPolicy::FILL(x) => {
                df = df
                    .lazy()
                    .with_columns([pl::col("*")
                        .exclude([y_name.clone()])
                        .cast(DataType::Float32)
                        .fill_null(lit(x))])
                    .collect()?;

                if y_has_null {
                    df = df.filter(&init_mask).unwrap();
                    Ok((df, init_mask))
                } else {
                    // all filled, no nulls
                    let mask = BooleanChunked::from_slice("".into(), &[true]);
                    Ok((df, mask))
                }
            }
            NullPolicy::FILL_WINDOW(x) => {
                df = df
                    .lazy()
                    .with_columns([pl::col("*")
                        .exclude([y_name.clone()])
                        .cast(DataType::Float32)
                        .fill_null(lit(x))])
                    .collect()?;

                if y_has_null {
                    // Unlike fill, this doesn't drop y's nulls
                    Ok((df, BooleanChunked::from_slice("".into(), &[false])))
                } else {
                    // all filled, no nulls
                    let mask = BooleanChunked::from_slice("".into(), &[true]);
                    Ok((df, mask))
                }
            }
        }
    } else {
        // In this case, the (!mask).any() is never true, which means there is no null.
        let mask = BooleanChunked::from_slice("".into(), &[true]);
        Ok((df, mask))
    }?;

    if df.height() < n_features {
        Err(PolarsError::ComputeError(
            "#Data < #features. No conclusive result.".into(),
        ))
    } else {
        let mat = df.to_ndarray::<Float32Type>(IndexOrder::Fortran)?;
        Ok((mat, mask))
    }
}

fn series_to_mat_for_multi_lstsq_f32(
    inputs: &[Series],
    last_target_idx: usize,
    has_bias: bool,
    null_policy: NullPolicy<f32>,
) -> PolarsResult<Array2<f32>> {
    let y_has_null = inputs[..last_target_idx]
        .iter()
        .fold(false, |acc, s| s.has_nulls() | acc);
    let has_null = inputs[last_target_idx..].iter().any(|s| s.has_nulls()) | y_has_null;

    let mut df = if has_null {
        match null_policy {
            NullPolicy::RAISE => Err(PolarsError::ComputeError("Nulls found in data".into())),

            NullPolicy::FILL(x) => {
                let df = DataFrame::new(inputs.iter().map(|s| s.clone().into_column()).collect())?;
                if y_has_null {
                    // This is because for predictions, it becomes too complicated when different targets have nulls.
                    // There will be too many masks we need to keep track of.
                    // This can be dealt with but I won't implement it for now.
                    Err(PolarsError::ComputeError(
                        "Filling null doesn't work for multi-target lstsq when there are nulls in any of the targets.".into(),
                    ))
                } else {
                    let y_names = inputs[..last_target_idx]
                        .iter()
                        .map(|s| s.name().clone())
                        .collect::<Vec<_>>();

                    df.lazy()
                        .with_columns([pl::col("*")
                            .exclude(y_names)
                            .cast(DataType::Float32)
                            .fill_null(lit(x))])
                        .collect()
                }
            }
            _ => Err(PolarsError::ComputeError(
                "The null policy is not supported by multi-target linear regression.".into(),
            )),
        }
    } else {
        DataFrame::new(inputs.iter().map(|s| s.clone().into_column()).collect())
    }?;

    if has_bias {
        df = df.lazy().with_column(lit(1f32)).collect()?;
    }

    if df.is_empty() {
        Err(PolarsError::ComputeError("Empty data".into()))
    } else {
        df.to_ndarray::<Float32Type>(IndexOrder::Fortran)
    }
}

// --------------------------------------------------------------------------------------------------

#[polars_expr(output_type_func=coeff_output)]
fn pl_lstsq_f32(inputs: &[Series], kwargs: LstsqKwargs) -> PolarsResult<Series> {
    let has_bias = kwargs.bias;
    let null_policy = NullPolicy::try_from(kwargs.null_policy)
        .map_err(|e| PolarsError::ComputeError(e.into()))?;

    let solver = kwargs.solver.as_str().into();
    let weighted = kwargs.weighted;
    let data_for_matrix = if weighted { &inputs[1..] } else { inputs };

    match series_to_mat_for_lstsq_f32(data_for_matrix, has_bias, null_policy) {
        Ok((mat, _)) => {
            // Solving Least Square
            let x = mat.slice(s![.., 1..]).into_faer();
            let y = mat.slice(s![.., 0..1]).into_faer();
            let coeffs = if weighted {
                let binding = inputs[0].cast(&DataType::Float32)?;
                let weights = binding.f32().unwrap();
                let weights = weights.cont_slice().unwrap();
                if weights.len() != mat.nrows() {
                    return Err(PolarsError::ComputeError(
                        "Shape of weights is not the same as the data.".into(),
                    ));
                }
                faer_weighted_lstsq(x, y, weights, solver)
            } else {
                match LRMethods::from((kwargs.l1_reg, kwargs.l2_reg)) {
                    LRMethods::Normal | LRMethods::L2 => {
                        faer_solve_lstsq(x, y, kwargs.l2_reg as f32, has_bias, solver)
                    }

                    LRMethods::L1 | LRMethods::ElasticNet => faer_coordinate_descent(
                        x,
                        y,
                        kwargs.l1_reg as f32,
                        kwargs.l2_reg as f32,
                        has_bias,
                        kwargs.tol as f32,
                        2000,
                    ),
                }
            };
            let mut builder: ListPrimitiveChunkedBuilder<Float32Type> =
                ListPrimitiveChunkedBuilder::new(
                    "coeffs".into(),
                    1,
                    coeffs.nrows(),
                    DataType::Float32,
                );

            builder.append_slice(coeffs.col_as_slice(0));
            let out = builder.finish();
            Ok(out.into_series())
        }
        Err(e) => Err(e),
    }
}

// Strictly speaking, this output type is not correct. Should be struct of last_target_idx many
// coeff_outputs
#[polars_expr(output_type_func=coeff_output)]
fn pl_lstsq_multi_f32(inputs: &[Series], kwargs: MultiLstsqKwargs) -> PolarsResult<Series> {
    let has_bias = kwargs.bias;
    let solver = kwargs.solver.as_str().into();
    let last_target_idx = kwargs.last_target_idx;
    let null_policy = NullPolicy::try_from(kwargs.null_policy)
        .map_err(|e| PolarsError::ComputeError(e.into()))?;

    let y_names = inputs[..last_target_idx]
        .iter()
        .map(|s| s.name())
        .collect::<Vec<_>>();
    let mat = series_to_mat_for_multi_lstsq_f32(inputs, last_target_idx, has_bias, null_policy)?;

    let y = mat.slice(s![.., 0..last_target_idx]).into_faer();
    let x = mat.slice(s![.., last_target_idx..]).into_faer();

    let coeffs = match LRMethods::from((0., kwargs.l2_reg)) {
        LRMethods::Normal | LRMethods::L2 => Ok(faer_solve_lstsq(
            x,
            y,
            kwargs.l2_reg as f32,
            has_bias,
            solver,
        )),
        _ => Err(PolarsError::ComputeError(
            "The method is not supported.".into(),
        )),
    }?;

    let df_out = DataFrame::new(
        y_names
            .into_iter()
            .enumerate()
            .map(|(i, y)| {
                let mut builder: ListPrimitiveChunkedBuilder<Float32Type> =
                    ListPrimitiveChunkedBuilder::new(
                        y.clone(),
                        1,
                        coeffs.nrows(),
                        DataType::Float32,
                    );
                builder.append_slice(coeffs.col_as_slice(i));
                let out = builder.finish();
                out.into_column()
            })
            .collect::<Vec<_>>(),
    )?;

    Ok(df_out.into_struct("coeffs".into()).into_series())
}

// Strictly speaking, this output type is not correct.
#[polars_expr(output_type_func=pred_residue_output)]
fn pl_lstsq_multi_pred_f32(inputs: &[Series], kwargs: MultiLstsqKwargs) -> PolarsResult<Series> {
    let has_bias = kwargs.bias;
    let solver = kwargs.solver.as_str().into();
    let last_target_idx = kwargs.last_target_idx;
    let null_policy = NullPolicy::try_from(kwargs.null_policy)
        .map_err(|e| PolarsError::ComputeError(e.into()))?;

    let y_names = inputs[..last_target_idx]
        .iter()
        .map(|s| s.name())
        .collect::<Vec<_>>();
    let mat = series_to_mat_for_multi_lstsq_f32(inputs, last_target_idx, has_bias, null_policy)?;

    let y = mat.slice(s![.., 0..last_target_idx]).into_faer();
    let x = mat.slice(s![.., last_target_idx..]).into_faer();

    let coeffs = match LRMethods::from((0., kwargs.l2_reg)) {
        LRMethods::Normal | LRMethods::L2 => Ok(faer_solve_lstsq(
            x,
            y,
            kwargs.l2_reg as f32,
            has_bias,
            solver,
        )),
        _ => Err(PolarsError::ComputeError(
            "The method is not supported.".into(),
        )),
    }?;

    let pred = x * &coeffs;
    let resid = y - &pred;

    let mut s = Vec::with_capacity(y_names.len() * 2);
    for (i, y) in y_names.into_iter().enumerate() {
        let pred_name = format!("{}_pred", y);
        let resid_name = format!("{}_resid", y);
        let p = Float32Chunked::from_slice(pred_name.into(), pred.col_as_slice(i));
        let r = Float32Chunked::from_slice(resid_name.into(), resid.col_as_slice(i));
        s.push(p.into_column());
        s.push(r.into_column());
    }
    let df_out = DataFrame::new(s)?;
    Ok(df_out.into_struct("all_preds".into()).into_series())
}

#[polars_expr(output_type_func=coeff_singular_values_output)]
fn pl_lstsq_w_rcond_f32(inputs: &[Series], kwargs: LstsqKwargs) -> PolarsResult<Series> {
    let has_bias = kwargs.bias;
    let null_policy = NullPolicy::try_from(kwargs.null_policy)
        .map_err(|e| PolarsError::ComputeError(e.into()))?;

    // Target y is at index 0
    match series_to_mat_for_lstsq_f32(inputs, has_bias, null_policy) {
        Ok((mat, _)) => {
            // rcond will be passed as tol
            let rcond =
                (kwargs.tol as f32).max(f32::EPSILON * (inputs.len().max(mat.len())) as f32);

            // Solving Least Square
            let x = mat.slice(s![.., 1..]).into_faer();
            let y = mat.slice(s![.., 0..1]).into_faer();

            //     // faer_solve_ridge_rcond
            let (coeffs, singular_values) =
                faer_solve_lstsq_rcond(x, y, kwargs.l2_reg as f32, has_bias, rcond);

            let mut builder: ListPrimitiveChunkedBuilder<Float32Type> =
                ListPrimitiveChunkedBuilder::new(
                    "coeffs".into(),
                    1,
                    coeffs.nrows(),
                    DataType::Float32,
                );

            builder.append_slice(coeffs.col_as_slice(0));
            let coeffs_ca = builder.finish();

            let mut sv_builder: ListPrimitiveChunkedBuilder<Float32Type> =
                ListPrimitiveChunkedBuilder::new(
                    "singular_values".into(),
                    1,
                    singular_values.len(),
                    DataType::Float32,
                );

            sv_builder.append_slice(&singular_values);
            let coeffs_sv = sv_builder.finish();

            let ca = StructChunked::from_columns(
                "".into(),
                coeffs_sv.len(),
                &[coeffs_ca.into_column(), coeffs_sv.into_column()],
            )?;
            Ok(ca.into_series())
        }
        Err(e) => Err(e),
    }
}

#[polars_expr(output_type_func=pred_residue_output)]
fn pl_lstsq_pred_f32(inputs: &[Series], kwargs: LstsqKwargs) -> PolarsResult<Series> {
    let has_bias = kwargs.bias;
    let null_policy = NullPolicy::try_from(kwargs.null_policy)
        .map_err(|e| PolarsError::ComputeError(e.into()))?;

    let solver = kwargs.solver.as_str().into();
    let weighted = kwargs.weighted;
    let data_for_matrix = if weighted { &inputs[1..] } else { inputs };

    match series_to_mat_for_lstsq_f32(data_for_matrix, has_bias, null_policy.clone()) {
        Ok((mat, mask)) => {
            let y = mat.slice(s![.., 0..1]).into_faer();
            let x = mat.slice(s![.., 1..]).into_faer();
            let coeffs = if weighted {
                let binding = inputs[0].cast(&DataType::Float32)?;
                let weights = binding.f32().unwrap();
                let weights = weights.cont_slice().unwrap();
                if weights.len() != mat.nrows() {
                    return Err(PolarsError::ComputeError(
                        "Length of weights and data in X must be the same.".into(),
                    ));
                }
                faer_weighted_lstsq(x, y, weights, solver)
            } else {
                match LRMethods::from((kwargs.l1_reg, kwargs.l2_reg)) {
                    LRMethods::Normal | LRMethods::L2 => {
                        faer_solve_lstsq(x, y, kwargs.l2_reg as f32, has_bias, solver)
                    }
                    LRMethods::L1 | LRMethods::ElasticNet => faer_coordinate_descent(
                        x,
                        y,
                        kwargs.l1_reg as f32,
                        kwargs.l2_reg as f32,
                        has_bias,
                        kwargs.tol as f32,
                        2000,
                    ),
                }
            };

            let pred = x * &coeffs;
            let resid = y - &pred;
            let pred = pred.col_as_slice(0);
            let resid = resid.col_as_slice(0);
            // If null policy is raise and we have nulls, we won't reach here
            // If null policy is raise and we are here, then (!&mask).any() will be false.
            // No need to check null policy here.
            // In the mask, true means is not null. In !&mask, true means is null
            let (p, r) = if (!&mask).any() {
                let mut p_builder: PrimitiveChunkedBuilder<Float32Type> =
                    PrimitiveChunkedBuilder::new("pred".into(), mask.len());
                let mut r_builder: PrimitiveChunkedBuilder<Float32Type> =
                    PrimitiveChunkedBuilder::new("resid".into(), mask.len());
                let mut i: usize = 0;
                for mm in mask.into_no_null_iter() {
                    // mask is always non-null, mm = true means it is not null
                    if mm {
                        p_builder.append_value(pred[i]);
                        r_builder.append_value(resid[i]);
                        i += 1;
                    } else {
                        p_builder.append_value(f32::NAN);
                        r_builder.append_value(f32::NAN);
                    }
                }
                (p_builder.finish(), r_builder.finish())
            } else {
                let pred = Float32Chunked::from_slice("pred".into(), pred);
                let residue = Float32Chunked::from_slice("resid".into(), resid);
                (pred, residue)
            };
            let p = p.into_series();
            let r = r.into_series();
            let out = StructChunked::from_series("".into(), p.len(), [&p, &r].into_iter())?;
            Ok(out.into_series())
        }
        Err(e) => Err(e),
    }
}

#[polars_expr(output_type_func=report_output)]
fn pl_lin_reg_report_f32(inputs: &[Series], kwargs: LstsqKwargs) -> PolarsResult<Series> {
    let has_bias = kwargs.bias;
    let se_type = StandardError::from(kwargs.std_err);
    let null_policy = NullPolicy::try_from(kwargs.null_policy)
        .map_err(|e| PolarsError::ComputeError(e.into()))?;
    // index 0 is y_var, 1 is target y. Skip
    let binding = inputs[0].cast(&DataType::Float32)?;
    let y_var = binding.f32().unwrap();
    let y_var = y_var.get(0).unwrap_or(f32::NAN);
    let mut name_builder = StringChunkedBuilder::new(
        "features".into(),
        inputs.len().abs_diff(2) + (has_bias) as usize,
    );
    for s in inputs[2..].iter().map(|s| s.name()) {
        name_builder.append_value(s);
    }
    if has_bias {
        name_builder.append_value("__bias__");
    }
    // Copy data
    // Target y is at index 0
    match series_to_mat_for_lstsq_f32(&inputs[1..], has_bias, null_policy) {
        Ok((mat, _)) => {
            let ncols = mat.ncols() - 1;
            let nrows = mat.nrows();

            let x = mat.slice(s![0..nrows, 1..]).into_faer();
            let y = mat.slice(s![0..nrows, 0..1]).into_faer();
            // Solving Least Square
            let xtx = x.transpose() * &x;
            let xtx_qr = xtx.col_piv_qr();
            let xtx_inv = xtx_qr.inverse();
            let xtx_inv_xt = &xtx_inv * x.transpose();
            let coeffs = &xtx_inv_xt * y;
            let betas = coeffs.col_as_slice(0);
            // Degree of Freedom
            let dof = nrows as f32 - ncols as f32;
            // Residue
            let res = y - x * &coeffs;

            // r2, adj_r2
            let nf32 = nrows as f32;
            let ratio = res.col(0).squared_norm_l2() / (y_var * nf32);
            let r2 = 1.0 - ratio;
            let adj_r2 = 1.0 - ratio * ((nrows - 1) as f32 / (dof - 1.0));

            // std err
            let std_err = match se_type {
                StandardError::SE => {
                    // total residue, sum of squares
                    let mse = (res.transpose() * &res).get(0, 0) / dof;
                    (0..ncols)
                        .map(|i| (mse * xtx_inv.get(i, i)).sqrt())
                        .collect_vec()
                }
                StandardError::HC0 | StandardError::HC1 => {
                    let temp_diag = res.get(.., 0).as_diagonal();
                    let diag = temp_diag * temp_diag;
                    let var_hc = &xtx_inv_xt * diag * xtx_inv_xt.transpose();
                    let factor = if se_type == StandardError::HC1 {
                        (nrows as f32) / ((nrows - ncols) as f32)
                    } else {
                        1f32
                    };
                    (0..ncols)
                        .map(|i| (var_hc.get(i, i) * factor).sqrt())
                        .collect_vec()
                }
                StandardError::HC2 | StandardError::HC3 => {
                    let temp_diag = res.get(.., 0).as_diagonal();
                    let temp_diag_scalers = Col::from_fn(nrows, |i| {
                        let d = x.get_r(i) * xtx_inv_xt.get(.., i);
                        if se_type == StandardError::HC3 {
                            1f32 / (1f32 - d).powi(2)
                        } else {
                            1f32 / (1f32 - d)
                        }
                    });
                    let diag_scalers = temp_diag_scalers.as_diagonal();
                    // (diagonal residue squared matrix) * (1 / (1 -h_ii)) or * (1 / (1 -h_ii)^2)
                    let diag = temp_diag * temp_diag * diag_scalers;

                    let var_hc = &xtx_inv_xt * diag * xtx_inv_xt.transpose();
                    (0..ncols).map(|i| (var_hc.get(i, i)).sqrt()).collect_vec()
                }
            };
            // T values
            let t_values = betas
                .iter()
                .zip(std_err.iter())
                .map(|(b, se)| b / se)
                .collect_vec();
            // P values
            let p_values = t_values
                .iter()
                .map(
                    |t| match crate::stats_utils::beta::student_t_sf(t.abs().into(), dof.into()) {
                        Ok(p) => 2.0f32 * (p as f32),
                        Err(_) => f32::NAN,
                    },
                )
                .collect_vec();

            let t_alpha = crate::stats_utils::beta::student_t_ppf(0.975, dof.into()) as f32;
            let ci_lower = betas
                .iter()
                .zip(std_err.iter())
                .map(|(b, se)| b - t_alpha * se)
                .collect_vec();
            let ci_upper = betas
                .iter()
                .zip(std_err.iter())
                .map(|(b, se)| b + t_alpha * se)
                .collect_vec();

            // Finalize
            let names_ca = name_builder.finish();
            let names_series = names_ca.into_series();
            let coeffs_series = Float32Chunked::from_slice("beta".into(), betas);
            let coeffs_series = coeffs_series.into_series();
            let stderr_series = Float32Chunked::from_vec(se_type.into(), std_err);
            let stderr_series = stderr_series.into_series();
            let t_series = Float32Chunked::from_vec("t".into(), t_values);
            let t_series = t_series.into_series();
            let p_series = Float32Chunked::from_vec("p>|t|".into(), p_values);
            let p_series = p_series.into_series();
            let lower = Float32Chunked::from_vec("0.025".into(), ci_lower);
            let lower = lower.into_series();
            let upper = Float32Chunked::from_vec("0.975".into(), ci_upper);
            let upper = upper.into_series();
            let r2 = Float32Chunked::from_vec("r2".into(), vec![r2]);
            let r2_series = r2.into_series();
            let adj_r2 = Float32Chunked::from_vec("adj_r2".into(), vec![adj_r2]);
            let adj_r2_series = adj_r2.into_series();
            let out = StructChunked::from_series(
                "lin_reg_report".into(),
                names_series.len(),
                [
                    &names_series,
                    &coeffs_series,
                    &stderr_series,
                    &t_series,
                    &p_series,
                    &lower,
                    &upper,
                    &r2_series,
                    &adj_r2_series,
                ]
                .into_iter(),
            )?;
            Ok(out.into_series())
        }
        Err(e) => Err(e),
    }
}

#[polars_expr(output_type_func=report_output)]
fn pl_wls_report_f32(inputs: &[Series], kwargs: LstsqKwargs) -> PolarsResult<Series> {
    let has_bias = kwargs.bias;
    let null_policy = NullPolicy::try_from(kwargs.null_policy)
        .map_err(|e| PolarsError::ComputeError(e.into()))?;

    let binding = inputs[0].cast(&DataType::Float32)?;
    let weights = binding.f32().unwrap();
    let weights = weights.cont_slice().unwrap();
    let binding2 = inputs[1].cast(&DataType::Float32)?;
    let y_var = binding2.f32().unwrap();
    let y_var = y_var.get(0).unwrap_or(f32::NAN);
    // index 0 is weights, 1 is y_var, 2 is target y. Skip them
    let mut name_builder = StringChunkedBuilder::new(
        "features".into(),
        inputs.len().abs_diff(3) + (has_bias) as usize,
    );
    for s in inputs[3..].iter().map(|s| s.name()) {
        name_builder.append_value(s);
    }
    if has_bias {
        name_builder.append_value("__bias__");
    }
    // Copy data
    // Target y is at index 2, weights at index 0
    match series_to_mat_for_lstsq_f32(&inputs[2..], has_bias, null_policy) {
        Ok((mat, _)) => {
            let ncols = mat.ncols() - 1;
            let nrows = mat.nrows();

            let x = mat.slice(s![0..nrows, 1..]).into_faer();
            let y = mat.slice(s![0..nrows, 0..1]).into_faer();

            // let w = faer::mat::from_row_major_slice(weights, x.nrows(), 1);
            let w = faer::ColRef::from_slice(weights);
            let w = w.as_diagonal();
            let xt = x.transpose();

            let xtwx = xt * w * x;
            let xtwy = xt * w * y;
            let qr = xtwx.col_piv_qr();
            let xtwx_inv = qr.inverse();
            let coeffs = qr.solve(xtwy);
            let betas = coeffs.col_as_slice(0);

            // Degree of Freedom
            let dof = nrows as f32 - ncols as f32;
            // Residue
            let res = y - x * &coeffs;
            let mse =
                (0..y.nrows()).fold(0., |acc, i| acc + weights[i] * res.get(i, 0).powi(2)) / dof;

            // r2, adj_r2
            let nf32 = nrows as f32;
            let ratio = res.col(0).squared_norm_l2() / (y_var * nf32);
            let r2 = 1.0 - ratio;
            let adj_r2 = 1.0 - ratio * ((nrows - 1) as f32 / (dof - 1.0));

            // std err
            let std_err = (0..ncols)
                .map(|i| (mse * xtwx_inv.get(i, i)).sqrt())
                .collect_vec();
            // T values
            let t_values = betas
                .iter()
                .zip(std_err.iter())
                .map(|(b, se)| b / se)
                .collect_vec();
            // P values
            let p_values = t_values
                .iter()
                .map(
                    |t| match crate::stats_utils::beta::student_t_sf(t.abs().into(), dof.into()) {
                        Ok(p) => 2.0 * (p as f32),
                        Err(_) => f32::NAN,
                    },
                )
                .collect_vec();

            let t_alpha = crate::stats_utils::beta::student_t_ppf(0.975, dof.into()) as f32;
            let ci_lower = betas
                .iter()
                .zip(std_err.iter())
                .map(|(b, se)| b - t_alpha * se)
                .collect_vec();
            let ci_upper = betas
                .iter()
                .zip(std_err.iter())
                .map(|(b, se)| b + t_alpha * se)
                .collect_vec();
            // Finalize
            let names_ca = name_builder.finish();
            let names_series = names_ca.into_series();
            let coeffs_series = Float32Chunked::from_slice("beta".into(), betas);
            let coeffs_series = coeffs_series.into_series();
            let stderr_series = Float32Chunked::from_vec("std_err".into(), std_err);
            let stderr_series = stderr_series.into_series();
            let t_series = Float32Chunked::from_vec("t".into(), t_values);
            let t_series = t_series.into_series();
            let p_series = Float32Chunked::from_vec("p>|t|".into(), p_values);
            let p_series = p_series.into_series();
            let lower = Float32Chunked::from_vec("0.025".into(), ci_lower);
            let lower = lower.into_series();
            let upper = Float32Chunked::from_vec("0.975".into(), ci_upper);
            let upper = upper.into_series();
            let r2 = Float32Chunked::from_vec("r2".into(), vec![r2]);
            let r2_series = r2.into_series();
            let adj_r2 = Float32Chunked::from_vec("adj_r2".into(), vec![adj_r2]);
            let adj_r2_series = adj_r2.into_series();
            let out = StructChunked::from_series(
                "lin_reg_report".into(),
                names_series.len(),
                [
                    &names_series,
                    &coeffs_series,
                    &stderr_series,
                    &t_series,
                    &p_series,
                    &lower,
                    &upper,
                    &r2_series,
                    &adj_r2_series,
                ]
                .into_iter(),
            )?;
            Ok(out.into_series())
        }
        Err(e) => Err(e),
    }
}

// --- Rolling and Recursive

#[polars_expr(output_type_func=coeff_pred_output)]
fn pl_recursive_lstsq_f32(inputs: &[Series], kwargs: SWWLstsqKwargs) -> PolarsResult<Series> {
    let n = kwargs.n; // Gauranteed n >= 1
    let has_bias = kwargs.bias;

    // Gauranteed in Python that this won't be SKIP. SKIP doesn't work now.
    let null_policy = NullPolicy::try_from(kwargs.null_policy)
        .map_err(|e| PolarsError::ComputeError(e.into()))?;

    // Target y is at index 0
    match series_to_mat_for_lstsq_f32(inputs, has_bias, null_policy) {
        Ok((mat, mask)) => {
            // Solving Least Square
            let x = mat.slice(s![.., 1..]).into_faer();
            let y = mat.slice(s![.., 0..1]).into_faer();

            let coeffs = faer_recursive_lstsq(x, y, n, kwargs.lambda as f32);
            let mut builder: ListPrimitiveChunkedBuilder<Float32Type> =
                ListPrimitiveChunkedBuilder::new(
                    "coeffs".into(),
                    mat.nrows(),
                    mat.ncols(),
                    DataType::Float32,
                );
            let mut pred_builder: PrimitiveChunkedBuilder<Float32Type> =
                PrimitiveChunkedBuilder::new("pred".into(), mat.nrows());

            // Fill or Skip strategy can drop nulls. Fill will drop null when y has nulls.
            // Skip will drop nulls whenever there is a null in the row.
            // Mask true means the row doesn't have nulls
            let has_nulls = (!&mask).any();
            if has_nulls {
                // Find the first index where we get n non-nulls.
                let mut new_n = 0;
                let mut m = 0;
                for v in mask.into_no_null_iter() {
                    new_n += v as usize;
                    if new_n >= n {
                        break;
                    }
                    m += 1;
                }
                for _ in 0..m {
                    builder.append_null();
                    pred_builder.append_null();
                }
                let mut i = 0;
                for should_keep in mask.into_no_null_iter().skip(m) {
                    if should_keep {
                        let coefficients = &coeffs[i];
                        let row = x.get(i..i + 1, ..);
                        let pred = *(row * coefficients).get(0, 0);
                        let coef = coefficients.col_as_slice(0);
                        builder.append_slice(coef);
                        pred_builder.append_value(pred);
                        i += 1;
                    } else {
                        builder.append_null();
                        pred_builder.append_null();
                    }
                }
            } else {
                // n always > 1, guaranteed in Python
                let m = n.abs_diff(1);
                for _ in 0..m {
                    builder.append_null();
                    pred_builder.append_null();
                }
                for (i, coefficients) in coeffs.into_iter().enumerate() {
                    let row = x.get(m + i..m + i + 1, ..);
                    let pred = *(row * &coefficients).get(0, 0);
                    let coef = coefficients.col_as_slice(0);
                    builder.append_slice(coef);
                    pred_builder.append_value(pred);
                }
            }

            let coef_out = builder.finish();
            let pred_out = pred_builder.finish();
            let ca = StructChunked::from_series(
                "".into(),
                coef_out.len(),
                [&coef_out.into_series(), &pred_out.into_series()].into_iter(),
            )?;
            Ok(ca.into_series())
        }
        Err(e) => Err(e),
    }
}

#[polars_expr(output_type_func=coeff_pred_output)] // They share the same output type
fn pl_rolling_lstsq_f32(inputs: &[Series], kwargs: SWWLstsqKwargs) -> PolarsResult<Series> {
    let n = kwargs.n; // Gauranteed n >= 2
    let has_bias = kwargs.bias;

    let mut null_policy = NullPolicy::try_from(kwargs.null_policy)
        .map_err(|e| PolarsError::ComputeError(e.into()))?;

    // For SKIP, we use SKIP_WINDOW. For FILL(x), use FILL_WINDOW
    null_policy = match null_policy {
        NullPolicy::SKIP => NullPolicy::SKIP_WINDOW,
        NullPolicy::FILL(x) => NullPolicy::FILL_WINDOW(x),
        _ => null_policy,
    };

    // Target y is at index 0
    match series_to_mat_for_lstsq_f32(inputs, has_bias, null_policy) {
        Ok((mat, mask)) => {
            let should_skip = match null_policy {
                NullPolicy::SKIP_WINDOW | NullPolicy::FILL_WINDOW(_) => (!&mask).any(),
                _ => false, // raise, ignore
            };
            let x = mat.slice(s![.., 1..]).into_faer();
            let y = mat.slice(s![.., 0..1]).into_faer();
            let coeffs = if should_skip {
                faer_rolling_skipping_lstsq(x, y, n, kwargs.min_size, kwargs.lambda as f32)
            } else {
                faer_rolling_lstsq(x, y, n, kwargs.lambda as f32)
            };

            let mut builder: ListPrimitiveChunkedBuilder<Float32Type> =
                ListPrimitiveChunkedBuilder::new(
                    "coeffs".into(),
                    mat.nrows(),
                    mat.ncols(),
                    DataType::Float32,
                );
            let mut pred_builder: PrimitiveChunkedBuilder<Float32Type> =
                PrimitiveChunkedBuilder::new("pred".into(), mat.nrows());

            let m = n - 1; // n >= 2 guaranteed in Python
            for _ in 0..m {
                builder.append_null();
                pred_builder.append_null();
            }

            if should_skip {
                // Skipped rows will have coeffs with shape (0, 0)
                for (i, coefficients) in coeffs.into_iter().enumerate() {
                    if coefficients.shape() == (0, 0) {
                        builder.append_null();
                        pred_builder.append_null();
                    } else {
                        let row = x.get(m + i..m + i + 1, ..);
                        let pred = *(row * &coefficients).get(0, 0);
                        let coef = coefficients.col_as_slice(0);
                        builder.append_slice(coef);
                        pred_builder.append_value(pred);
                    }
                }
            } else {
                // nothing is skipped. All coeffs must be valid.
                for (i, coefficients) in coeffs.into_iter().enumerate() {
                    let row = x.get(m + i..m + i + 1, ..);
                    let pred = *(row * &coefficients).get(0, 0);
                    let coef = coefficients.col_as_slice(0);
                    builder.append_slice(coef);
                    pred_builder.append_value(pred);
                }
            }

            let coef_out = builder.finish();
            let pred_out = pred_builder.finish();
            let ca = StructChunked::from_series(
                "".into(),
                coef_out.len(),
                [&coef_out.into_series(), &pred_out.into_series()].into_iter(),
            )?;
            Ok(ca.into_series())
        }
        Err(e) => Err(e),
    }
}
