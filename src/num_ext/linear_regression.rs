use core::f64;

/// Least Squares using Faer and ndarray.
use crate::linalg::lstsq::{
    faer_lasso_regression, faer_recursive_lstsq, faer_rolling_lstsq, faer_rolling_skipping_lstsq, faer_solve_lstsq, faer_solve_lstsq_rcond, faer_solve_ridge, faer_solve_ridge_rcond, faer_weighted_lstsq, ClosedFormLRMethods, LRMethods
};
use crate::utils::{to_frame, NullPolicy};
use faer::prelude::*;
use faer_ext::IntoFaer;
use itertools::Itertools;
use ndarray::{s, Array2};
use polars::prelude as pl;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub(crate) struct LstsqKwargs {
    pub(crate) bias: bool,
    pub(crate) null_policy: String,
    pub(crate) solver: String,
    pub(crate) l1_reg: f64,
    pub(crate) l2_reg: f64,
    pub(crate) tol: f64,
    #[serde(default)]
    pub(crate) weighted: bool,
}

// Sherman-William-Woodbury (Update, online versions) LstsqKwargs
#[derive(Deserialize, Debug)]
pub(crate) struct SWWLstsqKwargs {
    pub(crate) null_policy: String,
    pub(crate) n: usize,
    pub(crate) bias: bool,
    pub(crate) lambda: f64,
    pub(crate) min_size: usize,
}

fn report_output(_: &[Field]) -> PolarsResult<Field> {
    let features = Field::new("features", DataType::String); // index of feature
    let beta = Field::new("beta", DataType::Float64); // estimated value for this coefficient
    let stderr = Field::new("std_err", DataType::Float64); // Std Err for this coefficient
    let t = Field::new("t", DataType::Float64); // t value for this coefficient
    let p = Field::new("p>|t|", DataType::Float64); // p value for this coefficient
    let ci_lower = Field::new("0.025", DataType::Float64); // CI lower bound at 0.025
    let ci_upper = Field::new("0.975", DataType::Float64); // CI upper bound at 0.975
    let v: Vec<Field> = vec![features, beta, stderr, t, p, ci_lower, ci_upper]; //  ci_lower, ci_upper
    Ok(Field::new("lstsq_report", DataType::Struct(v)))
}

fn pred_residue_output(_: &[Field]) -> PolarsResult<Field> {
    let pred = Field::new("pred", DataType::Float64);
    let residue = Field::new("resid", DataType::Float64);
    let v = vec![pred, residue];
    Ok(Field::new("pred", DataType::Struct(v)))
}

fn coeff_pred_output(_: &[Field]) -> PolarsResult<Field> {
    let coeffs = Field::new("coeffs", DataType::List(Box::new(DataType::Float64)));
    let pred = Field::new("prediction", DataType::Float64);
    let v: Vec<Field> = vec![coeffs, pred];
    Ok(Field::new("", DataType::Struct(v)))
}

fn coeff_singular_values_output(_: &[Field]) -> PolarsResult<Field> {
    let coeffs = Field::new("coeffs", DataType::List(Box::new(DataType::Float64)));
    let singular_values = Field::new("singular_values", DataType::List(Box::new(DataType::Float64)));
    let v: Vec<Field> = vec![coeffs, singular_values];
    Ok(Field::new("", DataType::Struct(v)))
}

fn coeff_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "coeffs",
        DataType::List(Box::new(DataType::Float64)),
    ))
}

fn infer_regression_method(l1_reg:f64, l2_reg:f64) -> LRMethods {
    if l1_reg > 0. && l2_reg <= 0. {
        LRMethods::L1 
    } else if l1_reg <= 0. && l2_reg > 0. {
        LRMethods::L2
    } else {
        LRMethods::Normal
    }
}

/// Returns a Array2 ready for linear regression, and a mask, where true means the row doesn't contain null
#[inline(always)]
fn series_to_mat_for_lstsq(
    inputs: &[Series],
    add_bias: bool,
    null_policy: NullPolicy,
) -> PolarsResult<(Array2<f64>, BooleanChunked)> {
    let n_features = inputs.len().abs_diff(1);

    // minus 1 because target is also in inputs. Target is at position 0.
    let y_has_null = inputs[0].has_validity();
    let has_null = inputs[1..].iter().fold(false, |_, s| s.has_validity()) | y_has_null;

    let mut df = to_frame(inputs)?;
    if df.is_empty() {
        return Err(PolarsError::ComputeError("Lstsq: empty data".into()));
    }
    // Add a constant column if add_bias
    if add_bias {
        df = df.lazy().with_column(lit(1f64)).collect()?;
    }

    // In mask, true means not null.
    let y_name = inputs[0].name();
    let init_mask = inputs[0].is_not_null();
    let (df, mask) = if has_null {
        match null_policy {
            // Like ignore, skip_window takes the raw data. The actual skip is done in the underlying function in linalg.
            NullPolicy::IGNORE | NullPolicy::SKIP_WINDOW => {
                // false, because it has nulls
                Ok((df, BooleanChunked::from_slice("", &[false])))
            }
            NullPolicy::RAISE => Err(PolarsError::ComputeError(
                "Lstsq: nulls found in data".into(),
            )),
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
                        .exclude([y_name])
                        .cast(DataType::Float64)
                        .fill_null(lit(x))])
                    .collect()?;

                if y_has_null {
                    df = df.filter(&init_mask).unwrap();
                    Ok((df, init_mask))
                } else {
                    // all filled, no nulls
                    let mask = BooleanChunked::from_slice("", &[true]);
                    Ok((df, mask))
                }
            }
            NullPolicy::FILL_WINDOW(x) => {
                df = df
                    .lazy()
                    .with_columns([pl::col("*")
                        .exclude([y_name])
                        .cast(DataType::Float64)
                        .fill_null(lit(x))])
                    .collect()?;

                if y_has_null {
                    // Unlike fill, this doesn't drop y's nulls
                    Ok((df, BooleanChunked::from_slice("", &[false])))
                } else {
                    // all filled, no nulls
                    let mask = BooleanChunked::from_slice("", &[true]);
                    Ok((df, mask))
                }
            }
        }
    } else {
        // In this case, the (!mask).any() is never true, which means there is no null.
        let mask = BooleanChunked::from_slice("", &[true]);
        Ok((df, mask))
    }?;

    if df.height() < n_features {
        Err(PolarsError::ComputeError(
            "#Data < #features. No conclusive result.".into(),
        ))
    } else {
        let mat = df.to_ndarray::<Float64Type>(IndexOrder::Fortran)?;
        Ok((mat, mask))
    }
}

#[polars_expr(output_type_func=coeff_output)]
fn pl_lstsq(inputs: &[Series], kwargs: LstsqKwargs) -> PolarsResult<Series> {
    let add_bias = kwargs.bias;
    let null_policy = NullPolicy::try_from(kwargs.null_policy)
        .map_err(|e| PolarsError::ComputeError(e.into()))?;

    let solver = kwargs.solver.as_str().into();
    let weighted = kwargs.weighted;
    let data_for_matrix = if weighted {&inputs[1..]} else {inputs};
 
    match series_to_mat_for_lstsq(data_for_matrix, add_bias, null_policy) {
        Ok((mat, _)) => {
            // Solving Least Square
            let x = mat.slice(s![.., 1..]).into_faer();
            let y = mat.slice(s![.., 0..1]).into_faer();
            let coeffs = if weighted {
                let weights = inputs[0].f64().unwrap();
                let weights = weights.cont_slice().unwrap();
                if weights.len() != mat.nrows() {
                    return Err(PolarsError::ComputeError(
                        "Length of weights and data in X must be the same.".into(),
                    ));
                }
                faer_weighted_lstsq(x, y, weights, solver)
            } else {
                match infer_regression_method(kwargs.l1_reg, kwargs.l2_reg) {
                    LRMethods::Normal => faer_solve_lstsq(x, y, solver),
                    LRMethods::L1 => faer_lasso_regression(x, y, kwargs.l1_reg, add_bias, kwargs.tol),
                    LRMethods::L2 => faer_solve_ridge(x, y, kwargs.l2_reg, add_bias, solver)
                }
            };
            let mut builder: ListPrimitiveChunkedBuilder<Float64Type> =
                ListPrimitiveChunkedBuilder::new("coeffs", 1, coeffs.nrows(), DataType::Float64);

            builder.append_slice(coeffs.col_as_slice(0));
            let out = builder.finish();
            Ok(out.into_series())
        }
        Err(e) => Err(e),
    }
}

#[polars_expr(output_type_func=coeff_singular_values_output)]
fn pl_lstsq_w_rcond(inputs: &[Series], kwargs: LstsqKwargs) -> PolarsResult<Series> {
    let add_bias = kwargs.bias;
    let null_policy = NullPolicy::try_from(kwargs.null_policy)
        .map_err(|e| PolarsError::ComputeError(e.into()))?;

    let method = if kwargs.l2_reg > 0. {ClosedFormLRMethods::L2} else {ClosedFormLRMethods::Normal};

    // Target y is at index 0
    match series_to_mat_for_lstsq(inputs, add_bias, null_policy) {
        Ok((mat, _)) => {
            // rcond will be passed as tol 
            let rcond = kwargs.tol.max(
                f64::EPSILON * (inputs.len().max(mat.len())) as f64
            );

            // Solving Least Square
            let x = mat.slice(s![.., 1..]).into_faer();
            let y = mat.slice(s![.., 0..1]).into_faer();

            //     // faer_solve_ridge_rcond
            let (coeffs, singular_values) = match method {
                ClosedFormLRMethods::Normal => faer_solve_lstsq_rcond(x, y, rcond),
                ClosedFormLRMethods::L2 => faer_solve_ridge_rcond(x, y, kwargs.l2_reg, add_bias, rcond),
            };

            let mut builder: ListPrimitiveChunkedBuilder<Float64Type> =
                ListPrimitiveChunkedBuilder::new("coeffs", 1, coeffs.nrows(), DataType::Float64);

            builder.append_slice(coeffs.col_as_slice(0));
            let coeffs_ca = builder.finish();

            let mut sv_builder: ListPrimitiveChunkedBuilder<Float64Type> =
                ListPrimitiveChunkedBuilder::new("singular_values", 1, singular_values.len(), DataType::Float64);

            sv_builder.append_slice(&singular_values);
            let coeffs_sv = sv_builder.finish();

            let ca = StructChunked::new("", &[coeffs_ca.into_series(), coeffs_sv.into_series()])?;
            Ok(ca.into_series())
        }
        Err(e) => Err(e),
    }
}


#[polars_expr(output_type_func=pred_residue_output)]
fn pl_lstsq_pred(inputs: &[Series], kwargs: LstsqKwargs) -> PolarsResult<Series> {
    let add_bias = kwargs.bias;
    let null_policy = NullPolicy::try_from(kwargs.null_policy)
        .map_err(|e| PolarsError::ComputeError(e.into()))?;

    let solver = kwargs.solver.as_str().into();
    let weighted = kwargs.weighted;
    let data_for_matrix = if weighted {&inputs[1..]} else {inputs};

    match series_to_mat_for_lstsq(data_for_matrix, add_bias, null_policy.clone()) {
        Ok((mat, mask)) => {
            let y = mat.slice(s![.., 0..1]).into_faer();
            let x = mat.slice(s![.., 1..]).into_faer();
            let coeffs =if weighted {
                let weights = inputs[0].f64().unwrap();
                let weights = weights.cont_slice().unwrap();
                if weights.len() != mat.nrows() {
                    return Err(PolarsError::ComputeError(
                        "Length of weights and data in X must be the same.".into(),
                    ));
                }
                faer_weighted_lstsq(x, y, weights, solver)
            } else {
                match infer_regression_method(kwargs.l1_reg, kwargs.l2_reg) {
                    LRMethods::Normal => faer_solve_lstsq(x, y, solver),
                    LRMethods::L1 => faer_lasso_regression(x, y, kwargs.l1_reg, add_bias, kwargs.tol),
                    LRMethods::L2 => faer_solve_ridge(x, y, kwargs.l2_reg, add_bias, solver)
                }
            };

            let pred = x * &coeffs;
            let resid = y - &pred;
            let pred = pred.col_as_slice(0);
            let resid = resid.col_as_slice(0);
            // If null policy is raise and we have nulls, we won't reach here
            // If null policy is raise and we are here, then (!&mask).any() will be false.
            // No need to check null policy here.
            let (p, r) = if (!&mask).any() {
                let mut p_builder: PrimitiveChunkedBuilder<Float64Type> =
                    PrimitiveChunkedBuilder::new("pred", mask.len());
                let mut r_builder: PrimitiveChunkedBuilder<Float64Type> =
                    PrimitiveChunkedBuilder::new("resid", mask.len());
                let mut i: usize = 0;
                for mm in mask.into_no_null_iter() {
                    // mask is always non-null, mm = true means it is not null
                    if mm {
                        p_builder.append_value(pred[i]);
                        r_builder.append_value(resid[i]);
                        i += 1;
                    } else {
                        p_builder.append_value(f64::NAN);
                        r_builder.append_value(f64::NAN);
                    }
                }
                (p_builder.finish(), r_builder.finish())
            } else {
                let pred = Float64Chunked::from_slice("pred", pred);
                let residue = Float64Chunked::from_slice("resid", resid);
                (pred, residue)
            };
            let p = p.into_series();
            let r = r.into_series();
            let out = StructChunked::new("", &[p, r])?;
            Ok(out.into_series())
        }
        Err(e) => Err(e),
    }
}

#[polars_expr(output_type_func=report_output)]
fn pl_lstsq_report(inputs: &[Series], kwargs: LstsqKwargs) -> PolarsResult<Series> {
    let add_bias = kwargs.bias;
    let null_policy = NullPolicy::try_from(kwargs.null_policy)
        .map_err(|e| PolarsError::ComputeError(e.into()))?;
    // index 0 is target y. Skip
    let mut name_builder =
        StringChunkedBuilder::new("features", inputs.len() + (add_bias) as usize);
    for s in inputs[1..].iter().map(|s| s.name()) {
        name_builder.append_value(s);
    }
    if add_bias {
        name_builder.append_value("__bias__");
    }
    // Copy data
    // Target y is at index 0
    match series_to_mat_for_lstsq(inputs, add_bias, null_policy) {
        Ok((mat, _)) => {
            let ncols = mat.ncols() - 1;
            let nrows = mat.nrows();

            let x = mat.slice(s![0..nrows, 1..]).into_faer();
            let y = mat.slice(s![0..nrows, 0..1]).into_faer();
            // Solving Least Square
            let xtx = x.transpose() * &x;
            let xtx_qr = xtx.col_piv_qr();
            let xtx_inv = xtx_qr.inverse();
            let coeffs = &xtx_inv * x.transpose() * y;
            let betas = coeffs.col_as_slice(0);
            // Degree of Freedom
            let dof = nrows as f64 - ncols as f64;
            // Residue
            let res = y - x * &coeffs;
            // total residue, sum of squares
            let mse = (res.transpose() * &res).read(0, 0) / dof;
            // std err
            let std_err = (0..ncols)
                .map(|i| (mse * xtx_inv.read(i, i)).sqrt())
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
                    |t| match crate::stats_utils::beta::student_t_sf(t.abs(), dof) {
                        Ok(p) => 2.0 * p,
                        Err(_) => f64::NAN,
                    },
                )
                .collect_vec();

            let t_alpha = crate::stats_utils::beta::student_t_ppf(0.975, dof);
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
            let coeffs_series = Float64Chunked::from_slice("beta", betas);
            let coeffs_series = coeffs_series.into_series();
            let stderr_series = Float64Chunked::from_vec("std_err", std_err);
            let stderr_series = stderr_series.into_series();
            let t_series = Float64Chunked::from_vec("t", t_values);
            let t_series = t_series.into_series();
            let p_series = Float64Chunked::from_vec("p>|t|", p_values);
            let p_series = p_series.into_series();
            let lower = Float64Chunked::from_vec("0.025", ci_lower);
            let lower = lower.into_series();
            let upper = Float64Chunked::from_vec("0.975", ci_upper);
            let upper = upper.into_series();
            let out = StructChunked::new(
                "lstsq_report",
                &[
                    names_series,
                    coeffs_series,
                    stderr_series,
                    t_series,
                    p_series,
                    lower,
                    upper,
                ],
            )?;
            Ok(out.into_series())
        }
        Err(e) => Err(e),
    }
}

#[polars_expr(output_type_func=report_output)]
fn pl_wls_report(inputs: &[Series], kwargs: LstsqKwargs) -> PolarsResult<Series> {
    let add_bias = kwargs.bias;
    let null_policy = NullPolicy::try_from(kwargs.null_policy)
        .map_err(|e| PolarsError::ComputeError(e.into()))?;

    let weights = inputs[0].f64().unwrap();
    let weights = weights.cont_slice().unwrap();
    // index 0 is weights, 1 is target y. Skip them
    let mut name_builder =
        StringChunkedBuilder::new("features", inputs.len() + (add_bias) as usize);
    for s in inputs[2..].iter().map(|s| s.name()) {
        name_builder.append_value(s);
    }
    if add_bias {
        name_builder.append_value("__bias__");
    }
    // Copy data
    // Target y is at index 1, weights 0
    match series_to_mat_for_lstsq(&inputs[1..], add_bias, null_policy) {
        Ok((mat, _)) => {
            let ncols = mat.ncols() - 1;
            let nrows = mat.nrows();

            let x = mat.slice(s![0..nrows, 1..]).into_faer();
            let y = mat.slice(s![0..nrows, 0..1]).into_faer();

            let w = faer::mat::from_row_major_slice(weights, x.nrows(), 1);
            let w = w.column_vector_as_diagonal();
            let xt = x.transpose();

            let xtwx = xt * w * x;
            let xtwy = xt * w * y;
            let qr = xtwx.col_piv_qr();
            let xtwx_inv = qr.inverse();
            let coeffs = qr.solve(xtwy);
            let betas = coeffs.col_as_slice(0);

            // Degree of Freedom
            let dof = nrows as f64 - ncols as f64;
            // Residue
            let res = y - x * &coeffs;
            let mse =
                (0..y.nrows()).fold(0., |acc, i| acc + weights[i] * res.read(i, 0).powi(2)) / dof;
            // std err
            let std_err = (0..ncols)
                .map(|i| (mse * xtwx_inv.read(i, i)).sqrt())
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
                    |t| match crate::stats_utils::beta::student_t_sf(t.abs(), dof) {
                        Ok(p) => 2.0 * p,
                        Err(_) => f64::NAN,
                    },
                )
                .collect_vec();

            let t_alpha = crate::stats_utils::beta::student_t_ppf(0.975, dof);
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
            let coeffs_series = Float64Chunked::from_slice("beta", betas);
            let coeffs_series = coeffs_series.into_series();
            let stderr_series = Float64Chunked::from_vec("std_err", std_err);
            let stderr_series = stderr_series.into_series();
            let t_series = Float64Chunked::from_vec("t", t_values);
            let t_series = t_series.into_series();
            let p_series = Float64Chunked::from_vec("p>|t|", p_values);
            let p_series = p_series.into_series();
            let lower = Float64Chunked::from_vec("0.025", ci_lower);
            let lower = lower.into_series();
            let upper = Float64Chunked::from_vec("0.975", ci_upper);
            let upper = upper.into_series();
            let out = StructChunked::new(
                "lstsq_report",
                &[
                    names_series,
                    coeffs_series,
                    stderr_series,
                    t_series,
                    p_series,
                    lower,
                    upper,
                ],
            )?;
            Ok(out.into_series())
        }
        Err(e) => Err(e),
    }
}

// --- Rolling and Recursive

#[polars_expr(output_type_func=coeff_pred_output)]
fn pl_recursive_lstsq(inputs: &[Series], kwargs: SWWLstsqKwargs) -> PolarsResult<Series> {
    let n = kwargs.n; // Gauranteed n >= 1
    let has_bias = kwargs.bias;
    
    // Gauranteed in Python that this won't be SKIP. SKIP doesn't work now.
    let null_policy = NullPolicy::try_from(kwargs.null_policy)
        .map_err(|e| PolarsError::ComputeError(e.into()))?;

    // Target y is at index 0
    match series_to_mat_for_lstsq(inputs, has_bias, null_policy) {
        Ok((mat, mask)) => {
            // Solving Least Square
            let x = mat.slice(s![.., 1..]).into_faer();
            let y = mat.slice(s![.., 0..1]).into_faer();

            let coeffs = faer_recursive_lstsq(x, y, n, kwargs.lambda, has_bias);
            let mut builder: ListPrimitiveChunkedBuilder<Float64Type> =
                ListPrimitiveChunkedBuilder::new(
                    "coeffs",
                    mat.nrows(),
                    mat.ncols(),
                    DataType::Float64,
                );
            let mut pred_builder: PrimitiveChunkedBuilder<Float64Type> =
                PrimitiveChunkedBuilder::new("pred", mat.nrows());

            // Note, if has_null is true, and null policy is raise, then we won't reach here.
            // So in the has_null branch, only possibility is that null_policy != Raise

            // Fill or Skip strategy can drop nulls. Fill will drop null when y has nulls.
            // Skip will drop nulls whenever there is a null in the row.
            // Mask true means the row is good
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
                        let pred = (row * coefficients).read(0, 0);
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
                    let pred = (row * &coefficients).read(0, 0);
                    let coef = coefficients.col_as_slice(0);
                    builder.append_slice(coef);
                    pred_builder.append_value(pred);
                }
            }

            let coef_out = builder.finish();
            let pred_out = pred_builder.finish();
            let ca = StructChunked::new("", &[coef_out.into_series(), pred_out.into_series()])?;
            Ok(ca.into_series())
        }
        Err(e) => Err(e),
    }
}

#[polars_expr(output_type_func=coeff_pred_output)] // They share the same output type
fn pl_rolling_lstsq(inputs: &[Series], kwargs: SWWLstsqKwargs) -> PolarsResult<Series> {
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
    match series_to_mat_for_lstsq(inputs, has_bias, null_policy) {
        Ok((mat, mask)) => {
            let should_skip = match null_policy {
                NullPolicy::SKIP_WINDOW | NullPolicy::FILL_WINDOW(_) => (!&mask).any(),
                _ => false, // raise, ignore
            };
            let x = mat.slice(s![.., 1..]).into_faer();
            let y = mat.slice(s![.., 0..1]).into_faer();
            let coeffs = match should_skip {
                true => faer_rolling_skipping_lstsq(x, y, n, kwargs.min_size, kwargs.lambda, has_bias),
                false => faer_rolling_lstsq(x, y, n, kwargs.lambda, has_bias),
            };

            let mut builder: ListPrimitiveChunkedBuilder<Float64Type> =
                ListPrimitiveChunkedBuilder::new(
                    "coeffs",
                    mat.nrows(),
                    mat.ncols(),
                    DataType::Float64,
                );
            let mut pred_builder: PrimitiveChunkedBuilder<Float64Type> =
                PrimitiveChunkedBuilder::new("pred", mat.nrows());

            let m = n - 1; // n >= 2 guaranteed in Python
            for _ in 0..m {
                builder.append_null();
                pred_builder.append_null();
            }

            // Strictly speaking I don't need this branch
            if should_skip {
                // Skipped rows will have coeffs with shape (0, 0)
                for (i, coefficients) in coeffs.into_iter().enumerate() {
                    if coefficients.shape() == (0, 0) {
                        builder.append_null();
                        pred_builder.append_null();
                    } else {
                        let row = x.get(m + i..m + i + 1, ..);
                        let pred = (row * &coefficients).read(0, 0);
                        let coef = coefficients.col_as_slice(0);
                        builder.append_slice(coef);
                        pred_builder.append_value(pred);
                    }
                }
            } else {
                // nothing is skipped. All coeffs must be valid.
                for (i, coefficients) in coeffs.into_iter().enumerate() {
                    let row = x.get(m + i..m + i + 1, ..);
                    let pred = (row * &coefficients).read(0, 0);
                    let coef = coefficients.col_as_slice(0);
                    builder.append_slice(coef);
                    pred_builder.append_value(pred);
                }
            }

            let coef_out = builder.finish();
            let pred_out = pred_builder.finish();
            let ca = StructChunked::new("", &[coef_out.into_series(), pred_out.into_series()])?;
            Ok(ca.into_series())
        }
        Err(e) => Err(e),
    }
}
