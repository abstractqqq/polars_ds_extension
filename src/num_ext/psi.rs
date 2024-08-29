use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

fn psi_report_output(_: &[Field]) -> PolarsResult<Field> {
    let breakpoints = Field::new("cnt<=", DataType::Float64);
    let baseline_pct = Field::new("baseline_pct", DataType::Float64);
    let actual_pct = Field::new("actual_pct", DataType::Float64);
    let psi_bins = Field::new("psi_bin", DataType::Float64);
    let v: Vec<Field> = vec![breakpoints, baseline_pct, actual_pct, psi_bins];
    Ok(Field::new("psi_report", DataType::Struct(v)))
}

/// Computes counts in each bucket given by the breakpoints in
/// a PSI computation. This returns the count for the first series
/// and the count for the second series.
/// This assumes the breakpoints (bp)'s last value is always INF
#[inline(always)]
fn psi_with_bps_helper(s: &[f64], bp: &[f64]) -> Vec<u32> {
    // s: data
    // bp: breakpoints
    let mut c = vec![0u32; bp.len()];
    for x in s {
        let i = match bp.binary_search_by(|b| b.partial_cmp(x).unwrap()) {
            Ok(j) => j,
            Err(k) => k,
        };
        c[i] += 1;
    }
    c
}

/// Helper function to create PSI reports for numeric PSI computations.
#[inline(always)]
fn psi_frame(bp: &[f64], bp_name: &str, cnt1: &[u32], cnt2: &[u32]) -> PolarsResult<LazyFrame> {
    let b = Float64Chunked::from_slice("", bp);
    let c1 = UInt32Chunked::from_slice("", cnt1);
    let c2 = UInt32Chunked::from_slice("", cnt2);

    let df = df!(
        bp_name => b,
        "cnt_baseline" => c1,
        "cnt_actual" => c2,
    )?
    .lazy();

    Ok(df
        .with_columns([
            (col("cnt_baseline").cast(DataType::Float64)
                / col("cnt_baseline").sum().cast(DataType::Float64))
            .clip_min(lit(0.0001))
            .alias("baseline_pct"),
            (col("cnt_actual").cast(DataType::Float64)
                / col("cnt_actual").sum().cast(DataType::Float64))
            .clip_min(lit(0.0001))
            .alias("actual_pct"),
        ])
        .select([
            col(bp_name),
            col("baseline_pct"),
            col("actual_pct"),
            ((col("baseline_pct") - col("actual_pct"))
                * ((col("baseline_pct") / col("actual_pct")).log(std::f64::consts::E)))
            .alias("psi_bin"),
        ]))
}

/// Computs PSI with custom breakpoints and returns a report
#[polars_expr(output_type_func=psi_report_output)]
fn pl_psi_w_bps(inputs: &[Series]) -> PolarsResult<Series> {
    let data1 = inputs[0].f64().unwrap();
    let data2 = inputs[1].f64().unwrap();
    let breakpoints = inputs[2].f64().unwrap();

    let s1 = data1.cont_slice().unwrap();
    let s2 = data2.cont_slice().unwrap();

    let bp = breakpoints.cont_slice().unwrap();
    let c1 = psi_with_bps_helper(s1, bp);
    let c2 = psi_with_bps_helper(s2, bp);

    let psi_report = psi_frame(bp, "cnt<=", &c1, &c2)?.collect()?;
    Ok(psi_report.into_struct("").into_series())
}

/// Numeric PSI report
#[polars_expr(output_type_func=psi_report_output)]
fn pl_psi_report(inputs: &[Series]) -> PolarsResult<Series> {
    // The new data
    let new = inputs[0].f64().unwrap();
    // The breaks learned from baseline/reference
    let brk = inputs[1].f64().unwrap();
    // The cnts for the baseline/reference
    let cnt = inputs[2].u32().unwrap();

    let data_new = new.cont_slice().unwrap();
    let ref_brk = brk.cont_slice().unwrap();
    let ref_cnt = cnt.cont_slice().unwrap();

    let new_cnt = psi_with_bps_helper(data_new, ref_brk);
    let psi_report = psi_frame(ref_brk, "cnt<=", ref_cnt, &new_cnt)?.collect()?;

    Ok(psi_report.into_struct("").into_series())
}

/// Discrete PSI report
#[polars_expr(output_type_func=psi_report_output)]
fn pl_psi_discrete_report(inputs: &[Series]) -> PolarsResult<Series> {
    let df1 = df!(
        "actual_cat" => &inputs[0], // data cats
        "actual_cnt" => &inputs[1], // data cnt
    )?;
    let df2 = df!(
        "baseline_cat" => &inputs[2], // ref cats
        "baseline_cnt" => &inputs[3], // ref cnt
    )?;

    let psi_report = df1
        .lazy()
        .join(
            df2.lazy(),
            [col("actual_cat")],
            [col("baseline_cat")],
            JoinArgs::new(JoinType::Full),
        )
        .with_columns([
            col("baseline_cnt").fill_null(0),
            col("actual_cnt").fill_null(0),
        ])
        .with_columns([
            (col("baseline_cnt").cast(DataType::Float64)
                / col("baseline_cnt").sum().cast(DataType::Float64))
            .clip_min(lit(0.0001))
            .alias("baseline_pct"),
            (col("actual_cnt").cast(DataType::Float64)
                / col("actual_cnt").sum().cast(DataType::Float64))
            .clip_min(lit(0.0001))
            .alias("actual_pct"),
        ])
        .select([
            col("baseline_cat").alias("baseline_category"),
            col("actual_cat").alias("actual_category"),
            col("baseline_pct"),
            col("actual_pct"),
            ((col("baseline_pct") - col("actual_pct"))
                * ((col("baseline_pct") / col("actual_pct")).log(std::f64::consts::E)))
            .alias("psi_bin"),
        ])
        .collect()?;

    Ok(psi_report.into_struct("").into_series())
}
