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

/// Computs counts in each bucket given by the breakpoints in
/// a PSI computation. This returns the count for the first series
/// and the count for the second series.
/// This assumes the breakpoints (bp)'s last value is always INF
fn psi_with_bps_helper(s: &[f64], bp: &[f64]) -> Vec<u32> {
    // s1: data1, s2: data2
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
            .alias("baseline_pct"),
            (col("cnt_actual").cast(DataType::Float64)
                / col("cnt_actual").sum().cast(DataType::Float64))
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

/// Computs PSI with custom breakpoints
#[polars_expr(output_type_func=psi_report_output)]
fn pl_psi_w_bps(inputs: &[Series]) -> PolarsResult<Series> {
    let data1 = inputs[0].f64().unwrap();
    let data2 = inputs[1].f64().unwrap();
    let breakpoints = inputs[2].f64().unwrap();

    if breakpoints.len() == 0 {
        return Err(PolarsError::ComputeError(
            "PSI: breakpoints is empty.".into(),
        ));
    }

    let s1 = data1.cont_slice().unwrap();
    let s2 = data2.cont_slice().unwrap();

    let bp = breakpoints.cont_slice().unwrap();
    let c1 = psi_with_bps_helper(s1, bp);
    let c2 = psi_with_bps_helper(s2, bp);

    let psi_report = psi_frame(bp, "cnt<=", &c1, &c2)?.collect()?;
    Ok(psi_report.into_struct("").into_series())
}

#[polars_expr(output_type=Float64)]
fn pl_psi(inputs: &[Series]) -> PolarsResult<Series> {
    // The actual data
    let data = inputs[0].f64()?;
    let name = data.name();
    let data = data.cont_slice().unwrap();
    // breaks according to reference, already sorted, should be contiguous
    let brk = inputs[1].f64()?;
    if brk.len() < 2 {
        return Err(PolarsError::ComputeError(
            "PSI: Not enough bins can be created.".into(),
        ));
    }
    // cnts for each brk in ref
    let cnt_ref = inputs[2].u32()?;
    let cnt_ref = cnt_ref.cont_slice().unwrap();
    // slice to do binary search with.
    let brk_sl = brk.cont_slice().unwrap();
    // Compute the correct cnt (of values that are inside the bins defined by ref)
    let mut cnt_data = vec![0_u32; brk_sl.len()];

    for d in data {
        // values in brk_sl is guaranteed to be sorted, unique, and finite
        let idx = match brk_sl.binary_search_by(|x| x.partial_cmp(&d).unwrap()) {
            Ok(i) => i,
            Err(j) => j,
        };
        cnt_data[idx] += 1;
    }
    // Total cnt in ref
    let ref_total = cnt_ref.into_iter().sum::<u32>() as f64;
    // Total cnt in actual
    let act_total = data.len() as f64; // cnt_data.iter().sum::<u32>() as f64;

    let psi = cnt_ref.iter().zip(cnt_data.iter()).fold(0., |acc, (a, b)| {
        let aa = ((*a as f64) / ref_total).max(0.0001_f64);
        let bb = ((*b as f64) / act_total).max(0.0001_f64);
        acc + (aa - bb) * (aa / bb).ln()
    });
    let out = Float64Chunked::from_vec(name, vec![psi]);
    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn pl_psi_discrete(inputs: &[Series]) -> PolarsResult<Series> {
    let name = inputs[0].name();
    if inputs[0].len() == 1 && inputs[2].len() == 1 {
        let v1 = inputs[0].get(0).unwrap();
        let v2 = inputs[1].get(0).unwrap();
        let psi: f64 = if v1.eq(&v2) {
            0_f64
        } else {
            2.0_f64 * (1.0_f64 - 0.0001_f64) * (1.0_f64 / 0.0001_f64).ln()
        };
        let out = Float64Chunked::from_iter([Some(psi)]);
        return Ok(out.into_series());
    } else if inputs[0].len() < 2 || inputs[2].len() < 2 {
        // Only possible when either one is length 0
        return Err(PolarsError::ComputeError(
            "PSI: One of input columns is empty.".into(),
        ));
    }

    let df1 = df!(
        "data_discrete" => &inputs[0], // data cats
        "data_cnt" => &inputs[1], // data cnt
    )?;
    let df2 = df!(
        "ref_discrete" => &inputs[2], // ref cats
        "ref_cnt" => &inputs[3], // ref cnt
    )?;

    let mut df = df1
        .lazy()
        .join(
            df2.lazy(),
            [col("data_discrete")],
            [col("ref_discrete")],
            JoinArgs::new(JoinType::Outer { coalesce: false }),
        )
        .with_columns([col("data_cnt").fill_null(0), col("ref_cnt").fill_null(0)])
        .collect()?;

    let data_total = inputs[1].sum::<u32>().unwrap() as f64;
    let ref_total = inputs[3].sum::<u32>().unwrap() as f64;

    let cnt_data = df.drop_in_place("data_cnt").unwrap();
    let cnt_data = cnt_data.u32()?;
    let cnt_ref = df.drop_in_place("ref_cnt").unwrap();
    let cnt_ref = cnt_ref.u32()?;
    // PSI. The series should be all non null
    let psi = cnt_ref
        .into_no_null_iter()
        .zip(cnt_data.into_no_null_iter())
        .fold(0., |acc, (a, b)| {
            let aa = ((a as f64) / ref_total).max(0.0001_f64);
            let bb = ((b as f64) / data_total).max(0.0001_f64);
            acc + (aa - bb) * (aa / bb).ln()
        });
    let out = Float64Chunked::from_vec(name, vec![psi]);
    Ok(out.into_series())
}
