use faer::reborrow::IntoConst;
/// All things true positive, false positive related.
/// ROC AUC, Average Precision, precision, recall, etc. m
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

fn combo_output(_: &[Field]) -> PolarsResult<Field> {
    let roc_auc = Field::new("roc_auc".into(), DataType::Float64);
    let precision = Field::new("precision".into(), DataType::Float64);
    let recall = Field::new("recall".into(), DataType::Float64);
    let f = Field::new("f".into(), DataType::Float64);
    let avg_precision = Field::new("avg_precision".into(), DataType::Float64);
    let v: Vec<Field> = vec![precision, recall, f, avg_precision, roc_auc];
    Ok(Field::new("".into(), DataType::Struct(v)))
}

fn tpr_fpr_output(_: &[Field]) -> PolarsResult<Field> {
    let threshold = Field::new("threshold".into(), DataType::Float64);
    let tpr = Field::new("tpr".into(), DataType::Float64);
    let fpr = Field::new("fpr".into(), DataType::Float64);
    let v: Vec<Field> = vec![threshold, tpr, fpr];
    Ok(Field::new("tpr_fpr".into(), DataType::Struct(v)))
}

fn tp_fp_frame(
    predicted: &Series,
    actual: &Series,
    positive_count: u32,
    as_ratio: bool,
) -> PolarsResult<LazyFrame> {
    // Checking for data quality issues
    if (actual.len() != predicted.len())
        || actual.is_empty()
        || predicted.is_empty()
        || (!predicted.dtype().is_primitive_numeric())
        || actual.has_nulls()
        || predicted.has_nulls()
    {
        return Err(PolarsError::ComputeError(
            "ROC AUC: Input columns must be the same length, non-empty, numeric, and shouldn't contain nulls."
            .into(),
        ));
    }

    // Start computing
    let n = predicted.len() as u32;
    let df = df!(
        "threshold" => predicted,
        "actual" => actual
    )?;

    let temp = df
        .lazy()
        .group_by([col("threshold")])
        .agg([
            len().alias("cnt"),
            col("actual").sum().alias("pos_cnt_at_threshold"),
        ])
        .sort(["threshold"], Default::default())
        .with_columns([
            (lit(n) - col("cnt").cum_sum(false) + col("cnt")).alias("predicted_positive"),
            (lit(positive_count) - col("pos_cnt_at_threshold").cum_sum(false))
                .shift_and_fill(1, positive_count)
                .alias("tp"),
        ])
        .select([
            col("threshold"),
            col("tp"),
            (col("predicted_positive") - col("tp")).alias("fp"),
            (col("tp").cast(DataType::Float64) / col("predicted_positive").cast(DataType::Float64))
                .alias("precision"),
        ]);
    // col("cnt"),
    // col("predicted_positive")
    // col("pos_cnt_at_threshold"),
    if as_ratio {
        Ok(temp.select([
            col("threshold"),
            (col("tp").cast(DataType::Float64) / col("tp").first().cast(DataType::Float64))
                .alias("tpr"),
            (col("fp").cast(DataType::Float64) / col("fp").first().cast(DataType::Float64))
                .alias("fpr"),
            col("precision"),
        ]))
    } else {
        Ok(temp)
    }
}

#[polars_expr(output_type_func=combo_output)]
fn pl_combo_b(inputs: &[Series]) -> PolarsResult<Series> {
    // actual, when passed in, is always u32 (done in Python extension side)
    let actual = &inputs[0];
    let predicted = &inputs[1];
    // Threshold for precision and recall
    let threshold = inputs[2].f64()?;
    let threshold = threshold.get(0).unwrap_or(0.5);

    let positive_count = actual.sum::<u32>().unwrap_or(0);
    if positive_count == 0 {
        return Ok(Series::from_iter([f64::NAN]));
    }

    let mut binding = tp_fp_frame(predicted, actual, positive_count, true)?.collect()?;
    let frame = binding.align_chunks();

    let tpr = frame.drop_in_place("tpr").unwrap();
    let fpr = frame.drop_in_place("fpr").unwrap();

    let precision = frame.drop_in_place("precision").unwrap();
    let precision = precision.f64()?;
    let precision = precision.cont_slice().unwrap();

    let probs = frame.drop_in_place("threshold").unwrap();
    let probs = probs.f64()?;
    let probs = probs.cont_slice().unwrap();
    let index = match probs.binary_search_by(|x| x.partial_cmp(&threshold).unwrap()) {
        Ok(i) => i,
        Err(i) => i,
    };

    // ROC AUC
    let y = tpr.f64().unwrap();
    let x = fpr.f64().unwrap();

    let y = y.cont_slice()?; // Zero copy
    let x = x.cont_slice()?; // Zero copy

    let auc = -super::trapz::trapz(y, x);
    let auc: Series = Series::from_vec("roc_auc".into(), vec![auc]);

    // Average Precision
    let ap: f64 = -1.0
        * y.iter()
            .zip(y.iter().skip(1))
            .zip(precision)
            .fold(0., |acc, ((y, y_next), p)| (y_next - y).mul_add(*p, acc));
    let ap: Series = Series::from_vec("average_precision".into(), vec![ap]);

    // Precision & Recall & F
    let recall = y[index];
    let precision = precision[index];
    let f: f64 = 2.0 * (precision * recall) / (precision + recall);
    let recall: Series = Series::from_vec("recall".into(), vec![recall]);
    let precision: Series = Series::from_vec("precision".into(), vec![precision]);
    let f: Series = Series::from_vec("f".into(), vec![f]);

    let out = StructChunked::from_series(
        "metrics".into(),
        1,
        [&precision, &recall, &f, &ap, &auc].into_iter(),
    )?;
    Ok(out.into_series())
}

#[polars_expr(output_type_func=tpr_fpr_output)]
fn pl_tpr_fpr(inputs: &[Series]) -> PolarsResult<Series> {
    // actual, when passed in, is always u32 (done in Python extension side)
    let actual = &inputs[0];
    let predicted = &inputs[1];
    let positive_cnt = actual.sum::<u32>().unwrap_or(0);

    if positive_cnt == 0 {
        let tpr = Series::from_vec("tpr".into(), vec![f64::NAN]);
        let fpr = Series::from_vec("fpr".into(), vec![f64::NAN]);
        let ca = StructChunked::from_columns(
            "tpr_fpr".into(),
            1,
            &[tpr.into_column(), fpr.into_column()],
        )?;
        Ok(ca.into_series())
    } else {
        let frame = tp_fp_frame(predicted, actual, positive_cnt, true)?
            .select([col("threshold"), col("tpr"), col("fpr")])
            .collect()?;

        let ca = frame.into_struct("tpr_fpr".into());
        Ok(ca.into_series())
    }
}

fn binary_confusion_matrix(combined_series: &UInt32Chunked) -> [u32; 4] {
    // The combined_series is (lit(2) * col("y_true") + col("y_pred")).alias("y")
    // 0 is tn, 1 is fp, 2 is fn, 3 is tp
    // At Python side, actual and pred columns were turned into boolean first. So
    // no invalid indices will happen.
    let mut output = [0u32; 4];
    for i in combined_series.into_no_null_iter() {
        let j = i % 4; // to make this safe
        output[j as usize] += 1;
    }
    output
}

#[polars_expr(output_type=Float64)]
fn pl_mcc(inputs: &[Series]) -> PolarsResult<Series> {
    let combined_series = &inputs[0];
    let combined_series = combined_series.u32()?;
    let cm = binary_confusion_matrix(combined_series);

    let n00 = cm[0];
    let n01 = cm[1];
    let n10 = cm[2];
    let n11 = cm[3];

    let n1_ = (n11 + n10) as f64;
    let n_1 = (n11 + n01) as f64;
    let n0_ = (n01 + n00) as f64;
    let n_0 = (n10 + n00) as f64;

    let phi = ((n11 * n00) as f64 - (n10 * n01) as f64) / (n1_ * n_1 * n0_ * n_0).sqrt();
    Ok(Series::from_iter([phi]))
}

#[polars_expr(output_type=Float64)]
fn pl_roc_auc(inputs: &[Series]) -> PolarsResult<Series> {
    // actual, when passed in, is always u32 (done in Python extension side)
    let actual = &inputs[0];
    let predicted = &inputs[1];
    let positive_cnt = actual.sum::<u32>().unwrap_or(0);
    if positive_cnt == 0 {
        return Ok(Series::from_iter([f64::NAN]));
    }

    let mut binding = tp_fp_frame(predicted, actual, positive_cnt, true)?
        .select([col("tpr"), col("fpr")])
        .collect()?;
    let frame = binding.align_chunks();

    let tpr = frame.drop_in_place("tpr").unwrap();
    let fpr = frame.drop_in_place("fpr").unwrap();

    // Should be contiguous. No need to rechunk
    let y = tpr.f64().unwrap();
    let x = fpr.f64().unwrap();

    let y = y.cont_slice()?;
    let x = x.cont_slice()?;

    let auc = -super::trapz::trapz(y, x);
    Ok(Series::from_vec("roc_auc".into(), vec![auc]))
}

// bcm = binary confusion matrix
fn bcm_output(_: &[Field]) -> PolarsResult<Field> {
    let tp = Field::new("tp".into(), DataType::UInt32);
    let fp = Field::new("fp".into(), DataType::UInt32);
    let tn = Field::new("tn".into(), DataType::UInt32);
    let fn_ = Field::new("fn".into(), DataType::UInt32);

    let fpr = Field::new("fpr".into(), DataType::Float64);
    let fnr = Field::new("fnr".into(), DataType::Float64);
    let tnr = Field::new("tnr".into(), DataType::Float64);
    let prevalence = Field::new("prevalence".into(), DataType::Float64);
    let prevalence_threshold = Field::new("prevalence_threshold".into(), DataType::Float64);
    let tpr = Field::new("tpr".into(), DataType::Float64);
    let informedness = Field::new("informedness".into(), DataType::Float64);
    let precision = Field::new("precision".into(), DataType::Float64);
    let false_omission_rate = Field::new("false_omission_rate".into(), DataType::Float64);
    let plr = Field::new("plr".into(), DataType::Float64);
    let nlr = Field::new("nlr".into(), DataType::Float64);
    let acc = Field::new("acc".into(), DataType::Float64);
    let balanced_accuracy = Field::new("balanced_accuracy".into(), DataType::Float64);
    let f1 = Field::new("f1".into(), DataType::Float64);
    let folkes_mallows_index = Field::new("folkes_mallows_index".into(), DataType::Float64);
    let mcc = Field::new("mcc".into(), DataType::Float64);
    let threat_score = Field::new("threat_score".into(), DataType::Float64);
    let markedness = Field::new("markedness".into(), DataType::Float64);
    let fdr = Field::new("fdr".into(), DataType::Float64);
    let npv = Field::new("npv".into(), DataType::Float64);
    let dor = Field::new("dor".into(), DataType::Float64);

    Ok(Field::new(
        "confusion_matrix".into(),
        DataType::Struct(vec![
            tp,
            tn,
            fp,
            fn_,
            tpr,
            tnr,
            fpr,
            fnr,
            prevalence,
            prevalence_threshold,
            informedness,
            precision,
            false_omission_rate,
            plr,
            nlr,
            acc,
            balanced_accuracy,
            f1,
            folkes_mallows_index,
            mcc,
            threat_score,
            markedness,
            fdr,
            npv,
            dor,
        ]),
    ))
}

#[polars_expr(output_type_func=bcm_output)]
fn pl_binary_confusion_matrix(inputs: &[Series]) -> PolarsResult<Series> {
    // The combined_series is (lit(2) * col("y_true") + col("y_pred")).alias("y")
    // 0 is tn, 1 is fp, 2 is fn, 3 is tp
    let combined_series = inputs[0].u32()?;
    let confusion = binary_confusion_matrix(combined_series);
    let tn = UInt32Chunked::from_vec("tn".into(), vec![confusion[0]]);
    let tn = Column::Series(tn.into_series().into());

    let fp = UInt32Chunked::from_vec("fp".into(), vec![confusion[1]]);
    let fp = Column::Series(fp.into_series().into());

    let fn_ = UInt32Chunked::from_vec("fn".into(), vec![confusion[2]]);
    let fn_ = Column::Series(fn_.into_series().into());

    let tp = UInt32Chunked::from_vec("tp".into(), vec![confusion[3]]);
    let tp = Column::Series(tp.into_series().into());
    // All series have length 1 and no duplicate names

    let df = DataFrame::new(vec![tn, fp, fn_, tp])?;

    let result = df
        .lazy()
        .with_columns([
            (col("tp") + col("fn")).alias("p"),
            (col("fp") + col("tn")).alias("n"),
        ])
        .with_columns([
            (col("tp") / col("p")).alias("tpr"),
            (col("fp") / col("n")).alias("fpr"),
            (col("tp") / (col("tp") + col("fp"))).alias("precision"),
            (col("fn") / (col("fn") + col("tn"))).alias("false_omission_rate"),
            (col("p") / (col("p") + col("n"))).alias("prevalence"),
            (col("tp") / (col("tp") + col("fn") + col("fp"))).alias("threat_score"),
        ])
        .with_columns([
            (lit(1) - col("tpr")).alias("fnr"),
            (lit(1) - col("fpr")).alias("tnr"),
            (lit(1) - col("false_omission_rate")).alias("npv"),
            (lit(1) - col("precision")).alias("fdr"),
        ])
        .with_columns([
            (col("tpr") / col("fpr")).alias("plr"),
            (col("fnr") / col("tnr")).alias("nlr"),
            (col("tpr") + col("tnr") - lit(1)).alias("informedness"),
            (col("precision") - col("false_omission_rate")).alias("markedness"),
            (((col("tpr") * col("fpr")).sqrt() - col("fpr")) / (col("tpr") - col("fpr")))
                .alias("prevalence_threshold"),
            ((col("tpr") + col("tnr")) / lit(2)).alias("balanced_accuracy"),
            ((lit(2) * col("precision") * col("tpr")) / (col("precision") + col("tpr")))
                .alias("f1"),
            ((col("precision") * col("tpr")).sqrt()).alias("folkes_mallows_index"),
            ((col("tpr") * col("tnr") * col("precision") * col("npv")).sqrt()
                - (col("fnr") * col("fpr") * col("false_omission_rate") * col("fdr")).sqrt())
            .alias("mcc"),
            ((col("tp") + col("tn")) / (col("p") + col("n"))).alias("acc"),
        ])
        .with_columns([(col("plr") / col("nlr")).alias("dor")])
        .select(&[
            col("tp"),
            col("tn"),
            col("fp"),
            col("fn"),
            col("tpr"),
            col("tnr"),
            col("fpr"),
            col("fnr"),
            col("prevalence"),
            col("prevalence_threshold"),
            col("informedness"),
            col("precision"),
            col("false_omission_rate"),
            col("plr"),
            col("nlr"),
            col("acc"),
            col("balanced_accuracy"),
            col("f1"),
            col("folkes_mallows_index"),
            col("mcc"),
            col("threat_score"),
            col("markedness"),
            col("fdr"),
            col("npv"),
            col("dor"),
        ])
        .fill_null(f64::NAN) // In Rust dividing by 0 results in Null for some reason.
        .collect()?;

    let out = result.into_struct("confusion_matrix".into());
    Ok(out.into_series())
}
