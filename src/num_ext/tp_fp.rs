/// All things true positive, false positive related.
/// ROC AUC, Average Precision, precision, recall, etc. m
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

fn combo_output(_: &[Field]) -> PolarsResult<Field> {
    let roc_auc = Field::new("roc_auc", DataType::Float64);
    let precision = Field::new("precision", DataType::Float64);
    let recall = Field::new("recall", DataType::Float64);
    let f = Field::new("f", DataType::Float64);
    let avg_precision = Field::new("avg_precision", DataType::Float64);
    let v: Vec<Field> = vec![precision, recall, f, avg_precision, roc_auc];
    Ok(Field::new("", DataType::Struct(v)))
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
        || !predicted.dtype().is_numeric()
        || ((actual.null_count() + predicted.null_count()) > 0)
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

    let roc_auc: f64 = -super::trapz::trapz(y, x);
    let roc_auc: Series = Series::from_vec("roc_auc", vec![roc_auc]);

    // Average Precision
    let ap: f64 = -1.0
        * y.iter()
            .zip(y.iter().skip(1))
            .zip(precision)
            .fold(0., |acc, ((y, y_next), p)| (y_next - y).mul_add(*p, acc));
    let ap: Series = Series::from_vec("average_precision", vec![ap]);

    // Precision & Recall & F
    let recall = y[index];
    let precision = precision[index];
    let f: f64 = 2.0 * (precision * recall) / (precision + recall);
    let recall: Series = Series::from_vec("recall", vec![recall]);
    let precision: Series = Series::from_vec("precision", vec![precision]);
    let f: Series = Series::from_vec("f", vec![f]);

    let out = StructChunked::new("metrics", &[precision, recall, f, ap, roc_auc])?;
    Ok(out.into_series())
}

fn binary_confusion_matrix(combined_series: &UInt32Chunked) -> [u32; 4] {
    // The combined_series is (lit(2) * col("y_true") + col("y_pred")).alias("y")
    // 0 is tn, 1 is fp, 2 is fn, 3 is tp
    // At Python side, actual and pred columns were turned into boolean first. So
    // no invalid indices will happen.
    let mut output = [0u32; 4];
    for i in combined_series.into_no_null_iter() {
        output[i as usize] += 1;
    }
    output
}

#[polars_expr(output_type=Float64)]
fn pl_roc_auc(inputs: &[Series]) -> PolarsResult<Series> {
    // actual, when passed in, is always u32 (done in Python extension side)
    let actual = &inputs[0];
    let predicted = &inputs[1];

    let positive_count = actual.sum::<u32>().unwrap_or(0);
    if positive_count == 0 {
        return Ok(Series::from_iter([f64::NAN]));
    }

    let mut binding = tp_fp_frame(predicted, actual, positive_count, true)?
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

    let out: f64 = -super::trapz::trapz(y, x);
    Ok(Series::from_iter([out]))
}

// bcm = binary confusion matrix
fn bcm_output(_: &[Field]) -> PolarsResult<Field> {
    let tp = Field::new("tp", DataType::UInt32);
    let fp = Field::new("fp", DataType::UInt32);
    let tn = Field::new("tn", DataType::UInt32);
    let fn_ = Field::new("fn", DataType::UInt32);

    let fpr = Field::new("fpr", DataType::Float64);
    let fnr = Field::new("fnr", DataType::Float64);
    let tnr = Field::new("tnr", DataType::Float64);
    let prevalence = Field::new("prevalence", DataType::Float64);
    let prevalence_threshold = Field::new("prevalence_threshold", DataType::Float64);
    let tpr = Field::new("tpr", DataType::Float64);
    let informedness = Field::new("informedness", DataType::Float64);
    let precision = Field::new("precision", DataType::Float64);
    let false_omission_rate = Field::new("false_omission_rate", DataType::Float64);
    let plr = Field::new("plr", DataType::Float64);
    let nlr = Field::new("nlr", DataType::Float64);
    let acc = Field::new("acc", DataType::Float64);
    let balanced_accuracy = Field::new("balanced_accuracy", DataType::Float64);
    let f1 = Field::new("f1", DataType::Float64);
    let folkes_mallows_index = Field::new("folkes_mallows_index", DataType::Float64);
    let mcc = Field::new("mcc", DataType::Float64);
    let threat_score = Field::new("threat_score", DataType::Float64);
    let markedness = Field::new("markedness", DataType::Float64);
    let fdr = Field::new("fdr", DataType::Float64);
    let npv = Field::new("npv", DataType::Float64);
    let dor = Field::new("dor", DataType::Float64);

    Ok(Field::new(
        "confusion_matrix",
        DataType::Struct(vec![
            tn,
            fp,
            fn_,
            tp,
            tpr,
            fpr,
            fnr,
            tnr,
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
    let tn = UInt32Chunked::from_vec("tn", vec![confusion[0]]);
    let tn = tn.into_series();

    let fp = UInt32Chunked::from_vec("fp", vec![confusion[1]]);
    let fp = fp.into_series();

    let fn_ = UInt32Chunked::from_vec("fn", vec![confusion[2]]);
    let fn_ = fn_.into_series();

    let tp = UInt32Chunked::from_vec("tp", vec![confusion[3]]);
    let tp = tp.into_series();
    // All series have length 1 and no duplicate names
    let df = unsafe { DataFrame::new_no_checks(vec![tn, fp, fn_, tp]) };

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
        .fill_null(f64::NAN) // In Rust dividing by 0 results in Null for some reason.
        .collect()?;

    let out = result.into_struct("confusion_matrix");
    Ok(out.into_series())
}
