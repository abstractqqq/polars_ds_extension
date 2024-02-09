/// All things true positive, false positive related.
/// ROC AUC, Average Precision, precision, recall, etc. m
use ndarray::ArrayView1;
use polars::{lazy::dsl::len, prelude::*};
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

fn tp_fp_frame(predicted: &Series, actual: &Series, as_ratio: bool) -> PolarsResult<LazyFrame> {
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

    let positive_counts = actual.sum::<u32>().unwrap_or(0);
    if positive_counts == 0 {
        return Err(PolarsError::ComputeError(
            "No positives in actual, or actual cannot be turned into integers.".into(),
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
        .sort("threshold", Default::default())
        .with_columns([
            (lit(n) - col("cnt").cum_sum(false) + col("cnt")).alias("predicted_positive"),
            (lit(positive_counts) - col("pos_cnt_at_threshold").cum_sum(false))
                .shift_and_fill(1, positive_counts)
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

    let mut frame = tp_fp_frame(predicted, actual, true)?
        .collect()?
        .agg_chunks();

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

    let y: ArrayView1<f64> = y.to_ndarray()?; // Zero copy
    let x: ArrayView1<f64> = x.to_ndarray()?; // Zero copy

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
    let f: f64 = (precision * recall) / (precision + recall);
    let recall: Series = Series::from_vec("recall", vec![recall]);
    let precision: Series = Series::from_vec("precision", vec![precision]);
    let f: Series = Series::from_vec("f", vec![f]);

    let out = StructChunked::new("metrics", &[precision, recall, f, ap, roc_auc])?;
    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn pl_roc_auc(inputs: &[Series]) -> PolarsResult<Series> {
    // actual, when passed in, is always u32 (done in Python extension side)
    let actual = &inputs[0];
    let predicted = &inputs[1];

    let mut frame = tp_fp_frame(predicted, actual, true)?
        .select([col("tpr"), col("fpr")])
        .collect()?
        .agg_chunks();

    let tpr = frame.drop_in_place("tpr").unwrap();
    let fpr = frame.drop_in_place("fpr").unwrap();

    // Should be contiguous. No need to rechunk
    let y = tpr.f64().unwrap();
    let x = fpr.f64().unwrap();

    let y: ArrayView1<f64> = y.to_ndarray()?;
    let x: ArrayView1<f64> = x.to_ndarray()?;

    let out: f64 = -super::trapz::trapz(y, x);

    Ok(Series::from_iter([out]))
}
