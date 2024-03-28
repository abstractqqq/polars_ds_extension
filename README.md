<h1 align="center">
  <b>Polars for Data Science</b>
  <br>
</h1>

<p align="center">
  <a href="https://polars-ds-extension.readthedocs.io/en/latest/">Documentation</a>
  |
  <a href="https://github.com/abstractqqq/polars_ds_extension/blob/main/examples/basics.ipynb">User Guide</a>
  |
  <a href="https://github.com/abstractqqq/polars_ds_extension/blob/main/CONTRIBUTING.md">Want to Contribute?</a>
<br>
<b>pip install polars-ds</b>
</p>

# The Project

The goal of the project is to **reduce dependencies**, **improve code organization**, **simplify data pipelines** and overall **faciliate analysis of various kinds of tabular data** that a data scientist may encounter. It is a package built around your favorite **Polars dataframe**. Here are the current namespaces (Polars Extensions) provided by the package:

1. A numerical extension (num), which focuses on numerical quantities common in many fields of data analysis (credit modelling, time series, other well-known quantities, etc.), such as rfft, entropies, k-nearest-neighbors queries, Population Stability Index, Information Value, etc.

2. A metrics extension (metric), which contains a lot of common error/loss functions, model evaluation metrics. This module is mostly designed to generate model performance monitoring data.

3. A str extension (str2), which focuses on str distances/similarities, and other commonly used string manipulation procedures.

4. A stats extension (stats), which has common statistical tests such as t-test, chi2, and f-test, etc., and random sampling from a distribution, etc.

5. A complex extension (c), which treats complex numbers as a column of array of size 2. Sometimes complex numbers are needed for processing FFT outputs.

6. A graph extension (graph) for very simple graph queries, such as shortest path queries, eigenvector centrality computations. More will be added. (Usable but limited. Will to be refactored/redesigned.)

# But why? Why not use Sklearn? SciPy? NumPy?

The goal of the package is to **facilitate** data processes and analysis that go beyond standard SQL queries, and to **reduce** the number of dependencies in your project. It incorproates parts of SciPy, NumPy, Scikit-learn, and NLP (NLTK), etc., and treats them as Polars queries so that they can be run in parallel, in group_by contexts, all for almost no extra engineering effort. 

Let's see an example. Say we want to generate a model performance report. In our data, we have segments. We are not only interested in the ROC AUC of our model on the entire dataset, but we are also interested in the model's performance on different segments.

```python
import polars as pl
import polars_ds as pds

size = 100_000
df = pl.DataFrame({
    "a": np.random.random(size = size)
    , "b": np.random.random(size = size)
    , "x1" : range(size)
    , "x2" : range(size, size + size)
    , "y": range(-size, 0)
    , "actual": np.round(np.random.random(size=size)).astype(np.int32)
    , "predicted": np.random.random(size=size)
    , "segments":["a"] * (size//2 + 100) + ["b"] * (size//2 - 100) 
})
print(df.head())

shape: (5, 8)
┌──────────┬──────────┬─────┬────────┬─────────┬────────┬───────────┬──────────┐
│ a        ┆ b        ┆ x1  ┆ x2     ┆ y       ┆ actual ┆ predicted ┆ segments │
│ ---      ┆ ---      ┆ --- ┆ ---    ┆ ---     ┆ ---    ┆ ---       ┆ ---      │
│ f64      ┆ f64      ┆ i64 ┆ i64    ┆ i64     ┆ i32    ┆ f64       ┆ str      │
╞══════════╪══════════╪═════╪════════╪═════════╪════════╪═══════════╪══════════╡
│ 0.19483  ┆ 0.457516 ┆ 0   ┆ 100000 ┆ -100000 ┆ 0      ┆ 0.929007  ┆ a        │
│ 0.396265 ┆ 0.833535 ┆ 1   ┆ 100001 ┆ -99999  ┆ 1      ┆ 0.103915  ┆ a        │
│ 0.800558 ┆ 0.030437 ┆ 2   ┆ 100002 ┆ -99998  ┆ 1      ┆ 0.558918  ┆ a        │
│ 0.608023 ┆ 0.411389 ┆ 3   ┆ 100003 ┆ -99997  ┆ 1      ┆ 0.883684  ┆ a        │
│ 0.847527 ┆ 0.506504 ┆ 4   ┆ 100004 ┆ -99996  ┆ 1      ┆ 0.070269  ┆ a        │
└──────────┴──────────┴─────┴────────┴─────────┴────────┴───────────┴──────────┘
```

Traditionally, using the Pandas + Sklearn stack, we would do:

```
import pandas as pd
from sklearn.metrics import roc_auc_score

df_pd = df.to_pandas()

segments = []
rocaucs = []

for (segment, subdf) in df_pd.groupby("segments"):
    segments.append(segment)
    rocaucs.append(
        roc_auc_score(subdf["actual"], subdf["predicted"])
    )

report = pd.DataFrame({
    "segments": segments,
    "roc_auc": rocaucs
})
print(report)

  segments   roc_auc
0        a  0.497745
1        b  0.498801
```

This is ok, but not great, because (1) we are running for loops in Python, which tends to be slow. (2) We are writing more Python code, which leaves more room for errors in bigger projects. (3) The code is not very intuitive for beginners. Using Polars + Polars ds, one can do the following:

```
df.lazy().group_by("segments").agg(
    pl.col("actual").metric.roc_auc(pl.col("predicted")).alias("roc_auc"),
    pl.col("actual").metric.log_loss(pl.col("predicted")).alias("log_loss"),
).collect()

shape: (2, 3)
┌──────────┬──────────┬──────────┐
│ segments ┆ roc_auc  ┆ log_loss │
│ ---      ┆ ---      ┆ ---      │
│ str      ┆ f64      ┆ f64      │
╞══════════╪══════════╪══════════╡
│ a        ┆ 0.497745 ┆ 1.006438 │
│ b        ┆ 0.498801 ┆ 0.997226 │
└──────────┴──────────┴──────────┘
```

Notice a few things: (1) Computing ROC AUC on different segments is equivalent to an aggregation on segments! It is a concept everyone who knows SQL (aka everybody who works with data) will be familiar with! (2) There is no Python code. The extension is written in pure Rust and all complexities are hidden away from the end user. (3) Because Polars provides parallel execution for free, we can compute ROC AUC and log loss simultaneously on each segment! (In Pandas, one can do something like this in aggregations but is soooo much harder to write and way more confusing to reason about.)

The end result is simpler, more intuitive code that is also easier to reason about, and faster execution time. Because of Polars's extension (plugin) system, we are now blessed with both:

**Performance and elegance - something that is quite rare in the Python world.**

## Getting Started

```python
import polars_ds as pds
```
when you want to access the namespaces provided by the package.

```python
pip install "polars_ds[plot]"
```
for dataframe diagnosis related features.

## Examples

See this for Polars Extensions: [notebook](./examples/basics.ipynb)

See this for Native Polars DataFrame Explorative tools: [notebook](./examples/diagnosis.ipynb)

# Disclaimer

**Currently in Beta. Feel free to submit feature requests in the issues section of the repo. This library will only depend on python Polars and will try to be as stable as possible for polars>=0.20.6. Exceptions will be made when Polars's update forces changes in the plugins.**

This package is not tested with Polars streaming mode and is not designed to work with data so big that has to be streamed. 

The recommended usage will be for datasets of size 1k to 2-3mm rows, but actual performance will vary depending on dataset and hardware. Performance will only be a priority for datasets that fit in memory. It is a known fact that knn performance suffers greatly with a large k. Str-knn and Graph queries are only suitable for smaller data, of size ~1-5k for common computers.

# Credits

1. Rust Snowball Stemmer is taken from Tsoding's Seroost project (MIT). See [here](https://github.com/tsoding/seroost)
2. Some statistics functions are taken from Statrs (MIT) and internalized. See [here](https://github.com/statrs-dev/statrs/tree/master)
3. Graph functionalities are powered by the petgragh crate. See [here](https://crates.io/crates/petgraph)
4. Linear algebra routines are powered partly by [faer](https://crates.io/crates/faer)

# Other related Projects

1. Take a look at our friendly neighbor [functime](https://github.com/TracecatHQ/functime)
2. String similarity metrics is soooo fast and easy to use because of [RapidFuzz](https://github.com/maxbachmann/rapidfuzz-rs)