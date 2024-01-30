# Polars Extension for General Data Science Use

A Polars Plugin aiming to simplify common numerical/string data analysis procedures.

A comprehensive [walkthrough](./examples/basics.ipynb).

Read the [Docs](https://polars-ds-extension.readthedocs.io/en/latest/).

# The Project

Here are the current namespaces (Polars Extensions) provided by the package:

1. A numerical extension (num), which focuses on numerical quantities common in many fields of data analysis (credit modelling, time series, other well-known quantities, etc.), such as rfft, entropies, k-nearest-neighbors queries, Population Stability Index, Information Value, etc.

2. A metrics extension (metric), which contains a lot of common error/loss functions, model evaluation metrics. This module is mostly designed to generate model performance monitoring

3. A str extension (str2), which focuses on str distances/similarities, and other commonly used string manipulation procedures.

4. A stats extension (stats), which has common statistical tests such as t-test, chi2, and f-test, etc., and random sampling from a distribution without leaving dataframes.

5. A complex extension (c), which treats complex numbers as a column of array of size 2. Sometimes complex numbers are needed for processing FFT outputs.

6. A graph extension (graph) for very simple graph queries, such as shortest path queries, eigenvector centrality computations. More will be added.

# But why? Why not use Sklearn? SciPy? NumPy?

The goal of the package is to **facilitate** data processes and analysis that go beyond standard SQL queries. It incorproates parts of SciPy, NumPy, Scikit-learn, and NLP (NLTK), etc., and treats them as Polars queries so that they can be run in parallel, in group_by contexts, even in LazyFrames. 

Let's see an example. Say we want to generate a model performance report. In our data, we have segments. We are not only interested in the ROC AUC of our model on the entire dataset, but we are also interested in the model's performance on different segments.

```python
import polars as pl
import polars_ds as pld

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
```bash
pip install polars_ds
```

and 

```python
import polars_ds as pld
```
when you want to use the namespaces provided by the package.

## Examples

In-dataframe statistical testing
```python
df.select(
    pl.col("group1").stats.ttest_ind(pl.col("group2"), equal_var = True).alias("t-test"),
    pl.col("category_1").stats.chi2(pl.col("category_2")).alias("chi2-test"),
    pl.col("category_1").stats.f_test(pl.col("group1")).alias("f-test")
)

shape: (1, 3)
┌───────────────────┬──────────────────────┬────────────────────┐
│ t-test            ┆ chi2-test            ┆ f-test             │
│ ---               ┆ ---                  ┆ ---                │
│ struct[2]         ┆ struct[2]            ┆ struct[2]          │
╞═══════════════════╪══════════════════════╪════════════════════╡
│ {-0.004,0.996809} ┆ {37.823816,0.386001} ┆ {1.354524,0.24719} │
└───────────────────┴──────────────────────┴────────────────────┘
```

Generating random numbers according to reference column
```python
df.with_columns(
    # Sample from normal distribution, using reference column "a" 's mean and std
    pl.col("a").stats.sample_normal().alias("test1") 
    # Sample from uniform distribution, with low = 0 and high = "a"'s max, and respect the nulls in "a"
    , pl.col("a").stats.sample_uniform(low = 0., high = None, respect_null=True).alias("test2")
).head()

shape: (5, 3)
┌───────────┬───────────┬──────────┐
│ a         ┆ test1     ┆ test2    │
│ ---       ┆ ---       ┆ ---      │
│ f64       ┆ f64       ┆ f64      │
╞═══════════╪═══════════╪══════════╡
│ null      ┆ 0.459357  ┆ null     │
│ null      ┆ 0.038007  ┆ null     │
│ -0.826518 ┆ 0.241963  ┆ 0.968385 │
│ 0.737955  ┆ -0.819475 ┆ 2.429615 │
│ 1.10397   ┆ -0.684289 ┆ 2.483368 │
└───────────┴───────────┴──────────┘
```

Blazingly fast string similarity comparisons. (Thanks to [RapidFuzz](https://docs.rs/rapidfuzz/latest/rapidfuzz/))
```python
df.select(
    pl.col("word").str2.levenshtein("asasasa", return_sim=True).alias("asasasa"),
    pl.col("word").str2.levenshtein("sasaaasss", return_sim=True).alias("sasaaasss"),
    pl.col("word").str2.levenshtein("asdasadadfa", return_sim=True).alias("asdasadadfa"),
    pl.col("word").str2.fuzz("apples").alias("LCS based Fuzz match - apples"),
    pl.col("word").str2.osa("apples", return_sim = True).alias("Optimal String Alignment - apples"),
    pl.col("word").str2.jw("apples").alias("Jaro-Winkler - apples"),
)
shape: (5, 6)
┌──────────┬───────────┬─────────────┬────────────────┬───────────────────────────┬────────────────┐
│ asasasa  ┆ sasaaasss ┆ asdasadadfa ┆ LCS based Fuzz ┆ Optimal String Alignment  ┆ Jaro-Winkler - │
│ ---      ┆ ---       ┆ ---         ┆ match - apples ┆ - apple…                  ┆ apples         │
│ f64      ┆ f64       ┆ f64         ┆ ---            ┆ ---                       ┆ ---            │
│          ┆           ┆             ┆ f64            ┆ f64                       ┆ f64            │
╞══════════╪═══════════╪═════════════╪════════════════╪═══════════════════════════╪════════════════╡
│ 0.142857 ┆ 0.111111  ┆ 0.090909    ┆ 0.833333       ┆ 0.833333                  ┆ 0.966667       │
│ 0.428571 ┆ 0.333333  ┆ 0.272727    ┆ 0.166667       ┆ 0.0                       ┆ 0.444444       │
│ 0.111111 ┆ 0.111111  ┆ 0.090909    ┆ 0.555556       ┆ 0.444444                  ┆ 0.5            │
│ 0.875    ┆ 0.666667  ┆ 0.545455    ┆ 0.25           ┆ 0.25                      ┆ 0.527778       │
│ 0.75     ┆ 0.777778  ┆ 0.454545    ┆ 0.25           ┆ 0.25                      ┆ 0.527778       │
└──────────┴───────────┴─────────────┴────────────────┴───────────────────────────┴────────────────┘
```

Even in-dataframe nearest neighbors queries! 😲
```python
df.select(
    pl.col("id"),
    pl.col("id").num.query_radius_ptwise(
        pl.col("val1"), pl.col("val2"), pl.col("val3"), # Columns used as the coordinates in n-d space
        r = 0.1, 
        dist = "l2", # actually this is squared l2
        parallel = True
    ).alias("best friends"),
).with_columns( # -1 to remove the point itself
    (pl.col("best friends").list.len() - 1).alias("best friends count")
).head()

shape: (5, 3)
┌─────┬─────────────────┬────────────────────┐
│ id  ┆ best friends    ┆ best friends count │
│ --- ┆ ---             ┆ ---                │
│ u64 ┆ list[u64]       ┆ u32                │
╞═════╪═════════════════╪════════════════════╡
│ 0   ┆ [0, 681, … 90]  ┆ 84                 │
│ 1   ┆ [1, 232, … 20]  ┆ 144                │
│ 2   ┆ [2, 565, … 168] ┆ 137                │
│ 3   ┆ [3, 399, … 529] ┆ 58                 │
│ 4   ┆ [4, 389, … 898] ┆ 88                 │
└─────┴─────────────────┴────────────────────┘
```

# Disclaimers

**Currently in Beta. Feel free to submit feature requests in the issues section of the repo. This library will only depend on python Polars and will try to be as stable as possible for polars>=0.20.6. Exceptions will be made when Polars's update forces changes in the plugins.**

This package is not tested with Polars streaming mode and is not designed to work with data so big that has to be streamed. The recommended usage will be for datasets of size 1k to 2-3mm rows. Performance will only be a priority for datasets within this size.

# Credits

1. Rust Snowball Stemmer is taken from Tsoding's Seroost project (MIT). See [here](https://github.com/tsoding/seroost)
2. Some statistics functions are taken from Statrs (MIT). See [here](https://github.com/statrs-dev/statrs/tree/master)

# Other related Projects

1. Take a look at our friendly neighbor [functime](https://github.com/TracecatHQ/functime)
2. String similarity metrics is soooo fast and easy to use because of [RapidFuzz](https://github.com/maxbachmann/rapidfuzz-rs)