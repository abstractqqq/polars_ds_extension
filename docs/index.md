# Polars for Data Science (PDS)

A comprehensive [walkthrough](https://github.com/abstractqqq/polars_ds_extension/blob/knn_entropy/examples/basics.ipynb).

Read the [Docs](https://polars-ds-extension.readthedocs.io/en/latest/).

# Introduction

PDS is a modern data science package that

1. is fast and furious
2. is small and lean, with minimal dependencies
3. has an intuitive and concise API (if you know Polars already)
4. has dataframe friendly design
5. and covers a wide variety of data science topics, such as simple statistics, linear regression, string edit distances, tabular data transforms, feature extraction, traditional modelling pipelines, model evaluation metrics, etc., etc..

It stands on the shoulders of the great **Polars** dataframe. You can see [examples](./examples/basics.ipynb). Here are some highlights!

```python
import polars as pl
import polars_ds as pds
# Parallel evaluation of multiple ML metrics on different segments of data
df.lazy().group_by("segments").agg( 
    pds.query_roc_auc("actual", "predicted").alias("roc_auc"),
    pds.query_log_loss("actual", "predicted").alias("log_loss"),
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

Tabular Machine Learning Data Transformation Pipeline

```Python
import polars as pl
import polars.selectors as cs
from polars_ds.pipeline import Pipeline, Blueprint

bp = (
    # If we specify a target, then target will be excluded from any transformations.
    Blueprint(df, name = "example", target = "approved") 
    .lowercase() # lowercase all columns
    .select(cs.numeric() | cs.by_name(["gender", "employer_category1", "city_category"]))
    .linear_impute(features = ["var1", "existing_emi"], target = "loan_period") 
    .impute(["existing_emi"], method = "median")
    .append_expr( # generate some features
        pl.col("existing_emi").log1p().alias("existing_emi_log1p"),
        pl.col("loan_amount").log1p().alias("loan_amount_log1p"),
        pl.col("loan_amount").sqrt().alias("loan_amount_sqrt"),
        pl.col("loan_amount").shift(-1).alias("loan_amount_lag_1") # any kind of lag transform
    )
    .scale( 
        cs.numeric().exclude(["var1", "existing_emi_log1p"]), method = "standard"
    ) # Scale the columns up to this point. The columns below won't be scaled
    .append_expr( # Add missing flags
        pl.col("employer_category1").is_null().cast(pl.UInt8).alias("employer_category1_is_missing")
    )
    .one_hot_encode("gender", drop_first=True)
    .woe_encode("city_category")
    .target_encode("employer_category1", min_samples_leaf = 20, smoothing = 10.0) # same as above
)

pipe:Pipeline = bp.materialize()
# Check out the result in our example notebooks! (examples/pipeline.ipynb)
df_transformed = pipe.transform(df)
df_transformed.head()
```

Get all neighbors within radius r, call them best friends, and count the number

```python
df.select(
    pl.col("id"),
    pds.query_radius_ptwise(
        pl.col("var1"), pl.col("var2"), pl.col("var3"), # Columns used as the coordinates in 3d space
        index = pl.col("id"),
        r = 0.1, 
        dist = "sql2", # squared l2
        parallel = True
    ).alias("best friends"),
).with_columns( # -1 to remove the point itself
    (pl.col("best friends").list.len() - 1).alias("best friends count")
).head()

shape: (5, 3)
┌─────┬───────────────────┬────────────────────┐
│ id  ┆ best friends      ┆ best friends count │
│ --- ┆ ---               ┆ ---                │
│ u32 ┆ list[u32]         ┆ u32                │
╞═════╪═══════════════════╪════════════════════╡
│ 0   ┆ [0, 811, … 1435]  ┆ 152                │
│ 1   ┆ [1, 953, … 1723]  ┆ 159                │
│ 2   ┆ [2, 355, … 835]   ┆ 243                │
│ 3   ┆ [3, 102, … 1129]  ┆ 110                │
│ 4   ┆ [4, 1280, … 1543] ┆ 226                │
└─────┴───────────────────┴────────────────────┘
```

Run a linear regression on each category:

```Python

df = pds.random_data(size=5_000, n_cols=0).select(
    pds.random(0.0, 1.0).alias("x1"),
    pds.random(0.0, 1.0).alias("x2"),
    pds.random(0.0, 1.0).alias("x3"),
    pds.random_int(0, 3).alias("categories")
).with_columns(
    y = pl.col("x1") * 0.5 + pl.col("x2") * 0.25 - pl.col("x3") * 0.15 + pds.random() * 0.0001
)

df.group_by("categories").agg(
    pds.query_lstsq(
        "x1", "x2", "x3", 
        target = "y",
        method = "l2",
        l2_reg = 0.05,
        add_bias = False
    ).alias("coeffs")
) 

shape: (3, 2)
┌────────────┬─────────────────────────────────┐
│ categories ┆ coeffs                          │
│ ---        ┆ ---                             │
│ i32        ┆ list[f64]                       │
╞════════════╪═════════════════════════════════╡
│ 0          ┆ [0.499912, 0.250005, -0.149846… │
│ 1          ┆ [0.499922, 0.250004, -0.149856… │
│ 2          ┆ [0.499923, 0.250004, -0.149855… │
└────────────┴─────────────────────────────────┘
```

Various String Edit distances

```Python
df.select( # Column "word", compared to string in pl.lit(). It also supports column vs column comparison
    pds.str_leven("word", pl.lit("asasasa"), return_sim=True).alias("Levenshtein"),
    pds.str_osa("word", pl.lit("apples"), return_sim=True).alias("Optimal String Alignment"),
    pds.str_jw("word", pl.lit("apples")).alias("Jaro-Winkler"),
)
```

In-dataframe statistical tests

```Python
df.group_by("market_id").agg(
    pds.ttest_ind("var1", "var2", equal_var=False).alias("t-test"),
    pds.chi2("category_1", "category_2").alias("chi2-test"),
    pds.f_test("var1", group = "category_1").alias("f-test")
)

shape: (3, 4)
┌───────────┬──────────────────────┬──────────────────────┬─────────────────────┐
│ market_id ┆ t-test               ┆ chi2-test            ┆ f-test              │
│ ---       ┆ ---                  ┆ ---                  ┆ ---                 │
│ i64       ┆ struct[2]            ┆ struct[2]            ┆ struct[2]           │
╞═══════════╪══════════════════════╪══════════════════════╪═════════════════════╡
│ 0         ┆ {2.072749,0.038272}  ┆ {33.487634,0.588673} ┆ {0.312367,0.869842} │
│ 1         ┆ {0.469946,0.638424}  ┆ {42.672477,0.206119} ┆ {2.148937,0.072536} │
│ 2         ┆ {-1.175325,0.239949} ┆ {28.55723,0.806758}  ┆ {0.506678,0.730849} │
└───────────┴──────────────────────┴──────────────────────┴─────────────────────┘
```

Multiple Convolutions at once!

```Python
# Multiple Convolutions at once
# Modes: `same`, `left` (left-aligned same), `right` (right-aligned same), `valid` or `full`
# Method: `fft`, `direct`
# Currently slower than SciPy but provides parallelism because of Polars
df.select(
    pds.convolve("f", [-1, 0, 0, 0, 1], mode = "full", method = "fft"), # column f with the kernel given here
    pds.convolve("a", [-1, 0, 0, 0, 1], mode = "full", method = "direct"),
    pds.convolve("b", [-1, 0, 0, 0, 1], mode = "full", method = "direct"),
).head()
```

And more!

## Getting Started

```python
import polars_ds as pds
```

To make full use of the Diagnosis module, do

```python
pip install "polars_ds[plot]"
```

## How Fast is it?

Feel free to take a look at our [benchmark notebook](./benchmarks/benchmarks.ipynb)!

Generally speaking, the more expressions you want to evaluate simultaneously, the faster Polars + PDS will be than Pandas + (SciPy / Sklearn / NumPy). The more CPU cores you have on your machine, the bigger the time difference will be in favor of Polars + PDS. 

Why does speed matter? 

If your code already executes under 1s, then maybe it doesn't. But as your data grow, having a 5s run vs. a 1s run will make a lot of difference in your iterations for your project. Speed of execution becomes a bigger issues if you are building reports on demand, or if you need to pay extra for additional compute.  

## HELP WANTED!

1. Documentation writing, Doc Review, and Benchmark preparation

## Road Map

1. K-means, K-medoids clustering as expressions and also standalone modules.
2. Other improvement items. See issues.

# Disclaimer

**Currently in Beta. Feel free to submit feature requests in the issues section of the repo. This library will only depend on python Polars (for most of its core) and will try to be as stable as possible for polars>=1 (It currently supports polars>=0.20.16 but that will be dropped soon). Exceptions will be made when Polars's update forces changes in the plugins.**

This package is not tested with Polars streaming mode and is not designed to work with data so big that has to be streamed.

# Credits

1. Rust Snowball Stemmer is taken from Tsoding's Seroost project (MIT). See [here](https://github.com/tsoding/seroost)
2. Some statistics functions are taken from Statrs (MIT) and internalized. See [here](https://github.com/statrs-dev/statrs/tree/master)
3. Linear algebra routines are powered partly by [faer](https://crates.io/crates/faer)
4. String similarity metrics are soooo fast because of [RapidFuzz](https://github.com/maxbachmann/rapidfuzz-rs)

# Other related Projects

1. Take a look at our friendly neighbor [functime](https://github.com/TracecatHQ/functime)