# Polars Extension for General Data Science Use

A comprehensive [walkthrough](https://github.com/abstractqqq/polars_ds_extension/blob/knn_entropy/examples/basics.ipynb).

Read the [Docs](https://polars-ds-extension.readthedocs.io/en/latest/).

# The Project

PDS is a modern take on data science and traditional tabular machine learning. It is dataframe-centric in design, and provides parallelism for free via **Polars**. It offers Polars syntax that works both in normal and aggregation contexts, and provides these conveniences to the end user without any additional dependency. It includes the most common functions from NumPy, SciPy, edit distances, KNN-related queries, EDA tools, feature engineering queries, etc. Yes, it only depends on Polars (unless you want to use the plotting functionalities and want to interop with NumPy). Most of the code is rewritten in **Rust** and is on par or even faster than existing functions in SciPy and Scikit-learn. The following are some examples:


Parallel evaluations of classification metrics on segments

```python
import polars as pl
import polars_ds as pds

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

Ridge Regression on Categories

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
    pds.query_ttest_ind("var1", "var2", equal_var=False).alias("t-test"),
    pds.query_chi2("category_1", "category_2").alias("chi2-test"),
    pds.query_f_test("var1", group = "category_1").alias("f-test")
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
    # Impute loan_period by running a simple linear regression. 
    # Explicitly put target, since this is not the target for prediction. 
    .linear_impute(features = ["var1", "existing_emi"], target = "loan_period") 
    .impute(["existing_emi"], method = "median")
    .append_expr( # generate some features
        pl.col("existing_emi").log1p().alias("existing_emi_log1p"),
        pl.col("loan_amount").log1p().alias("loan_amount_log1p"),
        pl.col("loan_amount").sqrt().alias("loan_amount_sqrt"),
        pl.col("loan_amount").shift(-1).alias("loan_amount_lag_1") # any kind of lag transform
    )
    .scale( # target is numerical, but will be excluded automatically because bp is initialzied with a target
        cs.numeric().exclude(["var1", "existing_emi_log1p"]), method = "standard"
    ) # Scale the columns up to this point. The columns below won't be scaled
    .append_expr(
        # Add missing flags
        pl.col("employer_category1").is_null().cast(pl.UInt8).alias("employer_category1_is_missing")
    )
    .one_hot_encode("gender", drop_first=True)
    .woe_encode("city_category") # No need to specify target because we initialized bp with a target
    .target_encode("employer_category1", min_samples_leaf = 20, smoothing = 10.0) # same as above
)

pipe:Pipeline = bp.materialize()
# Check out the result in our example notebooks!
df_transformed = pipe.transform(df)
df_transformed.head()
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

## More Examples

See this for Polars Extensions: [notebook](./examples/basics.ipynb)

See this for Native Polars DataFrame Explorative tools: [notebook](./examples/diagnosis.ipynb)

# Disclaimer

**Currently in Beta. Feel free to submit feature requests in the issues section of the repo. This library will only depend on python Polars and will try to be as stable as possible for polars>=0.20.6. Exceptions will be made when Polars's update forces changes in the plugins.**

This package is not tested with Polars streaming mode and is not designed to work with data so big that has to be streamed.

# Credits

1. Rust Snowball Stemmer is taken from Tsoding's Seroost project (MIT). See [here](https://github.com/tsoding/seroost)
2. Some statistics functions are taken from Statrs (MIT) and internalized. See [here](https://github.com/statrs-dev/statrs/tree/master)
3. Graph functionalities are powered by the petgragh crate. See [here](https://crates.io/crates/petgraph)
4. Linear algebra routines are powered partly by [faer](https://crates.io/crates/faer)
