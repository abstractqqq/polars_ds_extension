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

### Parallel ML Metrics Calculation

```python
import polars as pl
import polars_ds as pds
# Parallel evaluation of multiple ML metrics on different segments of data
df.lazy().group_by("segments").agg( 
    # any other metrics you want in here
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

### In-dataframe linear regression + feature transformations

```python
import polars_ds as pds
from polars_ds.modeling.transforms import polynomial_features
# If you want the underlying computation to be done in f32, set pds.config.LIN_REG_EXPR_F64 = False
df.select(
    pds.lin_reg_report(
        *(
            ["x1", "x2", "x3"] +
            polynomial_features(["x1", "x2", "x3"], degree = 2, interaction_only=True)
        )
        , target = "target"
        , add_bias = False
    ).alias("result")
).unnest("result")

┌──────────┬───────────┬──────────┬───────────┬───────┬───────────┬──────────┬──────────┬──────────┐
│ features ┆ beta      ┆ std_err  ┆ t         ┆ p>|t| ┆ 0.025     ┆ 0.975    ┆ r2       ┆ adj_r2   │
│ ---      ┆ ---       ┆ ---      ┆ ---       ┆ ---   ┆ ---       ┆ ---      ┆ ---      ┆ ---      │
│ str      ┆ f64       ┆ f64      ┆ f64       ┆ f64   ┆ f64       ┆ f64      ┆ f64      ┆ f64      │
╞══════════╪═══════════╪══════════╪═══════════╪═══════╪═══════════╪══════════╪══════════╪══════════╡
│ x1       ┆ 0.26332   ┆ 0.000315 ┆ 835.68677 ┆ 0.0   ┆ 0.262703  ┆ 0.263938 ┆ 0.971087 ┆ 0.971085 │
│          ┆           ┆          ┆ 8         ┆       ┆           ┆          ┆          ┆          │
│ x2       ┆ 0.413824  ┆ 0.000311 ┆ 1331.9883 ┆ 0.0   ┆ 0.413216  ┆ 0.414433 ┆ 0.971087 ┆ 0.971085 │
│          ┆           ┆          ┆ 32        ┆       ┆           ┆          ┆          ┆          │
│ x3       ┆ 0.113688  ┆ 0.000315 ┆ 361.29924 ┆ 0.0   ┆ 0.113072  ┆ 0.114305 ┆ 0.971087 ┆ 0.971085 │
│ x1*x2    ┆ -0.097272 ┆ 0.000543 ┆ -179.0377 ┆ 0.0   ┆ -0.098337 ┆ -0.09620 ┆ 0.971087 ┆ 0.971085 │
│          ┆           ┆          ┆ 76        ┆       ┆           ┆ 7        ┆          ┆          │
│ x1*x3    ┆ -0.097266 ┆ 0.000542 ┆ -179.4486 ┆ 0.0   ┆ -0.098329 ┆ -0.09620 ┆ 0.971087 ┆ 0.971085 │
│          ┆           ┆          ┆ 32        ┆       ┆           ┆ 4        ┆          ┆          │
│ x2*x3    ┆ -0.097987 ┆ 0.000542 ┆ -180.7579 ┆ 0.0   ┆ -0.099049 ┆ -0.09692 ┆ 0.971087 ┆ 0.971085 │
│          ┆           ┆          ┆ 6         ┆       ┆           ┆ 4        ┆          ┆          │
└──────────┴───────────┴──────────┴───────────┴───────┴───────────┴──────────┴──────────┴──────────┘
```

- [x] Normal Linear Regression (pds.lin_reg)
- [x] Lasso, Ridge, Elastic Net (pds.lin_reg)
- [x] Rolling linear regression with skipping (pds.rolling_lin_reg)
- [x] Recursive linear regression (pds.recursive_lin_reg)
- [ ] Non-negative linear regression 
- [x] Statsmodel-like linear regression table (pds.lin_reg_report)
- [x] f32 support

### Tabular Machine Learning Data Transformation Pipeline

See [SKLEARN_COMPATIBILITY](SKLEARN_COMPATIBILITY.md) for more details.

```Python
import polars as pl
import polars.selectors as cs
from polars_ds.pipeline import Pipeline, Blueprint

bp = (
    Blueprint(df, name = "example", target = "approved", lowercase=True) # You can optionally 
    .filter( 
        "city_category is not null" # or equivalently, you can do: pl.col("city_category").is_not_null()
    )
    .linear_impute(features = ["var1", "existing_emi"], target = "loan_period") 
    .impute(["existing_emi"], method = "median")
    .append_expr( # generate some features
        pl.col("existing_emi").log1p().alias("existing_emi_log1p"),
        pl.col("loan_amount").log1p().alias("loan_amount_log1p"),
        pl.col("loan_amount").clip(lower_bound = 0, upper_bound = 1000).alias("loan_amount_log1p_clipped"),
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

print(bp)

pipe:Pipeline = bp.materialize()
# Check out the result in our example notebooks! (examples/pipeline.ipynb)
df_transformed = pipe.transform(df)
df_transformed.head()
```

### Nearest Neighbors Related Queries

Get all neighbors within radius r, call them best friends, and count the number. Due to limitations, 
this currently doesn't preserve the index, and is not fast when k or dimension of data is large.

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

### Various String Edit distances

```Python
df.select( # Column "word", compared to string in pl.lit(). It also supports column vs column comparison
    pds.str_leven("word", pl.lit("asasasa"), return_sim=True).alias("Levenshtein"),
    pds.str_osa("word", pl.lit("apples"), return_sim=True).alias("Optimal String Alignment"),
    pds.str_jw("word", pl.lit("apples")).alias("Jaro-Winkler"),
)
```

### In-dataframe statistical tests

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

### Multiple Convolutions at once!

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

Feel free to take a look at our [benchmark notebook](https://github.com/abstractqqq/polars_ds_extension/blob/main/benchmarks/benchmarks.ipynb)!

Generally speaking, the more expressions you want to evaluate simultaneously, the faster Polars + PDS will be than Pandas + (SciPy / Sklearn / NumPy). The more CPU cores you have on your machine, the bigger the time difference will be in favor of Polars + PDS. 

Why does speed matter? 

If your code already executes under 1s and you only use your code in non-production, ad-hoc environments, then maybe it doesn't. Even so, as your data grow, having a 5s run vs. a 1s run will make a lot of difference in your iterations for your project. Speed of execution becomes a bigger issues if you are building reports on demand, or if you need to pay extra for additional compute or when you have a production pipeline that has to deliver the data under a time constraint.  

## HELP WANTED!

1. Documentation writing, Doc Review, and Benchmark preparation

## Road Map

1. K-means, K-medoids clustering as expressions and also standalone modules.
2. Other improvement items. See issues.

# Disclaimer

**Currently in Beta. Feel free to submit feature requests in the issues section of the repo. This library will only depend on python Polars (for most of its core) and will try to be as stable as possible for polars>=1. Exceptions will be made when Polars's update forces changes in the plugins.**

This package is not tested with Polars streaming mode and is not designed to work with data so big that has to be streamed. This concerns the plugin expressions like `pds.lin_reg`, etc.. By the same token, Polars large index version is not intentionally supported at this point. However, non-plugin Polars utilities provided by the function should work with the streaming engine, as they are native Polars code.

## Polars LTS CPU Support / Build From Source

The guide here is not specific to LTS CPU, and can be used generally.

The best advice for LTS CPU is that you should compile the package yourself. First clone the repo and make sure Rust is installed on the system. Create a python virtual environment and install maturin in it. Next set the RUSTFLAG environment variable. The official polars-lts-cpu features are the following:
```
RUSTFLAGS=-C target-feature=+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt,+cmpxchg16b
```
If you simply want to compile from source, you may set target cpu to native, which autodetects CPU features.
```
RUSTFLAGS=-C target-cpu=native
```
If you are compiling for LTS CPU, then in pyproject.toml, update the polars dependency to polars-lts-cpu:
```
polars >= 1.4.0 # polars-lts-cpu >= 1.4.0
```
Lastly, run 
```
maturin develop --release
```
If you want to test the build locally, you may run 
```
# pip install -r requirements-test.txt
pytest tests/test_*
```
If you see this error in pytest, it means setuptools is not installed and you may ignore it. It is just a legacy python builtin package.
```
tests/test_many.py::test_xi_corr - ModuleNotFoundError: No module named 'pkg_resources'
```

You can then publish it to your private PYPI server, or just use it locally.

# Credits

1. Rust Snowball Stemmer is taken from Tsoding's Seroost project (MIT). See [here](https://github.com/tsoding/seroost)
2. Some statistics functions are taken from Statrs (MIT) and internalized. See [here](https://github.com/statrs-dev/statrs/tree/master)
3. Linear algebra routines are powered partly by [faer](https://crates.io/crates/faer)
4. String similarity metrics are soooo fast because of [RapidFuzz](https://github.com/maxbachmann/rapidfuzz-rs)

# Other related Projects

1. Take a look at our friendly neighbor [functime](https://github.com/TracecatHQ/functime)