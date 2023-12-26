# Polars Extension for General Data Science Use

A Polars Plugin aiming to simplify common numerical/string data analysis procedures. This means that the most basic data science, stats, NLP related tasks can be done natively inside a dataframe, thus minimizing the number of dependencies.

Its goal is not to replace SciPy, or NumPy, but rather it tries to improve runtime for common tasks, reduce Python code and UDFs.

See examples [here](./examples/basics.ipynb).

Read the docs [here](https://polars-ds-extension.readthedocs.io/en/latest/).

**Currently in Beta. Feel free to submit feature requests in the issues section of the repo.**

## Getting Started
```bash
pip install polars_ds
```

and 

```python
import polars_ds
```
when you want to use the namespaces provided by the package.

## Examples

Generating random numbers, and running t-test, normality test inside a dataframe
```python
df.with_columns(
    pl.col("a").stats.sample_normal(mean = 0.5, std = 1.).alias("test1")
    , pl.col("a").stats.sample_normal(mean = 0.5, std = 2.).alias("test2")
).select(
    pl.col("test1").stats.ttest_ind(pl.col("test2"), equal_var = False).alias("t-test")
    , pl.col("test1").stats.normal_test().alias("normality_test")
).select(
    pl.col("t-test").struct.field("statistic").alias("t-tests: statistics")
    , pl.col("t-test").struct.field("pvalue").alias("t-tests: pvalue")
    , pl.col("normality_test").struct.field("statistic").alias("normality_test: statistics")
    , pl.col("normality_test").struct.field("pvalue").alias("normality_test: pvalue")
)
```

Blazingly fast string similarity comparisons. (Thanks to [RapidFuzz](https://docs.rs/rapidfuzz/latest/rapidfuzz/))
```python
df2.select(
    pl.col("word").str2.levenshtein("world", return_sim = True)
).head()
```

Even in-dataframe nearest neighbors queries! 😲
```python
df.with_columns(
    pl.col("id").num.knn_ptwise(
        pl.col("val1"), pl.col("val2"), 
        k = 3, dist = "haversine", parallel = True
    ).alias("nearest neighbor ids")
)

shape: (5, 6)
┌─────┬──────────┬──────────┬──────────┬──────────┬──────────────────────┐
│ id  ┆ val1     ┆ val2     ┆ val3     ┆ val4     ┆ nearest neighbor ids │
│ --- ┆ ---      ┆ ---      ┆ ---      ┆ ---      ┆ ---                  │
│ i64 ┆ f64      ┆ f64      ┆ f64      ┆ f64      ┆ list[u64]            │
╞═════╪══════════╪══════════╪══════════╪══════════╪══════════════════════╡
│ 0   ┆ 0.804226 ┆ 0.937055 ┆ 0.401005 ┆ 0.119566 ┆ [0, 3, … 0]          │
│ 1   ┆ 0.526691 ┆ 0.562369 ┆ 0.061444 ┆ 0.520291 ┆ [1, 4, … 4]          │
│ 2   ┆ 0.225055 ┆ 0.080344 ┆ 0.425962 ┆ 0.924262 ┆ [2, 1, … 1]          │
│ 3   ┆ 0.697264 ┆ 0.112253 ┆ 0.666238 ┆ 0.45823  ┆ [3, 1, … 0]          │
│ 4   ┆ 0.227807 ┆ 0.734995 ┆ 0.225657 ┆ 0.668077 ┆ [4, 4, … 0]          │
└─────┴──────────┴──────────┴──────────┴──────────┴──────────────────────┘
```

And a lot more!

# Credits

1. Rust Snowball Stemmer is taken from Tsoding's Seroost project (MIT). See [here](https://github.com/tsoding/seroost)
2. Some statistics functions are taken from Statrs (MIT). See [here](https://github.com/statrs-dev/statrs/tree/master)

# Other related Projects

1. Take a look at our friendly neighbor [functime](https://github.com/TracecatHQ/functime)
2. My other project [dsds](https://github.com/abstractqqq/dsds). This is currently paused because I am developing polars-ds, but some modules in DSDS, such as the diagonsis one, is quite stable.
3. String similarity metrics is soooo fast and easy to use because of [RapidFuzz](https://github.com/maxbachmann/rapidfuzz-rs)