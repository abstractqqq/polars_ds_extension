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
        pl.col("val1"), pl.col("val2"), pl.col("val2"),
        k = 2, dist = "l1", parallel = False
    ).alias("nearest neighbor ids")
)

shape: (5, 4)
┌─────┬──────────┬──────────┬───────────────────────┐
│ id  ┆ val1     ┆ val2     ┆ nearest neighbor ids  │
│ --- ┆ ---      ┆ ---      ┆ ---                   │
│ i64 ┆ f64      ┆ f64      ┆ list[u64]             │
╞═════╪══════════╪══════════╪═══════════════════════╡
│ 0   ┆ 0.031918 ┆ 0.966439 ┆ [49187, 54820, 14596] │
│ 1   ┆ 0.824656 ┆ 0.765373 ┆ [55462, 74455, 78960] │
│ 2   ┆ 0.739483 ┆ 0.91011  ┆ [5264, 53423, 40837]  │
│ 3   ┆ 0.041424 ┆ 0.241765 ┆ [90369, 32434, 14777] │
│ 4   ┆ 0.956705 ┆ 0.964461 ┆ [27196, 53936, 58850] │
└─────┴──────────┴──────────┴───────────────────────┘
```

And a lot more!

# Credits

1. Rust Snowball Stemmer is taken from Tsoding's Seroost project (MIT). See [here](https://github.com/tsoding/seroost)
2. Some statistics functions are taken from Statrs (MIT). See [here](https://github.com/statrs-dev/statrs/tree/master)

# Other related Projects

1. Take a look at our friendly neighbor [functime](https://github.com/TracecatHQ/functime)
2. My other project [dsds](https://github.com/abstractqqq/dsds). This is currently paused because I am developing polars-ds, but some modules in DSDS, such as the diagonsis one, is quite stable.
3. String similarity metrics is soooo fast and easy to use because of [RapidFuzz](https://github.com/maxbachmann/rapidfuzz-rs)