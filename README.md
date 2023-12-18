# Polars Extension for General Data Science Use

A Polars Plugin aiming to simplify common numerical/string data analysis procedures. This means that the most basic data science, stats, NLP related tasks can be done natively inside a dataframe. 

Its goal is not to replace SciPy, or NumPy, but rather it tries reduce dependency for common workflows and simple analysis, and tries to reduce Python side code and UDFs.

See examples [here](./examples/basics.ipynb).

**Currently in Alpha. Feel free to submit feature requests in the issues section of the repo.**

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
    pl.col("a").stats_ext.sample_normal(mean = 0.5, std = 1.).alias("test1")
    , pl.col("a").stats_ext.sample_normal(mean = 0.5, std = 2.).alias("test2")
).select(
    pl.col("test1").stats_ext.ttest_ind(pl.col("test2"), equal_var = False).alias("t-test")
    , pl.col("test1").stats_ext.normal_test().alias("normality_test")
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
    pl.col("word").str_ext.levenshtein("world", return_sim = True)
).head()
```

And a lot more!

# Credits

1. Rust Snowball Stemmer is taken from Tsoding's Seroost project (MIT). See [here](https://github.com/tsoding/seroost)
2. Some statistics functions are taken from Statrs (MIT). See [here](https://github.com/statrs-dev/statrs/tree/master)

# Other related Projects

1. Take a look at our friendly neighbor [functime](https://github.com/TracecatHQ/functime)
2. My other project [dsds](https://github.com/abstractqqq/dsds). This is currently paused because I am developing polars-ds, but some modules in DSDS, such as the diagonsis one, is quite stable.
3. String similarity metrics is soooo fast and easy to use because of [RapidFuzz](https://github.com/maxbachmann/rapidfuzz-rs)