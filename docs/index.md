# Polars-ds

A Polars Plugin aiming to simplify common numerical/string data analysis procedures. This means that the most basic data science, stats, NLP related tasks can be done natively inside a dataframe, without leaving dataframe world. This also means that for simple data pipelines, you do not need to install NumPy/Scipy/Scikit-learn, which saves a lot of space, which is great under constrained resources.

Its goal is NOT to replace SciPy, or NumPy, but rather it tries reduce dependency for simple analysis, and tries to reduce Python side code and UDFs, which are often performance bottlenecks.

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