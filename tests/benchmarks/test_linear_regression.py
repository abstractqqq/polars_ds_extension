import polars as pl
import polars_ds as pds
import pytest
from sklearn.linear_model import Lasso, LinearRegression, Ridge

SEED = 208

size = 100_000
DF = (
    pds.random_data(size=size, n_cols=0)
    .select(
        pds.random(0.0, 1.0).alias("x1"),
        pds.random(0.0, 1.0).alias("x2"),
        pds.random(0.0, 1.0).alias("x3"),
        pds.random(0.0, 1.0).alias("x4"),
        pds.random(0.0, 1.0).alias("x5"),
        pds.random_int(0, 4).alias("code"),
        pl.Series(name="id", values=range(size)),
    )
    .with_columns(
        y=pl.col("x1") * 0.5
        + pl.col("x2") * 0.25
        - pl.col("x3") * 0.15
        + pl.col("x4") * 0.2
        - pl.col("x5") * 0.13
        + pds.random() * 0.0001,
    )
)


# Prepare data for Scikit-learn. We assume the Scikit-learn + Pandas combination.
# One can simply replace to_pandas() by to_numpy() to test the Scikit-learn + NumPy combination
PD_DF = DF.to_pandas()
X_VARS = ["x1", "x2", "x3", "x4", "x5"]
Y = ["y"]

SIZES = [1_000, 10_000, 50_000, 100_000]


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.benchmark(group="linear")
def test_linear_regression(benchmark, n):
    df = DF.sample(n=n, seed=SEED)

    @benchmark
    def func():
        df.select(
            pds.query_lstsq(
                "x1",
                "x2",
                "x3",
                "x4",
                "x5",
                target="y",
                method="normal",
            )
        )


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.benchmark(group="linear")
def test_sklearn_linear_regression(benchmark, n):
    df = PD_DF.sample(n=n, random_state=SEED)

    @benchmark
    def func():
        reg = LinearRegression(fit_intercept=False)
        reg.fit(df[X_VARS], df[Y])


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.benchmark(group="lasso")
def test_lasso(benchmark, n):
    df = DF.sample(n=n, seed=SEED)

    @benchmark
    def func():
        df.select(
            pds.query_lstsq(
                "x1", "x2", "x3", "x4", "x5", target="y", method="l1", l1_reg=0.1
            )
        )


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.benchmark(group="lasso")
def test_sklearn_lasso(benchmark, n):
    df = PD_DF.sample(n=n, random_state=SEED)

    @benchmark
    def func():
        reg = Lasso(alpha=0.1, fit_intercept=False)
        reg.fit(df[X_VARS], df[Y])


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.benchmark(group="ridge")
def test_ridge(benchmark, n):
    df = DF.sample(n=n, seed=SEED)

    @benchmark
    def func():
        df.select(
            pds.query_lstsq(
                "x1", "x2", "x3", "x4", "x5", target="y", method="l2", l2_reg=0.1
            )
        )


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.benchmark(group="ridge")
def test_sklearn_ridge(benchmark, n):
    df = PD_DF.sample(n=n, random_state=SEED)

    @benchmark
    def func():
        reg = Ridge(alpha=0.1, fit_intercept=False)
        reg.fit(df[X_VARS], df[Y])
