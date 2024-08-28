import polars as pl
import polars_ds as pds
import pytest
from polars_ds.linear_models import LR
from sklearn.linear_model import Lasso, LinearRegression, Ridge

SEED = 208

SIZE = 100_000
DF = (
    pds.frame(size=SIZE)
    .select(
        pds.random(0.0, 1.0).alias("x1"),
        pds.random(0.0, 1.0).alias("x2"),
        pds.random(0.0, 1.0).alias("x3"),
        pds.random(0.0, 1.0).alias("x4"),
        pds.random(0.0, 1.0).alias("x5"),
        pds.random_int(0, 4).alias("code"),
        pl.Series(name="id", values=range(SIZE)),
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
@pytest.mark.benchmark(group="linear_on_matrix")
def test_pds_linear_regression_on_matrix(benchmark, n):
    df = DF.sample(n=n, seed=SEED)
    X = df.select(*X_VARS).to_numpy()
    y = df.select(*Y).to_numpy()

    @benchmark
    def func():
        model = LR()
        model.fit(X, y)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.benchmark(group="linear_on_matrix")
def test_sklearn_linear_regression_on_matrix(benchmark, n):
    df = DF.sample(n=n, seed=SEED)
    X = df.select(*X_VARS).to_numpy()
    y = df.select(*Y).to_numpy()

    @benchmark
    def func():
        reg = LinearRegression(fit_intercept=False, n_jobs=-1)
        reg.fit(X, y)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.benchmark(group="linear_on_df")
def test_pds_linear_regression_on_df(benchmark, n):
    df = DF.sample(n=n, seed=SEED)

    @benchmark
    def func():
        df.select(
            pds.query_lstsq(
                *X_VARS,
                target=Y[0],
                method="normal",
            )
        )


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.benchmark(group="linear_on_df")
def test_sklearn_linear_regression_on_df(benchmark, n):
    df = PD_DF.sample(n=n, random_state=SEED)

    @benchmark
    def func():
        reg = LinearRegression(fit_intercept=False, n_jobs=-1)
        reg.fit(df[X_VARS], df[Y])


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.benchmark(group="lasso_on_df")
def test_pds_lasso_on_df(benchmark, n):
    df = DF.sample(n=n, seed=SEED)

    @benchmark
    def func():
        df.select(pds.query_lstsq(*X_VARS, target=Y[0], method="l1", l1_reg=0.1))


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.benchmark(group="lasso_on_df")
def test_sklearn_lasso_on_df(benchmark, n):
    df = PD_DF.sample(n=n, random_state=SEED)

    @benchmark
    def func():
        reg = Lasso(alpha=0.1, fit_intercept=False)
        reg.fit(df[X_VARS], df[Y])


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.benchmark(group="ridge_svd_on_df")
def test_ridge_svd_on_df(benchmark, n):
    df = DF.sample(n=n, seed=SEED)

    @benchmark
    def func():
        df.select(pds.query_lstsq(*X_VARS, target=Y[0], method="l2", l2_reg=0.1, solver="svd"))


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.benchmark(group="ridge_svd_on_df")
def test_sklearn_ridge_svd_on_df(benchmark, n):
    df = PD_DF.sample(n=n, random_state=SEED)

    @benchmark
    def func():
        reg = Ridge(alpha=0.1, fit_intercept=False, solver="svd")
        reg.fit(df[X_VARS], df[Y])


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.benchmark(group="ridge_cholesky")
def test_ridge_cholesky_on_df(benchmark, n):
    df = DF.sample(n=n, seed=SEED)

    @benchmark
    def func():
        df.select(pds.query_lstsq(*X_VARS, target=Y[0], method="l2", l2_reg=0.1, solver="cholesky"))


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.benchmark(group="ridge_cholesky")
def test_sklearn_ridge_cholesky_on_df(benchmark, n):
    df = PD_DF.sample(n=n, random_state=SEED)

    @benchmark
    def func():
        reg = Ridge(alpha=0.1, fit_intercept=False, solver="cholesky")
        reg.fit(df[X_VARS], df[Y])
