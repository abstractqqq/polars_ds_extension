import polars as pl
import polars_ds as pds
import pytest
import scipy.stats
import sklearn.metrics

SEED = 208

SIZE = 1_000_000
DF = (
    pds.random_data(size=SIZE, n_cols=2)
    .rename({"feature_1": "y_true_score", "feature_2": "y_score"})
    .with_columns(
        pl.col("y_true_score").ge(0.5).alias("y_true"),
    )
    .drop("row_num")
)

MUTLICLASS_DF = (
    pds.random_data(size=SIZE, n_cols=2)
    .with_columns(
        pds.random_int(0, 2).alias("y_true"),
        pl.concat_list("feature_1", "feature_2").alias("y_score"),
    )
    .drop("feature_1", "feature_2")
)

SIZES = [1_000, 10_000, 100_000, 1_000_000]


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.benchmark(group="mad")
def test_mad(benchmark, n):
    df = DF.sample(n=n, seed=SEED)

    @benchmark
    def func():
        df.select(pds.query_mad("y_score"))


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.benchmark(group="mad")
def test_scipy_mad(benchmark, n):
    df = DF.sample(n=n, seed=SEED)

    @benchmark
    def func():
        scipy.stats.median_abs_deviation(df["y_score"])


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.benchmark(group="r2")
def test_r2(benchmark, n):
    df = DF.sample(n=n, seed=SEED)

    @benchmark
    def func():
        df.select(pds.query_r2("y_true_score", "y_score"))


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.benchmark(group="r2")
def test_sklearn_r2(benchmark, n):
    df = DF.sample(n=n, seed=SEED)

    @benchmark
    def func():
        sklearn.metrics.r2_score(df["y_true_score"], df["y_score"])


@pytest.mark.parametrize("n", SIZES)
def test_log_cosh(benchmark, n):
    df = DF.sample(n=n, seed=SEED)

    @benchmark
    def func():
        df.select(pds.query_log_cosh("y_true_score", "y_score"))


@pytest.mark.parametrize("n", SIZES)
def test_hubor_loss(benchmark, n):
    df = DF.sample(n=n, seed=SEED)

    @benchmark
    def func():
        df.select(pds.query_hubor_loss("y_true_score", "y_score", delta=0.2))


@pytest.mark.parametrize("n", SIZES)
def test_l2(benchmark, n):
    df = DF.sample(n=n, seed=SEED)

    @benchmark
    def func():
        df.select(pds.query_l2("y_true_score", "y_score"))


@pytest.mark.parametrize("n", SIZES)
def test_l1(benchmark, n):
    df = DF.sample(n=n, seed=SEED)

    @benchmark
    def func():
        df.select(pds.query_l1("y_true_score", "y_score"))


@pytest.mark.parametrize("n", SIZES)
def test_l_inf(benchmark, n):
    df = DF.sample(n=n, seed=SEED)

    @benchmark
    def func():
        df.select(pds.query_l_inf("y_true_score", "y_score"))


@pytest.mark.parametrize("n", SIZES)
def test_log_loss(benchmark, n):
    df = DF.sample(n=n, seed=SEED)

    @benchmark
    def func():
        df.select(pds.query_log_loss("y_true_score", "y_score"))


@pytest.mark.parametrize("n", SIZES)
def test_mape(benchmark, n):
    df = DF.sample(n=n, seed=SEED)

    @benchmark
    def func():
        df.select(pds.query_mape("y_true_score", "y_score"))


@pytest.mark.parametrize("n", SIZES)
def test_smape(benchmark, n):
    df = DF.sample(n=n, seed=SEED)

    @benchmark
    def func():
        df.select(pds.query_smape("y_true_score", "y_score"))


@pytest.mark.parametrize("n", SIZES)
def test_msle(benchmark, n):
    df = DF.sample(n=n, seed=SEED)

    @benchmark
    def func():
        df.select(pds.query_msle("y_true_score", "y_score"))


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.benchmark(group="roc_auc")
def test_roc_auc(benchmark, n):
    df = DF.sample(n=n, seed=SEED)

    @benchmark
    def func():
        df.select(pds.query_roc_auc("y_true", "y_score"))


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.benchmark(group="roc_auc")
def test_sklearn_roc_auc(benchmark, n):
    df = DF.sample(n=n, seed=SEED)

    @benchmark
    def func():
        sklearn.metrics.roc_auc_score(df["y_true"], df["y_score"])


@pytest.mark.parametrize("n", SIZES)
def test_binary_metrics(benchmark, n):
    df = DF.sample(n=n, seed=SEED)

    @benchmark
    def func():
        df.select(pds.query_binary_metrics("y_true", "y_score"))


@pytest.mark.parametrize("n", SIZES)
def test_confusion_matrix(benchmark, n):
    df = DF.sample(n=n, seed=SEED)

    @benchmark
    def func():
        df.select(pds.query_confusion_matrix("y_true", "y_score"))


@pytest.mark.parametrize("n", SIZES)
def test_multi_roc_auc(benchmark, n):
    df = MUTLICLASS_DF.sample(n=n, seed=SEED)

    @benchmark
    def func():
        df.select(pds.query_multi_roc_auc("y_true", "y_score", n_classes=3))
