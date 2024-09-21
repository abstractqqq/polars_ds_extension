import polars as pl
import polars_ds as pds
import pytest

SEED = 208

SIZE = 1_000_000
DF = (
    pds.random_data(size=SIZE, n_cols=4)
    .rename(
        {
            "feature_1": "x",
            "feature_2": "y",
            "feature_3": "a",
            "feature_4": "b",
        }
    )
    .drop("row_num")
)

SIZES = [1_000, 10_000, 100_000, 1_000_000]


@pytest.mark.parametrize("n", SIZES)
def test_softmax(benchmark, n):
    df = DF.sample(n=n, seed=SEED)

    @benchmark
    def func():
        df.select(pds.softmax("x"))


@pytest.mark.parametrize("n", SIZES)
def test_gcd(benchmark, n):
    df = DF.sample(n=n, seed=SEED)

    @benchmark
    def func():
        df.select(pds.query_gcd("x", "y"))


@pytest.mark.parametrize("n", SIZES)
def test_lcm(benchmark, n):
    df = DF.sample(n=n, seed=SEED)

    @benchmark
    def func():
        df.select(pds.query_lcm("x", "y"))


@pytest.mark.parametrize("n", SIZES)
def test_haversine(benchmark, n):
    df = DF.sample(n=n, seed=SEED)

    @benchmark
    def func():
        df.select(pds.haversine("x", "y", "a", "b"))


@pytest.mark.parametrize("n", SIZES)
def test_singular_values(benchmark, n):
    df = DF.sample(n=n, seed=SEED)

    @benchmark
    def func():
        df.select(pds.query_singular_values("x", "y", "a", "b"))


@pytest.mark.parametrize("n", SIZES)
def test_pca(benchmark, n):
    df = DF.sample(n=n, seed=SEED)

    @benchmark
    def func():
        df.select(pds.query_pca("x", "y", "a", "b"))


@pytest.mark.parametrize("n", SIZES)
def test_principal_components(benchmark, n):
    df = DF.sample(n=n, seed=SEED)

    @benchmark
    def func():
        df.select(pds.query_principal_components("x", "y", "a", "b"))


@pytest.mark.parametrize("n", SIZES)
def test_knn_pointwise(benchmark, n):
    df = DF.sample(n=n, seed=SEED)

    @benchmark
    def func():
        df.select(pds.query_knn_ptwise("x", "y", "a", "b"))


@pytest.mark.parametrize("n", SIZES)
def test_within_dist_from(benchmark, n):
    df = DF.sample(n=n, seed=SEED)

    @benchmark
    def func():
        df.select(pds.within_dist_from("x", "y", "a", "b"))


@pytest.mark.parametrize("n", SIZES)
def test_is_knn_from(benchmark, n):
    df = DF.sample(n=n, seed=SEED)

    @benchmark
    def func():
        df.select(pds.is_knn_from("x", "y", "a", "b"))


@pytest.mark.parametrize("n", SIZES)
def test_radius_ptwise(benchmark, n):
    df = DF.sample(n=n, seed=SEED)

    @benchmark
    def func():
        df.select(pds.query_radius_ptwise("x", "y", "a", "b"))


@pytest.mark.parametrize("n", SIZES)
def test_nb_cnt(benchmark, n):
    df = DF.sample(n=n, seed=SEED)

    @benchmark
    def func():
        df.select(pds.query_nb_cnt(2, "x", "y", "a", "b"))


@pytest.mark.parametrize("n", SIZES)
def test_approx_entropy(benchmark, n):
    df = DF.sample(n=n, seed=SEED)

    @benchmark
    def func():
        df.select(pds.query_approx_entropy("x", 2, 0.1))


@pytest.mark.parametrize("n", SIZES)
def test_sample_entropy(benchmark, n):
    df = DF.sample(n=n, seed=SEED)

    @benchmark
    def func():
        df.select(pds.query_sample_entropy("x"))


@pytest.mark.parametrize("n", SIZES)
def test_cond_entropy(benchmark, n):
    df = DF.sample(n=n, seed=SEED)

    @benchmark
    def func():
        df.select(pds.query_cond_entropy("x", "y"))


@pytest.mark.parametrize("n", SIZES)
def test_knn_entropy(benchmark, n):
    df = DF.sample(n=n, seed=SEED)

    @benchmark
    def func():
        df.select(pds.query_knn_entropy("x", "y", "a", "b"))


@pytest.mark.parametrize("n", SIZES)
def test_copula_entropy(benchmark, n):
    df = DF.sample(n=n, seed=SEED)

    @benchmark
    def func():
        df.select(pds.query_copula_entropy("x", "y", "a", "b"))


@pytest.mark.parametrize("n", SIZES)
def test_cond_indep(benchmark, n):
    df = DF.sample(n=n, seed=SEED)

    @benchmark
    def func():
        df.select(pds.query_cond_indep("x", "y", "a"))


@pytest.mark.parametrize("n", SIZES)
def test_transfer_entropy(benchmark, n):
    df = DF.sample(n=n, seed=SEED)

    @benchmark
    def func():
        df.select(pds.query_transfer_entropy("x", "y"))


@pytest.mark.parametrize("n", SIZES)
def test_permute_entropy(benchmark, n):
    df = DF.sample(n=n, seed=SEED)

    @benchmark
    def func():
        df.select(pds.query_permute_entropy("x"))
