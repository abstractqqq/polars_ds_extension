import networkx as nx
import polars as pl
import polars_ds as pld  # noqa: F401
from typing import List, Optional
import pytest

# Don't run this automatically.
# If there are two shortest paths to the same target!
# The test will fail!
# The parameters here are chosen so that it doesn't happen very often.


def get_random_data(size: int = 2000) -> pl.DataFrame:
    df = pl.DataFrame(
        {
            "id": range(size),
        }
    ).with_columns(
        pl.col("id").cast(pl.UInt64),
        pl.col("id").stats.sample_uniform(low=0.0, high=1.0).alias("val1"),
        pl.col("id").stats.sample_uniform(low=0.0, high=1.0).alias("val2"),
        pl.col("id").stats.sample_uniform(low=0.0, high=1.0).alias("val3"),
    )
    return df.select(
        pl.col("id"),
        pl.col("id")
        .num.query_radius_ptwise(
            pl.col("val1"),
            pl.col("val2"),
            pl.col("val3"),  # Columns used as the coordinates in n-d space
            r=0.05,
            dist="inf",
            parallel=True,
        )
        .alias("friends"),
    )


@pytest.fixture
def random_data() -> pl.DataFrame:
    return get_random_data()


def shortest_path_pl_ds(df: pl.DataFrame, col: str = "friends") -> pl.DataFrame:
    return df.select(pl.col("friends").graph.shortest_path(target=0, parallel=False).alias("out"))


def shortest_path_pl_ds_parallel(df: pl.DataFrame, col: str = "friends") -> pl.DataFrame:
    return df.select(pl.col("friends").graph.shortest_path(target=0, parallel=True).alias("out"))


def shortest_path_networkx(edges: List[Optional[List[int]]]) -> List[Optional[List[int]]]:
    # Constructing Graph
    graph = nx.Graph()
    for i, e in enumerate(edges):
        if e is not None:
            for j in e:
                graph.add_edge(i, j, weight=1)

    # Generating output
    paths = []
    for i in range(len(edges)):
        try:
            path = nx.shortest_path(graph, i, 0, weight="weight")
            paths.append(path[1:])
        except Exception as _:  # No path
            paths.append(None)

    return paths


def test_shortest_path(benchmark):
    df = get_random_data()
    # Polars-ds answer
    res = shortest_path_pl_ds(df)
    ans_pl_ds = [list(s) if s is not None else None for s in res["out"]]
    # ans_pl_ds_par = shortest_path_pl_ds_parallel(df)
    # Networkx answer
    edges_as_list = [list(s) for s in df["friends"]]
    ans_networkx = shortest_path_networkx(edges_as_list)

    # Test each individual path. None means no path exists.
    for ans1, ans2 in zip(ans_pl_ds, ans_networkx):
        if ans1 is None:
            assert ans2 is None
        else:
            assert ans2 is not None
            # There might be two shortest paths to the same location
            # which happens often in randomly generated data.
            assert ans1 == ans2


@pytest.mark.benchmark(group="shortest_path")
def test_pl_ds_shortest_path(random_data, benchmark):
    # Warm up by running the function once
    _ = shortest_path_pl_ds(random_data)
    # Start benchmark
    _ = benchmark(shortest_path_pl_ds, random_data)
    assert True


@pytest.mark.benchmark(group="shortest_path")
def test_pl_ds_shortest_path_parallel(random_data, benchmark):
    # Warm up by running the function once
    _ = shortest_path_pl_ds_parallel(random_data)
    # Start benchmark
    _ = benchmark(shortest_path_pl_ds_parallel, random_data)
    assert True


@pytest.mark.benchmark(group="shortest_path")
def test_networkx_shortest_path(random_data, benchmark):
    edges_as_list = [list(s) for s in random_data["friends"]]
    # Warm up by running the function once
    _ = shortest_path_networkx(edges_as_list)
    # Start benchmark
    _ = benchmark(shortest_path_networkx, edges_as_list)
    assert True
