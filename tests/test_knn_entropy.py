import polars as pl
from polars_ds.num import query_knn_entropy
import pytest

def test_knn_entropy():
    df = pl.DataFrame(dict(x=[ 1,  2, 10], y=[ 2,  5, 10]))
    ent = df.select(query_knn_entropy('x', 'y', k=1))
    assert ent.item(0, 0) == pytest.approx(5.67, abs=0.01)
