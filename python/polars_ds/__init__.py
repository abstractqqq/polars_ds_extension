import polars as pl
from typing import Union
from .type_alias import Distance

from polars_ds.num_ext import NumExt  # noqa: E402
from polars_ds.complex_ext import ComplexExt  # noqa: E402
from polars_ds.str_ext import StrExt  # noqa: E402
from polars_ds.stats_ext import StatsExt  # noqa: E402

version = "0.2.0"
__all__ = ["NumExt", "StrExt", "StatsExt", "ComplexExt"]


def knn(
    *others: pl.Expr,
    pt: Union[list[float], "np.ndarray", pl.Series],  # noqa: F821
    k: int = 5,
    dist: Distance = "l2",
    leaf_size: int = 40,
) -> pl.Expr:
    point = pl.Series(pt, dtype=pl.Float64)
    return pl.lit(point).num._knn_pt(
        *others,
        k=k,
        leaf_size=40,
        dist=dist,
    )
