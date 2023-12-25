import polars as pl
from typing import Union
from .type_alias import Distance

from polars_ds.num import NumExt  # noqa: E402
from polars_ds.complex import ComplexExt  # noqa: E402
from polars_ds.str2 import StrExt  # noqa: E402
from polars_ds.stats import StatsExt  # noqa: E402

version = "0.2.0"
__all__ = ["NumExt", "StrExt", "StatsExt", "ComplexExt"]


def query_radius(
    x: Union[list[float], "np.ndarray", pl.Series],  # noqa: F821
    *others: pl.Expr,
    radius: float,
    dist: Distance = "l2",
) -> pl.Expr:
    """
    Returns an expression that queries the neighbors within (<=) radius from x. Note that
    this only queries around a single point x.

    Parameters
    ----------
    x
        The point, at which we filter using the radius.
    others
        Other columns used as features
    radius
        The radius to query with.
    dist
        The L^p distance to use or `h` for haversine. Default `l2`. Currently only
        supports `l1`, `l2` and `inf` which is L infinity or `h` or `haversine`. Any
        other string will be redirected to `l2`.
    """
    oth = list(others)
    if len(x) != len(oth):
        raise ValueError("Dimension of x must match the number of columns in `others`.")

    if dist == "l1":
        return (
            pl.sum_horizontal((e - pl.lit(x[i], dtype=pl.Float64)).abs() for i, e in enumerate(oth))
            <= radius
        )
    elif dist == "inf":
        return (
            pl.max_horizontal((e - pl.lit(x[i], dtype=pl.Float64)).abs() for i, e in enumerate(oth))
            <= radius
        )
    elif dist in ("h", "haversine"):
        if (len(x) != 2) or (len(oth) < 2):
            raise ValueError(
                "For Haversine distance, input x must have size 2 and 2 more columns"
                " must be provided as lat and long."
            )

        y_lat = pl.Series(values=[x[0]], dtype=pl.Float64)
        y_long = pl.Series(values=[x[1]], dtype=pl.Float64)
        dist = oth[0].num._haversine(oth[1], y_lat, y_long)
        return dist <= radius
    else:  # defaults to l2
        return (
            pl.sum_horizontal(
                (e - pl.lit(x[i], dtype=pl.Float64)).pow(2) for i, e in enumerate(oth)
            )
            <= radius
        )


def knn(
    x: Union[list[float], "np.ndarray", pl.Series],  # noqa: F821
    *others: pl.Expr,
    k: int = 5,
    dist: Distance = "l2",
    leaf_size: int = 40,
) -> pl.Expr:
    """
    Returns an expression that queries the k nearest neighbors to x.

    Note that this internally builds a kd-tree for fast querying and deallocates it once we
    are done. If you need to repeatedly run the same query on the same data, then it is not
    ideal to use this. A specialized external kd-tree structure would be better in that case.

    Parameters
    ----------
    x
        The point. It must be of the same length as the number of columns in `others`.
    others
        Other columns used as features
    k
        Number of neighbors to query
    leaf_size
        Leaf size for the kd-tree. Tuning this might improve performance.
    dist
        The L^p distance to use or `h` for haversine. Default `l2`. Currently only
        supports `l1`, `l2` and `inf` which is L infinity or `h` or `haversine`. Any
        other string will be redirected to `l2`.
    """
    pt = pl.Series(x, dtype=pl.Float64)
    return pl.lit(pt).num._knn_pt(
        *others,
        k=k,
        leaf_size=leaf_size,
        dist=dist,
    )


def haversine(
    x_lat: pl.Expr,
    x_long: pl.Expr,
    y_lat: Union[float, pl.Expr],
    y_long: Union[float, pl.Expr],
) -> pl.Expr:
    """
    Computes haversine distance using the naive method. The output unit is km.
    """
    ylat = pl.lit(y_lat) if isinstance(y_lat, float) else y_lat
    ylong = pl.lit(y_long) if isinstance(y_long, float) else y_long
    return x_lat.num._haversine(x_long, ylat, ylong)
