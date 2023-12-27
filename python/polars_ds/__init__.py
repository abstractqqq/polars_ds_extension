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
    radius: Union[float, pl.Expr],
    dist: Distance = "l2",
) -> pl.Expr:
    """
    Returns an expression that queries the neighbors within (<=) radius from x. Note that
    this only queries around a single point x.

    Parameters
    ----------
    x : A point
        The point, at which we filter using the radius.
    others : pl.Expr, positional arguments
        Other columns used as features
    radius : either a float or an expression
        The radius to query with.
    dist : One of `l1`, `l2`, `inf` or `h` or `haversine`
        Distance metric to use. Note `l2` is actually squared `l2` for computational
        efficiency. It defaults to `l2`.
    """
    oth = list(others)
    if len(x) != len(oth):
        raise ValueError("Dimension does not match.")

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
                "For Haversine distance, input x must have dimension 2 and 2 other columns"
                " must be provided as lat and long."
            )

        y_lat = pl.Series(values=[x[0]], dtype=pl.Float64)
        y_long = pl.Series(values=[x[1]], dtype=pl.Float64)
        dist = oth[0].num._haversine(oth[1], y_lat, y_long)
        return dist <= radius
    else:  # defaults to l2, actually squared l2
        return (
            pl.sum_horizontal(
                (e - pl.lit(x[i], dtype=pl.Float64)).pow(2) for i, e in enumerate(oth)
            )
            <= radius
        )


def query_nb_cnt(
    radius: Union[float, pl.Expr, list[float], "np.ndarray", pl.Series],  # noqa: F821
    *others: pl.Expr,
    leaf_size: int = 40,
    dist: Distance = "l2",
    parallel: bool = False,
) -> pl.Expr:
    """
    Return the number of neighbors within (<=) radius for each row under the given distance
    metric. The point itself is always a neighbor of itself.

    Parameters
    ----------
    radius : float | Iterable[float] | pl.Expr
        If this is a scalar, then it will run the query with fixed radius for all rows. If
        this is a list, then it must have the same height as the dataframe in which this is run. If
        this is an expression, it must be a pl.col() representing radius in the dataframe.
        A large radius (lots of neighbors) will slow down performance.
    others : pl.Expr, positional arguments
        Other columns used as features
    leaf_size : int, > 0
        Leaf size for the kd-tree. Tuning this might improve performance.
    dist : One of `l1`, `l2`, `inf` or `h` or `haversine`
        Distance metric to use. Note `l2` is actually squared `l2` for computational
        efficiency. It defaults to `l2`.
    parallel : bool
        Whether to run the distance query in parallel. This is recommended when you
        are running only this expression, and not in group_by context.
    """
    if isinstance(radius, (float, int)):
        rad = pl.lit(pl.Series(values=[radius], dtype=pl.Float64))
    elif isinstance(radius, pl.Expr):
        rad = radius
    else:
        rad = pl.lit(pl.Series(values=radius, dtype=pl.Float64))

    return rad.num._nb_cnt(*others, leaf_size=leaf_size, dist=dist, parallel=parallel)


def knn(
    x: Union[list[float], "np.ndarray", pl.Series],  # noqa: F821
    *others: pl.Expr,
    k: int = 5,
    leaf_size: int = 40,
    dist: Distance = "l2",
) -> pl.Expr:
    """
    Returns an expression that queries the k nearest neighbors to x.

    Note that this internally builds a kd-tree for fast querying and deallocates it once we
    are done. If you need to repeatedly run the same query on the same data, then it is not
    ideal to use this. A specialized external kd-tree structure would be better in that case.

    Parameters
    ----------
    x : A point
        The point. It must be of the same length as the number of columns in `others`.
    others : pl.Expr, positional arguments
        Other columns used as features
    k : int, > 0
        Number of neighbors to query
    leaf_size : int, > 0
        Leaf size for the kd-tree. Tuning this might improve performance.
    dist : One of `l1`, `l2`, `inf` or `h` or `haversine`
        Distance metric to use. Note `l2` is actually squared `l2` for computational
        efficiency. It defaults to `l2`.
    """
    if k <= 0:
        raise ValueError("Input `k` should be strictly positive.")

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
