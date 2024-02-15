import polars as pl
from typing import Union, Iterable, List
from .type_alias import Distance

from polars_ds.num import NumExt  # noqa: E402
from polars_ds.complex import ComplexExt  # noqa: E402
from polars_ds.str2 import StrExt  # noqa: E402
from polars_ds.stats import StatsExt  # noqa: E402
from polars_ds.metrics import MetricExt  # noqa: E402
from polars_ds.graph import GraphExt  # noqa: E402

__version__ = "0.3.2"
__all__ = ["NumExt", "StrExt", "StatsExt", "ComplexExt", "MetricExt", "GraphExt"]


def query_radius(
    x: Iterable[float],
    *others: pl.Expr,
    r: Union[float, pl.Expr],
    dist: Distance = "l2",
) -> pl.Expr:
    """
    Returns an expression that queries the neighbors within (<=) radius from x. Note that
    this only queries around a single point x and returns a boolean column.

    Parameters
    ----------
    x : A point
        The point, at which we filter using the radius.
    others : pl.Expr, positional arguments
        Other columns used as features
    r : either a float or an expression
        The radius to query with. If this is an expression, the radius will be applied row-wise.
    dist : One of `l1`, `l2`, `inf` or `h` or `haversine`
        Distance metric to use. Note `l2` is actually squared `l2` for computational
        efficiency. It defaults to `l2`. Note `cosine` is not implemented for a single
        point yet.
    """
    # For a single point, it is faster to just do it in native polars
    oth = list(others)
    if len(x) != len(oth):
        raise ValueError("Dimension does not match.")

    if dist == "l1":
        return (
            pl.sum_horizontal((e - pl.lit(xi, dtype=pl.Float64)).abs() for xi, e in zip(x, oth))
            <= r
        )
    elif dist == "inf":
        return (
            pl.max_horizontal((e - pl.lit(xi, dtype=pl.Float64)).abs() for xi, e in zip(x, oth))
            <= r
        )
    elif dist == "cosine":
        x_list = list(x)
        x_norm = sum(z * z for z in x_list)
        oth_norm = pl.sum_horizontal([e * e for e in oth])
        dist = (
            1.0
            - pl.sum_horizontal(xi * e for xi, e in zip(x_list, oth)) / (x_norm * oth_norm).sqrt()
        )
        return dist <= r
    elif dist in ("h", "haversine"):
        x_list = list(x)
        if (len(x_list) != 2) or (len(oth) < 2):
            raise ValueError(
                "For Haversine distance, input x must have dimension 2 and 2 other columns"
                " must be provided as lat and long."
            )

        y_lat = pl.Series(values=[x_list[0]], dtype=pl.Float64)
        y_long = pl.Series(values=[x_list[1]], dtype=pl.Float64)
        dist = oth[0].num._haversine(oth[1], y_lat, y_long)
        return dist <= r
    else:  # defaults to l2, actually squared l2
        return (
            pl.sum_horizontal((e - pl.lit(xi, dtype=pl.Float64)).pow(2) for xi, e in zip(x, oth))
            <= r
        )


def query_nb_cnt(
    r: Union[float, pl.Expr, List[float], "np.ndarray", pl.Series],  # noqa: F821
    *others: pl.Expr,
    leaf_size: int = 40,
    dist: Distance = "l2",
    parallel: bool = False,
) -> pl.Expr:
    """
    Return the number of neighbors within (<=) radius r for each row under the given distance
    metric. The point itself is always a neighbor of itself.

    Parameters
    ----------
    r : float | Iterable[float] | pl.Expr
        If this is a scalar, then it will run the query with fixed radius for all rows. If
        this is a list, then it must have the same height as the dataframe. If
        this is an expression, it must be an expression representing radius.
    others : pl.Expr, positional arguments
        Other columns used as features
    leaf_size : int, > 0
        Leaf size for the kd-tree. Tuning this might improve performance.
    dist : One of `l1`, `l2`, `inf`, `cosine` or `h` or `haversine`
        Distance metric to use. Note `l2` is actually squared `l2` for computational
        efficiency. It defaults to `l2`.
    parallel : bool
        Whether to run the distance query in parallel. This is recommended when you
        are running only this expression, and not in group_by context.
    """
    if isinstance(r, (float, int)):
        rad = pl.lit(pl.Series(values=[r], dtype=pl.Float64))
    elif isinstance(r, pl.Expr):
        rad = r
    else:
        rad = pl.lit(pl.Series(values=r, dtype=pl.Float64))

    return rad.num._nb_cnt(*others, leaf_size=leaf_size, dist=dist, parallel=parallel)


def knn(
    x: Union[List[float], "np.ndarray", pl.Series],  # noqa: F821
    *others: pl.Expr,
    k: int = 5,
    leaf_size: int = 40,
    dist: Distance = "l2",
) -> pl.Expr:
    """
    Returns an expression that queries the k nearest neighbors to a single point x.

    Note that this internally builds a kd-tree for fast querying and deallocates it once we
    are done. If you need to repeatedly run the same query on the same data, then it is not
    ideal to use this. A specialized external kd-tree structure would be better in that case.

    Parameters
    ----------
    x : A point
        The point. It must be of the same length as the number of columns in `others`.
    others : pl.Expr
        Other columns used as features
    k : int, > 0
        Number of neighbors to query
    leaf_size : int, > 0
        Leaf size for the kd-tree. Tuning this might improve performance.
    dist : One of `l1`, `l2`, `inf`, `cosine` or `h` or `haversine`
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


def l_inf_horizontal(*v: pl.Expr, normalize: bool = False) -> pl.Expr:
    """
    Horizontally L inf norm. Shorthand for pl.max_horizontal(pl.col(x).abs() for x in exprs).

    Parameters
    ----------
    *v
        Expressions to compute horizontal L infinity.
    normalize
        Whether to divide by the dimension
    """
    if normalize:
        exprs = list(v)
        return pl.max_horizontal(pl.col(x).abs() for x in exprs) / len(exprs)
    else:
        return pl.max_horizontal(pl.col(x).abs() for x in v)


def l2_sq_horizontal(*v: pl.Expr, normalize: bool = False) -> pl.Expr:
    """
    Horizontally computes L2 norm squared. Shorthand for pl.sum_horizontal(pl.col(x).pow(2) for x in exprs).

    Parameters
    ----------
    *v
        Expressions to compute horizontal L2.
    normalize
        Whether to divide by the dimension
    """
    if normalize:
        exprs = list(v)
        return pl.sum_horizontal(pl.col(x).pow(2) for x in exprs) / len(exprs)
    else:
        return pl.sum_horizontal(pl.col(x).pow(2) for x in v)


def l1_horizontal(*v: pl.Expr, normalize: bool = False) -> pl.Expr:
    """
    Horizontally computes L1 norm. Shorthand for pl.sum_horizontal(pl.col(x).abs() for x in exprs).

    Parameters
    ----------
    *v
        Expressions to compute horizontal L1.
    normalize
        Whether to divide by the dimension
    """
    if normalize:
        exprs = list(v)
        return pl.sum_horizontal(pl.col(x).abs() for x in exprs) / len(exprs)
    else:
        return pl.sum_horizontal(pl.col(x).abs() for x in v)


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
