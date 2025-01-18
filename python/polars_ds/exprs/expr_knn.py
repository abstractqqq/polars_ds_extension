"""
KNN related query expressions in Polars.
"""

from __future__ import annotations
import polars as pl
from typing import Any, Iterable, List, Sequence, cast
import warnings

# Internal dependencies
from polars_ds._utils import pl_plugin, str_to_expr
from polars_ds.typing import Distance

__all__ = [
    "query_knn_ptwise",
    "query_knn_freq_cnt",
    "query_knn_avg",
    "query_radius_ptwise",
    "query_radius_freq_cnt",
    "query_nb_cnt",
    "query_dist_from_kth_nb",
    "is_knn_from",
    "within_dist_from",
]


def warn_len_compare(item1: Iterable[Any], item2: Iterable[Any]) -> bool:
    """
    Compares the len of two Iterables if they have len returning true and warning if no len.

    Parameters
    ----------
    item1: Iterable[Any]
        Any iterable
    item2: Iterable[Any])
        Any iterable

    Returns:
        bool: If both items have __len__ then it will simply return whether or not
            they have equal size. If they don't have len then it returns True with a
            warning
    """
    # print()
    if hasattr(item1, "__len__") and hasattr(item2, "__len__"):
        return len(cast(Sequence, item1)) == len(cast(Sequence, item2))
    else:
        msg = "The inputs do not each have len so can't be compared, unexpected results may follow."
        warnings.warn(msg, stacklevel=2)
        return True


def query_dist_from_kth_nb(
    *features: str | pl.Expr,
    k: int,
    dist: Distance = "sql2",
    parallel: bool = False,
    epsilon: float = 0.0,
    max_bound: float = 99999.0,
) -> pl.Expr:
    """
    Computes the distance of each row to its k-th closest neighbor. This is useful for outlier detection.
    E.g. if the average distance to the 5th neighbor is 0.1, then a distance of 0.3 to the 5th neighbor might
    indicate that the point might be far away from neighboring points, or that it occupies a sparse region in which
    sample points typically do not appear.

    This can be 10% faster and more direct than getting the result from `query_knn_ptwise` with return_distance = True.

    Parameters
    ----------
    *features : str | pl.Expr
        Other columns used as features
    k : int
        Number of neighbors to query
    dist : Literal[`l1`, `l2`, `sql2`, `inf`]
        Note `sql2` stands for squared l2.
    parallel : bool
        Whether to run the k-nearest neighbor query in parallel. This is recommended when you
        are running only this expression, and not in group_by() or over() context.
    epsilon
        If > 0, then it is possible to miss a neighbor within epsilon distance away. This parameter
        should increase as the dimension of the vector space increases because higher dimensions
        allow for errors from more directions.
    max_bound
        Max distance the neighbors must be within
    """
    return pl_plugin(
        symbol="pl_dist_from_kth_nb",
        args=[str_to_expr(e) for e in features],
        kwargs={
            "k": k,
            "metric": str(dist).lower(),
            "parallel": parallel,
            "skip_eval": False,
            "max_bound": max_bound,
            "epsilon": epsilon,
        },
    )


def query_knn_ptwise(
    *features: str | pl.Expr,
    index: str | pl.Expr,
    k: int,
    dist: Distance = "sql2",
    parallel: bool = False,
    return_dist: bool = False,
    eval_mask: str | pl.Expr | None = None,
    data_mask: str | pl.Expr | None = None,
    epsilon: float = 0.0,
    max_bound: float = 99999.0,
) -> pl.Expr:
    """
    Takes the index column, and uses feature columns to determine the k nearest neighbors
    to each row. By default, this will return k + 1 neighbors, because the point (the row) itself
    is a neighbor to itself and this returns k additional neighbors. The only exception to this
    is when data_mask excludes the point from being a neighbor, in which case, k + 1 distinct neighbors will
    be returned. Any row with a null/NaN will never be a neighbor and will have null as its neighbor.

    Note that the index column must be convertible to u32. If you do not have a u32 column,
    you can generate one using pl.int_range(..), which should be a step before this. The index column
    must not contain nulls.

    Note that a default max distance bound of 99999.0 is applied. This means that if we cannot find
    k neighbors within `max_bound`, then there will be < k neighbors returned.

    Also note that this internally builds a kd-tree for fast querying and deallocates it once we
    are done. If you need to repeatedly run the same query on the same data, then it is not
    ideal to use this. A specialized external kd-tree structure would be better in that case.

    Parameters
    ----------
    *features : str | pl.Expr
        Other columns used as features
    index : str | pl.Expr
        The column used as index, must be castable to u32
    k : int
        Number of neighbors to query
    dist : Literal[`l1`, `l2`, `sql2`, `inf`]
        Note `sql2` stands for squared l2.
    parallel : bool
        Whether to run the k-nearest neighbor query in parallel. This is recommended when you
        are running only this expression, and not in group_by() or over() context.
    return_dist
        If true, return a struct with indices and distances.
    eval_mask
        Either None or a boolean expression or the name of a boolean column. If not none, this will
        only evaluate KNN for rows where this is true. This can speed up computation when only results on a
        subset are nedded.
    data_mask
        Either None or a boolean expression or the name of a boolean column. If none, all rows can be
        neighbors. If not None, the pool of possible neighbors will be rows where this is true.
    epsilon
        If > 0, then it is possible to miss a neighbor within epsilon distance away. This parameter
        should increase as the dimension of the vector space increases because higher dimensions
        allow for errors from more directions.
    max_bound
        Max distance the neighbors must be within
    """
    if k < 1:
        raise ValueError("Input `k` must be >= 1.")

    if dist in ("cosine", "h", "haversine"):
        raise ValueError(f"Distance {dist} doesn't work with current implementation.")

    idx = str_to_expr(index).cast(pl.UInt32).rechunk()
    cols = [idx]
    feats: List[pl.Expr] = [str_to_expr(e) for e in features]

    skip_data = data_mask is not None
    if skip_data:  # true means keep
        keep_mask = pl.all_horizontal(str_to_expr(data_mask), *(f.is_not_null() for f in feats))
    else:
        keep_mask = pl.all_horizontal(f.is_not_null() for f in feats)

    cols.append(keep_mask)
    skip_eval = eval_mask is not None
    if skip_eval:
        cols.append(str_to_expr(eval_mask))

    cols.extend(feats)
    kwargs = {
        "k": k,
        "metric": str(dist).lower(),
        "parallel": parallel,
        "skip_eval": skip_eval,
        "max_bound": max_bound,
        "epsilon": 0.0,
    }
    if return_dist:
        return pl_plugin(
            symbol="pl_knn_ptwise_w_dist",
            args=cols,
            kwargs=kwargs,
        )
    else:
        return pl_plugin(
            symbol="pl_knn_ptwise",
            args=cols,
            kwargs=kwargs,
        )


def query_knn_freq_cnt(
    *features: str | pl.Expr,
    index: str | pl.Expr,
    k: int,
    dist: Distance = "sql2",
    parallel: bool = False,
    eval_mask: str | pl.Expr | None = None,
    data_mask: str | pl.Expr | None = None,
    epsilon: float = 0.0,
    max_bound: float = 99999.0,
) -> pl.Expr:
    """
    Takes the index column, and uses feature columns to determine the k nearest neighbors
    to each row, and finally returns the number of times a row is a KNN of some other point.

    This calls `query_knn_ptwise` internally. See the docstring of `query_knn_ptwise` for more info.

    Parameters
    ----------
    *features : str | pl.Expr
        Other columns used as features
    index : str | pl.Expr
        The column used as index, must be castable to u32
    k : int
        Number of neighbors to query
    dist : Literal[`l1`, `l2`, `sql2`, `inf`]
        Note `sql2` stands for squared l2.
    parallel : bool
        Whether to run the k-nearest neighbor query in parallel. This is recommended when you
        are running only this expression, and not in group_by() or over() context.
    return_dist
        If true, return a struct with indices and distances.
    eval_mask
        Either None or a boolean expression or the name of a boolean column. If not none, this will
        only evaluate KNN for rows where this is true. This can speed up computation when only results on a
        subset are nedded.
    data_mask
        Either None or a boolean expression or the name of a boolean column. If none, all rows can be
        neighbors. If not None, the pool of possible neighbors will be rows where this is true.
    epsilon
        If > 0, then it is possible to miss a neighbor within epsilon distance away. This parameter
        should increase as the dimension of the vector space increases because higher dimensions
        allow for errors from more directions.
    max_bound
        Max distance the neighbors must be within
    """

    knn_expr: pl.Expr = query_knn_ptwise(
        *features,
        index=index,
        k=k,
        dist=dist,
        parallel=parallel,
        return_dist=False,
        eval_mask=eval_mask,
        data_mask=data_mask,
        epsilon=epsilon,
        max_bound=max_bound,
    )
    return knn_expr.explode().drop_nulls().value_counts(sort=True, parallel=parallel)


def query_knn_avg(
    *features: str | pl.Expr,
    target: str | pl.Expr,
    k: int,
    dist: Distance = "sql2",
    weighted: bool = False,
    parallel: bool = False,
    min_bound: float = 1e-9,
    max_bound: float = 99999.0,
) -> pl.Expr:
    """
    Takes the target column, and uses feature columns to determine the k nearest neighbors
    to each row. By default, this will return k + 1 neighbors, because the point (the row) itself
    is a neighbor to itself and this returns k additional neighbors. Any row with a null/NaN will
    never be a neighbor and will get null as the average.

    Note that a default max distance bound of 99999.0 is applied. This means that if we cannot find
    k neighbors within `max_bound`, then there will be < k neighbors returned.

    This is also known as KNN Regression, but really it is just the average of the K nearest neighbors.

    Parameters
    ----------
    *features : str | pl.Expr
        Other columns used as features
    target : str | pl.Expr
        Float, must be castable to f64. This should not contain null.
    k : int
        Number of neighbors to query
    dist : Literal[`l1`, `l2`, `sql2`, `inf`]
        Note `sql2` stands for squared l2.
    weighted : bool
        If weighted, it will use 1/distance as weights to compute the KNN average. If min_bound is
        an extremely small value, this will default to 1/(1+distance) as weights to avoid division by 0.
    parallel : bool
        Whether to run the k-nearest neighbor query in parallel. This is recommended when you
        are running only this expression, and not in group_by() or over() context.
    min_bound
        Min distance (>=) for a neighbor to be part of the average calculation. This prevents "identical"
        points from being part of the average and prevents division by 0. Note that this filter is applied
        after getting k nearest neighbors.
    max_bound
        Max distance the neighbors must be within (<)
    """
    if k < 1:
        raise ValueError("Input `k` must be >= 1.")

    if dist in ("cosine", "h", "haversine"):
        raise ValueError(f"Distance {dist} doesn't work with current implementation.")

    idx = str_to_expr(target).cast(pl.Float64).rechunk()
    feats = [str_to_expr(f) for f in features]
    keep_data = ~pl.any_horizontal(f.is_null() for f in feats)
    cols = [idx, keep_data]
    cols.extend(feats)

    kwargs = {
        "k": k,
        "metric": str(dist).lower(),
        "weighted": weighted,
        "parallel": parallel,
        "min_bound": min_bound,
        "max_bound": max_bound,
    }

    return pl_plugin(
        symbol="pl_knn_avg",
        args=cols,
        kwargs=kwargs,
    )


def within_dist_from(
    *features: str | pl.Expr,
    pt: Sequence[float] | Iterable[float],
    r: float | pl.Expr,
    dist: Distance = "sql2",
) -> pl.Expr:
    """
    Returns a boolean column that returns points that are within radius from the given point.

    Parameters
    ----------
    *features : str | pl.Expr
        Other columns used as features
    pt : Iterable[float]
        The point
    r : either a float or an expression
        The radius to query with. If this is an expression, the radius will be applied row-wise.
    dist : Literal[`l1`, `l2`, `sql2`, `inf`, `cosine`, `haversine`]
        Note `sql2` stands for squared l2.
    """
    # For a single point, it is faster to just do it in native polars
    oth = [str_to_expr(x) for x in features]
    if not warn_len_compare(pt, oth):
        raise ValueError("Dimension does not match.")

    if dist == "l1":
        return (
            pl.sum_horizontal((e - pl.lit(xi, dtype=pl.Float64)).abs() for xi, e in zip(pt, oth))
            <= r
        )
    elif dist in ("l2", "sql2"):
        return (
            pl.sum_horizontal((e - pl.lit(xi, dtype=pl.Float64)).pow(2) for xi, e in zip(pt, oth))
            <= r
        )
    elif dist == "inf":
        return (
            pl.max_horizontal((e - pl.lit(xi, dtype=pl.Float64)).abs() for xi, e in zip(pt, oth))
            <= r
        )
    elif dist == "cosine":
        x_list = list(pt)
        x_norm = sum(z * z for z in x_list)
        oth_norm = pl.sum_horizontal(e * e for e in oth)
        distN = (
            1.0
            - pl.sum_horizontal(xi * e for xi, e in zip(x_list, oth)) / (x_norm * oth_norm).sqrt()
        )
        return distN <= r
    elif dist in ("h", "haversine"):
        from . import haversine

        pt_as_list = list(pt)
        if (len(pt_as_list) != 2) or (len(oth) < 2):
            raise ValueError(
                "For Haversine distance, input x must have dimension 2 and 2 other columns"
                " must be provided as lat and long."
            )

        y_lat = pl.lit(pt_as_list[0], dtype=pl.Float64)
        y_long = pl.lit(pt_as_list[1], dtype=pl.Float64)
        dist_out = haversine(oth[0], oth[1], y_lat, y_long)
        return dist_out <= r
    else:
        raise ValueError(f"Unknown distance function: {dist}")


def is_knn_from(
    *features: str | pl.Expr,
    pt: Iterable[float],
    k: int,
    dist: Distance = "sql2",
) -> pl.Expr:
    """
    Returns a boolean column that returns points that are k nearest neighbors from the point.

    Parameters
    ----------
    *features : str | pl.Expr
        Other columns used as features
    pt : Iterable[float]
        The point
    k : int
        k nearest neighbor
    dist : Literal[`l1`, `l2`, `sql2`, `inf`]
        Note `sql2` stands for squared l2.
    """
    # For a single point, it is faster to just do it in native polars
    oth = [str_to_expr(x) for x in features]
    if not warn_len_compare(pt, oth):
        raise ValueError("Dimension does not match.")

    if dist == "l1":
        dist_out = pl.sum_horizontal(
            (e - pl.lit(xi, dtype=pl.Float64)).abs() for xi, e in zip(pt, oth)
        )
        return dist_out <= dist_out.bottom_k(k=k).max()
    elif dist in ("l2", "sql2"):
        dist_out = pl.sum_horizontal(
            (e - pl.lit(xi, dtype=pl.Float64)).pow(2) for xi, e in zip(pt, oth)
        )
        return dist_out <= dist_out.bottom_k(k=k).max()
    elif dist == "inf":
        dist_out = pl.max_horizontal(
            (e - pl.lit(xi, dtype=pl.Float64)).abs() for xi, e in zip(pt, oth)
        )
        return dist_out <= dist_out.bottom_k(k=k).max()
    elif dist == "cosine":
        x_list = list(pt)
        x_norm = sum(z * z for z in x_list)
        oth_norm = pl.sum_horizontal(e * e for e in oth)
        dist_out = (
            1.0
            - pl.sum_horizontal(xi * e for xi, e in zip(x_list, oth)) / (x_norm * oth_norm).sqrt()
        )
        return dist_out <= dist_out.bottom_k(k=k).max()
    elif dist in ("h", "haversine"):
        from . import haversine

        pt_as_list = list(pt)
        if (len(pt_as_list) != 2) or (len(oth) < 2):
            raise ValueError(
                "For Haversine distance, input x must have dimension 2 and 2 other columns"
                " must be provided as lat and long."
            )

        y_lat = pl.lit(pt_as_list[0], dtype=pl.Float64)
        y_long = pl.lit(pt_as_list[1], dtype=pl.Float64)
        dist_out = haversine(oth[0], oth[1], y_lat, y_long)
        return dist_out <= dist_out.bottom_k(k=k).max()
    else:
        raise ValueError(f"Unknown distance function: {dist}")


def query_radius_ptwise(
    *features: str | pl.Expr,
    index: str | pl.Expr,
    r: float,
    dist: Distance = "sql2",
    sort: bool = True,
    parallel: bool = False,
) -> pl.Expr:
    """
    Takes the index column, and uses features columns to determine distance, and finds all neighbors
    within distance r from each id. If you only care about neighbor count, you should use
    `query_nb_cnt`, which supports expression for radius and is way faster.

    Note that the index column must be convertible to u32. If you do not have a u32 ID column,
    you can generate one using pl.int_range(..), which should be a step before this.

    Also note that this internally builds a kd-tree for fast querying and deallocates it once we
    are done. If you need to repeatedly run the same query on the same data, then it is not
    ideal to use this. A specialized external kd-tree structure would be better in that case.

    Parameters
    ----------
    *features : str | pl.Expr
        Other columns used as features
    index : str | pl.Expr
        The column used as index, must be castable to u32
    r : float
        The radius. Must be a scalar value now.
    dist : Literal[`l1`, `l2`, `sql2`, `inf`]
        Note `sql2` stands for squared l2.
    sort
        Whether the neighbors returned should be sorted by the distance. Setting this to False can
        improve performance by 10-20%.
    parallel : bool
        Whether to run the k-nearest neighbor query in parallel. This is recommended when you
        are running only this expression, and not in group_by() or over() context.
    """

    if r <= 0.0:
        raise ValueError("Input `r` must be > 0.")
    elif isinstance(r, pl.Expr):
        raise ValueError("Input `r` must be a scalar now. Expression input is not implemented.")

    if dist in ("cosine", "h", "haversine"):
        raise ValueError(f"Distance {dist} doesn't work with current implementation.")

    idx = str_to_expr(index).cast(pl.UInt32).rechunk()
    metric = str(dist).lower()
    cols = [idx]
    cols.extend(str_to_expr(x) for x in features)
    return pl_plugin(
        symbol="pl_query_radius_ptwise",
        args=cols,
        kwargs={"r": r, "metric": metric, "parallel": parallel, "sort": sort},
    )


def query_radius_freq_cnt(
    *features: str | pl.Expr,
    index: str | pl.Expr,
    r: float,
    dist: Distance = "sql2",
    parallel: bool = False,
) -> pl.Expr:
    """
    Takes the index column, and uses features columns to determine distance, finds all neighbors
    within distance r from each index, and finally finds the count of the number of times the point is
    within distance r from other points.

    This calls `query_radius_ptwise` internally. See the docstring of `query_radius_ptwise` for more info.

    Parameters
    ----------
    *features : str | pl.Expr
        Other columns used as features
    index : str | pl.Expr
        The column used as index, must be castable to u32
    r : float
        The radius. Must be a scalar value now.
    dist : Literal[`l1`, `l2`, `sql2`, `inf`]
        Note `sql2` stands for squared l2.
    parallel : bool
        Whether to run the k-nearest neighbor query in parallel. This is recommended when you
        are running only this expression, and not in group_by() or over() context.
    """
    within_radius = query_radius_ptwise(
        *features, index=index, r=r, dist=dist, sort=False, parallel=parallel
    )

    return within_radius.explode().drop_nulls().value_counts(sort=True, parallel=parallel)


def query_nb_cnt(
    *features: str | pl.Expr,
    r: float | str | pl.Expr | Iterable[float],
    dist: Distance = "sql2",
    parallel: bool = False,
) -> pl.Expr:
    """
    Return the number of neighbors within (<=) radius r for each row under the given distance
    metric. The point itself is always a neighbor of itself.

    Parameters
    ----------
    *features : str | pl.Expr
        Other columns used as features
    r : float | Iterable[float] | pl.Expr | str
        If this is a scalar, then it will run the query with fixed radius for all rows. If
        this is a list, then it must have the same height as the dataframe. If
        this is an expression, it must be an expression representing radius. If this is a str,
        it must be the name of a column
    dist : Literal[`l1`, `l2`, `sql2`, `inf`]
        Note `sql2` stands for squared l2.
    parallel : bool
        Whether to run the distance query in parallel. This is recommended when you
        are running only this expression, and not in group_by() or over() context.
    """
    if dist in ("cosine", "h", "haversine"):
        raise ValueError(f"Distance `{dist}` doesn't work with current implementation.")

    if isinstance(r, (float, int)):
        rad = pl.lit(pl.Series(values=[r], dtype=pl.Float64))
    elif isinstance(r, pl.Expr):
        rad = r
    elif isinstance(r, str):
        rad = pl.col(r)
    else:
        rad = pl.lit(pl.Series(values=r, dtype=pl.Float64))

    return pl_plugin(
        symbol="pl_nb_cnt",
        args=[rad] + [str_to_expr(x) for x in features],
        kwargs={
            "k": 0,
            "metric": dist,
            "parallel": parallel,
            "skip_eval": False,
            "skip_data": False,
        },
    )
