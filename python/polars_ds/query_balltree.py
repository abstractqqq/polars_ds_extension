# from __future__ import annotations
# from typing import Iterable
# import polars as pl
# from .type_alias import StrOrExpr, str_to_expr, Distance
# from ._utils import pl_plugin


# def query_bt_knn_ptwise(
#     *features: str | pl.Expr,
#     index: str | pl.Expr,
#     r: float,
#     distance_metric: Distance = "euclidean",
#     sort: bool = True,
#     parallel: bool = False,
#     k: int = None,
#     return_dist: bool = False,
# ) -> pl.Expr:
#     """
#     Takes an index column, uses the feature columns to determine distance and finds all neighbors
#     within distance r from each id. It returns a list containing (neighbor_id, distance) tuples.
#     This is bounded by max k neighbors.
#     index columns must be convertible to u32

#     Parameters
#     ----------
#     *features : str | pl.Expr
#         The feature columns.
#     index : str | pl.Expr
#         The index column.
#     r : float
#         The radius.
#     distance_metric : Distance, optional
#         The distance metric to use, by default "euclidean". Currently the only available options are "euclidean" and "haversine".
#     sort : bool, optional
#         Whether to sort the output, by default True.
#     parallel : bool, optional
#         Whether to evaluate this in parallel, by default False.
#     k : int, optional
#     """
#     if r < 0:
#         raise ValueError("r must be positive")
#     elif isinstance(r, pl.Expr):
#         raise ValueError(
#             "r must be a scalar float. Expressions are not supported.")

#     if distance_metric.lower() not in ("euclidean", "haversine"):
#         raise ValueError(
#             "Invalid distance metric. Must be 'euclidean' or 'haversine'.")
#     if len(features) == 0:
#         raise ValueError("Must provide at least one feature column.")
#     elif len(features) != 2 and distance_metric == "haversine":
#         raise ValueError("Haversine distance requires exactly 2 features.")

#     idx = str_to_expr(index).cast(pl.UInt32).rechunk()

#     if not k:
#         # the length of any of the feature columns
#         k = pl.col(features[0]).len()
#     if k < 1:
#         raise ValueError("k must be positive.")

#     # Columns to send over to Rust as &[Series]
#     # First column will always be the index column
#     cols = [idx]
#     cols.extend([str_to_expr(f) for f in features])
#     if return_dist:
#         return pl_plugin(
#             symbol="pl_query_knn_ptwise_wdist",
#             args=cols,
#             kwargs={"r": r, "distance_metric": distance_metric.lower(
#             ), "sort": sort, "parallel": parallel, "k": k},
#             is_elementwise=False,
#         )
#     else:
#         return pl_plugin(
#             symbol="pl_query_knn_ptwise",
#             args=cols,
#             kwargs={"r": r, "distance_metric": distance_metric.lower(
#             ), "sort": sort, "parallel": parallel, "k": k},
#             is_elementwise=False,
#         )


# def query_bt_knn_radius_freq_cnt(
#     *features: str | pl.Expr,
#     index: str | pl.Expr,
#     r: float = None,
#     distance_metric: Distance = "euclidean",
#     sort: bool = True,
#     parallel: bool = False,
#     k: int = None,
# ) -> pl.Expr:
#     """
#     Returns the frequency count of neighbors within distance r, and within k nearest neighbors for each row using the given distance metric.
#     The point is always a neighbor of itself.
#     We need an index column that can be cast to u32.

#     Parameters
#     ----------

#     *features : str | pl.Expr
#         The feature columns.
#     index : str | pl.Expr
#         The index column.
#     r : float, optional
#         The radius. If None, it will be set to infinity, by default None.
#     distance_metric : Distance, optional
#         The distance metric to use, by default "euclidean". Currently the only available options are "euclidean" and "haversine".
#     sort : bool, optional
#         Whether to sort the output, by default True.
#     parallel : bool, optional
#         Whether to evaluate this in parallel, by default False.
#     k : int, optional
#         The number of nearest neighbors to consider, by default 1.
#     """
#     if not k:
#         # the length of any of the feature columns
#         k = pl.col(features[0]).len()
#     if not r:
#         r = float("inf")
#     if r < 0:
#         raise ValueError("r must be positive")
#     elif isinstance(r, pl.Expr):
#         raise ValueError(
#             "r must be a scalar float. Expressions are not supported.")

#     if distance_metric.lower() not in ("euclidean", "haversine"):
#         raise ValueError(
#             "Invalid distance metric. Must be 'euclidean' or 'haversine'.")
#     if len(features) == 0:
#         raise ValueError("Must provide at least one feature column.")
#     elif len(features) != 2 and distance_metric == "haversine":
#         raise ValueError("Haversine distance requires exactly 2 features.")

#     if k < 1:
#         raise ValueError("k must be positive.")
#     idx = str_to_expr(index).cast(pl.UInt32).rechunk()
#     # Columns to send over to Rust as &[Series]
#     # First column will always be the index column
#     cols = [idx]
#     cols.extend([str_to_expr(f) for f in features])
#     knn_expr: pl.Expr = query_bt_knn_ptwise(
#         *features, index=index, r=r, distance_metric=distance_metric, sort=sort, parallel=parallel, k=k, return_dist=False)
#     return knn_expr.explode().drop_nulls().value_counts(sort=True, parallel=parallel)


# def query_bt_knn_avg(
#     *features: str | pl.Expr,
#     index: str | pl.Expr,
#     r: float,
#     distance_metric: Distance = "euclidean",
#     sort: bool = True,
#     parallel: bool = False,
#     k: int = None,
# ) -> pl.Expr:
#     """Takes an index column, uses the feature columns to determine distance and finds all neighbors
#     within distance r from each id. It returns a list containing (neighbor_id, distance) tuples.
#     index columns must be convertible to u32

#     Args:
#         index (str | pl.Expr): _description_
#         r (float): _description_
#         dist (Distance, optional): _description_. Defaults to "euclidian".
#         sort (bool, optional): _description_. Defaults to True.
#         parallel (bool, optional): _description_. Defaults to False.

#     Returns:
#         pl.Expr: _description_
#     """
#     if not k:
#         # the length of any of the feature columns
#         k = pl.col(features[0]).len()
#     if r < 0:
#         raise ValueError("r must be positive")
#     elif isinstance(r, pl.Expr):
#         raise ValueError(
#             "r must be a scalar float. Expressions are not supported.")

#     if distance_metric.lower() not in ("euclidean", "haversine"):
#         raise ValueError(
#             "Invalid distance metric. Must be 'euclidean' or 'haversine'.")
#     if len(features) == 0:
#         raise ValueError("Must provide at least one feature column.")
#     elif len(features) != 2 and distance_metric == "haversine":
#         raise ValueError("Haversine distance requires exactly 2 features.")

#     if k < 1:
#         raise ValueError("k must be positive.")
#     idx = str_to_expr(index).cast(pl.UInt32).rechunk()
#     # Columns to send over to Rust as &[Series]
#     # First column will always be the index column
#     cols = [idx]
#     cols.extend([str_to_expr(f) for f in features])
#     return pl_plugin(
#         symbol="pl_ball_tree_knn_avg",
#         args=cols,
#         kwargs={"r": r, "distance_metric": distance_metric.lower(
#         ), "sort": sort, "parallel": parallel, "k": k},
#         is_elementwise=False,
#     )


# def bt_within_dist_from(
#     *features: str | pl.Expr,
#     pt: Iterable[float],
#     r: float | str | pl.Expr,
#     distance_metric: Distance = "euclidean",
#     parallel: bool = False,
# ) -> pl.Expr:
#     """
#     Returns a boolean column indicating if the provided point is within distance r.

#     Parameters
#     ----------
#     *features : str | pl.Expr
#         The feature columns.
#     pt : Iterable[float]
#         The point to compare against.
#     r : float | str | pl.Expr
#         The radius. Either a scalar float, or a 1d array with len = row_count(X).
#     distance_metric : Distance, optional
#         The distance metric to use, by default "euclidean". Currently the only available
#         options are "euclidean" and "haversine".
#     """
#     if distance_metric.lower() not in ("euclidean", "haversine"):
#         raise ValueError(
#             "Invalid distance metric. Must be 'euclidean' or 'haversine'.")
#     if len(features) < 2:
#         raise ValueError("Must provide at least two feature columns.")
#     if len(pt) != len(features):
#         raise ValueError(
#             "Number of features must match the number of dimensions")

#     if isinstance(r, (float, int)):
#         rad = pl.lit(pl.Series(values=[r], dtype=pl.Float64))
#     elif isinstance(r, pl.Expr):
#         rad = r
#     elif isinstance(r, str):
#         rad = pl.col(r)
#     else:
#         rad = pl.lit(pl.Series(values=r, dtype=pl.Float64))
#     cols = [rad]
#     cols.extend([str_to_expr(f) for f in features])
#     return pl_plugin(
#         symbol="pl_bt_within_dist_from",
#         args=cols,
#         kwargs={"point": pt, "distance_metric": distance_metric.lower(),
#                 "parallel": parallel},
#     )


# def is_bt_knn_from(
#     *features,
#     pt: Iterable[float],
#     k: int,
#     distance_metric: Distance = "euclidean",
#     parallel: bool = False,
#     epsilon: float = None,
# ) -> pl.Expr:
#     """
#     Returns a boolean column indicating if the provided point is within the k-nearest neighbors.

#     Parameters
#     ----------
#     *features : str | pl.Expr
#         The feature columns.
#     pt : Iterable[float]
#         The point to compare against.
#     k : int
#         Number of neighbors to consider.
#     distance_metric : Distance, optional
#         The distance metric to use, by default "euclidean". Currently the only available
#         options are "euclidean" and "haversine".
#     parallel : bool, optional
#     """
#     if distance_metric.lower() not in ("euclidean", "haversine"):
#         raise ValueError(
#             "Invalid distance metric. Must be 'euclidean' or 'haversine'.")
#     if len(features) < 2:
#         raise ValueError("Must provide at least two feature columns.")
#     if len(pt) != len(features):
#         raise ValueError(
#             "Number of features must match the number of dimensions")
#     if k < 1:
#         raise ValueError("k must be positive.")
#     cols = [str_to_expr(f) for f in features]
#     return pl_plugin(
#         symbol="pl_bt_knn_from",
#         args=cols,
#         kwargs={"point": pt, "distance_metric": distance_metric.lower(
#         ), "parallel": parallel, "k": k, "epsilon": epsilon},
#     )


# def query_bt_nb_cnt(
#     *features: str | pl.Expr,
#     r: float | str | Iterable[float],
#     index: str | pl.Expr,
#     distance_metric: Distance = "euclidean",
#     parallel: bool = False
# ) -> pl.Expr:
#     """
#     Returns the number of neighbors within ( <= ) radius r for each row using the given distance metric.
#     The point is always a neighbor of itself.

#     Parameters
#     ----------
#     r : float | str | Iterable[float]
#         The radius. Either a scalar float, or a 1d array with len = row_count(X).
#     index : str | pl.Expr
#         The index column.
#     *features : str | pl.Expr
#         The feature columns.
#     distance_metric : Distance, optional
#         The distance metric to use, by default "euclidean". Currently the only available
#         options are "euclidean" and "haversine".
#     parallel : bool, optional
#         Whether to evaluate this in parallel, by default False
#     """
#     if distance_metric.lower() not in ("euclidean", "haversine"):
#         raise ValueError(
#             "Invalid distance metric. Must be 'euclidean' or 'haversine'.")
#     if len(features) == 0:
#         raise ValueError("Must provide at least one feature column.")
#     elif len(features) != 2 and distance_metric == "haversine":
#         raise ValueError("Haversine distance requires exactly 2 features.")

#     if isinstance(r, (float, int)):
#         rad = pl.lit(pl.Series(values=[r], dtype=pl.Float64))
#     elif isinstance(r, pl.Expr):
#         rad = r
#     elif isinstance(r, str):
#         rad = pl.col(r)
#     else:
#         rad = pl.lit(pl.Series(values=r, dtype=pl.Float64))

#     idx = str_to_expr(index).cast(pl.UInt32).rechunk()
#     cols = [idx, rad]
#     cols.extend([str_to_expr(f) for f in features])
#     return pl_plugin(
#         symbol="pl_nb_count",
#         args=cols,
#         kwargs={
#             "distance_metric": distance_metric.lower(),
#             "parallel": parallel,
#             # Stubbed values so we can use the
#             # same kwargs struct in rust
#             "sort": True,
#             "r": 0,
#             "k": 1,
#         },
#     )


# def query_bt_radius_ptwise(
#     *features: str | pl.Expr,
#     index: str | pl.Expr,
#     r: float,
#     distance_metric: Distance = "euclidean",
#     sort: bool = True,
#     parallel: bool = False,
#     k=None,
#     return_dist: bool = False,
# ) -> pl.Expr:
#     """
#     Returns a list of neighbors within distance r from each id.
#     index columns must be convertible to u32

#     Parameters
#     ----------
#     *features : str | pl.Expr
#         The feature columns.
#     index : str | pl.Expr
#         The index column.
#     r : float
#         The radius.
#     distance_metric : Distance, optional
#         The distance metric to use, by default "euclidean". Currently the only available options are "euclidean" and "haversine".
#     sort : bool, optional
#         Whether to sort the output, by default True.
#     parallel : bool, optional
#         Whether to evaluate this in parallel, by default False.
#     k : int, optional
#         Max number of neighbors to consider.
#     """
#     return query_bt_knn_ptwise(*features, index=index, r=r, distance_metric=distance_metric, sort=sort, parallel=parallel, k=k, return_dist=return_dist)


# def query_bt_radius_freq_cnt(
#     *features: str | pl.Expr,
#     index: str | pl.Expr,
#     r: float = None,
#     distance_metric: Distance = "euclidean",
#     sort: bool = True,
#     parallel: bool = False,
#     k: int = None,
# ) -> pl.Expr:
#     """
#     Returns the frequency count of neighbors within distance r, and within k nearest neighbors for each row using the given distance metric.
#     The point is always a neighbor of itself.
#     We need an index column that can be cast to u32.

#     Parameters
#     ----------

#     *features : str | pl.Expr
#         The feature columns.
#     index : str | pl.Expr
#         The index column.
#     r : float, optional
#         The radius. If None, it will be set to infinity, by default None.
#     distance_metric : Distance, optional
#         The distance metric to use, by default "euclidean". Currently the only available options are "euclidean" and "haversine".
#     sort : bool, optional
#         Whether to sort the output, by default True.
#     parallel : bool, optional
#         Whether to evaluate this in parallel, by default False.
#     k : int, optional
#         The number of nearest neighbors to consider, by default 1.
#     """
#     return query_bt_knn_radius_freq_cnt(*features, index=index, r=r, distance_metric=distance_metric, sort=sort, parallel=parallel, k=k)
