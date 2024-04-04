from __future__ import annotations
import polars as pl
from typing import Union, Optional
from .type_alias import str_to_expr, StrOrExpr
from polars.utils.udfs import _get_shared_lib_location
from ._utils import pl_plugin

_lib = _get_shared_lib_location(__file__)

# @pl.api.register_expr_namespace("graph")
# class GraphExt:
#     """
#     This class contains tools for working with graphs inside a dataframe. Graphs are represented by two columns:
#     one node column (index, u32), and another u32 column representing connections (links).

#     Also note that some queries (e.g. shortest path) in this module is slow and is transcient, meaning that each
#     query will create a graph and use it only for the duration of the query. It does not persist
#     the graph. So it will be very slow if you are running multiple graph methods. The focus is not full coverage
#     of graph algorithms, because not all are suitable for dataframe queries. This module mainly provides the
#     convenience for common queires like deg, shortest path, etc.

#     To be decided: a separate, dedicated Graph module might be appropriate.

#     Polars Namespace: graph

#     Example: pl.col("node").graph.reachable(link = pl.col("connected_to"), target = 3)
#     """

#     def __init__(self, expr: pl.Expr):
#         self._expr: pl.Expr = expr

#     # def eigen_centrality(
#     #     self, n_iter: int = 15, normalize: bool = True, sparse: bool = True
#     # ) -> pl.Expr:
#     #     """
#     #     Treats self as a column of "edges" and computes the eigenvector centrality for the graph.

#     #     Self must be a column of list[u64]. It is the user's responsibility to ensure that edge list
#     #     does not contain duplicate node ids.

#     #     Parameters
#     #     ----------
#     #     n_iter
#     #         The number of iterations for the power iteration algorithm to compute eigenvecor centrality
#     #     normalize
#     #         Whether to normalize the eigenvector's elements by dividing by their sum
#     #     sparse
#     #         Whether the underlying adjacent matrix will be sparse or not. This is usually the case, and
#     #         using sparse matrix we can compute this significantly faster.
#     #     """
#     #     return self._expr.register_plugin(
#     #         lib=_lib,
#     #         symbol="pl_eigen_centrality",
#     #         kwargs={"n_iter": n_iter, "normalize": normalize, "sparse": sparse},
#     #         is_elementwise=True,
#     #     )


def query_shortest_path(
    node: StrOrExpr,
    link: StrOrExpr,
    target: Union[int, pl.Expr],
    cost: Optional[StrOrExpr] = None,
    parallel: bool = False,
) -> pl.Expr:
    """
    Treats self as a column of "edges" and computes the shortest path to the target by using the
    cost provided. This will treat the graph as a directed graph, and the edge (i, j) may have different
    cost than (j, i), depending on the data. This assumes all costs are positive.

    Note that this will not sort the nodes for the user. This assumes that nodes are indexed by the
    row numbers: 0, 1, ...

    Also note that if in a row, we have len(edges) != len(dist) or if either edge list has null or
    cost list has null, then this row will be disqualified, meaning that the node will be
    treated as having no connected edges.

    If you
    (1) do not need the actual path,
    (2) is fine with having constant cost for all edges, and
    (3) only cares about the cost of the shortest path,
    then `reachable` may be a better and much faster approach.

    Parameters
    ----------
    node
        A u32 column representing node identifiers
    link
        A polars expression representing connections (links)
    target
        If this is an int, then will try to find the shortest path from this node to the node with
        index = target. (NOT YET DECIDED): If this is an expression, then the expression must be a column
        of u64 representing points go from this node.
    cost
        If none, will use constant 1 cost for travelling from any edge. If this is an str or expression, it should
        represent a column of list[f64], with each list having the same length as
        the edge list, and they must not contain nulls. If these conditions are not met, this node will be
        treated as having no edges going from it. The values in the list will represent the cost (distance) to go
        from this node to the nodes in the edge list.
    parallel
        Whether to run the algorithm in parallel. The same cavaet as the one in value_counts() applies.
    """
    nd = str_to_expr(node)
    lk = str_to_expr(link)

    par = pl.lit(parallel, pl.Boolean)
    c = None if cost is None else str_to_expr(cost)
    if isinstance(target, int):
        to = pl.lit(target, pl.UInt32)
    else:
        return NotImplemented

    if c is None:  # use const cost impl
        return pl_plugin(
            lib=_lib,
            symbol="pl_shortest_path_const_cost",
            args=[nd, lk, to, par],
            changes_length=True,
        )
    else:
        return pl_plugin(
            lib=_lib,
            symbol="pl_shortest_path",
            args=[nd, lk, c, to, par],
            changes_length=True,
        )


def query_node_reachable(node: StrOrExpr, link: StrOrExpr, target: int) -> pl.Expr:
    """
    Determines whether the `target` node is reachable starting from all other nodes. It will return
    a struct with 3 fields: the first (node) is just the node, the second (reachable) tells if the node is reachable,
    and the third tells the length of the shortest path assuming constant edge cost.

    If the node is reachable and has steps = 0, the node must be the node in question itself.
    If reachable = False, and steps = 0, then it simply means the node is not reachable.

    Parameters
    ----------
    node
        A u32 column representing node identifiers
    link
        A polars expression representing connections (links)
    target
        An integer index for the node that we are insterested in testing reachability.
    """
    nd = str_to_expr(node)
    lk = str_to_expr(link)

    if target < 0:
        raise ValueError("Value for `target` can only be non-negative integer.")

    return pl_plugin(
        lib=_lib,
        symbol="pl_shortest_path_dijkstra",
        args=[nd, lk, pl.lit(target, pl.UInt32)],
        changes_length=True,
    )


def query_node_deg(node: StrOrExpr, link: StrOrExpr, directed: bool = False) -> pl.Expr:
    """
    Queries node degrees.

    Parameters
    ----------
    node
        A u32 column representing node identifiers
    link
        A polars expression representing connections (links)
    directed
        If true, will return in and out degree.
    """
    nd = str_to_expr(node)
    lk = str_to_expr(link)
    if directed:
        return pl_plugin(
            lib=_lib,
            symbol="pl_graph_deg",
            args=[nd, lk],
            changes_length=True,
        )
    else:
        return pl_plugin(
            lib=_lib,
            symbol="pl_graph_in_out_deg",
            args=[nd, link],
            changes_length=True,
        )
