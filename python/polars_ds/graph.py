from __future__ import annotations
import polars as pl
from typing import Union
from polars.utils.udfs import _get_shared_lib_location

_lib = _get_shared_lib_location(__file__)


@pl.api.register_expr_namespace("graph")
class GraphExt:
    """
    This class contains tools for working with graphs inside a dataframe. Graphs are represented by two columns:
    one node column (index, u64), which is usually implicit, and one edge column (list[u64]). Most algorithms
    here implicitly assumes the nodes are indexed by row number starting from 0. On row 5, say the value of the
    column of edges is [1,6,13], that means we have a connection from 5 to 1, 5 to 6, and 5 to 13.

    This module will only focus on undirected graph for now. Also note that this module is very slow and is transcient,
    meaning that each query will create a graph and use it only for the duration of the query. It does not persist
    the graph. So it will be very slow if you are running multiple graph methods. This module mainly provides the
    convenience. As a result, a lot of graph algorithms will not be implemented and won't have the most versatile API.

    To be decided: a separate, dedicated Graph module might be appropriate.

    Polars Namespace: graph

    Example: pl.col("neighbors").graph.eigen_centrality()
    """

    def __init__(self, expr: pl.Expr):
        self._expr: pl.Expr = expr

    def deg(self) -> pl.Expr:
        """
        Computes degree of each node. This will treat self as a column of "edges" and considers the graph undirected.

        Note that this will not sort the nodes for the user. This assumes that nodes
        are indexed by the natural numbers: 0, 1, ... If nodes are not sorted or if the u64 in edge list does not refer to
        the node's index, the result may be incorrect or may throw an error.
        """
        return self._expr.register_plugin(
            lib=_lib,
            symbol="pl_graph_deg",
            is_elementwise=True,
        )

    def in_out_deg(self) -> pl.Expr:
        """
        Computes in and out degree of each node. This will treat self as a column of "edges" and considers the graph directed.

        Note that this will not sort the nodes for the user. This assumes that nodes
        are indexed by the natural numbers: 0, 1, ... If nodes are not sorted or if the u64 in edge list does not refer to
        the node's index, the result may be incorrect or may throw an error.
        """
        return self._expr.register_plugin(
            lib=_lib,
            symbol="pl_graph_in_out_deg",
            is_elementwise=True,
        )

    def eigen_centrality(self, n_iter: int = 15, normalize: bool = True) -> pl.Expr:
        """
        Treats self as a column of "edges" and computes the eigenvector centrality for the graph.

        Self must be a column of list[u64].

        Note that this will not sort the nodes for the user. This assumes that nodes are indexed by the
        natural numbers: 0, 1, ... If nodes are not sorted or if the u64 in edge list does not refer to
        the node's index, the result may be incorrect or may throw an error.

        Parameters
        ----------
        n_iter
            The number of iterations for the power iteration algorithm to compute eigenvecor centrality
        normalize
            Whether to normalize the eigenvector's elements by dividing by their sum
        """
        return self._expr.register_plugin(
            lib=_lib,
            symbol="pl_eigen_centrality",
            kwargs={"n_iter": n_iter, "normalize": normalize},
            is_elementwise=True,
        )

    def shortest_path_const_cost(
        self, target: Union[int, pl.Expr], parallel: bool = False
    ) -> pl.Expr:
        """
        Treats self as a column of "edges" and computes the shortest path to the target assuming that
        the cost of traveling every edge is constant.

        Note that this will not sort the nodes for the user. This assumes that nodes are indexed by the
        row numbers: 0, 1, ...

        Parameters
        ----------
        target
            If this is an int, then will try to find the shortest path from this node to the node with
            index = target. (NOT YET DECIDED): If this is an expression, then the expression must be a column
            of u64 representing points go from this node.
        parallel
            Whether to run the algorithm in parallel. The same cavaet as the one in value_counts() applies.
        """

        par = pl.lit(parallel, pl.Boolean)
        if isinstance(target, int):
            to = pl.lit(target, pl.UInt64)
        else:
            return NotImplemented

        return self._expr.register_plugin(
            lib=_lib,
            symbol="pl_shortest_path_const_cost",
            args=[to, par],
            is_elementwise=True,
        )

    def shortest_path(
        self, target: Union[int, pl.Expr], cost: pl.Expr, parallel: bool = False
    ) -> pl.Expr:
        """
        Treats self as a column of "edges" and computes the shortest path to the target by using the
        cost provided. This will treat the graph as a directed graph, and the edge (i, j) may have different
        cost than (j, i), depending on the data.

        Note that this will not sort the nodes for the user. This assumes that nodes are indexed by the
        row numbers: 0, 1, ...

        Also note that if in a row, we have len(edges) != len(dist) or if either edge list has null or
        cost list has null, then this row will be disqualified, meaning that the node will be
        treated as having no connected edges.

        Parameters
        ----------
        target
            If this is an int, then will try to find the shortest path from this node to the node with
            index = target. (NOT YET DECIDED): If this is an expression, then the expression must be a column
            of u64 representing points go from this node.
        distance
            An expression representing a column of list[f64], with each list having the same length as
            the edge list. The values in the list will represent the cost (distance) to go from this node
            to the node in the edge list.
        parallel
            Whether to run the algorithm in parallel. The same cavaet as the one in value_counts() applies.
        """

        par = pl.lit(parallel, pl.Boolean)
        if isinstance(target, int):
            to = pl.lit(target, pl.UInt64)
        else:
            return NotImplemented

        return self._expr.register_plugin(
            lib=_lib,
            symbol="pl_shortest_path",
            args=[cost, to, par],
            is_elementwise=True,
        )
