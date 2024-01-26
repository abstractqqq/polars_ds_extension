import polars as pl
from typing import Union, Optional
from polars.utils.udfs import _get_shared_lib_location

_lib = _get_shared_lib_location(__file__)


@pl.api.register_expr_namespace("graph")
class GraphExt:
    """
    This class contains tools for working with graphs inside a dataframe. Graphs are represented by two columns:
    one node column (index, u64), which is usually implicit, and one edge column (list[u64]). Most algorithms
    here implicitly assumes the nodes are indexed by row number starting from 0.

    This module will only focus on undirected graph for now.

    Polars Namespace: graph

    Example: pl.col("neighbors").graph.eigen_centrality()
    """

    def __init__(self, expr: pl.Expr):
        self._expr: pl.Expr = expr

    def deg(self) -> pl.Expr:
        """
        Treat self as a column of "edges" and return the degree of each node in self. Note that this
        is simply an alias of `pl.col("edges").list.len()`.
        """
        return self._expr.list.len()

    def eigen_centrality(self, n_iter: int = 20, normalize: bool = True) -> pl.Expr:
        """
        Treats self as a column of "edges" and computes the eigenvector centrality for the graph.

        Self must be a column of list[u64].

        Note that this will not sort the nodes for the user. This assumes that nodes are indexed by the
        natural numbers: 0, 1, ... If nodes are not sorted or if the u64 in edge list does not refer to
        the node's index, the result may be incorrect or may thrown an error.

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

    def shortest_path(self, target: Union[int, pl.Expr], distance: Optional[pl.Expr] = None):
        """
        Treats self as a column of "edges" and computes the shortest path to the target.

        Self must be a column of list[u64].

        Note that this will not sort the nodes for the user. This assumes that nodes are indexed by the
        natural numbers: 0, 1, ... If nodes are not sorted or if the u64 in edge list does not refer to
        the node's index, the result may be incorrect or may thrown an error.

        Parameters
        ----------
        target
            If this is an int, then will try to find the shortest path from this node to the node with
            index = target. Tentative: If this is an expression, then the expression must be a column
            of u64 representing points go from this node.
        distance
            If none, constant cost will be applied to each edge. Tentative: If it is an expression, then
            it must represent a column of list[f64], with each list having the same length as the edge list.
            The values in the list will represent the cost (distance) to go from this node to the node in the
            edge list.
        """

        if isinstance(target, int):
            to = pl.lit(target, pl.UInt64)
        else:
            # to = target
            return NotImplemented

        if distance is None:
            return self._expr.register_plugin(
                lib=_lib,
                symbol="pl_shortest_path_const_cost",
                args=[to],
                is_elementwise=True,
            )
        else:
            return NotImplemented
