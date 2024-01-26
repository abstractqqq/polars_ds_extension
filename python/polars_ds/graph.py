import polars as pl
from polars.utils.udfs import _get_shared_lib_location

_lib = _get_shared_lib_location(__file__)


@pl.api.register_expr_namespace("graph")
class GraphExt:
    """
    This class contains tools for working with graphs inside a dataframe. Graphs are represented by two columns:
    one node column (index, u64) and one edge column (list[u64]). Most algorithms here implicitly assumes
    the nodes are indexed by row number.

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

        Note that this will not sort the nodes for the user. If nodes are not sorted or if the u64
        in edge list does not refer to the node's index, the result may be incorrect.

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
