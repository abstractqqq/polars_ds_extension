import polars as pl


@pl.api.register_expr_namespace("graph")
class GraphExt:
    """
    This class contains tools for working with graphs inside a dataframe. Graphs are represented by two columns:
    one node column (index, u64) and one edge column (list[u64]).

    Polars Namespace: graph

    Example: ...
    """

    def __init__(self, expr: pl.Expr):
        self._expr: pl.Expr = expr

    def deg(self) -> pl.Expr:
        """
        Treat self as `edges` and return the degree of each node. Note that this is simply an alias
        of `pl.col("edges").list.len()`.
        """
        return self._expr.list.len()
