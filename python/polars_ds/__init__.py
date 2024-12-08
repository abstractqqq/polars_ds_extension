from __future__ import annotations
import polars as pl
from ._utils import str_to_expr

from polars_ds.num import *  # noqa: F403
from polars_ds.metrics import *  # noqa: F403
from polars_ds.stats import *  # noqa: F403
from polars_ds.string import *  # noqa: F403
from polars_ds.ts_features import *  # noqa: F403
from polars_ds.expr_knn import *  # noqa: F403
from polars_ds.expr_linear import *  # noqa: F403

__version__ = "0.6.3"

def frame(size: int = 2_000, index_name: str = "row_num") -> pl.DataFrame:
    """
    Generates a frame with only an index (row number) column.
    This is a convenience function to be chained with pds.random(...) when running simulations and tests.

    Parameters
    ----------
    size
        The total number of rows in this dataframe
    index_name
        The name of the index column
    """
    return pl.DataFrame({index_name: range(size)})
