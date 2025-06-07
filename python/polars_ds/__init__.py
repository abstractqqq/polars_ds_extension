from __future__ import annotations
import importlib.metadata
import polars as pl

__version__ = importlib.metadata.version("polars_ds")

# Internal dependencies
from polars_ds.exprs import *  # noqa F403

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
