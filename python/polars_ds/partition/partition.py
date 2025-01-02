from __future__ import annotations

import polars as pl
import polars.selectors as cs
import warnings
import sys

from .._utils import _IS_POLARS_V1
if _IS_POLARS_V1:
    from polars._typing import IntoExpr
else:
    raise ValueError("You must be on Polars >= v1.0.0 to use this module.")

# Typing
from collections.abc import Callable
if sys.version_info >= (3, 11):
    from typing import Self
else:  # 3.10, 3.9, 3.8
    from typing_extensions import Self
from ..typing import PolarsFrame
from typing import List, Dict, Any

class PartitionResult():
    """
    A transitory convenience class.
    """

    def __init__(
        self, 
        df: PolarsFrame, 
        by: str | List[str] | None, 
        separator: str = "|",
        whole_df_name: str = "df"
    ):
        """
        Creates a Partition Result

        Parameters
        ----------
        df
            Either a Polars dataframe or a Lazyframe
        by
            Either None, or a string or a list of strings representing column names. If by
            is None, the entire df will be considered a partition.
        separator 
            Separator for concatenating the names of different parts, if the partition is done 
            by multiple columns
        whole_df_name
            If by is None, the name for the whole df.
        """
        if by is None:
            self.parts: Dict[str, pl.DataFrame] = {whole_df_name: df.lazy().collect()}
        else:
            cols = df.select(
                (cs.by_name(by)) & (cs.string() | cs.categorical())
            ).collect_schema().names()

            all_ok = cols[0] == by if isinstance(by, str) else sorted(cols) == sorted(by)
            if not all_ok:
                raise ValueError("Currently this only supports partitions by str or categorical columns.")

            self.parts = {
                separator.join((str(k) for k in keys)): value
                for keys, value in df.lazy().collect().partition_by(by=by, as_dict=True).items()
            }

    def __repr__(self) -> str:
        output = ""
        for part, df in self.parts.items():
            output += f"Paritition: {part}\n"
            output += df.__repr__() + "\n"
        return output

    def head(self, n:int = 5) -> Dict[str, pl.DataFrame]:
        return {k: df.head(n) for k, df in self.parts.items()}

    def parts(self) -> List[str]:
        return list(self.parts.keys())

    def apply(self, func: Callable[[str, pl.DataFrame], Any]) -> Dict[str, Any]:
        """
        Apply the function to all of the parts in the partition.

        Parameters
        ----------
        func
            A function that takes in a str and a pl.DataFrame and outputs anything. The string
            represents the name of the segment. Note: this is usually a partial function with 
            all other arguments provided.
        """
        output = {}
        for part, df in self.parts.items():
            try:
                output[part] = func(part, df)
            except Exception as e:
                warnings.warn(
                    f"An error occured while processing for the part: {part}. This partition is omitted.\nOriginal Error Message: {e}"
                    , stacklevel = 2
                )

        return output