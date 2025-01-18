import polars as pl
import numpy as np
from typing import Any, Callable
import polars_ds as pds

# Everything in __init__.py of polars_ds that this shouldn't be able to call
CANNOT_CALL = {
    "frame",
    "str_to_expr",
    "pl",
    "annotations",
    "__version__",
    "warn_len_compare"
}

__all__ = ["compat"]

class _Compat():

    @staticmethod
    def _try_into_series(x:Any, name:str) -> Any:
        """
        Try to map the input to a Polars Series by going through a NumPy array. If
        this is not possible, return the original input.
        """
        if isinstance(x, np.ndarray):
            return pl.lit(pl.Series(name=name, values=x))
        elif isinstance(x, pl.Series):
            return pl.lit(x)
        elif hasattr(x, "__array__"):
            return pl.lit(pl.Series(name=name, values=x.__array__()))
        else: 
            return x

    def __getattr__(self, name:str) -> pl.Series:
        if name in CANNOT_CALL:
            raise ValueError(f"`{name}` exists but doesn't work in compat mode.")

        func = getattr(pds, name)
        def compat_wrapper(*args, **kwargs) -> Callable:
            positionals = list(args)
            if len(positionals) <= 0:
                raise ValueError("There must be at least 1 positional argument!")
            
            new_args = (
                _Compat._try_into_series(x, name = str(i))
                for i, x in enumerate(positionals)
            )
            new_kwargs = {
                n: _Compat._try_into_series(v, name = n)
                for n, v in kwargs.items()
            }
            # An eager df, drop output col, so a pl.Series
            return (
                pl.select(
                    func(*new_args, **new_kwargs).alias("__output__")
                ).drop_in_place("__output__")
                .rename(name.replace("query_", ""))
            )

        return compat_wrapper

compat: _Compat = _Compat()

