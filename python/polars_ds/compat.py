import polars as pl
import numpy as np
import sys
from typing import Any, Callable
if sys.version_info >= (3, 11):
    from typing import Self
else:  # 3.10, 3.9, 3.8
    from typing_extensions import Self

class Compat():

    @staticmethod
    def try_into_series(x:Any, name:str) -> Any:
        """
        Try to map the input to a Polars Series by going through a NumPy array. If
        this is not possible, return the original input.
        """
        if isinstance(x, np.ndarray):
            return pl.lit(pl.Series(name=name, values=x))
        elif hasattr(x, "__array__"):
            return pl.lit(pl.Series(name=name, values=x.__array__()))
        else: 
            return x

    @classmethod
    def register(cls, pds_module:Any) -> Self:
        """
        Registers a Polars DS compatibility layer by passing in the alias to the PDS package.

        Parameters
        ----------
        pds_module
            If polars_ds is imported as `import polars_ds as pds`, then the variable alias `pds`
            should be passed. 
        """
        return cls(pds_module)

    def __init__(self, pds_module:Any):
        """
        Initiates a Polars DS compatibility layer by passing in the alias to the PDS package.

        Parameters
        ----------
        pds_module
            If polars_ds is imported as `import polars_ds as pds`, then the variable alias `pds`
            should be passed. 
        """
        import warnings
        warnings.warn(
            "This class is considered unstable. Functions requiring complicated inputs "
            "may not work as intended. Use at your caution."
            , stacklevel = 2
        )

        self._pds = pds_module

    def __getattr__(self, name:str) -> Any:
        func = getattr(self._pds, name)
        def compatibility_wrapper(*args, **kwargs) -> Callable:
            positionals = list(args)
            if len(positionals) <= 0:
                raise ValueError("There must be at least 1 positional argument!")
            
            first = positionals[0]
            if not (hasattr(first, "__array__") or isinstance(first, np.ndarray)):
                raise ValueError("The first positional argument must be convertible to a NumPy array.")

            new_args = (
                Compat.try_into_series(x, name = str(i))
                for i, x in enumerate(positionals)
            )
            pl_result = pl.select(func(*new_args, **kwargs).alias("__output__"))
            result = first.__class__(pl_result.drop_in_place("__output__").to_numpy())
            if hasattr(result, "rename"):
                return result.rename(name.replace("query_", ""))
            return result

        return compatibility_wrapper