import polars as pl
from typing import Any, Optional, List, Dict


def pl_plugin(
    *,
    lib: str,
    symbol: str,
    args: List[pl.Expr],
    kwargs: Optional[Dict[str, Any]] = None,
    is_elementwise: bool = False,
    returns_scalar: bool = False,
    changes_length: bool = False,
    cast_to_supertype: bool = False,
) -> pl.Expr:
    # pl.__version__ should always be a valid version number, so split returns always 3 strs
    if tuple(int(x) for x in pl.__version__.split(".")) < (0, 20, 16):
        # This will eventually be deprecated?
        return args[0].register_plugin(
            lib=lib,
            symbol=symbol,
            args=args[1:],
            kwargs=kwargs,
            is_elementwise=is_elementwise,
            returns_scalar=returns_scalar,
            changes_length=changes_length,
            cast_to_supertype=cast_to_supertype,
        )

    from polars.plugins import register_plugin_function

    return register_plugin_function(
        plugin_path=lib,
        args=args,
        function_name=symbol,
        kwargs=kwargs,
        is_elementwise=is_elementwise,
        returns_scalar=returns_scalar,
        changes_length=changes_length,
        cast_to_supertype=cast_to_supertype,
    )
