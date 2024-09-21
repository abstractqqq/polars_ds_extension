import polars as pl
from typing import Any, Optional, List, Dict, Union
from pathlib import Path
from polars.plugins import register_plugin_function

# Only need this
_PLUGIN_PATH = Path(__file__).parent
# FLAG FOR v1 polars
_IS_POLARS_V1 = pl.__version__.startswith("1.")


def pl_plugin(
    *,
    symbol: str,
    args: List[Union[pl.Series, pl.Expr]],
    kwargs: Optional[Dict[str, Any]] = None,
    is_elementwise: bool = False,
    returns_scalar: bool = False,
    changes_length: bool = False,
    cast_to_supertype: bool = False,
    pass_name_to_apply: bool = False,
) -> pl.Expr:
    return register_plugin_function(
        plugin_path=_PLUGIN_PATH,
        args=args,
        function_name=symbol,
        kwargs=kwargs,
        is_elementwise=is_elementwise,
        returns_scalar=returns_scalar,
        changes_length=changes_length,
        cast_to_supertype=cast_to_supertype,
        pass_name_to_apply=pass_name_to_apply,
    )
