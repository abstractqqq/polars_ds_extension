import polars as pl
import re
from typing import Sequence, Any


def parse_version(version: Sequence[str | int]) -> tuple[int, ...]:
    # Simple version parser; split into a tuple of ints for comparison.
    # vendored from Polars
    if isinstance(version, str):
        version = version.split(".")
    return tuple(int(re.sub(r"\D", "", str(v))) for v in version)


def pl_plugin(
    *,
    lib: str,
    symbol: str,
    args: list[pl.Expr],
    kwargs: dict[str, Any] | None = None,
    is_elementwise: bool = False,
    returns_scalar: bool = False,
    changes_length: bool = False,
) -> pl.Expr:
    if parse_version(pl.__version__) < parse_version("0.20.16"):
        # This will eventually be deprecated?
        assert isinstance(args[0], pl.Expr)
        assert isinstance(lib, str)
        return args[0].register_plugin(
            lib=lib,
            symbol=symbol,
            args=args[1:],
            kwargs=kwargs,
            is_elementwise=is_elementwise,
            returns_scalar=returns_scalar,
            changes_length=changes_length,
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
    )
