"""Run the changed contracts with the minimum Polars' collect API."""

from pathlib import Path
import sys
import polars as pl
import pytest

original_collect = pl.LazyFrame.collect


def collect_with_legacy_engine(frame, *args, **kwargs):
    engine = kwargs.pop("engine", None)
    if engine is not None:
        if engine not in {"auto", "streaming", "in-memory"}:
            raise ValueError(f"unsupported test engine: {engine}")
        kwargs["streaming"] = engine == "streaming"
    return original_collect(frame, *args, **kwargs)


if __name__ == "__main__":
    pl.LazyFrame.collect = collect_with_legacy_engine
    tests = Path(__file__).resolve().parent
    raise SystemExit(
        pytest.main(
            [
                "-q",
                "--tb=short",
                str(tests / "test_polars2_compat.py"),
                str(tests / "test_polars2_contracts.py"),
                *sys.argv[1:],
            ]
        )
    )
