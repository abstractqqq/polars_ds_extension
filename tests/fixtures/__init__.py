"""
Fixture loader for the polars-ds parity oracle.

Usage
-----
    from tests.fixtures import load

    df = load("tiny_clean")          # Schema{x: Float64, y: Float64, ...}
    df = load("medium_multichunk")   # n_chunks() > 1  (reconstructed via concat)
"""

from pathlib import Path
import polars as pl

_FIXTURES_DIR = Path(__file__).parent

# Fixtures stored as a single parquet file.
_SINGLE_FILE = {
    "tiny_clean",
    "tiny_with_nulls",
    "tiny_with_specials",
    "medium_with_nulls",
}

# medium_multichunk is stored as 4 separate parts (p0..p3) because parquet
# readers rechunk on load; we reconstruct with rechunk=False to preserve
# multiple chunks in memory.
_MULTICHUNK = {
    "medium_multichunk": [f"medium_multichunk_p{i}.parquet" for i in range(4)],
}

_ALL = _SINGLE_FILE | set(_MULTICHUNK)


def load(name: str) -> pl.DataFrame:
    """Load a named fixture and return a ``pl.DataFrame``.

    For ``medium_multichunk`` the frame is reconstructed from 4 parquet
    part files and concatenated with ``rechunk=False``, so
    ``df.n_chunks()`` is > 1.
    """
    if name not in _ALL:
        raise ValueError(f"Unknown fixture {name!r}. Available: {sorted(_ALL)}")

    if name in _MULTICHUNK:
        parts = [pl.read_parquet(_FIXTURES_DIR / fname) for fname in _MULTICHUNK[name]]
        return pl.concat(parts, rechunk=False)

    return pl.read_parquet(_FIXTURES_DIR / f"{name}.parquet")
