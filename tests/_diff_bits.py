"""
_diff_bits.py — byte-equality checker for polars Series/DataFrame outputs.

Motivation: Polars Series.equals() and numpy.array_equal() can mask NaN-vs-Inf
divergences (e.g. 0x7ff0000000000000 +Inf vs 0x7ff8000000000000 canonical NaN).
Raw uint64/uint32 bit comparison is the only correct approach for floats.
"""

from __future__ import annotations

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fmt_float64(val: float) -> str:
    bits = np.array([val], dtype=np.float64).view(np.uint64)[0]
    return f"{val!r} (0x{bits:016x})"


def _fmt_float32(val: float) -> str:
    bits = np.array([val], dtype=np.float32).view(np.uint32)[0]
    return f"{val!r} (0x{bits:08x})"


def _first_mismatch(a: np.ndarray, b: np.ndarray) -> int:
    """Return index of first differing element (-1 if identical)."""
    ne = np.where(a != b)[0]
    return int(ne[0]) if len(ne) else -1


def _assert_series(
    old: pl.Series,
    new: pl.Series,
    context: str,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> None:
    tag = context or repr(old.name)

    # Length
    if len(old) != len(new):
        raise AssertionError(f"[{tag}] length mismatch: old={len(old)}, new={len(new)}")

    # Dtype
    if old.dtype != new.dtype:
        raise AssertionError(f"[{tag}] dtype mismatch: old={old.dtype}, new={new.dtype}")

    # Null mask
    old_null = old.is_null()
    new_null = new.is_null()
    if not old_null.equals(new_null):
        ne = (old_null != new_null).arg_true()
        n_diff = len(ne)
        first = int(ne[0])
        raise AssertionError(
            f"[{tag}] null-mask differs at {n_diff} position(s); first at row {first}"
        )

    dtype = old.dtype

    # Float64
    if dtype == pl.Float64:
        a_f = old.to_numpy(allow_copy=True)
        b_f = new.to_numpy(allow_copy=True)
        if rtol > 0 or atol > 0:
            mask = ~np.isclose(a_f, b_f, rtol=rtol, atol=atol, equal_nan=True)
            ne = np.where(mask)[0]
            if len(ne):
                first = int(ne[0])
                raise AssertionError(
                    f"[{tag}] float64 mismatch (rtol={rtol}, atol={atol}) at "
                    f"{len(ne)} position(s); first at row {first}: "
                    f"old={_fmt_float64(a_f[first])}, new={_fmt_float64(b_f[first])}"
                )
            return
        a = a_f.view(np.uint64)
        b = b_f.view(np.uint64)
        ne = np.where(a != b)[0]
        if len(ne):
            n_diff = len(ne)
            first = int(ne[0])
            ov = old[first]
            nv = new[first]
            raise AssertionError(
                f"[{tag}] float64 bit mismatch at {n_diff} position(s); "
                f"first at row {first}: "
                f"old={_fmt_float64(ov)}, new={_fmt_float64(nv)}"
            )
        return

    # Float32
    if dtype == pl.Float32:
        a_f = old.to_numpy(allow_copy=True)
        b_f = new.to_numpy(allow_copy=True)
        if rtol > 0 or atol > 0:
            mask = ~np.isclose(a_f, b_f, rtol=rtol, atol=atol, equal_nan=True)
            ne = np.where(mask)[0]
            if len(ne):
                first = int(ne[0])
                raise AssertionError(
                    f"[{tag}] float32 mismatch (rtol={rtol}, atol={atol}) at "
                    f"{len(ne)} position(s); first at row {first}: "
                    f"old={_fmt_float32(a_f[first])}, new={_fmt_float32(b_f[first])}"
                )
            return
        a = a_f.view(np.uint32)
        b = b_f.view(np.uint32)
        ne = np.where(a != b)[0]
        if len(ne):
            n_diff = len(ne)
            first = int(ne[0])
            ov = old[first]
            nv = new[first]
            raise AssertionError(
                f"[{tag}] float32 bit mismatch at {n_diff} position(s); "
                f"first at row {first}: "
                f"old={_fmt_float32(ov)}, new={_fmt_float32(nv)}"
            )
        return

    # Struct: recurse field by field
    if isinstance(dtype, pl.Struct):
        for field in dtype.fields:
            fname = field.name
            _assert_series(
                old.struct.field(fname),
                new.struct.field(fname),
                context=f"{tag}.{fname}",
                rtol=rtol,
                atol=atol,
            )
        return

    # List / Array: explode and recurse
    if isinstance(dtype, (pl.List, pl.Array)):
        old_exp = old.explode()
        new_exp = new.explode()
        _assert_series(old_exp, new_exp, context=f"{tag}[exploded]", rtol=rtol, atol=atol)
        return

    # Boolean / integers: numpy exact
    if dtype in (pl.Boolean,) or dtype.is_integer():
        a = old.to_numpy(allow_copy=True)
        b = new.to_numpy(allow_copy=True)
        if not np.array_equal(a, b):
            ne = np.where(a != b)[0]
            n_diff = len(ne)
            first = int(ne[0])
            raise AssertionError(
                f"[{tag}] {dtype} mismatch at {n_diff} position(s); "
                f"first at row {first}: old={old[first]!r}, new={new[first]!r}"
            )
        return

    # String / Categorical: compare as string lists
    if dtype in (pl.String, pl.Utf8, pl.Categorical) or isinstance(dtype, pl.Enum):
        ol = old.cast(pl.String).to_list()
        nl = new.cast(pl.String).to_list()
        if ol != nl:
            mismatches = [(i, ol[i], nl[i]) for i in range(len(ol)) if ol[i] != nl[i]]
            first_i, first_o, first_n = mismatches[0]
            raise AssertionError(
                f"[{tag}] string/cat mismatch at {len(mismatches)} position(s); "
                f"first at row {first_i}: old={first_o!r}, new={first_n!r}"
            )
        return

    # Fallback
    if not old.series_equal(new, null_equal=True):
        raise AssertionError(f"[{tag}] Series differ (dtype={dtype}, fallback series_equal check)")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assert_byte_equal(
    old: pl.Series | pl.DataFrame,
    new: pl.Series | pl.DataFrame,
    *,
    context: str = "",
    rtol: float = 0.0,
    atol: float = 0.0,
) -> None:
    """
    Assert that `old` and `new` are byte-for-byte identical.

    old, new: pl.Series or pl.DataFrame. Must match exactly.
    Raises AssertionError with first-diverging-row diagnostic.

    Float comparison: ndarray.view(np.uint64 / np.uint32) catches NaN-vs-Inf.
    Struct: recurse into fields by schema order.
    List/Array: explode + recurse.
    Bool/Int/String/Cat: exact equality.
    Length, dtype, null-mask checked first.
    """
    if isinstance(old, pl.DataFrame) and isinstance(new, pl.DataFrame):
        tag = context or "<DataFrame>"
        if list(old.schema.keys()) != list(new.schema.keys()):
            raise AssertionError(
                f"[{tag}] column names differ: "
                f"old={list(old.schema.keys())}, new={list(new.schema.keys())}"
            )
        if list(old.dtypes) != list(new.dtypes):
            raise AssertionError(
                f"[{tag}] dtypes differ: old={list(old.dtypes)}, new={list(new.dtypes)}"
            )
        for col in old.schema.keys():
            col_ctx = f"{context}.{col}" if context else col
            _assert_series(old[col], new[col], context=col_ctx, rtol=rtol, atol=atol)
        return

    if isinstance(old, pl.Series) and isinstance(new, pl.Series):
        _assert_series(old, new, context=context or old.name, rtol=rtol, atol=atol)
        return

    raise TypeError(
        f"assert_byte_equal: both args must be Series or both DataFrame; "
        f"got {type(old).__name__} and {type(new).__name__}"
    )


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    passed = 0
    failed = 0

    def _expect_pass(label: str, fn):
        global passed, failed
        try:
            fn()
            print(f"  PASS  {label}")
            passed += 1
        except Exception:
            print(f"  FAIL  {label}")
            traceback.print_exc()
            failed += 1

    def _expect_fail(label: str, fn, must_contain: str = ""):
        global passed, failed
        try:
            fn()
            print(f"  FAIL  {label}  (no error raised)")
            failed += 1
        except AssertionError as e:
            msg = str(e)
            if must_contain and must_contain not in msg:
                print(f"  FAIL  {label}  (missing {must_contain!r} in: {msg})")
                failed += 1
            else:
                print(f"  PASS  {label}  → {msg}")
                passed += 1
        except Exception:
            print(f"  FAIL  {label}  (unexpected exception)")
            traceback.print_exc()
            failed += 1

    print("\n=== _diff_bits.py self-tests ===\n")

    # 1. Identical Series → no error
    _expect_pass(
        "identical float64 series",
        lambda: assert_byte_equal(
            pl.Series("a", [1.0, 2.0, 3.0]),
            pl.Series("a", [1.0, 2.0, 3.0]),
        ),
    )

    # 2. NaN at same position → no error
    nan = float("nan")
    _expect_pass(
        "NaN==NaN same position (no error)",
        lambda: assert_byte_equal(
            pl.Series("a", [1.0, nan, 3.0]),
            pl.Series("a", [1.0, nan, 3.0]),
        ),
    )

    # 3. NaN vs Inf → AssertionError with hex bits
    inf = float("inf")
    _expect_fail(
        "NaN vs Inf → hex bits in message",
        lambda: assert_byte_equal(
            pl.Series("a", [1.0, inf, 3.0]),
            pl.Series("a", [1.0, nan, 3.0]),
        ),
        must_contain="0x7ff",
    )

    # 4. DataFrame with one differing column → names that column
    df_old = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    df_new = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 99.0]})
    _expect_fail(
        "DataFrame differing column named",
        lambda: assert_byte_equal(df_old, df_new),
        must_contain="y",
    )

    # 5. Struct with one differing field → error names field path
    s_old = pl.Series("s", [{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0}])
    s_new = pl.Series("s", [{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 99.0}])
    _expect_fail(
        "Struct differing field names path",
        lambda: assert_byte_equal(s_old, s_new),
        must_contain=".b",
    )

    # 6. Empty DataFrames → no error
    _expect_pass(
        "empty DataFrames",
        lambda: assert_byte_equal(
            pl.DataFrame({"x": pl.Series([], dtype=pl.Float64)}),
            pl.DataFrame({"x": pl.Series([], dtype=pl.Float64)}),
        ),
    )

    print(f"\n{'all self-tests pass' if failed == 0 else f'{failed} test(s) FAILED'}")
    print(f"({passed} passed, {failed} failed)\n")
    if failed:
        raise SystemExit(1)
