"""
Parity oracle: compare old vs new Rust impls of polars-ds expressions for
byte-identical output.

HOW TO ADD A NEW ParityCase
----------------------------
1. In the relevant Rust source, keep the original impl as `pl_xxx_old` and
   the rewritten impl as `pl_xxx_new`.  The production `#[polars_expr]` fn
   `pl_xxx` stays wired to `_old`.  Add a second `#[polars_expr]` entry
   point `pl_xxx_new_expr` wired to `_new`.

2. Register a case here using the @register decorator:

    from polars_ds._utils import pl_plugin          # just for reference, not used here
    import polars as pl

    @register("pl_haversine", fixtures=["tiny_clean", "medium_multichunk"])
    def _haversine_call(df: pl.DataFrame):
        # Return (args_list, kwargs_dict, extra_call_kwargs).
        # args_list: list of pl.Expr consumed positionally by the Rust fn.
        # kwargs_dict: serialised to the kwargs struct on the Rust side.
        # extra_call_kwargs: forwarded verbatim to register_plugin_function
        #     (e.g. is_elementwise=True, returns_scalar=False).
        args = [df["x"].cast(pl.Float64), df["y"].cast(pl.Float64),
                pl.lit(0.0), pl.lit(0.0)]
        kwargs = {}
        extra = {"is_elementwise": True}
        return args, kwargs, extra

3. Run:
    python tests/parity_oracle.py --list          # show registered cases
    python tests/parity_oracle.py pl_haversine    # test one
    python tests/parity_oracle.py                 # test all

PLACEHOLDER CASE
----------------
`pl_fract_placeholder` below exercises the dispatch loop against an existing
symbol (`pl_fract`) and its not-yet-added `_new_expr` variant (expected to
fail gracefully with a "function not found" error).  Remove once the first
real B-phase rewrite lands.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Allow running as `python tests/parity_oracle.py ...` (script mode)
# by ensuring the repo root is on sys.path so `tests.parity_cases` resolves.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import polars as pl
from polars.plugins import register_plugin_function

# ---------------------------------------------------------------------------
# .so location
# ---------------------------------------------------------------------------
# The canonical compiled plugin lives alongside the Python package.
# Rename-conflict copies ("_polars_ds.abi3 2.so", etc.) are ignored.
_PKG_PATH = Path(__file__).parent.parent / "python" / "polars_ds"
_SO = _PKG_PATH / "_polars_ds.abi3.so"

if not _SO.exists():
    raise FileNotFoundError(
        f"Plugin .so not found at {_SO}.\n"
        "Run `maturin develop` (debug) or `maturin develop --release` first."
    )

# ---------------------------------------------------------------------------
# Registry — lives in tests/_parity_registry.py so case files and the CLI
# share the same dict instance regardless of script-vs-module import path.
# ---------------------------------------------------------------------------

from tests._parity_registry import ParityCase, REGISTRY, register  # noqa: E402,F401

# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------

def _call(symbol: str, df: pl.DataFrame, build_call: Callable) -> pl.DataFrame:
    """Call a single plugin symbol and return the resulting DataFrame."""
    args, kwargs, extra = build_call(df)
    expr = register_plugin_function(
        plugin_path=_SO,
        function_name=symbol,
        args=args,
        kwargs=kwargs,
        **extra,
    )
    return df.select(expr.alias("out"))


def run_one(
    case: ParityCase,
    fixtures: dict[str, pl.DataFrame],
) -> list[tuple[str, bool, str]]:
    """Run old vs new for every fixture in the case.  Returns (fix_name, ok, msg)."""
    from tests._diff_bits import assert_byte_equal  # built in parallel (A5)

    results = []
    for fix_name in case.fixtures:
        if fix_name not in fixtures:
            results.append((fix_name, False, f"fixture {fix_name!r} not found"))
            continue
        df = fixtures[fix_name]
        try:
            old = _call(case.name, df, case.build_call)
            new = _call(f"{case.name}_new_expr", df, case.build_call)
            assert_byte_equal(
                old, new,
                context=f"{case.name}/{fix_name}",
                rtol=case.rtol, atol=case.atol,
            )
            results.append((fix_name, True, ""))
        except AssertionError as e:
            results.append((fix_name, False, str(e)))
        except Exception as e:
            results.append((fix_name, False, f"EXCEPTION: {type(e).__name__}: {e}"))
    return results

# ---------------------------------------------------------------------------
# Auto-discover ParityCase modules under tests/parity_cases/
# Each Phase B/C/... agent drops a self-contained registration file there
# (e.g. tests/parity_cases/pl_query_radius_ptwise.py) — no edits to this file.
# ---------------------------------------------------------------------------

def _autoload_cases() -> None:
    import importlib
    import pkgutil
    try:
        import tests.parity_cases as _cases_pkg
    except ImportError:
        return
    for _finder, mod_name, _is_pkg in pkgutil.iter_modules(_cases_pkg.__path__):
        importlib.import_module(f"tests.parity_cases.{mod_name}")

_autoload_cases()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Polars-DS parity oracle: old vs new Rust impls."
    )
    p.add_argument("fns", nargs="*", help="Names of registered cases to run (default: all).")
    p.add_argument("--list", action="store_true", help="List registered cases and exit.")
    p.add_argument(
        "--bench",
        action="store_true",
        help="Also include the large_bench fixture (not loaded by default).",
    )
    args = p.parse_args()

    if args.list:
        if not REGISTRY:
            print("(no cases registered)")
        else:
            for n in sorted(REGISTRY):
                print(f"  {n}  fixtures={REGISTRY[n].fixtures}")
        return

    from tests.fixtures import load  # A2 provides this

    fixture_names = [
        "tiny_clean",
        "tiny_with_nulls",
        "tiny_with_specials",
        "medium_multichunk",
        "medium_with_nulls",
    ]
    if args.bench:
        fixture_names.append("large_bench")

    fixtures: dict[str, pl.DataFrame] = {}
    for name in fixture_names:
        try:
            fixtures[name] = load(name)
        except Exception as e:
            # Non-fatal: individual cases will report "fixture not found".
            print(f"  [WARN] Could not load fixture {name!r}: {e}", file=sys.stderr)

    targets = args.fns if args.fns else list(REGISTRY.keys())
    missing = [n for n in targets if n not in REGISTRY]
    if missing:
        sys.exit(f"Unknown cases: {missing}. Try --list.")

    if not targets:
        print("No cases registered.  Nothing to run.")
        return

    n_pass = n_fail = 0
    t0 = time.perf_counter()

    for name in targets:
        results = run_one(REGISTRY[name], fixtures)
        for fix_name, ok, err in results:
            tag = "PASS" if ok else "FAIL"
            suffix = f" — {err}" if err else ""
            print(f"  [{tag}] {name} / {fix_name}{suffix}")
            n_pass += int(ok)
            n_fail += int(not ok)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"\n{n_pass}/{n_pass + n_fail} passed in {elapsed_ms:.0f}ms")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
