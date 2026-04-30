"""
Standalone REGISTRY + @register decorator for the parity oracle.
Lives in its own module so case files and the oracle CLI share the same
dict instance regardless of script-vs-module import path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class ParityCase:
    name: str
    fixtures: list[str]
    build_call: Callable
    rtol: float = 0.0  # 0 = bit-identical required; >0 = relative tolerance (per element)
    atol: float = 0.0


REGISTRY: dict[str, ParityCase] = {}


def register(name: str, *, fixtures: list[str], rtol: float = 0.0, atol: float = 0.0):
    """Register a parity case.

    rtol/atol > 0 relax the float comparison to numpy.allclose semantics. Use
    sparingly — default is bit-identity. Suitable for parallel-reduction code
    paths whose partition order is non-deterministic and produces sub-ULP
    differences (e.g. par_bridge → split_offsets refactors of float sums).
    """

    def deco(fn: Callable) -> Callable:
        REGISTRY[name] = ParityCase(
            name=name, fixtures=fixtures, build_call=fn, rtol=rtol, atol=atol
        )
        return fn

    return deco
