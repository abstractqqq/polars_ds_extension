from __future__ import annotations

import numpy as np
import polars as pl
import polars_ds as pds
import pytest
from polars.testing import assert_frame_equal


def test_replace_non_ascii():
    df = pl.DataFrame({"x": ["mercy", "xbĤ", "ĤŇƏ"]})

    assert_frame_equal(
        df.select(pds.replace_non_ascii("x")), pl.DataFrame({"x": ["mercy", "xb", ""]})
    )

    assert_frame_equal(
        df.select(pds.replace_non_ascii("x", "?")),
        pl.DataFrame({"x": ["mercy", "xb?", "???"]}),
    )

    assert_frame_equal(
        df.select(pds.replace_non_ascii("x", "??")),
        pl.DataFrame({"x": ["mercy", "xb??", "??????"]}),
    )
