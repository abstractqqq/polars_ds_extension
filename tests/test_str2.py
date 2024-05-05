from __future__ import annotations

import polars as pl
import polars_ds as pds
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


def test_remove_diacritics():
    df = pl.DataFrame({"x": ["mercy", "mèrcy", "françoise", "über"]})

    assert_frame_equal(
        df.select(pds.remove_diacritics("x")),
        pl.DataFrame({"x": ["mercy", "mercy", "francoise", "uber"]}),
    )
