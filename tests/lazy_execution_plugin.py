import os
from pathlib import Path
from collections import Counter
import json
import polars as pl

mode = os.environ.get("PDS_AUDIT_ENGINE", "auto")
original_select = pl.DataFrame.select
original_with_columns = pl.DataFrame.with_columns
active = False
calls = Counter()


def make_wrapper(method_name, original):
    def wrapped(frame, *args, **kwargs):
        global active
        if active:
            return original(frame, *args, **kwargs)
        active = True
        try:
            calls[method_name] += 1
            return getattr(frame.lazy(), method_name)(*args, **kwargs).collect(engine=mode)
        finally:
            active = False

    return wrapped


def pytest_configure(config):
    pl.DataFrame.select = make_wrapper("select", original_select)
    pl.DataFrame.with_columns = make_wrapper("with_columns", original_with_columns)


def pytest_sessionfinish(session, exitstatus):
    out = os.environ.get("PDS_AUDIT_COUNTERS")
    if out:
        Path(out).write_text(
            json.dumps({"engine": mode, "calls": dict(calls), "exitstatus": exitstatus}, indent=2)
        )
