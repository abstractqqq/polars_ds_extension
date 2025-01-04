from .._utils import _IS_POLARS_V1
if not _IS_POLARS_V1:
    raise ValueError("You must be on Polars >= v1.0.0 to use this module.")
