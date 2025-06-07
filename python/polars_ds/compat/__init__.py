"""
Compatibility with other Dataframes. 

This module provides compatibility with other dataframe libraries that:

1. Have a notion of Series
2. The Series implements the array protocal, which means it can be translated to NumPy array via 
.__array__() method.

Since most dataframe libraries can turn their Series into NumPy (or vice versa) with 0 copy, 
this compatibility layer has very little overhead. The only constraint is that the dataframe
must be eager, in the sense that data is already loaded in memory. The reason for this is that
the notion of a Series doesn't really exist in the lazy world, and lazy columns cannot be turned 
to NumPy arrays.

When using this compatibility, the output is always a Polars Series. This is because the output 
type could be Polars struct/list Series, which are Polars-specific types. It is up to the user
what to do with the output.

For example, in order to use PDS with Pandas dataframe, say df:pd.DataFrame, one needs to write

>>> from polars_ds.compat import compat as pds2
>>> # Output is a Polars Series. 
>>> pds2.query_roc_auc(df_pd["actual"], df_pd["predicted"])
>>> # For more advanced queries
>>> pds2.lin_reg(
>>>     df_pd["x1"], df_pd["x2"], df_pd["x3"]
>>>     target = df_pd["y"],
>>>     return_pred = True
>>> )
"""

from ._compat import compat

import warnings
warnings.warn(
    "The compatibility layer is considered experimental.", 
    stacklevel=2
)