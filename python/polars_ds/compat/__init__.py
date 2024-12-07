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
>>>     df["x1"], df["x2"], df["x3"]
>>>     target = df["y"],
>>>     return_pred = True
>>> )

Question: if output is still Polars, then the user must still use both Polars and Pandas.
Why bother with compatibility?

Here are some answers I consider to be true (or self-promotion :))

1. PDS is a very light weight package that can reduce dependencies in your project.
2. For projects with mixed dataframes, it is sometimes not a good idea to cast the 
entire Pandas (or other) dataframe to Polars.
3. Some PDS functions are faster than SciPy / Sklearn equivalents.
4. For ad-hoc analysis that involves say something like linear regression, PDS is easier to 
use than other package.
"""

from ._compat import compat

import warnings
warnings.warn(
    "The compatibility layer is considered experimental.", 
    stacklevel=2
)