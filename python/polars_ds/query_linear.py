from __future__ import annotations
import polars as pl
import warnings
from .type_alias import LRMethods, NullPolicy
from ._utils import pl_plugin

__all__ = [
    "query_lstsq",
    "query_lstsq_report",
    "query_rolling_lstsq",
    "query_recursive_lstsq",
    "query_wls_ww",
]


def linear_formula(s: str | pl.Expr) -> pl.Expr:
    if isinstance(s, str):
        return pl.sql_expr(s).alias(s)
    elif isinstance(s, pl.Expr):
        return s
    else:
        raise ValueError(
            "Input can only be str or polars expression. The str must be valid SQL strings that polars can understand."
        )


def query_lstsq(
    *x: str | pl.Expr,
    target: str | pl.Expr,
    add_bias: bool = False,
    skip_null: bool = False,
    return_pred: bool = False,
    method: LRMethods = "normal",
    l1_reg: float = 0.0,
    l2_reg: float = 0.0,
    tol: float = 1e-5,
    null_policy: NullPolicy = "raise",
) -> pl.Expr:
    """
    Computes least squares solution to the equation Ax = y where y is the target.

    All positional arguments should be expressions representing predictive variables. This
    does not support composite expressions like pl.col(["a", "b"]), pl.all(), etc.

    If add_bias is true, it will be the last coefficient in the output
    and output will have len(variables) + 1. Bias term will not be regularized if method is l1 or l2.

    Memory hint: if data takes 100MB of memory, you need to have at least 200MB of memory to run this.

    Parameters
    ----------
    x : str | pl.Expr
        The variables used to predict target
    target : str | pl.Expr
        The target variable
    add_bias
        Whether to add a bias term
    skip_null
        Deprecated. Use null_policy = 'skip'. Whether to skip a row if there is a null value in row
    return_pred
        If true, return prediction and residue. If false, return coefficients. Note that
        for coefficients, it reduces to one output (like max/min), but for predictions and
        residue, it will return the same number of rows as in input.
    method
        Linear Regression method. One of "normal" (normal equation), "l2" or "ridge" (l2 regularized, Ridge),
        "l1" or "lasso" (l1 regularized, Lasso).
    l1_reg
        Regularization factor for Lasso. Should be nonzero when method = l1.
    l2_reg
        Regularization factor for Ridge. Should be nonzero when method = l2.
    tol
        When method = l1, if maximum coordinate update is < tol, the algorithm is considered to have
        converged. If not, it will run for at most 2000 iterations. This stopping criterion is not as
        good as the dual gap.
    null_policy: Literal['raise', 'skip', 'zero', 'one', 'ignore']
        One of options shown here, but you can also pass in any numeric string. E.g you may pass '1.25' to mean
        fill nulls with 1.25. If the string cannot be converted to a float, an error will be thrown. Note: if
        the target column has null, the rows with nulls will always be dropped. Null-fill only applies to non-target
        columns.
    """
    if skip_null:
        warnings.warn(
            "`skip_null` is deprecated. Please use null_policy = 'skip'.",
            DeprecationWarning,
            stacklevel=2,
        )
        null_policy = "skip"

    t = linear_formula(target)
    cols = [t]
    cols.extend(linear_formula(z) for z in x)

    if method == "l1" and l1_reg <= 0.0:
        raise ValueError("For Lasso regression, `l1_reg` must be positive.")
    if method == "l2" and l2_reg <= 0.0:
        raise ValueError("For Ridge regression, `l2_reg` must be positive.")

    lr_kwargs = {
        "bias": add_bias,
        "null_policy": null_policy,
        "method": str(method).lower(),
        "l1_reg": l1_reg,
        "l2_reg": l2_reg,
        "tol": tol,
    }
    if return_pred:
        return pl_plugin(
            symbol="pl_lstsq_pred",
            args=cols,
            kwargs=lr_kwargs,
            pass_name_to_apply=True,
        )
    else:
        return pl_plugin(
            symbol="pl_lstsq",
            args=cols,
            kwargs=lr_kwargs,
            returns_scalar=True,
            pass_name_to_apply=True,
        )


def query_wls_ww(
    *x: str | pl.Expr,
    target: str | pl.Expr,
    weights: str | pl.Expr,
    add_bias: bool = False,
    return_pred: bool = False,
    null_policy: NullPolicy = "raise",
) -> pl.Expr:
    """
    Computes weighted least squares with weights given by the user (ww stands for with weights). This
    only supports ordinary weighted least squares. The weights are presumed to be (proportional to) the
    inverse of the variance of the observations (See page 4 of reference).

    Memory hint: if data takes 100MB of memory, you need to have at least 200MB of memory to run this.

    Parameters
    ----------
    x : str | pl.Expr
        The variables used to predict target
    target : str | pl.Expr
        The target variable
    weights : str | pl.Expr
        The column representing weights
    add_bias
        Whether to add a bias term
    return_pred
        If true, return prediction and residue. If false, return coefficients. Note that
        for coefficients, it reduces to one output (like max/min), but for predictions and
        residue, it will return the same number of rows as in input.
    null_policy: Literal['raise', 'skip', 'zero', 'one', 'ignore']
        One of options shown here, but you can also pass in any numeric string. E.g you may pass '1.25' to mean
        fill nulls with 1.25. If the string cannot be converted to a float, an error will be thrown. Note: if
        the target column has null, the rows with nulls will always be dropped. Null-fill only applies to non-target
        columns.

    Reference
    ---------
    https://www.stat.uchicago.edu/~yibi/teaching/stat224/L14.pdf
    """

    w = linear_formula(weights)
    cols = [
        w.cast(pl.Float64).rechunk(),
        linear_formula(target),
    ]  # weights are at index 0, then target
    cols.extend(linear_formula(z) for z in x)

    lr_kwargs = {
        "bias": add_bias,
        "null_policy": null_policy,
        "method": "",
        "l1_reg": 0.0,
        "l2_reg": 0.0,
        "tol": 0.0,
    }
    if return_pred:
        return pl_plugin(
            symbol="pl_wls_ww_pred",
            args=cols,
            kwargs=lr_kwargs,
            pass_name_to_apply=True,
        )
    else:
        return pl_plugin(
            symbol="pl_wls_ww",
            args=cols,
            kwargs=lr_kwargs,
            returns_scalar=True,
            pass_name_to_apply=True,
        )


def query_recursive_lstsq(
    *x: str | pl.Expr,
    target: str | pl.Expr,
    start_with: int,
    add_bias: bool = False,
    method: LRMethods = "normal",
    l2_reg: float = 0.1,
    null_policy: NullPolicy = "raise",
):
    """
    Using the first `start_with` rows of data as basis, start computing the least square solutions
    by updating the betas per row. A prediction for that row will also be included in the output.
    This uses the famous Sherman-Morrison-Woodbury Formula under the hood.

    Note: You have to be careful about the order of data when using this in aggregation contexts.

    In the author's opinion, this should be called "cumulative" instead of resursive because of
    its similarity with other cumulative operations on sequences of data. However, I will go with
    the academia's name.

    Parameters
    ----------
    x : str | pl.Expr
        The variables used to predict target
    target : str | pl.Expr
        The target variable
    start_with: int
        Must be >= 1. You `start_with` n rows of data to train the first linear regression. If `start_with` = 2,
        the first row will be null, etc. If you start with N < # features, result will be numerically very
        unstable and potentially wrong.
    add_bias
        Whether to add a bias term
    method : Literal['normal', 'l2']
        Linear Regression method. One of "normal" (normal equation), "l2" (l2 regularized, Ridge). "l1" does
        not work for now.
    l2_reg
        The L2 regularization factor
    null_policy: Literal['raise', 'skip', 'zero', 'one', 'ignore']
        One of options shown here, but you can also pass in any numeric string. E.g you may pass '1.25' to mean
        fill nulls with 1.25. If the string cannot be converted to a float, an error will be thrown. Note: if
        the target column has null, the rows with nulls will always be dropped. Null-fill only applies to non-target
        columns. If null_policy is `skip` or `fill`, and nulls actually exist, it will keep skipping until we have
        scanned `start_at` many valid rows. And if subsequently we get a row with null values, then null will
        be returned for that row.
    """

    if method == "l1":
        raise NotImplementedError

    if start_with < 1:
        raise ValueError("You must start with >= 1 rows for recursive lstsq.")

    cols = [linear_formula(target)]
    features = [linear_formula(z) for z in x]
    if len(features) > start_with:
        warnings.warn(
            "# features > number of rows for the initial fit. Outputs may be off.", stacklevel=2
        )

    cols.extend(features)
    kwargs = {
        "null_policy": null_policy,
        "n": start_with,
        "bias": add_bias,
        "method": method,
        "lambda": abs(l2_reg),
        "min_size": 0,  # Not used for recursive
    }
    return pl_plugin(
        symbol="pl_recursive_lstsq",
        args=cols,
        kwargs=kwargs,
        pass_name_to_apply=True,
    )


def query_rolling_lstsq(
    *x: str | pl.Expr,
    target: str | pl.Expr,
    window_size: int,
    add_bias: bool = False,
    method: LRMethods = "normal",
    l2_reg: float = 0.1,
    min_valid_rows: int | None = None,
    null_policy: NullPolicy = "raise",
):
    """
    Using every `window_size` rows of data as feature matrix, and computes least square solutions
    by rolling the window. A prediction for that row will also be included in the output.
    This uses the famous Sherman-Morrison-Woodbury Formula under the hood.

    Note: You have to be careful about the order of data when using this in aggregation contexts.

    Parameters
    ----------
    x : str | pl.Expr
        The variables used to predict target
    target : str | pl.Expr
        The target variable
    window_size: int
        Must be >= 2. Window size for the rolling regression
    add_bias
        Whether to add a bias term
    method : Literal['normal', 'l2']
        Linear Regression method. One of "normal" (normal equation), "l2" (l2 regularized, Ridge). "l1" does
        not work for now.
    l2_reg
        The L2 regularization factor
    min_valid_rows
        Minimum number of valid rows to evaluate the model. This is only used when null policy is `skip`. E.g.
        if there are nulls in the windows, the window must have at least `min_valid_rows` valid rows in order to
        produce a result. Otherwise, null will be returned.
    null_policy: Literal['raise', 'skip', 'zero', 'one', 'ignore']
        One of options shown here, but you can also pass in any numeric string. E.g you may pass '1.25' to mean
        fill nulls with 1.25. If the string cannot be converted to a float, an error will be thrown. Note: For
        rolling lstsq, null-fill only works when target doesn't have nulls, and WILL NOT drop rows where the
        target is null.
    """

    if method == "l1":
        raise NotImplementedError

    if window_size < 2:
        raise ValueError("`window_size` must be >= 2.")

    cols = [linear_formula(target)]
    features = [linear_formula(z) for z in x]
    if len(features) > window_size:
        raise ValueError("# features > window size. Linear regression is not well-defined.")

    if min_valid_rows is None:
        min_size = min(len(features), window_size)
    else:
        if min_valid_rows < len(features):
            warnings.warn(
                "# features > min_window_size. Linear regression may not always be well-defined.",
                stacklevel=2,
            )
        min_size = min_valid_rows

    cols.extend(features)
    kwargs = {
        "null_policy": null_policy,
        "n": window_size,
        "bias": add_bias,
        "method": method,
        "lambda": abs(l2_reg),
        "min_size": min_size,
    }
    return pl_plugin(
        symbol="pl_rolling_lstsq",
        args=cols,
        kwargs=kwargs,
        pass_name_to_apply=True,
    )


def query_lstsq_report(
    *x: str | pl.Expr,
    target: str | pl.Expr,
    weights: str | pl.Expr | None = None,
    add_bias: bool = False,
    skip_null: bool = False,
    null_policy: NullPolicy = "raise",
) -> pl.Expr:
    """
    Creates an ordinary least square report with more stats about each coefficient.

    Note: if columns are not linearly independent, some numerical issue may occur. This uses
    the closed form solution to compute the least square report.

    Parameters
    ----------
    x : str | pl.Expr
        The variables used to predict target
    target : str | pl.Expr
        The target variable
    weights : str | pl.Expr | None
        If not None, this will then compute the stats for a weights least square.
    add_bias
        Whether to add a bias term. If bias is added, it is always the last feature.
    skip_null
        Deprecated. Use null_policy = 'skip'. Whether to skip a row if there is a null value in row
    null_policy: Literal['raise', 'skip', 'zero', 'one', 'ignore']
        One of options shown here, but you can also pass in any numeric string. E.g you may pass '1.25' to mean
        fill nulls with 1.25. If the string cannot be converted to a float, an error will be thrown. Note: if
        the target column has null, the rows with nulls will always be dropped. Null-fill only applies to non-target
        columns.
    """
    if skip_null:
        import warnings  # noqa: E401

        warnings.warn(
            "`skip_null` is deprecated. Please use null_policy = 'skip'.",
            DeprecationWarning,
            stacklevel=2,
        )
        null_policy = "skip"

    lr_kwargs = {
        "bias": add_bias,
        "null_policy": null_policy,
        "method": "normal",
        "l1_reg": 0.0,
        "l2_reg": 0.0,
        "tol": 0.0,
    }

    t = linear_formula(target)
    if weights is None:
        cols = [t]
        cols.extend(linear_formula(z) for z in x)
        return pl_plugin(
            symbol="pl_lstsq_report",
            args=cols,
            kwargs=lr_kwargs,
            changes_length=True,
            pass_name_to_apply=True,
        )
    else:
        w = linear_formula(weights)
        cols = [w.cast(pl.Float64).rechunk(), t]
        cols.extend(linear_formula(z) for z in x)
        return pl_plugin(
            symbol="pl_wls_report",
            args=cols,
            kwargs=lr_kwargs,
            changes_length=True,
            pass_name_to_apply=True,
        )
