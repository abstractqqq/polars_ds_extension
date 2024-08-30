from __future__ import annotations
import polars as pl
import warnings
from .type_alias import LRSolverMethods, NullPolicy
from ._utils import pl_plugin

__all__ = [
    "query_lstsq",
    "query_lstsq_report",
    "query_rolling_lstsq",
    "query_recursive_lstsq",
    "query_lstsq_w_rcond",
]


def lr_formula(s: str | pl.Expr) -> pl.Expr:
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
    weights: str | pl.Expr | None = None,
    skip_null: bool = False,
    return_pred: bool = False,
    l1_reg: float = 0.0,
    l2_reg: float = 0.0,
    tol: float = 1e-5,
    solver: LRSolverMethods = "qr",
    null_policy: NullPolicy = "skip",
) -> pl.Expr:
    """
    Computes least squares solution to the equation Ax = y where y is the target. If l1_reg is > 0,
    then this performs Lasso regression. If l2_reg is > 0, this performs Ridge regression. Only
    one of l1_reg or l2_reg can be greater than 0 and any other case will result in normal regression. (This
    is because Elastic net hasn't been implemented.)

    All positional arguments should be expressions representing predictive variables. This
    does not support composite expressions like pl.col(["a", "b"]), pl.all(), etc.

    If add_bias is true, it will be the last coefficient in the output
    and output will have len(variables) + 1.

    Memory hint: if data takes 100MB of memory, you need to have at least 200MB of memory to run this.

    Parameters
    ----------
    x : str | pl.Expr
        The variables used to predict target
    target : str | pl.Expr
        The target variable
    add_bias
        Whether to add a bias term
    weights
        Whether to perform a weighted least squares or not. If this is weighted, then it will ignore
        l1_reg or l2_reg parameters.
    skip_null
        Deprecated. Use null_policy = 'skip'. Whether to skip a row if there is a null value in row
    return_pred
        If true, return prediction and residue. If false, return coefficients. Note that
        for coefficients, it reduces to one output (like max/min), but for predictions and
        residue, it will return the same number of rows as in input.
    l1_reg
        Regularization factor for Lasso. Should be nonzero when method = l1.
    l2_reg
        Regularization factor for Ridge. Should be nonzero when method = l2.
    tol
        When method = 'l1', if maximum coordinate update is < tol, the algorithm is considered to have
        converged. If not, it will run for at most 2000 iterations. This stopping criterion is not as
        good as the dual gap.
    solver
        Only applies when this is normal or l2 regression. One of ['svd', 'qr', 'cholesky'].
        Both 'svd' and 'qr' can handle rank deficient cases relatively well, while cholesky may fail or
        slow down. When cholesky fails, it falls back to svd.
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

    weighted = weights is not None

    lr_kwargs = {
        "bias": add_bias,
        "null_policy": null_policy,
        "l1_reg": l1_reg,
        "l2_reg": l2_reg,
        "solver": solver,
        "tol": tol,
        "weighted": weighted,
    }

    if weighted:
        cols = [lr_formula(weights).cast(pl.Float64).rechunk(), lr_formula(target)]
    else:
        cols = [lr_formula(target)]

    cols.extend(lr_formula(z) for z in x)

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


def query_lstsq_w_rcond(
    *x: str | pl.Expr,
    target: str | pl.Expr,
    add_bias: bool = False,
    rcond: float = 0.0,
    l2_reg: float = 0.0,
    null_policy: NullPolicy = "raise",
) -> pl.Expr:
    """
    Uses SVD to compute least squares. During the process, singular values will be set to 0
    if it is smaller than rcond * max singular value (of X). This will return the coefficients as well
    as singular values of X as the output. The number of nonzero singular values is the rank of X.

    Note: the singular values return will be the values before applying the rcond cut off.

    Parameters
    ----------
    x : str | pl.Expr
        The variables used to predict target
    target : str | pl.Expr
        The target variable
    add_bias
        Whether to add a bias term
    rcond
        Cut-off ratio for small singular values. If rcond < machine precision * MAX(M,N),
        it will be set to machine precision * MAX(M,N).
    l2_reg
        Regularization factor for Ridge. Should be nonzero when method = l2.
    null_policy: Literal['raise', 'skip', 'zero', 'one', 'ignore']
        One of options shown here, but you can also pass in any numeric string. E.g you may pass '1.25' to mean
        fill nulls with 1.25. If the string cannot be converted to a float, an error will be thrown. Note: if
        the target column has null, the rows with nulls will always be dropped. Null-fill only applies to non-target
        columns.
    """

    cols = [lr_formula(target)]
    cols.extend(lr_formula(z) for z in x)
    lr_kwargs = {
        "bias": add_bias,
        "null_policy": null_policy,
        "l1_reg": 0.0,
        "l2_reg": l2_reg,
        "solver": "",
        "tol": abs(rcond),
        "weighted": False,
    }
    return pl_plugin(
        symbol="pl_lstsq_w_rcond",
        args=cols,
        kwargs=lr_kwargs,
        pass_name_to_apply=True,
    )


def query_recursive_lstsq(
    *x: str | pl.Expr,
    target: str | pl.Expr,
    start_with: int,
    add_bias: bool = False,
    l2_reg: float = 0.0,
    null_policy: NullPolicy = "raise",
) -> pl.Expr:
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
        Must be >= 1. You `start_with` n rows of data to train the first linear regression. If `start_with` = N,
        the first N-1 rows will be null. If you start with N < # features, result will be numerically very
        unstable and potentially wrong.
    add_bias
        Whether to add a bias term
    l2_reg
        The L2 regularization factor. This performs Ridge regression if this is > 0.
    null_policy: Literal['raise', 'skip', 'zero', 'one', 'ignore']
        One of options shown here, but you can also pass in any numeric string. E.g you may pass '1.25' to mean
        fill nulls with 1.25. If the string cannot be converted to a float, an error will be thrown. Note: if
        the target column has null, the rows with nulls will always be dropped. Null-fill only applies to non-target
        columns. If null_policy is `skip` or `fill`, and nulls actually exist, it will keep skipping until we have
        scanned `start_at` many valid rows. And if subsequently we get a row with null values, then null will
        be returned for that row.
    """

    if start_with < 1:
        raise ValueError("You must start with >= 1 rows for recursive lstsq.")

    cols = [lr_formula(target)]
    features = [lr_formula(z) for z in x]
    if len(features) > start_with:
        warnings.warn(
            "# features > number of rows for the initial fit. Outputs may be off.", stacklevel=2
        )

    cols.extend(features)
    kwargs = {
        "null_policy": null_policy,
        "n": start_with,
        "bias": add_bias,
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
    l2_reg: float = 0.0,
    min_valid_rows: int | None = None,
    null_policy: NullPolicy = "raise",
) -> pl.Expr:
    """
    Using every `window_size` rows of data as feature matrix, and computes least square solutions
    by rolling the window. A prediction for that row will also be included in the output.
    This uses the famous Sherman-Morrison-Woodbury Formula under the hood.

    Note: You have to be careful about the order of data when using this in aggregation contexts.
    Rows with null will not contribute to the update.

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
    l2_reg
        The L2 regularization factor. This performs rolling Ridge if this is > 0.
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

    if window_size < 2:
        raise ValueError("`window_size` must be >= 2.")

    cols = [lr_formula(target)]
    features = [lr_formula(z) for z in x]
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
        "l1_reg": 0.0,
        "l2_reg": 0.0,
        "solver": "qr",
        "tol": 0.0,
    }

    t = lr_formula(target)
    if weights is None:
        cols = [t]
        cols.extend(lr_formula(z) for z in x)
        return pl_plugin(
            symbol="pl_lstsq_report",
            args=cols,
            kwargs=lr_kwargs,
            changes_length=True,
            pass_name_to_apply=True,
        )
    else:
        w = lr_formula(weights)
        cols = [w.cast(pl.Float64).rechunk(), t]
        cols.extend(lr_formula(z) for z in x)
        return pl_plugin(
            symbol="pl_wls_report",
            args=cols,
            kwargs=lr_kwargs,
            changes_length=True,
            pass_name_to_apply=True,
        )
