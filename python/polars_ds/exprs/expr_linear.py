"""Linear Regression Related Expressions in Polars."""

from __future__ import annotations
import polars as pl
import warnings
from typing import List, Any, Literal

# Internal dependencies
from polars_ds.typing import LRSolverMethods, NullPolicy
from polars_ds._utils import pl_plugin
from polars_ds.config import _lin_reg_expr_symbol

__all__ = [
    "lin_reg",
    "lin_reg_w_rcond",
    "simple_lin_reg",
    "recursive_lin_reg",
    "rolling_lin_reg",
    "lin_reg_report",
    "query_lstsq",
    "query_lstsq_w_rcond",
    "query_recursive_lstsq",
    "query_rolling_lstsq",
    "query_lstsq_report",
]


# Despite the typing requirments in the function signatures, we allow some slack
# by accepting the most common Series/Array types.
def lr_formula(s: Any) -> pl.Expr:
    if isinstance(s, str):
        return pl.sql_expr(s).alias(s)
    elif isinstance(s, pl.Series):
        return pl.lit(s)
    elif isinstance(s, pl.Expr):
        return s
    elif hasattr(s, "__array__"):
        return pl.lit(pl.Series(values=s.__array__()))
    else:
        raise ValueError(
            "Input can only be str or polars expression. The str must be valid SQL strings that polars can understand."
        )


def simple_lin_reg(
    x: str | pl.Expr,
    target: str | pl.Expr,
    add_bias: bool = False,
    weights: str | pl.Expr | None = None,
    return_pred: bool = False,
) -> pl.Expr:
    """
    Simple least square with 1 predictive variable and 1 target.

    Parameters
    ----------
    x
        The variables used to predict target
    target
        The target variable
    add_bias
        Whether to add a bias term
    weights
        Whether to perform a weighted least squares or not.
    return_pred
        If true, return prediction and residue. If false, return coefficients. Note that
        for coefficients, it reduces to one output (like max/min), but for predictions and
        residue, it will return the same number of rows as in input.
    """
    # No test. All forumla here are mathematically correct.
    xx = lr_formula(x)
    yy = lr_formula(target)
    if add_bias:
        if weights is None:
            x_mean = xx.mean()
            y_mean = yy.mean()
            beta = (xx - x_mean).dot(yy - y_mean) / (xx - x_mean).dot(xx - x_mean)
            alpha = y_mean - beta * x_mean
        else:
            w = lr_formula(weights)
            w_sum = w.sum()
            x_wmean = w.dot(xx) / w_sum
            y_wmean = w.dot(yy) / w_sum
            beta = w.dot((xx - x_wmean) * (yy - y_wmean)) / (w.dot((xx - x_wmean).pow(2)))
            alpha = y_wmean - beta * x_wmean

        if return_pred:
            return pl.struct(pred=beta * xx + alpha, resid=yy - (beta * xx + alpha)).alias(
                "lr_pred"
            )
        else:
            return (beta.append(alpha)).implode().alias("coeffs")
    else:
        if weights is None:
            beta = xx.dot(yy) / xx.dot(xx)
        else:
            w = lr_formula(weights)
            beta = w.dot(xx * yy) / w.dot(xx.pow(2))

        if return_pred:
            return pl.struct(pred=beta * xx, resid=yy - (beta * xx)).alias("lr_pred")
        else:
            return beta.implode().alias("coeffs")


def lin_reg(
    *x: str | pl.Expr,
    target: str | pl.Expr | List[str | pl.Expr],
    add_bias: bool = False,
    weights: str | pl.Expr | None = None,
    return_pred: bool = False,
    l1_reg: float = 0.0,
    l2_reg: float = 0.0,
    tol: float = 1e-5,
    solver: LRSolverMethods = "qr",
    null_policy: NullPolicy = "skip",
) -> pl.Expr:
    """
    Computes least squares solution to the equation Ax = y where y is the target (or multiple targets).
    If l1_reg is > 0, then this performs Lasso regression. If l2_reg is > 0, this performs Ridge regression.
    If both are > 0, then this is elastic net regression. If none of the cases above is true, as is the default case,
    then a normal regression will be performed.

    If add_bias is true, it will be the last coefficient in the output and output will have len(variables) + 1.

    If you only want to do simple lstsq (one predictive x variable and one target) and null policy doesn't matter,
    then `query_simple_lstsq` is a faster alternative.

    Memory hint: if data takes 100MB of memory, you need to have at least 200MB of memory to run this.

    Parameters
    ----------
    x
        The variables used to predict target
    target
        The target variable, or a list of targets for a multi-target linear regression
    add_bias
        Whether to add a bias term
    weights
        Whether to perform a weighted least squares or not. If this is weighted, then it will ignore
        l1_reg or l2_reg parameters. This doesn't work if this is multi-target.
    return_pred
        If true, return prediction and residue. If false, return coefficients. Note that
        for coefficients, it reduces to one output (like max/min), but for predictions and
        residue, it will return the same number of rows as in input.
    l1_reg
        Regularization factor for Lasso. Should be nonzero when method = l1.
        This is ignored if this is multi-target.
    l2_reg
        Regularization factor for Ridge. Should be nonzero when method = l2.
    tol
        For Lasso or elastic net regression, if maximum coordinate update is < tol, the algorithm is considered
        to have converged. If not, it will run for at most 2000 iterations. This doesn't work if this is multi-target.
    solver
        Only applies when this is normal or l2 regression. One of ['svd', 'qr'].
        Both 'svd' and 'qr' can handle rank deficient cases relatively well.
    null_policy: Literal['raise', 'skip', 'zero', 'one', 'ignore']
        One of options shown here, but you can also pass in any numeric string. E.g you may pass '1.25' to mean
        fill nulls with 1.25. If the string cannot be converted to a float, an error will be thrown. Note: if
        the target column has null, the rows with nulls will always be dropped. Null-fill only applies to non-target
        columns. If this is multi-target, fill will fail if there are nulls in any of the targets.
    """

    if isinstance(target, list):
        n_targets = len(target)
        if n_targets == 0:
            raise ValueError("If `target` is a list, it cannot be empty.")
        elif n_targets == 1:
            return lin_reg(
                *x,
                target=target[0],
                add_bias=add_bias,
                weights=weights,
                return_pred=return_pred,
                l1_reg=l1_reg,
                l2_reg=l2_reg,
                tol=tol,
                solver=solver,
                null_policy=null_policy,
            )
        else:
            cols = [lr_formula(t).alias(f"target_{i}") for i, t in enumerate(target)]
            multi_target_lr_kwargs = {
                "bias": add_bias,
                "null_policy": null_policy,
                "solver": solver,
                "last_target_idx": n_targets,
                "l2_reg": l2_reg,
            }
            cols.extend(lr_formula(z) for z in x)
            if return_pred:
                return pl_plugin(
                    symbol=_lin_reg_expr_symbol("pl_lstsq_multi_pred"),
                    args=cols,
                    kwargs=multi_target_lr_kwargs,
                    pass_name_to_apply=True,
                ).alias("lr_pred")
            else:
                return pl_plugin(
                    symbol=_lin_reg_expr_symbol("pl_lstsq_multi"),
                    args=cols,
                    kwargs=multi_target_lr_kwargs,
                    returns_scalar=True,
                    pass_name_to_apply=True,
                ).alias("coeffs")
    else:
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
                symbol=_lin_reg_expr_symbol("pl_lstsq_pred"),
                args=cols,
                kwargs=lr_kwargs,
                pass_name_to_apply=True,
            ).alias("lr_pred")
        else:
            return pl_plugin(
                symbol=_lin_reg_expr_symbol("pl_lstsq"),
                args=cols,
                kwargs=lr_kwargs,
                returns_scalar=True,
                pass_name_to_apply=True,
            ).alias("coeffs")


def query_lstsq(
    *x: str | pl.Expr,
    target: str | pl.Expr | List[str | pl.Expr],
    add_bias: bool = False,
    weights: str | pl.Expr | None = None,
    return_pred: bool = False,
    l1_reg: float = 0.0,
    l2_reg: float = 0.0,
    tol: float = 1e-5,
    solver: LRSolverMethods = "qr",
    null_policy: NullPolicy = "skip",
) -> pl.Expr:
    warnings.warn(
        "`query_lstsq` has been renamed to `lin_reg` and will be deprecated in future versions.",
        DeprecationWarning,
        stacklevel=2,
    )

    return lin_reg(
        *x,
        target=target,
        add_bias=add_bias,
        weights=weights,
        return_pred=return_pred,
        l1_reg=l1_reg,
        l2_reg=l2_reg,
        tol=tol,
        solver=solver,
        null_policy=null_policy,
    )


def lin_reg_w_rcond(
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
    x
        The variables used to predict target
    target
        The target variable
    add_bias
        Whether to add a bias term
    rcond
        Cut-off ratio for small singular values. If rcond < machine precision * MAX(M,N),
        it will be set to machine precision * MAX(M,N).
    l2_reg
        The L2 regularization factor. If this is > 0, then a Ridge regression will be performed.
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
    }
    return pl_plugin(
        symbol=_lin_reg_expr_symbol("pl_lstsq_w_rcond"),
        args=cols,
        kwargs=lr_kwargs,
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
    warnings.warn(
        "`query_lstsq_w_rcond` has been renamed to `lin_reg_w_rcond` and will be deprecated in future versions.",
        DeprecationWarning,
        stacklevel=2,
    )

    return lin_reg_w_rcond(
        *x, target=target, add_bias=add_bias, rcond=rcond, l2_reg=l2_reg, null_policy=null_policy
    )


def recursive_lin_reg(
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
    x:
        The variables used to predict target
    target:
        The target variable
    start_with:
        Must be >= 1. You `start_with` n rows of data to train the first linear regression. If `start_with` = N,
        the first N-1 rows will be null. If you start with N < # features, result will be numerically very
        unstable and potentially wrong.
    add_bias
        Whether to add a bias term
    l2_reg
        The L2 regularization factor. If this is > 0, then a Ridge regression will be performed.
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
        symbol=_lin_reg_expr_symbol("pl_recursive_lstsq"),
        args=cols,
        kwargs=kwargs,
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
    warnings.warn(
        "`query_recursive_lstsq` has been renamed to `recursive_lin_reg` and will be deprecated in future versions.",
        DeprecationWarning,
        stacklevel=2,
    )

    return recursive_lin_reg(
        *x,
        target=target,
        start_with=start_with,
        add_bias=add_bias,
        l2_reg=l2_reg,
        null_policy=null_policy,
    )


def rolling_lin_reg(
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
    x
        The variables used to predict target
    target
        The target variable
    window_size
        Must be >= 2. Window size for the rolling regression
    add_bias
        Whether to add a bias term
    l2_reg
        The L2 regularization factor. If this is > 0, then a Ridge regression will be performed.
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
        symbol=_lin_reg_expr_symbol("pl_rolling_lstsq"),
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
    warnings.warn(
        "`query_rolling_lstsq` has been renamed to `rolling_lin_reg` and will be deprecated in future versions.",
        DeprecationWarning,
        stacklevel=2,
    )

    return rolling_lin_reg(
        *x,
        target=target,
        window_size=window_size,
        add_bias=add_bias,
        l2_reg=l2_reg,
        min_valid_rows=min_valid_rows,
        null_policy=null_policy,
    )


def lin_reg_report(
    *x: str | pl.Expr,
    target: str | pl.Expr,
    weights: str | pl.Expr | None = None,
    add_bias: bool = False,
    null_policy: NullPolicy = "raise",
    std_err: Literal["se", "hc0", "hc1", "hc2", "hc3"] = "se",
) -> pl.Expr:
    """
    Creates an ordinary least square report with more stats about each coefficient.

    Note: if columns are not linearly independent, some numerical issue may occur. This uses
    the closed form solution to compute the least square report.

    Parameters
    ----------
    x
        The variables used to predict target
    target
        The target variable
    weights
        If not None, this will then compute the stats for a weights least square.
    add_bias
        Whether to add a bias term. If bias is added, it is always the last feature.
    null_policy: Literal['raise', 'skip', 'zero', 'one', 'ignore']
        One of options shown here, but you can also pass in any numeric string. E.g you may pass '1.25' to mean
        fill nulls with 1.25. If the string cannot be converted to a float, an error will be thrown. Note: if
        the target column has null, the rows with nulls will always be dropped. Null-fill only applies to non-target
        columns.
    std_err
        One of "se", "hc0", "hc1", "hc2", "hc3", where "se" means we compute the standard error
        under the assumption of homoskedasticity, and the hc options are different options for
        heteroskedasticity. The hc0-hc3 are called Heteroskedasticity-Consistent Standard Errors, and their
        formulas can be found here: https://jslsoc.sitehost.iu.edu/files_research/testing_tests/hccm/00TAS.pdf.
        This won't be used if weights are used (The author is not super familiar with the theory). If any other
        string is provided, it will default to "se".
    """

    lr_kwargs = {
        "bias": add_bias,
        "null_policy": null_policy,
        "l1_reg": 0.0,
        "l2_reg": 0.0,
        "solver": "qr",
        "tol": 0.0,
        "std_err": std_err.lower(),
    }

    t = lr_formula(target)
    if weights is None:
        cols = [t.var(), t]
        cols.extend(lr_formula(z) for z in x)
        symbol = _lin_reg_expr_symbol("pl_lin_reg_report")

    else:
        w = lr_formula(weights)
        cols = [w.cast(pl.Float64).rechunk(), t.var(), t]
        cols.extend(lr_formula(z) for z in x)
        symbol = _lin_reg_expr_symbol("pl_wls_report")

    return pl_plugin(
        symbol=symbol,
        args=cols,
        kwargs=lr_kwargs,
        changes_length=True,
        pass_name_to_apply=True,
    )


def query_lstsq_report(
    *x: str | pl.Expr,
    target: str | pl.Expr,
    weights: str | pl.Expr | None = None,
    add_bias: bool = False,
    null_policy: NullPolicy = "raise",
) -> pl.Expr:
    warnings.warn(
        "`query_lstsq_report` has been renamed to `lin_reg_report` and will be deprecated in future versions.",
        DeprecationWarning,
        stacklevel=2,
    )

    return lin_reg_report(
        *x, target=target, weights=weights, add_bias=add_bias, null_policy=null_policy
    )
