"""
Linear models. Currently, only supports Polars DataFrame Inputs or NumPy matrices as inputs. This module is in 
very early development and is subject to frequent breaking changes.

This module requires the NumPy package. PDS only requires Polars, but you can get all the optional dependencies by

`pip install polars_ds[all]`

"""

# Currently skipping tests for this module because the underlying functions are all tested in one way or
# another.

import polars as pl
import numpy as np
from .type_alias import LinearRegressionMethod
from ._polars_ds import pds_linear_regression
from typing import List
from .type_alias import PolarsFrame

import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:  # 3.10, 3.9, 3.8
    from typing_extensions import Self


def linear_regression_report(X: np.ndarray, y: np.ndarray) -> pl.DataFrame:
    """
    Fits a one-time linear regression model using X and y and returns the regression report as a dataframe.

    This only works with NumPy inputs. If you have a dataframe input, use pds.query_lstsq_report directly.

    Parameters
    ----------
    X
        The feature Matrix. NumPy 2D matrix only.
    y
        The target data. NumPy array.
    """

    from . import query_lstsq_report

    schema = [f"x{i+1}" for i in range(X.shape[1])]
    df = pl.from_numpy(X, schema=schema).with_columns(
        pl.Series(name="__target__", values=y.flatten())
    )

    return df.select(
        query_lstsq_report(*schema, target="__target__", skip_null=True).alias("report")
    ).unnest("report")


class LinearRegression:

    """
    Normal, L1 and L2 regression models.
    """

    def __init__(
        self,
        add_bias: bool = False,
        method: LinearRegressionMethod = "normal",
        lambda_: float = 0.0,
        tol: float = 1e-5,
    ):
        """
        Parameters
        ----------
        add_bias
            Whether to add a bias term. Also known as intercept in other packages.
        method
            One of 'normal' (normal lstsq), 'l1' (Lasso), 'l2' (Ridge).
        lambda_
            The regularization parameters for Lasso or Ridge
        tol
            The tolerance parameters when method = 'l1'. This controls when coordinate descent will stop.
        """
        self.coeffs: np.ndarray = np.array([])
        self.add_bias: bool = add_bias
        self.bias: float = 0.0

        _VALID_METHODS = ["normal", "l1", "l2"]
        if method in _VALID_METHODS:
            self.method: LinearRegressionMethod = method
        else:
            raise ValueError(f"Input `method` is not valid. Valid values are {_VALID_METHODS}.")

        if method != "normal" and lambda_ <= 0.0:
            raise ValueError("Input `lambda_` must be > 0 when method is `l1` or `l2`.")

        self.lambda_: float = lambda_
        self.tol: float = tol
        self.feature_names_in_: List[str] = []

    def set_input_features(self, features: List[str]) -> Self:
        """
        Sets the names of input features.

        Parameters
        ----------
        features
            List of strings.
        """
        self.feature_names_in_ = list(features)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """
        Fit the linear regression model on NumPy data.

        Parameters
        ----------
        X
            The feature Matrix. NumPy 2D matrix only.
        y
            The target data. NumPy array. Must be reshape-able to (-1, 1).
        """

        new_y = y.reshape((-1, 1))
        n_features = X.shape[0]  # no bias
        if self.add_bias:
            new_x = np.hstack((X, np.ones((n_features, 1))))
        else:
            new_x = X

        temp = pds_linear_regression(
            new_x, new_y, self.add_bias, self.method, self.lambda_, self.tol
        )

        self.coeffs = temp[:n_features]
        if self.add_bias:
            self.bias = temp[-1]

        return self

    def fit_df(self, df: PolarsFrame, features: List[str], target: str) -> Self:
        """
        Fit the linear regression model on a dataframe. This will overwrite previously set feature names.

        Parameters
        ----------
        df
            Either an eager or a lazy Polars dataframe.
        features
            List of strings of column names.
        target
            The target column's name.
        """
        df_x = df.lazy().select(features).collect()
        self.feature_names_in_ = list(df_x.columns)
        X = df_x.to_numpy()
        y = df.select(target).to_numpy()
        return self.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Returns the prediction of this linear model.

        Parameters
        ----------
        X
            New feature matrix
        """
        return X @ (self.coeffs.reshape((-1, 1))) + self.bias

    def predict_df(self, df: PolarsFrame, name: str = "prediction") -> PolarsFrame:
        """
        Computes the prediction of the linear model and append it as a column in the dataframe. If input
        is lazy, output will be lazy.

        Parameters
        ----------
        df
            Either an eager or a lazy Polars dataframe.
        name
            The name of the prediction column
        """
        if len(self.feature_names_in_) <= 0:
            raise ValueError(
                "The linear model is not fitted on a dataframe, and therefore cannot predict on a dataframe. Hint: try .fit_df() first."
            )

        pred = pl.sum_horizontal(
            beta * pl.col(c) for c, beta in zip(self.feature_names_in_, self.coeffs)
        )
        if self.add_bias:
            pred = pred + self.bias

        return df.with_columns(pred.alias(name))
