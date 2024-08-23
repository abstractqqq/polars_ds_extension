"""
Linear models. Currently, only supports Polars DataFrame Inputs or NumPy matrices as inputs. This module is in 
very early development and is subject to frequent breaking changes.

This module requires the NumPy package. PDS only requires Polars, but you can get all the optional dependencies by

`pip install polars_ds[all]`

"""

# Currently skipping tests for this module because the underlying functions are all tested in one way or
# another.

from __future__ import annotations

import polars as pl
import numpy as np
from typing import List
from .type_alias import LRSolverMethods, PolarsFrame

from polars_ds._polars_ds import PyLR

import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:  # 3.10, 3.9, 3.8
    from typing_extensions import Self


class LR:

    """
    Normal and Ridge Regression.
    """

    def __init__(
        self,
        fit_bias: bool = False,
        lambda_: float = 0.0,
        solver: LRSolverMethods = "qr",
        feature_names_in_: List[str] | None = None,
    ):
        """
        Parameters
        ----------
        fit_bias
            Whether to add a bias term. Also known as intercept in other packages.
        solver
            Use one of 'svd', 'cholesky' and 'qr' method to solve the least square equation. Default is 'qr'.
        lambda_
            The regularization parameters for ridge. If this is positive, then this class will solve Ridge.
        feature_names_in_
            Names for the incoming features, if available. If None, the names will be empty. They will be
            learned if .fit_df() is run later, or .set_input_features() is set later.
        """
        self._lr = PyLR(fit_bias=fit_bias, solver=solver, lambda_=lambda_)
        self.feature_names_in_ = [] if feature_names_in_ is None else list(feature_names_in_)

    @classmethod
    def from_values(
        cls, coeffs: List[float], bias: float, feature_names_in_: List[str] | None = None
    ) -> Self:
        """
        Constructs a LR class instance from coefficients and bias. This always assumes the coefficients come
        from a normal linear regression, not Ridge.

        Parameters
        ----------
        coeffs
            Iterable of numbers representing the coefficients
        bias
            Value for the bias term
        feature_names_in_
            Names for the incoming features, if available. If None, the names will be empty. They will be
            learned if .fit_df() is run later, or .set_input_features() is set later.
        """
        coefficients = np.ascontiguousarray(coeffs, dtype=np.float64).flatten()
        lr = cls(
            fit_bias=(bias != 0.0), lambda_=0.0, solver="qr", feature_names_in_=feature_names_in_
        )
        lr._lr.set_coeffs_and_bias(coefficients, bias)
        return lr

    def is_fit(self) -> bool:
        return self._lr.is_fit()

    def __repr__(self) -> str:
        if self._lr.lambda_ > 0.0:
            output = "Linear Regression (Ridge) Model\n"
        else:
            output = "Linear Regression Model\n"

        if self._lr.is_fit():
            output += f"Coefficients: {list(self._lr.coeffs)}\n"
            output += f"Bias/Intercept: {self._lr.bias}\n"
        else:
            output += "Not fitted yet."
        return output

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
        self._lr.fit(X, y.reshape((-1, 1)))
        return self

    def fit_df(
        self, df: PolarsFrame, features: List[str], target: str, show_report: bool = False
    ) -> Self:
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
        show_report
            Whether to print out a regression report. This will duplicate work and will not work for Ridge
            regression. E.g. Nothing will be printed if lambda_ > 0.
        """
        if show_report and self._lr.lambda_ == 0.0:
            from . import query_lstsq_report

            print(
                df.lazy()
                .select(
                    query_lstsq_report(
                        *features,
                        target=target,
                    ).alias("report")
                )
                .unnest("report")
                .collect()
            )

        df2 = df.lazy().select(*features, target).collect()
        X = df2.select(features).to_numpy()
        y = df2.select(target).to_numpy()
        self.feature_names_in_ = list(features)
        return self.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Returns the prediction of this linear model.

        Parameters
        ----------
        X
            New feature matrix
        """
        return self._lr.predict(X)

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
                "The linear model is not fitted on a dataframe, or no feature names have been given."
                "Not enough info to predict on a dataframe. Hint: try .fit_df() or .set_input_features()."
            )

        pred = pl.sum_horizontal(
            beta * pl.col(c) for c, beta in zip(self.feature_names_in_, self._lr.coeffs)
        )
        bias = self._lr.bias
        if bias != 0:
            pred = pred + bias

        return df.with_columns(pred.alias(name))
