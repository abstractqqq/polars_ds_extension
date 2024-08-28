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
from typing import List, Tuple
from .type_alias import LRSolverMethods, NullPolicy, PolarsFrame

from polars_ds._polars_ds import PyOnlineLR

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
        lambda_
            The regularization parameters for ridge. If this is positive, then this class will solve Ridge.
        solver
            Use one of 'svd', 'cholesky' and 'qr' method to solve the least square equation. Default is 'qr'.
        feature_names_in_
            Names for the incoming features, if available. If None, the names will be empty. They will be
            learned if .fit_df() is run later, or .set_input_features() is set later.
        """
        self._lr = PyOnlineLR()

    @staticmethod
    def _handle_nulls_in_df(
        df: PolarsFrame, features: List[str], target: str, null_policy: NullPolicy
    ) -> PolarsFrame:
        if null_policy == "ignore":
            return df
        elif null_policy == "raise":
            total_null_count = (
                df.lazy().select(*features, target).null_count().collect().sum_horizontal()[0]
            )
            if total_null_count > 0:
                raise ValueError("Nulls found in Dataframe.")
            return df
        elif null_policy == "skip":
            return df.drop_nulls(subset=features + [target])
        elif null_policy == "zero":
            return df.with_columns(pl.col(features).fill_null(0.0)).drop_nulls(subset=target)
        elif null_policy == "one":
            return df.with_columns(pl.col(features).fill_null(1.0)).drop_nulls(subset=target)
        else:
            try:
                fill_value = float(null_policy)
                if np.isfinite(fill_value):
                    return df.with_columns(pl.col(features).fill_null(fill_value)).drop_nulls(
                        subset=target
                    )
                raise ValueError("When null_policy is a number, it cannot be nan or infinite.")
            except Exception as e:
                raise ValueError(f"Unknown null_policy. Error: {e}")

    @staticmethod
    def _handle_nans_in_np(
        X: np.ndarray,  # N x M
        y: np.ndarray,  # N x 1
        null_policy: NullPolicy,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if null_policy == "ignore":
            return (X, y)
        elif null_policy == "raise":
            if np.any(np.isnan(X)) | np.any(np.isnan(y)):
                raise ValueError("Nulls found in X or y.")
            return (X, y)
        elif null_policy == "skip":
            row_has_nan = np.any(np.isnan(X), axis=1) | np.any(np.isnan(y), axis=1)
            return (X[~row_has_nan], y[~row_has_nan])
        elif null_policy == "zero":
            y_nans = np.any(np.isnan(y), axis=1)
            return (np.nan_to_num(X, nan=0.0)[~y_nans], y[~y_nans])
        elif null_policy == "one":
            y_nans = np.any(np.isnan(y), axis=1)
            return (np.nan_to_num(X, nan=1.0)[~y_nans], y[~y_nans])
        else:
            try:
                fill_value = float(null_policy)
                if np.isfinite(fill_value):
                    y_nans = np.any(np.isnan(y), axis=1)
                    return (np.nan_to_num(X, nan=fill_value)[~y_nans], y[~y_nans])
                raise ValueError("When null_policy is a number, it cannot be nan or infinite.")
            except Exception as e:
                raise ValueError(f"Unknown null_policy. Error: {e}")

    @classmethod
    def from_coeffs(
        cls, coeffs: List[float], bias: float = 0.0, feature_names_in_: List[str] | None = None
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
            output += f"Coefficients: {list(round(x, 5) for x in self._lr.coeffs)}\n"
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

    def coeffs(self) -> np.ndarray:
        """
        Returns a copy of the coefficients.
        """
        return self._lr.coeffs

    def fit(self, X: np.ndarray, y: np.ndarray, null_policy: NullPolicy = "ignore") -> Self:
        """
        Fit the linear regression model on NumPy data.

        Parameters
        ----------
        X
            The feature Matrix. NumPy 2D matrix only.
        y
            The target data. NumPy array. Must be reshape-able to (-1, 1).
        null_policy: Literal['raise', 'skip', 'zero', 'one', 'ignore']
            One of options shown here, but you can also pass in any numeric string. E.g you may pass '1.25' to mean
            fill nulls with 1.25. If the string cannot be converted to a float, an error will be thrown. Note: if
            the target column has null, the rows with nulls will always be dropped. Null-fill only applies to non-target
            columns. If target has null, then the row will still be dropped.
        """
        self._lr.fit(*LR._handle_nans_in_np(X, y.reshape((-1, 1)), null_policy))
        return self

    def fit_df(
        self,
        df: PolarsFrame,
        features: List[str],
        target: str,
        null_policy: NullPolicy = "skip",
        show_report: bool = False,
    ) -> Self:
        """
        Fit the linear regression model on a dataframe. This will overwrite previously set feature names.
        The null policy only handles null values in df, not NaN values. It is the user's responsibility to handle
        NaN values if they exist in their pipeline.

        Parameters
        ----------
        df
            Either an eager or a lazy Polars dataframe.
        features
            List of strings of column names.
        target
            The target column's name.
        null_policy: Literal['raise', 'skip', 'zero', 'one', 'ignore']
            One of options shown here, but you can also pass in any numeric string. E.g you may pass '1.25' to mean
            fill nulls with 1.25. If the string cannot be converted to a float, an error will be thrown. Note: if
            the target column has null, the rows with nulls will always be dropped. Null-fill only applies to non-target
            columns. If target has null, then the row will still be dropped.
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

        df2 = (
            LR._handle_nulls_in_df(df.lazy(), features, target, null_policy)
            .select(*features, target)
            .collect()
        )
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
            Data to predict on, as a matrix
        """
        if X.ndim == 1:
            return self._lr.predict(X.reshape((1, -1)))
        elif X.ndim == 2:
            return self._lr.predict(X)
        else:
            return ValueError("Dimension not supported.")

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


class OnlineLR:
    """
    Normal or Ridge Online Regression. Currently doesn't support dataframe inputs.
    """

    def __init__(
        self,
        fit_bias: bool = False,
        lambda_: float = 0.0,
    ):
        """
        Parameters
        ----------
        fit_bias
            Whether to add a bias term. Also known as intercept in other packages.
        lambda_
            The regularization parameters for ridge. If this is positive, then this class will solve Ridge.
        """
        self._lr = PyOnlineLR(fit_bias=fit_bias, lambda_=lambda_)

    @classmethod
    def from_coeffs_and_inv(cls, coeffs: List[float], inv: np.ndarray, bias: float = 0.0) -> Self:
        """
        Constructs an online linear regression instance from coefficients, inverse and bias.

        Parameters
        ----------
        coeffs
            Iterable of numbers representing the coefficients
        inv
            2D NumPy matrix representing the inverse of XtX in a regression problem.
        bias
            Value for the bias term
        """
        coefficients = np.ascontiguousarray(coeffs, dtype=np.float64).flatten()
        lr = cls(fit_bias=(bias != 0.0), lambda_=0.0)
        lr._lr.set_coeffs_bias_inverse(coefficients, inv, bias)
        return lr

    def is_fit(self) -> bool:
        return self._lr.is_fit()

    def __repr__(self) -> str:
        if self._lr.lambda_ > 0.0:
            output = "Online Linear Regression (Ridge) Model\n"
        else:
            output = "Online Linear Regression Model\n"

        if self._lr.is_fit():
            output += f"Coefficients: {list(round(x, 5) for x in self._lr.coeffs)}\n"
            output += f"Bias/Intercept: {self._lr.bias}\n"
        else:
            output += "Not fitted yet."
        return output

    def coeffs(self) -> np.ndarray:
        """
        Returns a copy of the current coefficients.
        """
        return self._lr.coeffs

    def inv(self) -> np.ndarray:
        """
        Returns a copy of the current inverse matrix (inverse of XtX in a linear regression).
        """
        return self._lr.inv

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """
        Initial Fit for the online linear regression model on NumPy data.

        Parameters
        ----------
        X
            The feature Matrix. NumPy 2D matrix only.
        y
            The target data. NumPy array. Must be reshape-able to (-1, 1).
        """
        if np.any(np.isnan(X)) | np.any(np.isnan(y)):
            raise ValueError(
                "Online regression currently must fit with data without null for the initial fit."
            )

        self._lr.fit(X, y)
        return self

    def update(self, X: np.ndarray, y: np.ndarray | float, c: float = 1.0) -> Self:
        """
        Updates the online linear regression model with one row of data. If the row contains np.nan,
        it will be ignored.

        Parameters
        ----------
        X
            Either a a 1d array or a 2d array with 1 row. Must be reshapeable to a matrix with 1 row.
        y
            Either a scalar, or a 1d array with 1 element, or a 2d array of size 1x1.
        c
            The middle term (C) in the woodbury matrix identity. A value of 1.0 means we add
            the impact of the new data, and a value of -1.0 means we remove the impact of the
            data. Any other value will `scale` the impact of the data.
        """
        x_2d = X.reshape((1, -1))
        if isinstance(y, float):
            y_2d = np.array([[y]])
        else:
            y_2d = y.reshape((1, 1))

        self._lr.update(x_2d, y_2d, c)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Returns the prediction of this online linear model.

        Parameters
        ----------
        X
            Data to predict on, as a matrix
        """
        if X.ndim == 1:
            return self._lr.predict(X.reshape((1, -1)))
        elif X.ndim == 2:
            return self._lr.predict(X)
        else:
            return ValueError("Dimension not supported.")
