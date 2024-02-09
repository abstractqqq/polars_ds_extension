from __future__ import annotations
import polars as pl
from typing import Union
import math


@pl.api.register_expr_namespace("c")
class ComplexExt:

    """
    This class contains tools for dealing with complex numbers columns inside Polars dataframe.

    Polars Namespace: c

    Example: pl.col("a").c.modulus()

    Complex number columns are represented as a column of size-2 Arrays. By default, an element will look like [re, im],
    which is in coordinate form. All operations (except powi, which turns it into polar form internally) assume the number
    is in coordinate form. There is a to_coord function provided for complex numbers in polar form [r, theta].
    """

    def __init__(self, expr: pl.Expr):
        self._expr: pl.Expr = expr

    def re(self) -> pl.Expr:
        """Returns the real part of the complex number."""
        return self._expr.arr.first()

    def im(self) -> pl.Expr:
        """Returns the imaginary part of the complex number."""
        return self._expr.arr.last()

    def to_complex(self) -> pl.Expr:
        """Turns a column of floats into a column of complex with im = 0."""
        return pl.concat_list(self._expr, pl.lit(0.0, dtype=pl.Float64)).list.to_array()

    def with_imag(self, im: pl.Expr) -> pl.Expr:
        """
        Treats self as the real part, and the other as the imaginary part and combines
        them into a complex column. An alias for pl.concat_list(self._expr, im)

        Parameters
        ----------
        im
            Another polars expression represeting imaginary part
        """
        return pl.concat_list(self._expr, im).list.to_array()

    def modulus(self) -> pl.Expr:
        """Returns the modulus of the complex number."""
        return (self._expr.arr.first().pow(2) + self._expr.arr.last().pow(2)).sqrt()

    def squared_modulus(self) -> pl.Expr:
        """Returns the squared modulus of the complex number."""
        return self._expr.arr.first().pow(2) + self._expr.arr.last().pow(2)

    def theta(self, degree: bool = False) -> pl.Expr:
        """
        Returns the polar angle (in radians by default) of the complex number.

        Parameters
        ----------
        degree
            If true, use degree. If false, use radians.
        """
        x = self._expr.arr.first()
        y = self._expr.arr.last()
        if degree:
            return (
                pl.when((x > 0) | (y != 0))
                .then(pl.arctan2d(y, x))
                .when((x < 0) & (y == 0))
                .then(pl.lit(180.0, dtype=pl.Float64))
                .otherwise(pl.lit(math.nan, dtype=pl.Float64))
            )
        else:
            return (
                pl.when((x > 0) | (y != 0))
                .then(pl.arctan2(y, x))
                .when((x < 0) & (y == 0))
                .then(pl.lit(math.pi, dtype=pl.Float64))
                .otherwise(pl.lit(math.nan, dtype=pl.Float64))
            )

    def to_polar(self) -> pl.Expr:
        """Turns a complex number in coordinate form into polar form."""
        return pl.concat_list(self.modulus(), self.theta()).list.to_array()

    def to_coord(self) -> pl.Expr:
        """Turns a complex number in polar form into coordinate form."""
        r = self._expr.arr.first()
        theta = self._expr.arr.last()
        return pl.concat_list(r * theta.cos(), r * theta.sin()).list.to_array()

    def conj(self) -> pl.Expr:
        """Returns complex conjugate."""
        return pl.concat_list(self._expr.arr.first(), -self._expr.arr.last()).list.to_array()

    def add(self, rhs: Union[float, complex, pl.Expr]) -> pl.Expr:
        """
        Add either a single real, complex, or another col of complex to self. If other is
        an expression, it must be another col of complex numbers.

        Parameters
        ----------
        rhs
            The right hand side.
        """
        x = self._expr.arr.first()
        y = self._expr.arr.last()
        if isinstance(rhs, float):
            return pl.concat_list(x + rhs, y).list.to_array()
        if isinstance(rhs, complex):
            return pl.concat_list(x + rhs.real, y + rhs.imag).list.to_array()
        else:  # Expression must be another complex col
            return pl.concat_list(
                x + rhs.arr.first(),
                y + rhs.arr.last(),
            ).list.to_array()

    def sub(self, rhs: Union[float, complex, pl.Expr]) -> pl.Expr:
        """
        Subtract either a single real, complex, or another col of complex to self. If other is
        an expression, it must be another col of complex numbers.

        Parameters
        ----------
        rhs
            The right hand side.
        """
        return self.add(-rhs)

    def mul(self, rhs: Union[float, complex, pl.Expr]) -> pl.Expr:
        """
        Multiply either a single real, complex, or another col of complex to self. If other is
        an expression, it must be another col of complex numbers.

        Parameters
        ----------
        rhs
            The right hand side.
        """
        x = self._expr.arr.first()
        y = self._expr.arr.last()
        if isinstance(rhs, float):
            return pl.concat_list(x * rhs, y * rhs).list.to_array()
        if isinstance(rhs, complex):
            new_real = x * rhs.real - y * rhs.imag
            new_imag = x * rhs.imag + y * rhs.real
            return pl.concat_list(new_real, new_imag).list.to_array()
        else:
            x2 = rhs.arr.first()
            y2 = rhs.arr.last()
            new_real = x * x2 - y * y2
            new_imag = x * y2 + y * x2
            return pl.concat_list(new_real, new_imag).list.to_array()

    def inv(self) -> pl.Expr:
        """Returns 1/z for a complex number z."""
        x = self._expr.arr.first()
        y = self._expr.arr.last()
        denom = x.pow(2) + y.pow(2)
        return pl.concat_list(x / denom, -y / denom).list.to_array()

    def div(self, rhs: Union[float, complex, pl.Expr]) -> pl.Expr:
        """
        Divide either a single real, complex, or another col of complex to self. If other is
        an expression, it must be another col of complex numbers.

        Parameters
        ----------
        rhs
            The right hand side.
        """
        return self.mul(1 / rhs)

    def mul_by_i(self) -> pl.Expr:
        """Multiplies self by i."""
        x = self._expr.arr.first()
        y = self._expr.arr.last()
        return pl.concat_list(-y, x).list.to_array()

    def pow(self, x: float) -> pl.Expr:
        """
        Raises a complex number to the x power.

        Parameters
        ----------
        x
            Only supports real power now.
        """
        if x == 0.0:
            return pl.concat_list(
                pl.when(self.modulus() == 0.0).then(math.nan).otherwise(1.0),
                pl.lit(0.0, dtype=pl.Float64),
            ).list.to_array()
        elif x == 1.0:
            return self._expr
        elif x == 2.0:
            return self.mul(self._expr)
        elif x == -1.0:
            return self.inv()
        else:
            polar = self.to_polar()
            r = polar.arr.first()
            theta = polar.arr.last()
            return pl.concat_list(
                r.pow(x) * (x * theta).cos(), r.pow(x) * (x * theta).sin()
            ).list.to_array()
