import polars as pl
from typing import Union
import math


@pl.api.register_expr_namespace("c")
class ComplexExt:

    """
    This class contains tools for dealing with complex numbers columns inside Polars dataframe.

    Polars Namespace: c

    Example: pl.col("a").c.modulus()

    Complex number columns are represented as a column of size-2 lists. By default, an element will look like [re, im],
    which is in coordinate form. All operations (except powi, which turns it into polar form internally) assume the number
    is in coordinate form. There is a to_coord function provided for complex numbers in polar form [r, theta].
    """

    def __init__(self, expr: pl.Expr):
        self._expr: pl.Expr = expr

    def re(self) -> pl.Expr:
        """Returns the real part of the complex number."""
        return self._expr.list.first()

    def im(self) -> pl.Expr:
        """Returns the imaginary part of the complex number."""
        return self._expr.list.last()

    def to_complex(self) -> pl.Expr:
        """Turns a column of floats into a column of complex with im = 0."""
        return pl.concat_list(self._expr, pl.lit(0.0, dtype=pl.Float64))

    def with_imag(self, other: pl.Expr) -> pl.Expr:
        """
        Treats self as the real part, and other as the imaginary part and combines
        them into a complex column. An alias for pl.concat_list(self._expr, other)
        """
        return pl.concat_list(self._expr, other)

    def modulus(self) -> pl.Expr:
        """Returns the modulus of the complex number."""
        return self._expr.list.eval(pl.element().dot(pl.element()).sqrt()).list.first()

    def squared_modulus(self) -> pl.Expr:
        """Returns the squared modulus of the complex number."""
        return self._expr.list.eval(pl.element().dot(pl.element())).list.first()

    def theta(self, degree: bool = False) -> pl.Expr:
        """Returns the polar angle (in radians by default) of the complex number."""
        x = self._expr.list.first()
        y = self._expr.list.last()
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
        return pl.concat_list(self.modulus(), self.theta())

    def to_coord(self) -> pl.Expr:
        """Turns a complex number in polar form into coordinate form."""
        r = self._expr.list.first()
        theta = self._expr.list.last()
        return pl.concat_list(r * theta.cos(), r * theta.sin())

    def conj(self) -> pl.Expr:
        """Returns complex conjugate."""
        return pl.concat_list(self._expr.list.first(), -self._expr.list.last())

    def add(self, other: Union[float, complex, pl.Expr]) -> pl.Expr:
        """
        Add either a single real, complex, or another col of complex to self. If other is
        an expression, it must be another col of complex numbers.
        """
        if isinstance(other, float):
            return self._expr.list.eval(pl.element() + pl.Series([other, 0]))
        if isinstance(other, complex):
            return self._expr.list.eval(pl.element() + pl.Series([other.real, other.imag]))
        else:
            return pl.concat_list(
                self._expr.list.first() + other.list.first(),
                self._expr.list.last() + other.list.last(),
            )

    def sub(self, other: Union[float, complex, pl.Expr]) -> pl.Expr:
        """
        Subtract either a single real, complex, or another col of complex to self. If other is
        an expression, it must be another col of complex numbers.
        """
        if isinstance(other, float):
            return self._expr.list.eval(pl.element() - pl.Series([other, 0]))
        if isinstance(other, complex):
            return self._expr.list.eval(pl.element() - pl.Series([other.real, other.imag]))
        else:
            return pl.concat_list(
                self._expr.list.first() - other.list.first(),
                self._expr.list.last() - other.list.last(),
            )

    def mul(self, other: Union[float, complex, pl.Expr]) -> pl.Expr:
        """
        Multiply either a single real, complex, or another col of complex to self. If other is
        an expression, it must be another col of complex numbers.
        """
        if isinstance(other, float):
            return self._expr.list.eval(pl.element() * pl.lit(other))
        if isinstance(other, complex):
            x = self._expr.list.first()
            y = self._expr.list.last()
            new_real = x * other.real - y * other.imag
            new_imag = x * other.imag + y * other.real
            return pl.concat_list(new_real, new_imag)
        else:
            x = self._expr.list.first()
            y = self._expr.list.last()
            x2 = other.list.first()
            y2 = other.list.last()
            new_real = x * x2 - y * y2
            new_imag = x * y2 + y * x2
            return pl.concat_list(new_real, new_imag)

    def inv(self) -> pl.Expr:
        """Returns 1/z for a complex number z."""
        x = self._expr.list.first()
        y = self._expr.list.last()
        denom = x.pow(2) + y.pow(2)
        return pl.concat_list(x / denom, -y / denom)

    def div(self, other: Union[float, complex, pl.Expr]) -> pl.Expr:
        """
        Divide either a single real, complex, or another col of complex to self. If other is
        an expression, it must be another col of complex numbers.
        """
        if isinstance(other, float):
            return self._expr.list.eval(pl.element() / pl.lit(other))
        if isinstance(other, complex):
            x = self._expr.list.first()
            y = self._expr.list.last()
            inverse = 1 / other
            new_real = x * inverse.real - y * inverse.imag
            new_imag = x * inverse.imag + y * inverse.real
            return pl.concat_list(new_real, new_imag)
        else:
            x = self._expr.list.first()
            y = self._expr.list.last()
            x2 = other.list.first()
            y2 = other.list.last()
            denom = x2.pow(2) + y2.pow(2)
            x_inv = x2 / denom
            y_inv = -y2 / denom
            new_real = x * x_inv - y * y_inv
            new_imag = x * y_inv + y * x_inv
            return pl.concat_list(new_real, new_imag)

    def mul_by_i(self) -> pl.Expr:
        """Multiplies self by i."""
        x = self._expr.list.first()
        y = self._expr.list.last()
        return pl.concat_list(-y, x)

    def pow(self, x: float) -> pl.Expr:
        """Raises a complex number to the x power."""
        if x == 0.0:
            return pl.concat_list(
                pl.when(self.modulus() == 0.0).then(math.nan).otherwise(1.0),
                pl.lit(0.0, dtype=pl.Float64),
            )
        elif x == 1.0:
            return self._expr
        elif x == 2.0:
            return self.mul(self._expr)
        elif x == -1.0:
            return self.inv()
        else:
            polar = self.to_polar()
            r = polar.list.first()
            theta = polar.list.last()
            return pl.concat_list(r.pow(x) * (x * theta).cos(), r.pow(x) * (x * theta).sin())
