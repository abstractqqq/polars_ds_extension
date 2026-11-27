## Linear Models Related Queries

### Quantile regression

`quantile_reg` fits a non-rolling conditional quantile, rather than the conditional
mean fitted by least squares. For example, `quantile=0.9` can describe the upper
part of a response distribution. Coefficients are returned in predictor order,
with the optional bias last, matching `lin_reg`.

```python
import polars as pl
import polars_ds as pds

df = pl.DataFrame({
    "group": ["a", "a", "a", "a", "b", "b", "b", "b"],
    "x": [0., 1., 2., 3., 0., 1., 2., 3.],
    "y": [1., 3., 4., 8., 2., 4., 7., 9.],
})
expr = pds.quantile_reg("x", target="y", quantile=0.9, add_bias=True)
coefficients = df.lazy().group_by("group").agg(expr).collect()
predictions = df.select(
    pds.quantile_reg("x", target="y", add_bias=True, return_pred=True)
).unnest("quantile_pred")  # pred, resid (= target - pred)
```

This first version uses Float64 and a full-column-rank design, with at least as
many retained samples as coefficients. It supports the existing null policies:
`skip` drops rows with any null, `raise` rejects nulls, and `zero`, `one` or a
numeric string fills predictors while dropping null targets. Predictions and
residuals are null at dropped rows. `ignore` requires null-free input. Non-finite
values remaining after null handling, empty fits and rank-deficient designs raise
`ComputeError`, also inside groups. Rolling fits, weights, regularization and
multiple targets are outside this version.

#### Objective and IRLS derivation

For residuals `r_i = y_i - x_i.T @ beta`, the objective is the mean pinball loss:

```text
P(beta) = sum_i rho_q(r_i) / n
rho_q(r) = max(q*r, (q-1)*r) = |r|/2 + (q-1/2)*r
```

The implementation uses the smooth surrogate
`rho_q_delta(r) = sqrt(r*r + delta*delta)/2 + (q-1/2)*r`.
Its difference from the pinball loss lies between zero and `delta/2`.
Concavity of the square root gives, at current residual `r_old` and
`s = sqrt(r_old*r_old + delta*delta)`,

```text
sqrt(r*r + delta*delta) <= s + (r*r - r_old*r_old)/(2*s)
```

After completing the square, minimizing this tangent majorizer is weighted
least squares with weights `1/s_i` and working targets `y_i + (2*q-1)*s_i`.
The surrogate is tangent at the current iterate, so an exact weighted solve
cannot increase the smooth objective. Each update uses column-pivoted QR in
faer, without forming `X.T @ X` or adding a runtime dependency.

Columns and the response are divided by their maximum absolute values before
solving. The response scale is one for an all-zero response. Coefficients are
then converted back to the original units. The rank threshold is
`machine_epsilon * max(n,p) * abs(R[0,0])` on the normalized design's pivoted QR
diagonal. The smoothing parameter in normalized units is
`delta = max(0.05 * tol, 1e-12)`.

#### Stopping rule and limitations

Small coefficient changes alone do not establish optimality. The pinball
problem's dual maximizes `D(u) = y.T @ u / n` subject to
`X.T @ u = 0` and `q-1 <= u_i <= q`. Weak duality gives
`D(u) <= P(beta_optimal) <= P(beta)`, so `P(beta)-D(u)` bounds the objective error.
Indeed, the box gives `u_i*r_i <= rho_q(r_i)`, and stationarity gives
`y.T @ u = (y-X@beta).T @ u`, proving the bound for every feasible dual.
For a numerical dual candidate, start from the smooth score
`psi_i = q - 1/2 + r_i / (2*sqrt(r_i*r_i + delta*delta))`, remove its projection
onto the column space of `X` using the thin QR basis, and scale toward zero until
it fits the box. Projection is repeated once to reduce roundoff.

Success requires a numerical primal-dual gap at most
`tol * (1 + mean_loss)` in normalized response units. In original units this is
`tol * (response_scale + mean_loss)`, up to floating-point error. Thus `tol` is a
loss tolerance; it does not guarantee coefficient accuracy when solutions are
non-unique. Tests compare loss with a separate SciPy HiGHS linear program, and
compare identifiable fits with statsmodels.

IRLS may converge slowly for extreme quantiles or difficult designs. If the
criterion is not met after `max_iter` updates (default 10,000), the expression
raises `ComputeError` rather than returning a partial fit. Increase `max_iter`
or relax `tol` when appropriate. Very small tolerances can reach floating-point
or smoothing limits. Rank-deficient designs are rejected explicitly; full rank
does not imply a unique pinball minimizer.

Background: [statsmodels QuantReg (IRLS)](https://www.statsmodels.org/stable/generated/statsmodels.regression.quantile_regression.QuantReg.html),
[Hunter and Lange, Quantile Regression via an MM Algorithm (2000)](https://doi.org/10.1080/10618600.2000.10474866),
and [SciPy's HiGHS linear-programming interface](https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs.html).
The square-root surrogate and gap construction used here are derived above;
they are not a copy of the statsmodels solver or the paper's specific smoothing.

::: polars_ds.exprs.expr_linear
