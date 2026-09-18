# Finite-window exponentially weighted least squares

For row position $t$, window length $W$ and positive half-life $h$, fit

$$\hat\theta_t = \arg\min_\theta \sum_{j=t-W+1}^{t}
   m_j 2^{-(t-j)/h}(y_j-\tilde x_j^T\theta)^2.$$

Here $m_j$ is one only when every predictor and the target are finite and non-null.
With an intercept, $\tilde x_j$ appends a constant one; its coefficient is last.
Missing observations keep their original positions. For positions 0, 2 and 3 in
a window ending at 3, the ages are 3, 1 and 0. The first $W-1$ rows never produce
a fit, even when `min_valid_rows` is smaller than $W$. Groups shorter than $W$
produce only null fields; empty inputs produce empty outputs.

## Updating the window

Let $\rho=2^{-1/h}$ and define

$$A_t = \sum_j m_j\rho^{t-j}\tilde x_j\tilde x_j^T,\qquad
  b_t = \sum_j m_j\rho^{t-j}\tilde x_j y_j.$$

Moving forward one row multiplies all previous contributions by $\rho$. Add the
current row with weight one, and remove row $t-W$, whose weight after aging is
$\rho^W$:

$$A_t = \rho A_{t-1} + m_t\tilde x_t\tilde x_t^T
                     - \rho^W m_{t-W}\tilde x_{t-W}\tilde x_{t-W}^T,$$
$$b_t = \rho b_{t-1} + m_t\tilde x_t y_t
                     - \rho^W m_{t-W}\tilde x_{t-W}y_{t-W}.$$

Invalid contributions are skipped before arithmetic; multiplying NaN by zero
would still produce NaN. The valid-observation count is updated separately and
does not count effective sample size. Solving $A_t\hat\theta_t=b_t$ gives the
weighted least-squares solution when the design is estimable. Weight
normalization does not change this unregularized solution.

## Numerical behavior

Accumulation, solving and output dtype follow `LIN_REG_EXPR_F64`, as in the other
regression kernels. Each solve scales
the Gram matrix by the square roots of its diagonal and uses Cholesky
factorization. This avoids treating a change in predictor units as a change in
rank. Failed/nonfinite factorizations or coefficients produce null fields.

As in `lin_reg`'s `singular_x_tol` gate, the relative determinant
$\det(A_t)/\prod_i(A_t)_{ii}$ must exceed a tolerance: $10^{-12}$ for Float64
output and $10^{-6}$ for Float32 output. The calculation uses the Cholesky
diagonal in log space and reuses the solve's factorization. This is a numerical
degeneracy policy, not an exact symbolic rank test or NumPy's SVD `rcond` rule.
All predictors, including the optional intercept, count toward the minimum
number of observations needed for an identifiable fit.

The implementation normalizes weights to the newest valid row $s_t$ in the
window, using $2^{-(s_t-j)/h}$. This divides every weight by the same positive
factor $2^{-(t-s_t)/h}$ and preserves the minimizer. Between valid observations,
the remaining state does not decay into subnormal numbers. When a new valid row
arrives, the state decays by $2^{-(s_t-s_{t-1})/h}$ before adding that row. An
expired row $t-W$ is removed with weight $2^{-(s_t-(t-W))/h}$. Both differences
use original row positions, so gaps retain their full age and window width.
An empty window clears the state. Relative weights can still underflow when
valid observations are very far apart compared with the half-life.

Cross-products are rebuilt from the current window every $W$ rows. A downdate
that loses nearly all of a diagonal or right-hand-side entry also triggers a
rebuild, as do nonfinite accumulated statistics and a transition from an estimable
to a numerically degenerate window. A zero decay factor clears the previous state
directly, avoiding `0 * inf` after overflow. Persistently degenerate windows with
finite statistics do not force a full-window rebuild on every row. The state
continues to update while outputs are null, allowing a later window to recover.
The cancellation threshold accounts for machine epsilon and rebuilds earlier
for Float32. Large expired outliers are included in regression tests
because subtracting their contributions can otherwise leave roundoff artifacts.

Normal-equation methods square the conditioning of the weighted design;
ill-conditioned problems may be rejected by the gate. Extremely small
half-lives can underflow older weights and make a window numerically singular.
Periodic rebuilding limits drift but does not guarantee accuracy for arbitrarily
scaled or ill-conditioned data.

For $p$ coefficients, ordinary updates cost $O(p^2)$ and solves $O(p^3)$ per row.
The scheduled $O(Wp^2)$ rebuild every $W$ rows adds $O(p^2)$ amortized work.
For fixed $p$, the usual total cost is linear in input length; exceptional
rebuilds caused by numerical instability can increase that cost. Input and
output storage is $O(Np)$, with $O(p^2)$ cross-product state per active group.

## Output and validation

`coeffs` and `pred` are null during warm-up, when too few observations remain,
or when the fit is numerically degenerate. With a valid fit, an invalid current
target does not prevent prediction from valid current predictors. An invalid
current predictor yields a null `pred`. Predictions use the current window fit,
which includes the current target when it is valid.

The independent reference assigns ages by original window position, applies
the same joint validity mask to X, y and ages, subtracts the smallest valid age
before exponentiation to normalize the weights without underflow, and solves
`numpy.linalg.lstsq(sqrt(w)[:, None] * X, sqrt(w) * y, rcond=None)`.
Well-conditioned windows are also compared with `lin_reg(weights=...)` after
filtering all three inputs together. Tests cover both precisions, missingness,
expiry and causality, singularity and recovery, grouped/lazy queries, slices,
multiple chunks, and long histories. `half_life=None` uses the existing rolling
implementation unchanged.

Run `benchmarks/rolling_ewls.py` for isolated-process timings and peak RSS.
The default matrix is 5,000 groups by 1,500 rows, windows 252/504/1040, one/three
predictors, and complete/randomly missing/contiguously missing data. Each result
records the actual number of full-window positions, valid fits, software,
hardware and thread count. Use smaller matched cases for the per-window baseline
and long groups to separate steady-state scaling from warm-up effects.
Measured results and reproduction commands are in the
[benchmark report](../benchmarks/rolling_ewls.md).
