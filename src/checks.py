"""
Residual computation and validation utilities for all experiments.

Provides functions to compute:
  - Affine identity residual (exp_1)
  - Derivative identity residual (exp_1)
  - Bell-polynomial truncation errors (exp_2, exp_3)
  - Convergence rate estimation via log-log regression
  - Normalized residuals for two-parameter sensitivity (exp_5)
"""

from __future__ import annotations

import numpy as np
from typing import List, Sequence, Tuple


def affine_residual(
    v_full: float,
    v0: float,
    v1: float,
    eps: float,
) -> float:
    """
    Affine identity residual: R = v(y, eps) - v0(y) - eps * v1(y).

    Should be zero up to numerical precision.
    """
    return v_full - v0 - eps * v1


def derivative_identity_residual(
    neg_vy: float,
    g0: float,
    g1: float,
    eps: float,
) -> float:
    """
    Derivative identity residual: R = -v_y(y, eps) - (g0(y) + eps * g1(y)).

    -v_y(y, eps) is approximated by finite difference or computed analytically.
    """
    return neg_vy - (g0 + eps * g1)


def truncation_error(
    y_exact: float,
    y_coeffs: List[float],
    eps: float,
    order: int,
) -> float:
    """
    Compute |y_exact(eps) - y^{(m)}(eps)| where
    y^{(m)}(eps) = sum_{j=0}^{m} eps^j / j! * y_j.
    """
    y_approx = sum(y_coeffs[j] * eps ** j / _factorial(j) for j in range(order + 1))
    return abs(y_exact - y_approx)


def truncation_series(
    y_exact: float,
    y_coeffs: List[float],
    eps: float,
    max_order: int,
) -> List[float]:
    """
    Compute truncation errors for orders 1..max_order.
    """
    return [truncation_error(y_exact, y_coeffs, eps, m) for m in range(1, max_order + 1)]


def convergence_rate(
    eps_vals: np.ndarray,
    errors: np.ndarray,
) -> float:
    """
    Estimate convergence rate via log-log regression: error ~ C * eps^rate.

    Only uses points where both eps and errors are positive and finite.

    Returns
    -------
    rate : estimated slope of log(error) vs log(eps)
    """
    mask = (eps_vals > 0) & (errors > 0) & np.isfinite(errors)
    if mask.sum() < 2:
        return float("nan")
    log_eps = np.log(eps_vals[mask])
    log_err = np.log(errors[mask])
    slope, _ = np.polyfit(log_eps, log_err, 1)
    return float(slope)


def normalized_multiplier_residual(
    y_exact: float,
    y0: float,
    delta_n: float,
    eps_n: float,
    y_mu: float,
    y_I: float,
) -> float:
    """
    Normalized multiplier residual for exp_5:
        r_y = |y_n - y - delta_n * y_mu - eps_n * y_I| / (|delta_n| + |eps_n|)
    """
    denom = abs(delta_n) + abs(eps_n)
    if denom == 0:
        return 0.0
    num = abs(y_exact - y0 - delta_n * y_mu - eps_n * y_I)
    return num / denom


def normalized_optimizer_residual(
    X_exact: np.ndarray,
    X0: np.ndarray,
    X_mu: np.ndarray,
    X_I: np.ndarray,
    delta_n: float,
    eps_n: float,
) -> dict:
    """
    Pathwise normalized optimizer residual for exp_5:
        r_X(omega) = |X_n - X0 - delta_n * X_mu - eps_n * X_I| / (|delta_n| + |eps_n|)

    Returns a dict with sup, mean, L2, and selected quantiles.
    """
    denom = abs(delta_n) + abs(eps_n)
    if denom == 0:
        return {"sup": 0.0, "mean": 0.0, "L2": 0.0, "q50": 0.0, "q90": 0.0}
    r = np.abs(X_exact - X0 - delta_n * X_mu - eps_n * X_I) / denom
    return {
        "sup": float(np.max(r)),
        "mean": float(np.mean(r)),
        "L2": float(np.sqrt(np.mean(r ** 2))),
        "q50": float(np.quantile(r, 0.50)),
        "q90": float(np.quantile(r, 0.90)),
    }


def _factorial(n: int) -> int:
    """Integer factorial."""
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def build_truncated_series(coeffs: List[float], eps: float, order: int) -> float:
    """
    Evaluate truncated series: sum_{j=0}^{order} eps^j / j! * coeffs[j].
    """
    return sum(coeffs[j] * eps ** j / _factorial(j) for j in range(order + 1))
