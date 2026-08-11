"""
CRRA (Constant Relative Risk Aversion) utilities for exp_4.

For CRRA utility with power p (p < 1, p != 0):
    U^p(x) = x^p / p  (utility)
    (U^p)'(x) = x^{p-1}  (marginal utility)
    Inverse marginal: I^p(y) = y^{1/(p-1)}

Budget function:
    g^p(y) = E[H_T * I^p(y H_T)] = E[H_T * (y H_T)^{1/(p-1)}]
           = y^{1/(p-1)} * E[H_T^{p/(p-1)}]
           = y^{1/(p-1)} * M(p/(p-1))

where M(alpha) = E[H_T^alpha] is the moment function.

For BSM with H_T lognormal:
    M(alpha) = exp(-alpha*r*T + 0.5*(alpha^2 - alpha)*theta^2*T)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from typing import Tuple

from .model import BSMParams, bsm_moment


def crra_inverse_marginal(y: np.ndarray, p: float) -> np.ndarray:
    """
    Inverse marginal utility for CRRA: I^p(y) = y^{1/(p-1)}.

    Parameters
    ----------
    y : array of positive values
    p : risk aversion parameter (p < 1, p != 0)

    Returns
    -------
    array of same shape as y
    """
    assert p < 1 and p != 0, f"p must satisfy p < 1, p != 0; got p={p}"
    return y ** (1.0 / (p - 1))


def crra_inverse_marginal_deriv(y: np.ndarray, p: float, order: int = 1) -> np.ndarray:
    """
    r-th derivative of I^p w.r.t. y:
        d^r/dy^r y^{1/(p-1)} = prod_{j=0}^{r-1} (1/(p-1) - j) * y^{1/(p-1) - r}
    """
    alpha = 1.0 / (p - 1)
    coeff = 1.0
    for j in range(order):
        coeff *= alpha - j
    return coeff * y ** (alpha - order)


def crra_baseline_y0(
    x: float, p0: float, params: BSMParams
) -> float:
    """
    Closed-form baseline Lagrange multiplier for CRRA(p0):
        x = y0^{1/(p0-1)} * M(a)  where a = p0/(p0-1)
        y0 = (x / M(a))^{p0-1}

    Returns y0 > 0.
    """
    a = p0 / (p0 - 1)
    Ma = bsm_moment(params, a)
    y0 = (x / Ma) ** (p0 - 1)
    return float(y0)


def crra_g0(y: float, p0: float, params: BSMParams) -> float:
    """
    Budget function g0(y) = E[H_T * I^{p0}(y H_T)] = y^{1/(p0-1)} * M(p0/(p0-1)).
    """
    a = p0 / (p0 - 1)
    Ma = bsm_moment(params, a)
    return float(y ** (1.0 / (p0 - 1)) * Ma)


def crra_g0_deriv(y: float, p0: float, params: BSMParams, order: int = 1) -> float:
    """
    r-th derivative of g0(y) = y^{1/(p0-1)} * M(a):
        d^r g0/dy^r = prod_{j=0}^{r-1} (1/(p0-1) - j) * y^{1/(p0-1) - r} * M(a)
    """
    a = p0 / (p0 - 1)
    Ma = bsm_moment(params, a)
    alpha = 1.0 / (p0 - 1)
    coeff = 1.0
    for j in range(order):
        coeff *= alpha - j
    return float(coeff * y ** (alpha - order) * Ma)


def crra_g1(y: float, p1: float, params: BSMParams) -> float:
    """
    Perturbation budget function g1(y) = E[H_T * I^{p1}(y H_T)] = y^{1/(p1-1)} * M(b)
    where b = 1 + 1/(p1-1) = p1/(p1-1).
    """
    b = p1 / (p1 - 1)
    Mb = bsm_moment(params, b)
    return float(y ** (1.0 / (p1 - 1)) * Mb)


def crra_g1_deriv(y: float, p1: float, params: BSMParams, order: int = 1) -> float:
    """r-th derivative of g1(y)."""
    b = p1 / (p1 - 1)
    Mb = bsm_moment(params, b)
    alpha = 1.0 / (p1 - 1)
    coeff = 1.0
    for j in range(order):
        coeff *= alpha - j
    return float(coeff * y ** (alpha - order) * Mb)


def crra_y1_closed_form(y0: float, p0: float, p1: float, params: BSMParams) -> float:
    """
    Closed-form first-order multiplier correction (from paper Section 4):
        y1 = (1 - p0) * y0^{1/(p1-1) - 1/(p0-1) + 1} * M(b) / M(a)

    where a = p0/(p0-1) and b = p1/(p1-1).
    """
    a = p0 / (p0 - 1)
    b = p1 / (p1 - 1)
    Ma = bsm_moment(params, a)
    Mb = bsm_moment(params, b)
    exponent = 1.0 / (p1 - 1) - 1.0 / (p0 - 1) + 1.0
    return float((1.0 - p0) * y0 ** exponent * Mb / Ma)


def crra_solve_perturbed_y(
    x: float,
    eps: float,
    p0: float,
    p1: float,
    params: BSMParams,
    bracket: Tuple[float, float] = (1e-12, 1e12),
) -> float:
    """
    Solve E[H_T(I^{p0}(yH_T) + eps * I^{p1}(yH_T))] = x for y_exact(eps).

    Using analytical moments:
        E[H_T * (y H_T)^{1/(p-1)}] = y^{1/(p-1)} * M(p/(p-1))

    So the equation becomes:
        y^{1/(p0-1)} * M(a) + eps * y^{1/(p1-1)} * M(b) = x
    """
    a = p0 / (p0 - 1)
    b = p1 / (p1 - 1)
    Ma = bsm_moment(params, a)
    Mb = bsm_moment(params, b)

    def f(y: float) -> float:
        return y ** (1.0 / (p0 - 1)) * Ma + eps * y ** (1.0 / (p1 - 1)) * Mb - x

    fa = f(bracket[0])
    fb = f(bracket[1])
    if fa * fb >= 0:
        # Try to find a valid bracket automatically
        for lo, hi in [(1e-15, 1e-5), (1e-5, 1e0), (1e0, 1e5), (1e5, 1e15)]:
            if f(lo) * f(hi) < 0:
                bracket = (lo, hi)
                fa, fb = f(lo), f(hi)
                break

    assert fa * fb < 0, (
        f"Cannot bracket root for perturbed CRRA budget at eps={eps}: "
        f"f({bracket[0]:.2e})={fa:.4g}, f({bracket[1]:.2e})={fb:.4g}"
    )
    return float(brentq(f, bracket[0], bracket[1], xtol=1e-14, rtol=1e-14))
