"""
Dual value functions and budget equation solvers.

For a CMIM dual potential V^eps and the BSM state price density H_T:
  v(y, eps) = E[V^eps(y * H_T)]
  v0(y)     = E[V^0(y * H_T)]
  v1(y)     = E[V^1(y * H_T)]

Budget functions:
  g0(y) = E[H_T * I^0(y * H_T)]
  g1(y) = E[H_T * J(y * H_T)]

These expectations are computed by weighted sums over the quadrature/MC representation.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from typing import Callable, Tuple

from .bernstein import AtomicBernsteinSetup


# ──────────────────────────────────────────────────────────────────────────────
# Expectation helpers
# ──────────────────────────────────────────────────────────────────────────────

def weighted_expectation(f_vals: np.ndarray, weights: np.ndarray) -> float:
    """Compute E[f(H_T)] = sum_i w_i * f_i."""
    return float(np.dot(weights, f_vals))


# ──────────────────────────────────────────────────────────────────────────────
# Dual value functions
# ──────────────────────────────────────────────────────────────────────────────

def v_eps(
    y: float,
    eps: float,
    bm: AtomicBernsteinSetup,
    H_T: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Dual value function v(y, eps) = E[V^eps(y * H_T)].

    Parameters
    ----------
    y       : dual argument (positive)
    eps     : perturbation parameter
    bm      : AtomicBernsteinSetup instance
    H_T     : state price density samples
    weights : probability weights
    """
    yH = y * H_T
    Vvals = bm.V_eps(yH, eps)
    return weighted_expectation(Vvals, weights)


def v0_fn(
    y: float,
    bm: AtomicBernsteinSetup,
    H_T: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Baseline dual value v0(y) = E[V^0(y H_T)]."""
    return v_eps(y, 0.0, bm, H_T, weights)


def v1_fn(
    y: float,
    bm: AtomicBernsteinSetup,
    H_T: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Perturbation dual value v1(y) = E[V^1(y H_T)]."""
    return v_eps(y, 1.0, bm, H_T, weights) - v_eps(y, 0.0, bm, H_T, weights)


def v0_deriv(
    y: float,
    order: int,
    bm: AtomicBernsteinSetup,
    H_T: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    r-th derivative of v0 with respect to y:
        d^r v0 / dy^r = E[H_T^{r-1} * (d^{r-1} V^0/dy^{r-1})(y H_T)] ... simplified:
        = E[(d^r V^0/dy^r)(y H_T)] evaluated via chain rule.

    For V^0(y) = sum_i a_i (e^{-y z_i} - e^{-y* z_i})/z_i:
      d^r V^0/dy^r = (-1)^r * sum_i a_i * z_i^{r-1} * e^{-y z_i}   (for r >= 1)

    The chain rule from composing with y -> y*H_T adds a factor H_T^r when the
    outer variable is y (the scalar argument), not yH:
      dv0/dy = E[H_T * (dV0/d(yH))(yH)] = -E[H_T * I^0(yH_T)] = -g0(y)

    For general order, we need:
      d^r v0/dy^r = E[H_T^r * (d^r V^0/d(yH)^r)(yH_T)]
                 = E[H_T^r * dV0_dy^r(yH_T)]
    """
    yH = y * H_T
    # d^r V^0 / d(yH)^r = sum_i a_i * (-z_i)^{r-1} * exp(-yH * z_i)  for r>=1
    dV0_r_vals = bm.dV0_dy(yH, order=order)   # uses dV0_dy from bernstein
    return weighted_expectation(H_T ** order * dV0_r_vals, weights)


def v1_deriv(
    y: float,
    order: int,
    bm: AtomicBernsteinSetup,
    H_T: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    r-th derivative of v1 w.r.t. y:
        d^r v1 / dy^r = E[H_T^r * dV1_dy^r(y H_T)]
    """
    yH = y * H_T
    dV1_r_vals = bm.dV1_dy(yH, order=order)
    return weighted_expectation(H_T ** order * dV1_r_vals, weights)


# ──────────────────────────────────────────────────────────────────────────────
# Budget functions
# ──────────────────────────────────────────────────────────────────────────────

def g0_fn(
    y: float,
    bm: AtomicBernsteinSetup,
    H_T: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Budget function g0(y) = E[H_T * I^0(y H_T)]."""
    yH = y * H_T
    I0_vals = bm.I0(yH)
    return weighted_expectation(H_T * I0_vals, weights)


def g1_fn(
    y: float,
    bm: AtomicBernsteinSetup,
    H_T: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Perturbation budget function g1(y) = E[H_T * J(y H_T)]."""
    yH = y * H_T
    J_vals = bm.J(yH)
    return weighted_expectation(H_T * J_vals, weights)


def g0_deriv(
    y: float,
    order: int,
    bm: AtomicBernsteinSetup,
    H_T: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    r-th derivative of g0 w.r.t. y:
        g0(y) = E[H_T * I^0(y H_T)]
        d^r g0/dy^r = E[H_T^{r+1} * (I^0)^{(r)}(y H_T)]
    """
    yH = y * H_T
    dI0_r_vals = bm.dI0_dy(yH, order=order)
    return weighted_expectation(H_T ** (order + 1) * dI0_r_vals, weights)


def g1_deriv(
    y: float,
    order: int,
    bm: AtomicBernsteinSetup,
    H_T: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    r-th derivative of g1 w.r.t. y:
        d^r g1/dy^r = E[H_T^{r+1} * J^{(r)}(y H_T)]
    """
    yH = y * H_T
    dJ_r_vals = bm.dJ_dy(yH, order=order)
    return weighted_expectation(H_T ** (order + 1) * dJ_r_vals, weights)


# ──────────────────────────────────────────────────────────────────────────────
# Budget equation solver
# ──────────────────────────────────────────────────────────────────────────────

def solve_baseline_budget(
    x: float,
    bm: AtomicBernsteinSetup,
    H_T: np.ndarray,
    weights: np.ndarray,
    bracket: Tuple[float, float] = (1e-8, 1e6),
) -> float:
    """
    Solve g0(y0) = x for y0 using Brent's method.

    The budget function g0(y) = E[H_T I^0(y H_T)] is strictly decreasing
    in y (since I^0 is a decreasing Bernstein function), so there is at
    most one root.

    Parameters
    ----------
    x       : initial wealth (target budget)
    bm      : AtomicBernsteinSetup
    H_T     : state price density samples
    weights : probability weights
    bracket : bracketing interval for Brent's method

    Returns
    -------
    y0 : float, the baseline Lagrange multiplier
    """
    def f(y: float) -> float:
        return g0_fn(y, bm, H_T, weights) - x

    fa = f(bracket[0])
    fb = f(bracket[1])
    assert fa * fb < 0, (
        f"Bracket does not straddle root: f({bracket[0]})={fa:.4g}, "
        f"f({bracket[1]})={fb:.4g}. Adjust bracket."
    )
    y0 = brentq(f, bracket[0], bracket[1], xtol=1e-12, rtol=1e-12)
    return float(y0)


def solve_perturbed_budget(
    x: float,
    eps: float,
    bm: AtomicBernsteinSetup,
    H_T: np.ndarray,
    weights: np.ndarray,
    bracket: Tuple[float, float] = (1e-8, 1e6),
) -> float:
    """
    Solve E[H_T * (I^0(y H_T) + eps * J(y H_T))] = x for y_exact(eps).

    Returns
    -------
    y_exact : float
    """
    def f(y: float) -> float:
        yH = y * H_T
        integrand = H_T * (bm.I0(yH) + eps * bm.J(yH))
        return weighted_expectation(integrand, weights) - x

    fa = f(bracket[0])
    fb = f(bracket[1])
    if fa * fb >= 0:
        # Expand bracket
        bracket = (1e-10, 1e8)
        fa = f(bracket[0])
        fb = f(bracket[1])
    assert fa * fb < 0, (
        f"Perturbed bracket problem at eps={eps}: "
        f"f({bracket[0]})={fa:.4g}, f({bracket[1]})={fb:.4g}"
    )
    return float(brentq(f, bracket[0], bracket[1], xtol=1e-12, rtol=1e-12))


def collect_derivatives(
    y0: float,
    M: int,
    bm: AtomicBernsteinSetup,
    H_T: np.ndarray,
    weights: np.ndarray,
) -> Tuple[list, list, list, list]:
    """
    Collect all required derivatives at y0 up to order M.

    Returns
    -------
    g0_derivs : [g0(y0), g0'(y0), ..., g0^{(M)}(y0)]
    g1_derivs : [g1(y0), g1'(y0), ..., g1^{(M-1)}(y0)]
    v0_derivs : [v0(y0), v0'(y0), ..., v0^{(M)}(y0)]
    v1_derivs : [v1(y0), v1'(y0), ..., v1^{(M-1)}(y0)]
    """
    g0_derivs = [g0_fn(y0, bm, H_T, weights)]
    for r in range(1, M + 1):
        g0_derivs.append(g0_deriv(y0, r, bm, H_T, weights))

    g1_derivs = [g1_fn(y0, bm, H_T, weights)]
    for r in range(1, M):
        g1_derivs.append(g1_deriv(y0, r, bm, H_T, weights))

    v0_derivs = [v0_fn(y0, bm, H_T, weights)]
    v0_derivs.append(-g0_derivs[0])  # v0'(y0) = -g0(y0) = -x
    for r in range(2, M + 1):
        v0_derivs.append(v0_deriv(y0, r, bm, H_T, weights))

    v1_derivs = [v1_fn(y0, bm, H_T, weights)]
    v1_derivs.append(-g1_derivs[0])  # v1'(y0) = -g1(y0)
    for r in range(2, M):
        v1_derivs.append(v1_deriv(y0, r, bm, H_T, weights))

    return g0_derivs, g1_derivs, v0_derivs, v1_derivs
