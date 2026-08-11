"""
Partial Bell polynomials and Bell-polynomial-based coefficient recursions.

Partial Bell polynomial (second kind) B_{n,k}(x_1,...,x_{n-k+1}):
    B_{n,k} = sum_{partitions} n!/(j1! j2! ... ) * prod_i (x_i/i!)^{j_i}
    where sum j_l = k and sum l*j_l = n.

Convention: B_{0,0} = 1, B_{n,0} = 0 for n >= 1, B_{0,k} = 0 for k >= 1.

Recursion (from the partition expansion):
    B_{n,k}(x1,...) = sum_{j=1}^{n-k+1} C(n-1, j-1) * x_j * B_{n-j, k-1}(x1,...)

This module also provides:
  - bell_poly_matrix: precomputed B_{n,k} table for all n,k up to max_order
  - lagrange_multiplier_recursion: compute y_0,...,y_M from g0/g1 derivatives
  - primal_value_recursion: compute u_m from v0/v1 derivatives and y_m
  - optimizer_recursion: compute pathwise X_m from I0/J derivatives and Z_m
"""

from __future__ import annotations

import numpy as np
from math import comb, factorial
from typing import Callable, List, Sequence, Tuple, Union

ArrayLike = Union[float, np.ndarray]


def partial_bell(n: int, k: int, xs: Sequence[float]) -> float:
    """
    Compute the scalar partial Bell polynomial B_{n,k}(x_1,...,x_{n-k+1}).

    Uses the recursion:
        B_{n,k}(x) = sum_{j=1}^{n-k+1} C(n-1,j-1) * x_j * B_{n-j,k-1}(x)

    Parameters
    ----------
    n : total weight (n >= 0)
    k : number of parts (0 <= k <= n)
    xs : sequence where xs[i] = x_{i+1} (0-indexed), length >= n-k+1

    Returns
    -------
    float : value of B_{n,k}
    """
    # Base cases
    if n == 0 and k == 0:
        return 1.0
    if k == 0 or k > n or n < 0 or k < 0:
        return 0.0

    result = 0.0
    upper = n - k + 1  # j ranges from 1 to n-k+1
    for j in range(1, upper + 1):
        if j - 1 >= len(xs):
            break
        x_j = xs[j - 1]  # x_j is xs[j-1] in 0-indexed
        result += comb(n - 1, j - 1) * x_j * partial_bell(n - j, k - 1, xs)
    return result


def partial_bell_array(n: int, k: int, xs: np.ndarray) -> np.ndarray:
    """
    Pathwise version of B_{n,k}: xs is a 2D array of shape (n-k+1, n_paths).
    Returns an array of shape (n_paths,).

    Same recursion as partial_bell but applied elementwise along paths.
    """
    if n == 0 and k == 0:
        return np.ones(xs.shape[1] if xs.ndim == 2 else 1)
    if k == 0 or k > n or n < 0 or k < 0:
        return np.zeros(xs.shape[1] if xs.ndim == 2 else 1)

    n_paths = xs.shape[1] if xs.ndim == 2 else len(xs[0])
    result = np.zeros(n_paths)
    upper = n - k + 1
    for j in range(1, upper + 1):
        if j - 1 >= xs.shape[0]:
            break
        x_j = xs[j - 1]  # shape (n_paths,)
        result += comb(n - 1, j - 1) * x_j * partial_bell_array(n - j, k - 1, xs)
    return result


def bell_poly_table(max_n: int, xs: Sequence[float]) -> np.ndarray:
    """
    Precompute B_{n,k} for all 0 <= n <= max_n, 0 <= k <= n.

    Returns a 2D array B of shape (max_n+1, max_n+1) where B[n,k] = B_{n,k}(xs).
    """
    B = np.zeros((max_n + 1, max_n + 1))
    B[0, 0] = 1.0
    for n in range(1, max_n + 1):
        for k in range(1, n + 1):
            B[n, k] = partial_bell(n, k, xs)
    return B


def lagrange_multiplier_recursion(
    g0_derivs: List[float],
    g1_derivs: List[float],
    M: int,
) -> List[float]:
    """
    Compute Taylor coefficients y_0, y_1, ..., y_M of the Lagrange multiplier
    y(x, epsilon) = sum_{m=0}^{M} epsilon^m / m! * y_m.

    Inputs
    ------
    g0_derivs : [g0(y0), g0'(y0), g0''(y0), ..., g0^{(M)}(y0)]
                length M+1
    g1_derivs : [g1(y0), g1'(y0), ..., g1^{(M-1)}(y0)]
                length M (index r corresponds to g1^{(r)}(y0))
    M : maximum order

    Algorithm
    ---------
    y_0 = y0 (provided as first element of g0_derivs, used to solve budget; passed separately)
    For m >= 1:
        A_m = sum_{r=2}^{m} g0^{(r)}(y0) * B_{m,r}(y1,...,y_{m-r+1})
            + m * sum_{r=0}^{m-1} g1^{(r)}(y0) * B_{m-1,r}(y1,...,y_{m-r})
        y_m = -A_m / g0'(y0)

    g0'(y0) < 0 is required (strict monotonicity / admissibility).

    Returns
    -------
    y_coeffs : list of floats [y_0, y_1, ..., y_M]
                where y_0 is set to 0 (caller must inject the actual y0 value)
    """
    assert len(g0_derivs) >= M + 1, "Need g0 derivatives up to order M"
    assert len(g1_derivs) >= M, "Need g1 derivatives up to order M-1"

    g0_prime = g0_derivs[1]
    assert g0_prime < 0, (
        f"g0'(y0) = {g0_prime:.6g} must be negative (budget function monotonicity)"
    )

    # y_coeffs[m] = y_m
    y_coeffs: List[float] = [0.0] * (M + 1)
    # y_0 is the baseline root, filled by caller; we initialize to 0 here
    y_coeffs[0] = 0.0  # placeholder; caller injects actual y0

    for m in range(1, M + 1):
        xs_y = tuple(y_coeffs[1 : m + 1])  # (y_1, ..., y_m)

        # First sum: r from 2 to m, g0^{(r)}(y0) * B_{m,r}(y1,...,y_{m-r+1})
        A_m = 0.0
        for r in range(2, m + 1):
            if r < len(g0_derivs):
                g0_r = g0_derivs[r]
                bell_val = partial_bell(m, r, xs_y)
                A_m += g0_r * bell_val

        # Second sum: r from 0 to m-1, m * g1^{(r)}(y0) * B_{m-1,r}(y1,...,y_{m-r})
        for r in range(0, m):
            if r < len(g1_derivs):
                g1_r = g1_derivs[r]
                bell_val = partial_bell(m - 1, r, xs_y)
                A_m += m * g1_r * bell_val

        y_coeffs[m] = -A_m / g0_prime

    return y_coeffs


def primal_value_recursion(
    v0_derivs: List[float],
    v1_derivs: List[float],
    y_coeffs: List[float],
    x: float,
    M: int,
) -> List[float]:
    """
    Compute Taylor coefficients u_0, u_1, ..., u_M of the primal value:
        u(x, eps) = sum_{m=0}^{M} eps^m / m! * u_m(x)

    Using the simplified recursion (exploiting v0'(y0) + x = 0):
        u_0 = v0(y0) + x * y0
        For m >= 1:
          u_m = sum_{r=2}^{m} v0^{(r)}(y0) * B_{m,r}(y1,...,y_{m-r+1})
              + m * sum_{r=0}^{m-1} v1^{(r)}(y0) * B_{m-1,r}(y1,...,y_{m-r})
              + x * y_m

    Parameters
    ----------
    v0_derivs : [v0(y0), v0'(y0), v0''(y0), ..., v0^{(M)}(y0)]
    v1_derivs : [v1(y0), v1'(y0), ..., v1^{(M-1)}(y0)]
    y_coeffs  : [y0, y1, ..., y_M] from lagrange_multiplier_recursion
    x : initial wealth
    M : maximum order
    """
    assert abs(v0_derivs[1] + x) < 1e-6, (
        f"v0'(y0) + x = {v0_derivs[1] + x:.2e} should be ~0 (duality relation)"
    )

    u_coeffs = [0.0] * (M + 1)
    y0 = y_coeffs[0]
    u_coeffs[0] = float(v0_derivs[0]) + x * y0

    for m in range(1, M + 1):
        xs_y = tuple(y_coeffs[1 : m + 1])  # (y1, ..., y_m)
        u_m = 0.0

        # sum_{r=2}^{m} v0^{(r)}(y0) * B_{m,r}(y1,...,y_{m-r+1})
        for r in range(2, m + 1):
            if r < len(v0_derivs):
                bell_val = partial_bell(m, r, xs_y)
                u_m += v0_derivs[r] * bell_val

        # m * sum_{r=0}^{m-1} v1^{(r)}(y0) * B_{m-1,r}(y1,...,y_{m-r})
        for r in range(0, m):
            if r < len(v1_derivs):
                bell_val = partial_bell(m - 1, r, xs_y)
                u_m += m * v1_derivs[r] * bell_val

        # Note: the full formula has +x*y_m and +v0'(y0)*B_{m,1}(y_m) = -x*y_m;
        # they cancel exactly (duality: v0'(y0) = -x), so neither term appears here.
        u_coeffs[m] = u_m

    return u_coeffs


def optimizer_recursion_pathwise(
    dI0_dy_fns: List[Callable],
    dJ_dy_fns: List[Callable],
    y_coeffs: List[float],
    H_T: np.ndarray,
    M: int,
) -> List[np.ndarray]:
    """
    Compute pathwise Taylor coefficients X_0, X_1, ..., X_M of the optimizer:
        X_hat_T(x, eps) = sum_{m=0}^{M} eps^m/m! * X_m

    Recursion:
        Z_j = y_j * H_T  (pathwise)
        X_0 = I^0(Z_0) = I^0(y0 * H_T)
        X_m = sum_{r=1}^{m} (I^0)^{(r)}(Z_0) * B_{m,r}(Z_1,...,Z_{m-r+1})
            + m * sum_{r=0}^{m-1} J^{(r)}(Z_0) * B_{m-1,r}(Z_1,...,Z_{m-r})

    Parameters
    ----------
    dI0_dy_fns : list of callables, where dI0_dy_fns[r](y) = d^r I^0/dy^r(y)
                 (r=0 is I^0 itself, r=1 is first derivative, etc.)
    dJ_dy_fns  : list of callables similarly for J
    y_coeffs   : [y0, y1, ..., y_M]
    H_T        : array of shape (n_paths,)
    M          : maximum order
    """
    n_paths = len(H_T)

    # Z_j = y_j * H_T, shape (M+1, n_paths)
    Z = np.array([y_coeffs[j] * H_T for j in range(M + 1)])  # (M+1, n_paths)

    X_coeffs: List[np.ndarray] = []

    # X_0 = I^0(Z_0)
    X_coeffs.append(dI0_dy_fns[0](Z[0]))

    for m in range(1, M + 1):
        X_m = np.zeros(n_paths)
        Z_stack = Z[1 : m + 1]  # rows Z_1,...,Z_m, shape (m, n_paths)

        # sum_{r=1}^{m} (I^0)^{(r)}(Z_0) * B_{m,r}(Z_1,...,Z_{m-r+1})
        for r in range(1, m + 1):
            if r < len(dI0_dy_fns):
                dI0_r = dI0_dy_fns[r](Z[0])  # (n_paths,)
                bell_val = partial_bell_array(m, r, Z_stack)
                X_m += dI0_r * bell_val

        # m * sum_{r=0}^{m-1} J^{(r)}(Z_0) * B_{m-1,r}(Z_1,...,Z_{m-r})
        for r in range(0, m):
            if r < len(dJ_dy_fns):
                dJ_r = dJ_dy_fns[r](Z[0])  # (n_paths,)
                bell_val = partial_bell_array(m - 1, r, Z_stack)
                X_m += m * dJ_r * bell_val

        X_coeffs.append(X_m)

    return X_coeffs
