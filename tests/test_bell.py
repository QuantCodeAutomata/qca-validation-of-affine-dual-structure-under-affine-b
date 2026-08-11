"""
Tests for partial Bell polynomials and Bell-polynomial recursion utilities.

Validates:
- Standard Bell polynomial values against known combinatorial formulas.
- lagrange_multiplier_recursion low-order formulas y1, y2, y3.
- primal_value_recursion low-order identity u1 = v1(y0).
- optimizer_recursion_pathwise against explicit formulas X1, X2.
"""

from __future__ import annotations

import numpy as np
import pytest
from math import factorial

from src.bell import (
    partial_bell,
    partial_bell_array,
    bell_poly_table,
    lagrange_multiplier_recursion,
    primal_value_recursion,
    optimizer_recursion_pathwise,
)


# ─── Partial Bell polynomial scalar tests ─────────────────────────────────────

class TestPartialBellScalar:
    """Standard partial Bell polynomial values."""

    def test_B00_is_one(self):
        assert partial_bell(0, 0, []) == 1.0

    def test_B10_is_zero(self):
        assert partial_bell(1, 0, [1.0]) == 0.0

    def test_B11(self):
        # B_{1,1}(x1) = x1
        assert partial_bell(1, 1, [3.5]) == pytest.approx(3.5)

    def test_B21(self):
        # B_{2,1}(x1, x2) = x2
        assert partial_bell(2, 1, [2.0, 5.0]) == pytest.approx(5.0)

    def test_B22(self):
        # B_{2,2}(x1) = x1^2
        x1 = 3.0
        assert partial_bell(2, 2, [x1]) == pytest.approx(x1 ** 2)

    def test_B31(self):
        # B_{3,1}(x1, x2, x3) = x3
        assert partial_bell(3, 1, [1.0, 2.0, 7.0]) == pytest.approx(7.0)

    def test_B32(self):
        # B_{3,2}(x1, x2) = 3 * x1 * x2
        x1, x2 = 2.0, 3.0
        assert partial_bell(3, 2, [x1, x2]) == pytest.approx(3 * x1 * x2)

    def test_B33(self):
        # B_{3,3}(x1) = x1^3
        x1 = 4.0
        assert partial_bell(3, 3, [x1]) == pytest.approx(x1 ** 3)

    def test_B43(self):
        # B_{4,3}(x1, x2) = 6*x1^2*x2  (from combinatorics)
        x1, x2 = 2.0, 3.0
        expected = 6 * x1 ** 2 * x2
        assert partial_bell(4, 3, [x1, x2]) == pytest.approx(expected)

    def test_B44(self):
        # B_{4,4}(x1) = x1^4
        x1 = 2.0
        assert partial_bell(4, 4, [x1]) == pytest.approx(x1 ** 4)

    def test_k_greater_than_n_is_zero(self):
        assert partial_bell(3, 5, [1.0, 2.0, 3.0]) == 0.0

    def test_n_negative_is_zero(self):
        assert partial_bell(-1, 0, []) == 0.0

    def test_all_ones_argument(self):
        # B_{n,k}(1,1,...,1) = number of set partitions of [n] into k parts = Stirling(n,k)
        # Stirling(3,2) = 3; Stirling(4,2) = 7
        xs = [1.0] * 10
        assert partial_bell(3, 2, xs) == pytest.approx(3.0)
        assert partial_bell(4, 2, xs) == pytest.approx(7.0)


class TestBellPolyTable:
    def test_shape_and_base_cases(self):
        xs = [2.0, 3.0, 4.0, 5.0]
        B = bell_poly_table(4, xs)
        assert B.shape == (5, 5)
        assert B[0, 0] == pytest.approx(1.0)
        assert B[1, 0] == 0.0
        assert B[0, 1] == 0.0

    def test_matches_scalar_function(self):
        xs = [1.5, 2.5, 0.7, 1.1]
        B = bell_poly_table(4, xs)
        for n in range(5):
            for k in range(n + 1):
                expected = partial_bell(n, k, xs)
                assert B[n, k] == pytest.approx(expected, rel=1e-12), \
                    f"B[{n},{k}] mismatch"


# ─── Partial Bell polynomial array (pathwise) tests ────────────────────────────

class TestPartialBellArray:
    def test_B00_returns_ones(self):
        xs = np.array([[1.0, 2.0, 3.0]])  # shape (1, 3)
        result = partial_bell_array(0, 0, xs)
        np.testing.assert_array_equal(result, np.ones(3))

    def test_B11_returns_first_row(self):
        xs = np.array([[5.0, 6.0, 7.0]])  # shape (1, 3)
        result = partial_bell_array(1, 1, xs)
        np.testing.assert_allclose(result, xs[0])

    def test_B22_is_first_row_squared(self):
        xs = np.array([[2.0, 3.0, 4.0], [1.0, 1.0, 1.0]])  # shape (2, 3)
        result = partial_bell_array(2, 2, xs)
        np.testing.assert_allclose(result, xs[0] ** 2)

    def test_B21_is_second_row(self):
        xs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # shape (2, 3)
        result = partial_bell_array(2, 1, xs)
        np.testing.assert_allclose(result, xs[1])

    def test_B32_is_3_x1_x2(self):
        xs = np.array([[2.0, 1.5], [3.0, 2.5], [4.0, 3.5]])
        result = partial_bell_array(3, 2, xs)
        expected = 3 * xs[0] * xs[1]
        np.testing.assert_allclose(result, expected)

    def test_k_greater_n_returns_zeros(self):
        xs = np.array([[1.0, 2.0]])
        result = partial_bell_array(1, 3, xs)
        np.testing.assert_array_equal(result, np.zeros(2))


# ─── Lagrange multiplier recursion ─────────────────────────────────────────────

class TestLagrangeMultiplierRecursion:
    """Validate y1, y2, y3 against explicit low-order formulas."""

    def _make_inputs(self):
        """Simple synthetic g0 and g1 derivatives."""
        # g0'(y0) < 0 is required
        g0_derivs = [-1.0, -2.0, 0.5, -0.3, 0.1]  # g0^{(0)},...,g0^{(4)}
        g1_derivs = [3.0, -1.5, 0.8, -0.2]        # g1^{(0)},...,g1^{(3)}
        return g0_derivs, g1_derivs

    def test_y1_matches_explicit(self):
        g0, g1 = self._make_inputs()
        y_coeffs = lagrange_multiplier_recursion(g0, g1, M=3)
        y1_expected = -g1[0] / g0[1]
        assert y_coeffs[1] == pytest.approx(y1_expected, rel=1e-12)

    def test_y2_matches_explicit(self):
        g0, g1 = self._make_inputs()
        y_coeffs = lagrange_multiplier_recursion(g0, g1, M=3)
        y1 = y_coeffs[1]
        y2_expected = -(g0[2] * y1 ** 2 + 2 * g1[1] * y1) / g0[1]
        assert y_coeffs[2] == pytest.approx(y2_expected, rel=1e-10)

    def test_y3_matches_explicit(self):
        g0, g1 = self._make_inputs()
        y_coeffs = lagrange_multiplier_recursion(g0, g1, M=3)
        y1 = y_coeffs[1]
        y2 = y_coeffs[2]
        y3_expected = -(
            g0[3] * y1 ** 3
            + 3 * g0[2] * y1 * y2
            + 3 * g1[2] * y1 ** 2
            + 3 * g1[1] * y2
        ) / g0[1]
        assert y_coeffs[3] == pytest.approx(y3_expected, rel=1e-9)

    def test_fails_if_g0_prime_positive(self):
        g0 = [1.0, 0.5, 0.1]  # g0' > 0 — invalid
        g1 = [0.3, 0.1]
        with pytest.raises(AssertionError):
            lagrange_multiplier_recursion(g0, g1, M=2)

    def test_returns_correct_length(self):
        g0, g1 = self._make_inputs()
        y_coeffs = lagrange_multiplier_recursion(g0, g1, M=3)
        assert len(y_coeffs) == 4  # y0, y1, y2, y3


# ─── Primal value recursion ────────────────────────────────────────────────────

class TestPrimalValueRecursion:
    """Validate u1 = v1(y0) and u2 <= 0 identities."""

    def _make_data(self):
        # Synthetic data consistent with duality: v0'(y0) = -x
        x = 2.0
        v0_derivs = [5.0, -x, 1.5, -0.8, 0.3]   # v0, -x, v0'', v0''', ...
        v1_derivs = [0.7, -0.4, 0.2, -0.1]        # v1, v1', v1'', ...
        g0_derivs = [x, -1.2, 0.5, -0.3, 0.1]    # g0(y0)=x, g0'(y0)=-1.2, ...
        g1_derivs = [0.4, -0.3, 0.2, -0.1]
        y_coeffs = lagrange_multiplier_recursion(g0_derivs, g1_derivs, M=3)
        y_coeffs[0] = 1.5  # inject synthetic y0
        return x, v0_derivs, v1_derivs, y_coeffs

    def test_u1_equals_v1_at_y0(self):
        x, v0d, v1d, y_coeffs = self._make_data()
        u_coeffs = primal_value_recursion(v0d, v1d, y_coeffs, x, M=3)
        assert u_coeffs[1] == pytest.approx(v1d[0], rel=1e-10)

    def test_u2_formula_and_sign(self):
        x, v0d, v1d, y_coeffs = self._make_data()
        u_coeffs = primal_value_recursion(v0d, v1d, y_coeffs, x, M=3)
        y1 = y_coeffs[1]
        u2_expected = v0d[2] * y1 ** 2 + 2 * v1d[1] * y1
        assert u_coeffs[2] == pytest.approx(u2_expected, rel=1e-9)

    def test_u0_duality_relation(self):
        x, v0d, v1d, y_coeffs = self._make_data()
        u_coeffs = primal_value_recursion(v0d, v1d, y_coeffs, x, M=3)
        assert u_coeffs[0] == pytest.approx(v0d[0] + x * y_coeffs[0], rel=1e-12)

    def test_fails_if_duality_violated(self):
        x = 2.0
        v0d = [5.0, -1.0]  # v0'(y0) = -1.0, but x = 2.0 → -1.0 + 2.0 = 1.0 ≠ 0
        v1d = [0.5, 0.1]
        y_coeffs = [1.0, 0.1, 0.05]
        with pytest.raises(AssertionError):
            primal_value_recursion(v0d, v1d, y_coeffs, x, M=1)


# ─── Optimizer recursion pathwise ──────────────────────────────────────────────

class TestOptimizerRecursionPathwise:
    """Validate pathwise X1 and X2 explicit low-order formulas."""

    def _make_bernstein_fns(self):
        """Simple exponential I^0: I^0(y) = 2*e^{-y}, J(y) = 0.5*e^{-2y}."""
        def I0(y):
            return 2.0 * np.exp(-y)
        def dI0_1(y):
            return -2.0 * np.exp(-y)
        def dI0_2(y):
            return 2.0 * np.exp(-y)
        def J(y):
            return 0.5 * np.exp(-2.0 * y)
        def dJ_1(y):
            return -1.0 * np.exp(-2.0 * y)

        dI0_fns = [I0, dI0_1, dI0_2, lambda y: -2.0 * np.exp(-y)]
        dJ_fns = [J, dJ_1, lambda y: 2.0 * np.exp(-2.0 * y)]
        return dI0_fns, dJ_fns

    def test_X0_equals_I0_Z0(self):
        dI0_fns, dJ_fns = self._make_bernstein_fns()
        y_coeffs = [1.0, 0.5, 0.2, 0.1]
        H_T = np.array([0.5, 1.0, 1.5, 2.0])
        X_coeffs = optimizer_recursion_pathwise(dI0_fns, dJ_fns, y_coeffs, H_T, M=2)
        Z0 = y_coeffs[0] * H_T
        np.testing.assert_allclose(X_coeffs[0], dI0_fns[0](Z0))

    def test_X1_explicit_formula(self):
        """X1 = (I^0)'(Z0)*Z1 + J(Z0)."""
        dI0_fns, dJ_fns = self._make_bernstein_fns()
        y_coeffs = [1.0, 0.5, 0.2, 0.1]
        H_T = np.array([0.5, 1.0, 1.5, 2.0])
        X_coeffs = optimizer_recursion_pathwise(dI0_fns, dJ_fns, y_coeffs, H_T, M=2)
        Z0 = y_coeffs[0] * H_T
        Z1 = y_coeffs[1] * H_T
        X1_expected = dI0_fns[1](Z0) * Z1 + dJ_fns[0](Z0)
        np.testing.assert_allclose(X_coeffs[1], X1_expected, rtol=1e-10)

    def test_X2_explicit_formula(self):
        """X2 = (I^0)''(Z0)*Z1^2 + (I^0)'(Z0)*Z2 + 2*J'(Z0)*Z1."""
        dI0_fns, dJ_fns = self._make_bernstein_fns()
        y_coeffs = [1.0, 0.5, 0.2, 0.1]
        H_T = np.array([0.5, 1.0, 1.5, 2.0])
        X_coeffs = optimizer_recursion_pathwise(dI0_fns, dJ_fns, y_coeffs, H_T, M=2)
        Z0 = y_coeffs[0] * H_T
        Z1 = y_coeffs[1] * H_T
        Z2 = y_coeffs[2] * H_T
        X2_expected = (
            dI0_fns[2](Z0) * Z1 ** 2
            + dI0_fns[1](Z0) * Z2
            + 2 * dJ_fns[1](Z0) * Z1
        )
        np.testing.assert_allclose(X_coeffs[2], X2_expected, rtol=1e-9)

    def test_returns_M_plus_1_elements(self):
        dI0_fns, dJ_fns = self._make_bernstein_fns()
        y_coeffs = [1.0, 0.2, 0.1, 0.05]
        H_T = np.linspace(0.5, 2.0, 20)
        X_coeffs = optimizer_recursion_pathwise(dI0_fns, dJ_fns, y_coeffs, H_T, M=3)
        assert len(X_coeffs) == 4
        for X in X_coeffs:
            assert X.shape == (20,)
