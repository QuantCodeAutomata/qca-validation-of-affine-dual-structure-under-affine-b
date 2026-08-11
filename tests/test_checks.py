"""
Tests for validation utility functions in src/checks.py.

Covers:
- affine_residual: zero for exact linear decomposition.
- derivative_identity_residual: zero when identity holds.
- truncation_error: decreasing with order, zero for exact series.
- convergence_rate: log-log slope estimation.
- normalized_multiplier_residual: scaling and zero case.
- normalized_optimizer_residual: all output fields present.
- build_truncated_series: exactness of evaluation.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.checks import (
    affine_residual,
    derivative_identity_residual,
    truncation_error,
    truncation_series,
    convergence_rate,
    normalized_multiplier_residual,
    normalized_optimizer_residual,
    build_truncated_series,
)


class TestAffineResidual:
    def test_zero_when_exact(self):
        """R = v - v0 - eps*v1 must be 0 when v = v0 + eps*v1."""
        v0, v1 = 3.0, 1.5
        eps = 0.2
        v = v0 + eps * v1
        assert affine_residual(v, v0, v1, eps) == pytest.approx(0.0, abs=1e-15)

    def test_nonzero_for_perturbation(self):
        v0, v1 = 3.0, 1.5
        eps = 0.2
        v = v0 + eps * v1 + 0.01  # residual 0.01
        assert abs(affine_residual(v, v0, v1, eps)) == pytest.approx(0.01, rel=1e-12)

    def test_zero_at_eps_zero(self):
        """At eps=0, residual = v - v0."""
        v0 = 4.0
        assert affine_residual(v0, v0, 2.0, 0.0) == pytest.approx(0.0, abs=1e-15)

    def test_linearity_in_eps(self):
        """Residual should be linear when v = v0 + eps*v1 exactly."""
        v0, v1 = 2.0, 1.0
        eps_vals = [0.1, 0.2, 0.5]
        for eps in eps_vals:
            v = v0 + eps * v1
            r = affine_residual(v, v0, v1, eps)
            assert abs(r) < 1e-14


class TestDerivativeIdentityResidual:
    def test_zero_when_identity_holds(self):
        g0, g1, eps = 2.0, 1.0, 0.3
        neg_vy = g0 + eps * g1
        assert derivative_identity_residual(neg_vy, g0, g1, eps) == pytest.approx(0.0, abs=1e-14)

    def test_nonzero_for_mismatch(self):
        g0, g1, eps = 2.0, 1.0, 0.3
        neg_vy = g0 + eps * g1 + 0.05
        assert abs(derivative_identity_residual(neg_vy, g0, g1, eps)) == pytest.approx(0.05)


class TestTruncationError:
    def test_zero_for_exact_coefficients(self):
        """If the Taylor series terminates, truncation error is zero."""
        coeffs = [1.0, 2.0, 3.0, 0.0, 0.0]  # series = 1 + 2*eps + 3/2*eps^2
        eps = 0.1
        y_exact = coeffs[0] + coeffs[1] * eps + coeffs[2] * eps**2 / 2
        err = truncation_error(y_exact, coeffs, eps, order=2)
        assert err == pytest.approx(0.0, abs=1e-13)

    def test_error_decreases_with_order(self):
        """Higher-order truncation should have smaller error for small eps."""
        # Exact series: y(eps) = 1 + eps + eps^2/2 + eps^3/6 (exponential)
        eps = 0.1
        y_exact = np.exp(eps)
        coeffs = [1.0, 1.0, 1.0, 1.0, 1.0]  # k-th factorial term = 1 → y_k = 1
        errs = [truncation_error(y_exact, coeffs, eps, m) for m in range(1, 4)]
        assert errs[0] > errs[1] > errs[2]

    def test_negative_error_impossible(self):
        """Truncation error is absolute value, always nonneg."""
        coeffs = [2.0, -1.0, 0.5]
        eps = 0.2
        err = truncation_error(3.0, coeffs, eps, order=2)
        assert err >= 0.0

    def test_single_element_coefficient(self):
        """With only y0 and order=0, error = |y_exact - y0|."""
        # truncation_error uses range(order+1) = range(1) = [0]
        err = truncation_error(5.0, [4.0, 1.0], 0.5, order=0)
        assert err == pytest.approx(1.0, rel=1e-12)


class TestTruncationSeries:
    def test_returns_list_of_correct_length(self):
        coeffs = [1.0, 1.0, 1.0, 1.0]
        eps = 0.1
        result = truncation_series(1.2, coeffs, eps, max_order=3)
        assert len(result) == 3

    def test_errors_decreasing_for_small_eps(self):
        eps = 0.05
        y_exact = np.exp(eps)
        coeffs = [1.0, 1.0, 1.0, 1.0]
        result = truncation_series(y_exact, coeffs, eps, max_order=3)
        assert result[0] > result[1] > result[2]


class TestConvergenceRate:
    def test_known_quadratic_rate(self):
        """Error ~ eps^2 should give rate ≈ 2."""
        eps_vals = np.array([0.1, 0.05, 0.02, 0.01, 0.005])
        errors = 3.0 * eps_vals ** 2
        rate = convergence_rate(eps_vals, errors)
        assert rate == pytest.approx(2.0, abs=0.05)

    def test_known_cubic_rate(self):
        eps_vals = np.array([0.1, 0.05, 0.02, 0.01])
        errors = 2.0 * eps_vals ** 3
        rate = convergence_rate(eps_vals, errors)
        assert rate == pytest.approx(3.0, abs=0.05)

    def test_returns_nan_for_too_few_points(self):
        eps_vals = np.array([0.1])
        errors = np.array([0.01])
        rate = convergence_rate(eps_vals, errors)
        assert np.isnan(rate)

    def test_handles_zero_errors(self):
        """Points with zero error are excluded; result should not crash."""
        eps_vals = np.array([0.1, 0.05, 0.0, 0.01])
        errors = np.array([0.01, 0.005, 0.0, 0.001])
        rate = convergence_rate(eps_vals, errors)
        assert np.isfinite(rate)

    def test_handles_nonpositive_eps(self):
        eps_vals = np.array([-0.1, 0.0, 0.1, 0.05])
        errors = np.array([0.01, 0.001, 0.01, 0.005])
        rate = convergence_rate(eps_vals, errors)
        assert np.isfinite(rate)


class TestNormalizedMultiplierResidual:
    def test_zero_when_first_order_exact(self):
        y0, y_mu, y_I = 1.5, 0.3, -0.2
        delta_n, eps_n = 0.1, 0.05
        y_exact = y0 + delta_n * y_mu + eps_n * y_I
        r = normalized_multiplier_residual(y_exact, y0, delta_n, eps_n, y_mu, y_I)
        assert r == pytest.approx(0.0, abs=1e-14)

    def test_returns_zero_for_zero_perturbation(self):
        r = normalized_multiplier_residual(1.5, 1.5, 0.0, 0.0, 0.3, -0.2)
        assert r == 0.0

    def test_correct_scaling(self):
        y0, y_mu, y_I = 1.0, 1.0, 0.0
        delta_n, eps_n = 0.2, 0.0
        y_exact = y0 + delta_n * y_mu + 0.05  # 0.05 extra error
        r = normalized_multiplier_residual(y_exact, y0, delta_n, eps_n, y_mu, y_I)
        expected = 0.05 / (abs(delta_n) + abs(eps_n))
        assert r == pytest.approx(expected, rel=1e-12)


class TestNormalizedOptimizerResidual:
    def _make_arrays(self, n=100):
        rng = np.random.default_rng(0)
        X0 = rng.standard_normal(n)
        X_mu = rng.standard_normal(n)
        X_I = rng.standard_normal(n)
        delta_n, eps_n = 0.1, 0.05
        X_exact = X0 + delta_n * X_mu + eps_n * X_I  # first-order exact
        return X_exact, X0, X_mu, X_I, delta_n, eps_n

    def test_zero_when_first_order_exact(self):
        X_exact, X0, X_mu, X_I, delta_n, eps_n = self._make_arrays()
        result = normalized_optimizer_residual(X_exact, X0, X_mu, X_I, delta_n, eps_n)
        assert result["sup"] == pytest.approx(0.0, abs=1e-12)
        assert result["mean"] == pytest.approx(0.0, abs=1e-12)

    def test_result_keys_present(self):
        X_exact, X0, X_mu, X_I, delta_n, eps_n = self._make_arrays()
        result = normalized_optimizer_residual(X_exact, X0, X_mu, X_I, delta_n, eps_n)
        for key in ["sup", "mean", "L2", "q50", "q90"]:
            assert key in result

    def test_nonneg_residuals(self):
        X_exact, X0, X_mu, X_I, delta_n, eps_n = self._make_arrays()
        # Add noise
        X_exact = X_exact + 0.01 * np.random.default_rng(1).standard_normal(100)
        result = normalized_optimizer_residual(X_exact, X0, X_mu, X_I, delta_n, eps_n)
        assert result["sup"] >= 0
        assert result["mean"] >= 0
        assert result["L2"] >= 0

    def test_zero_perturbation_returns_zeros(self):
        X0 = np.ones(50)
        result = normalized_optimizer_residual(X0, X0, X0, X0, 0.0, 0.0)
        assert result["sup"] == 0.0


class TestBuildTruncatedSeries:
    def test_order_zero_gives_constant_term(self):
        coeffs = [5.0, 1.0, 2.0]
        result = build_truncated_series(coeffs, 0.3, order=0)
        assert result == pytest.approx(5.0, rel=1e-12)

    def test_order_one_matches_linear(self):
        coeffs = [5.0, 2.0, 1.0]
        eps = 0.2
        expected = 5.0 + 2.0 * eps  # j=0: 5*eps^0/0! + j=1: 2*eps/1!
        result = build_truncated_series(coeffs, eps, order=1)
        assert result == pytest.approx(expected, rel=1e-12)

    def test_order_two_matches_quadratic(self):
        coeffs = [1.0, 2.0, 3.0]
        eps = 0.5
        expected = 1.0 + 2.0 * eps + 3.0 * eps**2 / 2.0
        result = build_truncated_series(coeffs, eps, order=2)
        assert result == pytest.approx(expected, rel=1e-12)

    def test_factorials_correct_at_order_three(self):
        coeffs = [0.0, 0.0, 0.0, 6.0]  # only y3 term
        eps = 1.0
        # contribution = 6 * 1^3 / 3! = 1
        result = build_truncated_series(coeffs, eps, order=3)
        assert result == pytest.approx(1.0, rel=1e-12)
