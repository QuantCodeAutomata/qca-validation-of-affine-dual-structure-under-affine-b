"""
Tests for CRRA utilities: inverse marginals, moments, closed-form
baseline multiplier, first-order correction y1, and the perturbed budget solver.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.model import BSMParams, bsm_moment
from src.crra import (
    crra_inverse_marginal,
    crra_inverse_marginal_deriv,
    crra_baseline_y0,
    crra_g0,
    crra_g0_deriv,
    crra_g1,
    crra_y1_closed_form,
    crra_solve_perturbed_y,
)


@pytest.fixture()
def bsm_params() -> BSMParams:
    return BSMParams(x=1.0, T=1.0, r=0.02, mu=0.08, sigma=0.20)


@pytest.fixture()
def bsm_params2() -> BSMParams:
    return BSMParams(x=2.0, T=2.0, r=0.03, mu=0.10, sigma=0.25)


class TestCRRAInverseMarginal:
    def test_positive_output(self):
        y = np.array([0.5, 1.0, 2.0])
        result = crra_inverse_marginal(y, p=-1.0)
        assert np.all(result > 0)

    def test_explicit_value_p_minus1(self):
        """I^{-1}(y) = y^{1/(-1-1)} = y^{-1/2}."""
        y = np.array([4.0])
        result = crra_inverse_marginal(y, p=-1.0)
        assert result[0] == pytest.approx(0.5, rel=1e-12)

    def test_explicit_value_p_half(self):
        """I^{0.5}(y) = y^{1/(0.5-1)} = y^{-2}."""
        y = np.array([2.0])
        result = crra_inverse_marginal(y, p=0.5)
        assert result[0] == pytest.approx(0.25, rel=1e-12)

    def test_invalid_p_equals_1_raises(self):
        with pytest.raises(AssertionError):
            crra_inverse_marginal(np.array([1.0]), p=1.0)

    def test_invalid_p_greater_than_1_raises(self):
        with pytest.raises(AssertionError):
            crra_inverse_marginal(np.array([1.0]), p=1.5)

    def test_decreasing_in_y_for_negative_p(self):
        """I^p is decreasing in y when 1/(p-1) < 0, i.e., p < 1."""
        y = np.array([0.5, 1.0, 2.0, 4.0])
        result = crra_inverse_marginal(y, p=-0.5)
        assert np.all(np.diff(result) < 0)


class TestCRRAInverseMarginalDeriv:
    def test_first_deriv_matches_finite_diff(self):
        y = np.array([1.5])
        p = -0.5
        h = 1e-7
        fd = (crra_inverse_marginal(y + h, p) - crra_inverse_marginal(y - h, p)) / (2 * h)
        analytic = crra_inverse_marginal_deriv(y, p, order=1)
        np.testing.assert_allclose(analytic, fd, rtol=1e-5)

    def test_second_deriv_matches_finite_diff(self):
        y = np.array([1.0])
        p = 0.5
        h = 1e-5
        d1_plus = crra_inverse_marginal_deriv(y + h, p, order=1)
        d1_minus = crra_inverse_marginal_deriv(y - h, p, order=1)
        fd2 = (d1_plus - d1_minus) / (2 * h)
        analytic2 = crra_inverse_marginal_deriv(y, p, order=2)
        np.testing.assert_allclose(analytic2, fd2, rtol=1e-4)

    def test_first_deriv_is_negative_for_p_less_than_1(self):
        """I^p is decreasing, so first derivative is negative."""
        y = np.array([0.5, 1.0, 2.0])
        for p in [-1.0, -0.5, 0.5]:
            assert np.all(crra_inverse_marginal_deriv(y, p, order=1) < 0)


class TestCRRABaselineY0:
    def test_y0_satisfies_budget_equation(self, bsm_params):
        """g0(y0) should equal x exactly (closed form)."""
        p0 = -0.5
        y0 = crra_baseline_y0(bsm_params.x, p0, bsm_params)
        g0_val = crra_g0(y0, p0, bsm_params)
        assert g0_val == pytest.approx(bsm_params.x, rel=1e-10)

    def test_y0_is_positive(self, bsm_params):
        for p0 in [-1.0, -0.5, 0.5]:
            y0 = crra_baseline_y0(bsm_params.x, p0, bsm_params)
            assert y0 > 0

    def test_y0_scales_with_x(self, bsm_params):
        """Larger wealth → larger y0? Not necessarily; depends on sign of 1/(p-1)."""
        p0 = -1.0  # 1/(p0-1) = 1/(-2) = -0.5: I^{p0}(y) = y^{-0.5}, decreasing in y
        # g0 is decreasing → larger x means smaller y0
        x1, x2 = 0.5, 2.0
        y0_1 = crra_baseline_y0(x1, p0, bsm_params)
        y0_2 = crra_baseline_y0(x2, p0, bsm_params)
        assert y0_1 > y0_2, "For p0=-1 (decreasing g0), larger x → smaller y0"

    def test_y0_equals_second_config(self, bsm_params2):
        p0 = 0.5
        y0 = crra_baseline_y0(bsm_params2.x, p0, bsm_params2)
        g0_val = crra_g0(y0, p0, bsm_params2)
        assert g0_val == pytest.approx(bsm_params2.x, rel=1e-10)


class TestCRRAG0Derivatives:
    def test_g0_deriv_order1_is_negative(self, bsm_params):
        """g0 is decreasing: g0'(y) < 0 for p0 < 1."""
        p0 = -0.5
        y = crra_baseline_y0(bsm_params.x, p0, bsm_params)
        d = crra_g0_deriv(y, p0, bsm_params, order=1)
        assert d < 0

    def test_g0_deriv_matches_finite_diff(self, bsm_params):
        p0 = -1.0
        y = 1.5
        h = 1e-7
        fd = (crra_g0(y + h, p0, bsm_params) - crra_g0(y - h, p0, bsm_params)) / (2 * h)
        analytic = crra_g0_deriv(y, p0, bsm_params, order=1)
        assert analytic == pytest.approx(fd, rel=1e-5)


class TestCRRAY1ClosedForm:
    def test_y1_closed_matches_recursion(self, bsm_params):
        """y1_closed = -g1(y0)/g0'(y0): should match closed-form formula."""
        for p0, p1 in [(-0.5, -1.0), (0.5, -0.5), (-1.0, 0.5)]:
            y0 = crra_baseline_y0(bsm_params.x, p0, bsm_params)
            g0_prime = crra_g0_deriv(y0, p0, bsm_params, order=1)
            g1_val = crra_g1(y0, p1, bsm_params)
            y1_rec = -g1_val / g0_prime
            y1_cf = crra_y1_closed_form(y0, p0, p1, bsm_params)
            assert y1_rec == pytest.approx(y1_cf, rel=1e-8), \
                f"y1 mismatch for p0={p0}, p1={p1}"

    def test_y1_closed_form_numerical_value(self, bsm_params):
        """Spot-check with p0=-0.5, p1=-1.0."""
        p0, p1 = -0.5, -1.0
        y0 = crra_baseline_y0(bsm_params.x, p0, bsm_params)
        y1 = crra_y1_closed_form(y0, p0, p1, bsm_params)
        # y1 must be finite and real
        assert np.isfinite(y1)


class TestCRRASolvePerturbedY:
    def test_perturbed_at_eps0_equals_baseline(self, bsm_params):
        p0, p1 = -0.5, -1.0
        y0 = crra_baseline_y0(bsm_params.x, p0, bsm_params)
        y_exact = crra_solve_perturbed_y(bsm_params.x, 0.0, p0, p1, bsm_params)
        assert y_exact == pytest.approx(y0, rel=1e-8)

    def test_perturbed_satisfies_budget(self, bsm_params):
        """Verify the solved y satisfies the perturbed budget equation."""
        from src.model import bsm_moment
        p0, p1 = -0.5, -1.0
        eps = 0.1
        y_exact = crra_solve_perturbed_y(bsm_params.x, eps, p0, p1, bsm_params)
        a = p0 / (p0 - 1)
        b = p1 / (p1 - 1)
        Ma = bsm_moment(bsm_params, a)
        Mb = bsm_moment(bsm_params, b)
        budget_val = (
            y_exact ** (1.0 / (p0 - 1)) * Ma
            + eps * y_exact ** (1.0 / (p1 - 1)) * Mb
        )
        assert budget_val == pytest.approx(bsm_params.x, rel=1e-9)

    def test_perturbed_y_is_positive(self, bsm_params):
        p0, p1 = -1.0, 0.5
        for eps in [0.0, 0.05, 0.1, 0.2]:
            y = crra_solve_perturbed_y(bsm_params.x, eps, p0, p1, bsm_params)
            assert y > 0


class TestBSMMomentConsistency:
    def test_moment_a_used_in_g0(self, bsm_params):
        """g0(y) = y^{1/(p0-1)} * M(a) where a = p0/(p0-1)."""
        p0 = -0.5
        a = p0 / (p0 - 1)
        Ma = bsm_moment(bsm_params, a)
        y = 1.0
        expected = y ** (1.0 / (p0 - 1)) * Ma
        assert crra_g0(y, p0, bsm_params) == pytest.approx(expected, rel=1e-12)

    def test_moment_positive_for_range_of_alpha(self, bsm_params):
        for alpha in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]:
            assert bsm_moment(bsm_params, alpha) > 0
