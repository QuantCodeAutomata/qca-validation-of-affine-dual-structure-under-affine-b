"""
Tests for dual value functions, budget equation solvers, and
the collect_derivatives helper.

Validates:
- v_eps, v0_fn, v1_fn: expectation of normalized dual potential.
- Affine identity: v(y,eps) = v0(y) + eps * v1(y) exactly.
- g0_fn, g1_fn: budget functions.
- Duality relation: v0'(y0) = -g0(y0) = -x.
- solve_baseline_budget: root finder accuracy.
- solve_perturbed_budget: perturbed root matches baseline at eps=0.
- collect_derivatives: derivative arrays internally consistent.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.model import BSMParams, generate_HT_quadrature
from src.bernstein import AtomicBernsteinSetup
from src.dual import (
    v_eps, v0_fn, v1_fn, v0_deriv, v1_deriv,
    g0_fn, g0_deriv, g1_fn, g1_deriv,
    solve_baseline_budget, solve_perturbed_budget,
    collect_derivatives, weighted_expectation,
)


@pytest.fixture()
def env():
    """Standard BSM environment + Bernstein measure."""
    params = BSMParams(x=0.5, T=1.0, r=0.02, mu=0.08, sigma=0.20)
    bm = AtomicBernsteinSetup(
        z=[1.0, 2.0, 3.0],
        a=[0.5, 0.3, 0.2],
        b=[0.08, -0.03, 0.01],
        y_star=1.0,
    )
    H_T, weights = generate_HT_quadrature(params, n_points=200)
    return params, bm, H_T, weights


class TestWeightedExpectation:
    def test_constant_function(self):
        f = np.ones(10)
        w = np.ones(10) / 10.0
        assert weighted_expectation(f, w) == pytest.approx(1.0, rel=1e-12)

    def test_weighted_sum(self):
        f = np.array([2.0, 4.0])
        w = np.array([0.25, 0.75])
        assert weighted_expectation(f, w) == pytest.approx(3.5, rel=1e-12)


class TestDualValueFunctions:
    def test_v_eps_at_zero_eps_equals_v0(self, env):
        params, bm, H_T, weights = env
        y = 1.5
        assert v_eps(y, 0.0, bm, H_T, weights) == pytest.approx(
            v0_fn(y, bm, H_T, weights), rel=1e-12
        )

    def test_affine_identity_v_eps(self, env):
        """v(y,eps) = v0(y) + eps * v1(y) exactly."""
        params, bm, H_T, weights = env
        for y in [0.5, 1.0, 2.0, 3.0]:
            for eps in [0.0, 0.05, 0.1, 0.2]:
                if not bm.check_positivity([eps]):
                    continue
                v_full = v_eps(y, eps, bm, H_T, weights)
                v0 = v0_fn(y, bm, H_T, weights)
                v1 = v1_fn(y, bm, H_T, weights)
                expected = v0 + eps * v1
                assert v_full == pytest.approx(expected, rel=1e-12, abs=1e-14), \
                    f"Affine identity failed for y={y}, eps={eps}"

    def test_v0_normalization_at_y_star(self, env):
        """v0(y_star) relates to E[V^0(y_star * H_T)].
        Since V^0(y_star) = 0, v0(y_star) = 0 only if H_T = y_star/y_star = 1 a.s.
        In general v0(y_star) ≠ 0. Just check it is finite and consistent.
        """
        params, bm, H_T, weights = env
        v0_val = v0_fn(bm.y_star, bm, H_T, weights)
        assert np.isfinite(v0_val)

    def test_v1_fn_is_expectation_of_V1(self, env):
        params, bm, H_T, weights = env
        y = 1.0
        v1_val = v1_fn(y, bm, H_T, weights)
        V1_vals = bm.V1(y * H_T)
        expected = weighted_expectation(V1_vals, weights)
        assert v1_val == pytest.approx(expected, rel=1e-12)


class TestBudgetFunctions:
    def test_g0_is_positive(self, env):
        params, bm, H_T, weights = env
        for y in [0.1, 0.5, 1.0, 2.0]:
            assert g0_fn(y, bm, H_T, weights) > 0

    def test_g0_is_decreasing(self, env):
        """g0 is strictly decreasing in y."""
        params, bm, H_T, weights = env
        y_vals = [0.1, 0.5, 1.0, 2.0, 5.0]
        g0_vals = [g0_fn(y, bm, H_T, weights) for y in y_vals]
        assert all(g0_vals[i] > g0_vals[i + 1] for i in range(len(g0_vals) - 1))

    def test_g0_deriv_order1_is_negative(self, env):
        params, bm, H_T, weights = env
        for y in [0.5, 1.0, 2.0]:
            assert g0_deriv(y, 1, bm, H_T, weights) < 0

    def test_g0_deriv_order2_is_positive(self, env):
        """g0''(y) > 0 because it involves sum H_T^3 * a_i * z_i^2 * exp(...) > 0."""
        params, bm, H_T, weights = env
        for y in [0.5, 1.0, 2.0]:
            assert g0_deriv(y, 2, bm, H_T, weights) > 0

    def test_g0_deriv_matches_finite_difference(self, env):
        """g0'(y) ≈ [g0(y+h) - g0(y-h)]/(2h)."""
        params, bm, H_T, weights = env
        y, h = 1.0, 1e-6
        fd = (g0_fn(y + h, bm, H_T, weights) - g0_fn(y - h, bm, H_T, weights)) / (2 * h)
        analytic = g0_deriv(y, 1, bm, H_T, weights)
        assert analytic == pytest.approx(fd, rel=1e-5)

    def test_g1_is_finite(self, env):
        params, bm, H_T, weights = env
        for y in [0.5, 1.0, 2.0]:
            assert np.isfinite(g1_fn(y, bm, H_T, weights))


class TestBudgetSolver:
    def test_solve_baseline_satisfies_equation(self, env):
        """g0(y0) should equal x exactly."""
        params, bm, H_T, weights = env
        y0 = solve_baseline_budget(params.x, bm, H_T, weights)
        g0_val = g0_fn(y0, bm, H_T, weights)
        assert g0_val == pytest.approx(params.x, rel=1e-9)

    def test_solve_baseline_is_positive(self, env):
        params, bm, H_T, weights = env
        y0 = solve_baseline_budget(params.x, bm, H_T, weights)
        assert y0 > 0

    def test_solve_perturbed_at_eps0_equals_baseline(self, env):
        """Perturbed budget with eps=0 should return y0."""
        params, bm, H_T, weights = env
        y0 = solve_baseline_budget(params.x, bm, H_T, weights)
        y_pert = solve_perturbed_budget(params.x, 0.0, bm, H_T, weights)
        assert y_pert == pytest.approx(y0, rel=1e-8)

    def test_solve_perturbed_satisfies_equation(self, env):
        params, bm, H_T, weights = env
        eps = 0.1
        if not bm.check_positivity([eps]):
            pytest.skip("eps not admissible")
        y_exact = solve_perturbed_budget(params.x, eps, bm, H_T, weights)
        yH = y_exact * H_T
        integrand = H_T * (bm.I0(yH) + eps * bm.J(yH))
        residual = abs(weighted_expectation(integrand, weights) - params.x)
        assert residual < 1e-9


class TestDualityRelation:
    def test_v0_prime_equals_neg_g0(self, env):
        """Duality: dv0/dy(y0) = -g0(y0) = -x."""
        params, bm, H_T, weights = env
        y0 = solve_baseline_budget(params.x, bm, H_T, weights)
        # v0_deriv(y, 1) = E[H_T * dV0_dy(y*H_T, 1)]
        # dV0_dy(u, 1) = (-1)^1 * sum_i a_i * z_i^0 * exp(-u z_i) = -I^0(u)
        # so v0_deriv(y, 1) = E[H_T * (-I^0(y*H_T))] = -g0(y)  => equals -x at root
        v0d = v0_deriv(y0, 1, bm, H_T, weights)
        g0_val = g0_fn(y0, bm, H_T, weights)
        assert v0d == pytest.approx(-g0_val, rel=1e-9)

    def test_collect_derivatives_v0_prime_plus_x_near_zero(self, env):
        params, bm, H_T, weights = env
        y0 = solve_baseline_budget(params.x, bm, H_T, weights)
        g0d, g1d, v0d, v1d = collect_derivatives(y0, 3, bm, H_T, weights)
        # In collect_derivatives, v0'(y0) is stored as -g0(y0) = -x
        # so v0_derivs[1] + x = -x + x = 0
        assert abs(v0d[1] + params.x) < 1e-9

    def test_collect_derivatives_lengths(self, env):
        params, bm, H_T, weights = env
        y0 = solve_baseline_budget(params.x, bm, H_T, weights)
        M = 3
        g0d, g1d, v0d, v1d = collect_derivatives(y0, M, bm, H_T, weights)
        assert len(g0d) == M + 1
        assert len(g1d) == M  # g1 up to order M-1
        assert len(v0d) == M + 1
        assert len(v1d) == M  # v1 up to order M-1

    def test_g0_deriv0_equals_x_at_y0(self, env):
        params, bm, H_T, weights = env
        y0 = solve_baseline_budget(params.x, bm, H_T, weights)
        g0d, _, _, _ = collect_derivatives(y0, 3, bm, H_T, weights)
        assert g0d[0] == pytest.approx(params.x, rel=1e-9)


class TestV0DerivViaFiniteDifference:
    def test_v0_deriv_order2_matches_fd(self, env):
        """v0''(y0) ≈ [v0'(y+h) - v0'(y-h)] / (2h)."""
        params, bm, H_T, weights = env
        y = 1.0
        h = 1e-5
        vd_plus = v0_deriv(y + h, 1, bm, H_T, weights)
        vd_minus = v0_deriv(y - h, 1, bm, H_T, weights)
        fd = (vd_plus - vd_minus) / (2 * h)
        analytic = v0_deriv(y, 2, bm, H_T, weights)
        assert analytic == pytest.approx(fd, rel=1e-4)
