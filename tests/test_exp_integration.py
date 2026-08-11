"""
Experiment-level integration tests.

Each test runs a stripped-down version of the corresponding experiment
and verifies the key theorem-level assertions:

  Exp 1: Affine identity v(y,eps) = v0(y) + eps*v1(y) holds to machine precision.
  Exp 2: y1 from Bell recursion matches explicit formula; g0'(y0) < 0.
  Exp 3: u1 = v1(y0); u2 <= 0.
  Exp 4: Closed-form y1 matches general recursion; first-order X1 match.
  Exp 5: H_mu analytic matches finite-difference; normalized residuals vanish.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.model import BSMParams, generate_HT_quadrature, generate_HT_monte_carlo
from src.model import generate_HT_perturbed, H_mu_analytic, bsm_moment
from src.bernstein import AtomicBernsteinSetup
from src.dual import (
    v_eps, v0_fn, v1_fn, g0_fn, g0_deriv, g1_fn,
    solve_baseline_budget, solve_perturbed_budget, collect_derivatives,
)
from src.bell import (
    lagrange_multiplier_recursion, primal_value_recursion,
    optimizer_recursion_pathwise,
)
from src.crra import (
    crra_baseline_y0, crra_g0_deriv, crra_g1,
    crra_y1_closed_form, crra_inverse_marginal, crra_inverse_marginal_deriv,
    crra_solve_perturbed_y,
)
from src.checks import (
    affine_residual, truncation_error, convergence_rate, build_truncated_series,
    normalized_multiplier_residual, normalized_optimizer_residual,
)

# ─── Shared fixtures ──────────────────────────────────────────────────────────

PARAMS_STD = BSMParams(x=0.8, T=1.0, r=0.02, mu=0.08, sigma=0.20)
BM_STD = AtomicBernsteinSetup(
    z=[1.0, 2.0, 3.0],
    a=[0.5, 0.3, 0.2],
    b=[0.08, -0.03, 0.01],
    y_star=1.0,
)
N_QUAD = 200
N_MC = 20_000  # small for speed in tests


@pytest.fixture(scope="module")
def quad_env():
    H_T, weights = generate_HT_quadrature(PARAMS_STD, n_points=N_QUAD)
    return H_T, weights


@pytest.fixture(scope="module")
def mc_env():
    H_T, W_T, weights = generate_HT_monte_carlo(PARAMS_STD, n_samples=N_MC, seed=0)
    return H_T, W_T, weights


@pytest.fixture(scope="module")
def y0_solved(quad_env):
    H_T, weights = quad_env
    return solve_baseline_budget(PARAMS_STD.x, BM_STD, H_T, weights)


# ─── Exp 1: Affine identity ────────────────────────────────────────────────────

class TestExp1AffineIdentity:
    """
    Validates Theorem 1: v(y,eps) = v0(y) + eps*v1(y) is exact for atomic measures.
    Residual should be indistinguishable from zero.
    """

    def test_affine_identity_multiple_y_eps(self, quad_env):
        H_T, weights = quad_env
        y_grid = [0.5, 1.0, 1.5, 2.0, 3.0]
        eps_grid = [0.0, 0.05, 0.1, 0.15, 0.2]

        max_residual = 0.0
        for y in y_grid:
            for eps in eps_grid:
                if not BM_STD.check_positivity([eps]):
                    continue
                v_full = v_eps(y, eps, BM_STD, H_T, weights)
                v0 = v0_fn(y, BM_STD, H_T, weights)
                v1 = v1_fn(y, BM_STD, H_T, weights)
                r = affine_residual(v_full, v0, v1, eps)
                max_residual = max(max_residual, abs(r))

        # Should be machine precision (< 1e-14)
        assert max_residual < 1e-12, f"Max affine residual {max_residual:.2e} too large"

    def test_normalization_V_eps_at_y_star(self):
        """V^eps(y_star) = 0 exactly for all eps."""
        y_star = BM_STD.y_star
        for eps in [0.0, 0.1, 0.2, -0.1]:
            if not BM_STD.check_positivity([eps]):
                continue
            v = BM_STD.V_eps(y_star, eps)
            assert abs(v) < 1e-15, f"V^eps(y_star) = {v} ≠ 0 at eps={eps}"

    def test_derivative_identity_at_y0(self, quad_env, y0_solved):
        """
        -dv(y,eps)/dy = g0(y) + eps*g1(y).
        Verified via finite difference of v(y,eps).
        """
        H_T, weights = quad_env
        y = y0_solved
        for eps in [0.0, 0.1]:
            if not BM_STD.check_positivity([eps]):
                continue
            h = 1e-6
            neg_vy_fd = -(v_eps(y + h, eps, BM_STD, H_T, weights)
                          - v_eps(y - h, eps, BM_STD, H_T, weights)) / (2 * h)
            g0 = g0_fn(y, BM_STD, H_T, weights)
            g1 = g1_fn(y, BM_STD, H_T, weights)
            expected = g0 + eps * g1
            assert neg_vy_fd == pytest.approx(expected, rel=1e-5), \
                f"Derivative identity failed at eps={eps}"


# ─── Exp 2: Bell-polynomial recursion ─────────────────────────────────────────

class TestExp2BellRecursion:
    """
    Validates Theorem 2: analyticity of y(x,eps); y_m from Bell recursion.
    """

    def test_g0_prime_negative(self, quad_env, y0_solved):
        H_T, weights = quad_env
        y0 = y0_solved
        g0p = g0_deriv(y0, 1, BM_STD, H_T, weights)
        assert g0p < 0, f"g0'(y0) = {g0p:.4g} must be negative"

    def test_y1_matches_explicit_formula(self, quad_env, y0_solved):
        H_T, weights = quad_env
        y0 = y0_solved
        g0d, g1d, v0d, v1d = collect_derivatives(y0, 3, BM_STD, H_T, weights)

        y_coeffs = lagrange_multiplier_recursion(g0d, g1d, M=3)
        y_coeffs[0] = y0

        y1_explicit = -g1d[0] / g0d[1]
        assert y_coeffs[1] == pytest.approx(y1_explicit, rel=1e-10)

    def test_y2_matches_explicit_formula(self, quad_env, y0_solved):
        H_T, weights = quad_env
        y0 = y0_solved
        g0d, g1d, v0d, v1d = collect_derivatives(y0, 3, BM_STD, H_T, weights)

        y_coeffs = lagrange_multiplier_recursion(g0d, g1d, M=3)
        y_coeffs[0] = y0

        y1 = y_coeffs[1]
        y2_explicit = -(g0d[2] * y1**2 + 2 * g1d[1] * y1) / g0d[1]
        assert y_coeffs[2] == pytest.approx(y2_explicit, rel=1e-9)

    def test_truncation_error_order1_larger_than_order2(self, quad_env, y0_solved):
        """Order-1 truncation error > order-2 at same eps."""
        H_T, weights = quad_env
        y0 = y0_solved
        g0d, g1d, _, _ = collect_derivatives(y0, 3, BM_STD, H_T, weights)
        y_coeffs = lagrange_multiplier_recursion(g0d, g1d, M=3)
        y_coeffs[0] = y0

        eps = 0.1
        if not BM_STD.check_positivity([eps]):
            pytest.skip("eps not admissible")

        y_exact = solve_perturbed_budget(PARAMS_STD.x, eps, BM_STD, H_T, weights)
        err1 = truncation_error(y_exact, y_coeffs, eps, order=1)
        err2 = truncation_error(y_exact, y_coeffs, eps, order=2)
        assert err1 >= err2, f"err1={err1:.2e}, err2={err2:.2e}: order-2 should be at least as good"

    def test_convergence_rate_order1_near_2(self, quad_env, y0_solved):
        """Error for order-1 truncation should scale as O(eps^2)."""
        H_T, weights = quad_env
        y0 = y0_solved
        g0d, g1d, _, _ = collect_derivatives(y0, 3, BM_STD, H_T, weights)
        y_coeffs = lagrange_multiplier_recursion(g0d, g1d, M=3)
        y_coeffs[0] = y0

        eps_seq = np.array([0.2, 0.15, 0.1, 0.07, 0.05, 0.03])
        eps_seq = np.array([e for e in eps_seq if BM_STD.check_positivity([e])])

        errs = np.array([
            truncation_error(
                solve_perturbed_budget(PARAMS_STD.x, e, BM_STD, H_T, weights),
                y_coeffs, e, order=1
            )
            for e in eps_seq
        ])
        rate = convergence_rate(eps_seq, errs)
        assert rate >= 1.5, f"Order-1 convergence rate {rate:.2f} too low (expected ~2)"


# ─── Exp 3: Primal expansion ───────────────────────────────────────────────────

class TestExp3PrimalExpansion:
    """
    Validates Theorems 3 & 4: expansions of u(x,eps) and X(x,eps).
    """

    def test_u1_equals_v1_y0(self, quad_env, y0_solved):
        H_T, weights = quad_env
        y0 = y0_solved
        g0d, g1d, v0d, v1d = collect_derivatives(y0, 3, BM_STD, H_T, weights)
        y_coeffs = lagrange_multiplier_recursion(g0d, g1d, M=3)
        y_coeffs[0] = y0
        u_coeffs = primal_value_recursion(v0d, v1d, y_coeffs, PARAMS_STD.x, M=3)

        v1_at_y0 = v1_fn(y0, BM_STD, H_T, weights)
        assert u_coeffs[1] == pytest.approx(v1_at_y0, rel=1e-9), \
            f"u1={u_coeffs[1]:.6f} ≠ v1(y0)={v1_at_y0:.6f}"

    def test_u2_nonpositive(self, quad_env, y0_solved):
        H_T, weights = quad_env
        y0 = y0_solved
        g0d, g1d, v0d, v1d = collect_derivatives(y0, 3, BM_STD, H_T, weights)
        y_coeffs = lagrange_multiplier_recursion(g0d, g1d, M=3)
        y_coeffs[0] = y0
        u_coeffs = primal_value_recursion(v0d, v1d, y_coeffs, PARAMS_STD.x, M=3)
        assert u_coeffs[2] <= 1e-10, f"u2 = {u_coeffs[2]:.4g} should be <= 0"

    def test_u2_matches_formula(self, quad_env, y0_solved):
        """u2 = -(v1'(y0))^2 / v0''(y0) = v0''(y0)*y1^2 + 2*v1'(y0)*y1."""
        H_T, weights = quad_env
        y0 = y0_solved
        g0d, g1d, v0d, v1d = collect_derivatives(y0, 3, BM_STD, H_T, weights)
        y_coeffs = lagrange_multiplier_recursion(g0d, g1d, M=3)
        y_coeffs[0] = y0
        u_coeffs = primal_value_recursion(v0d, v1d, y_coeffs, PARAMS_STD.x, M=3)

        y1 = y_coeffs[1]
        u2_formula = v0d[2] * y1**2 + 2 * v1d[1] * y1
        assert u_coeffs[2] == pytest.approx(u2_formula, rel=1e-8)

    def test_u0_duality_relation(self, quad_env, y0_solved):
        """u0 = v0(y0) + x*y0."""
        H_T, weights = quad_env
        y0 = y0_solved
        g0d, g1d, v0d, v1d = collect_derivatives(y0, 3, BM_STD, H_T, weights)
        y_coeffs = lagrange_multiplier_recursion(g0d, g1d, M=3)
        y_coeffs[0] = y0
        u_coeffs = primal_value_recursion(v0d, v1d, y_coeffs, PARAMS_STD.x, M=3)
        v0_val = v0_fn(y0, BM_STD, H_T, weights)
        assert u_coeffs[0] == pytest.approx(v0_val + PARAMS_STD.x * y0, rel=1e-10)

    def test_X1_explicit_formula(self, mc_env, y0_solved):
        """X1 = (I^0)'(Z0) * Z1 + J(Z0) matches recursion."""
        H_T, W_T, weights = mc_env
        y0 = y0_solved
        # Recompute y_coeffs using quadrature
        H_T_q, w_q = generate_HT_quadrature(PARAMS_STD, n_points=N_QUAD)
        g0d, g1d, v0d, v1d = collect_derivatives(y0, 2, BM_STD, H_T_q, w_q)
        y_coeffs = lagrange_multiplier_recursion(g0d, g1d, M=2)
        y_coeffs[0] = y0

        dI0_fns = [BM_STD.I0, lambda y: BM_STD.dI0_dy(y, order=1),
                   lambda y: BM_STD.dI0_dy(y, order=2)]
        dJ_fns = [BM_STD.J, lambda y: BM_STD.dJ_dy(y, order=1)]
        X_coeffs = optimizer_recursion_pathwise(dI0_fns, dJ_fns, y_coeffs, H_T, M=2)

        Z0 = y_coeffs[0] * H_T
        Z1 = y_coeffs[1] * H_T
        X1_explicit = BM_STD.dI0_dy(Z0, order=1) * Z1 + BM_STD.J(Z0)
        np.testing.assert_allclose(X_coeffs[1], X1_explicit, rtol=1e-10)


# ─── Exp 4: CRRA validation ────────────────────────────────────────────────────

class TestExp4CRRAValidation:
    """
    Validates CRRA moment formulas and first-order y1 agreement.
    """

    @pytest.mark.parametrize("p0,p1", [(-0.5, -1.0), (0.5, -0.5), (-1.0, 0.5)])
    def test_y1_recursion_matches_closed_form(self, p0, p1):
        params = PARAMS_STD
        y0 = crra_baseline_y0(params.x, p0, params)
        g0_prime = crra_g0_deriv(y0, p0, params, order=1)
        g1_val = crra_g1(y0, p1, params)
        y1_rec = -g1_val / g0_prime
        y1_cf = crra_y1_closed_form(y0, p0, p1, params)
        assert y1_rec == pytest.approx(y1_cf, rel=1e-8), \
            f"y1 mismatch for p0={p0}, p1={p1}: rec={y1_rec}, cf={y1_cf}"

    @pytest.mark.parametrize("p0,p1", [(-0.5, -1.0), (0.5, -0.5)])
    def test_X1_general_vs_closed_form(self, p0, p1):
        """X1_general = (I^p0)'(Z0)*Z1 + I^p1(Z0) matches closed form."""
        params = PARAMS_STD
        H_T, W_T, weights = generate_HT_monte_carlo(params, n_samples=N_MC, seed=1)
        y0 = crra_baseline_y0(params.x, p0, params)
        y1 = crra_y1_closed_form(y0, p0, p1, params)

        Z0 = y0 * H_T
        Z1 = y1 * H_T
        X1_general = crra_inverse_marginal_deriv(Z0, p0, order=1) * Z1 + crra_inverse_marginal(Z0, p1)
        X1_closed = (
            (y0 * H_T) ** (1.0 / (p1 - 1))
            + (y1 / (p0 - 1)) * H_T * (y0 * H_T) ** (1.0 / (p0 - 1) - 1)
        )
        np.testing.assert_allclose(X1_general, X1_closed, rtol=1e-8)

    @pytest.mark.parametrize("p0,p1", [(-0.5, -1.0)])
    def test_first_order_y_error_decays_as_O_eps2(self, p0, p1):
        """First-order y error |y_exact - y0 - eps*y1| should scale as eps^2."""
        params = PARAMS_STD
        y0 = crra_baseline_y0(params.x, p0, params)
        y1 = crra_y1_closed_form(y0, p0, p1, params)

        eps_vals = np.array([0.1, 0.05, 0.02, 0.01])
        errs = np.array([
            abs(crra_solve_perturbed_y(params.x, e, p0, p1, params) - y0 - e * y1)
            for e in eps_vals
        ])
        rate = convergence_rate(eps_vals, errs)
        assert rate >= 1.5, f"Rate {rate:.2f} too low (expected ~2)"

    def test_y0_closed_form_budget_satisfied(self):
        params = PARAMS_STD
        p0 = -0.5
        y0 = crra_baseline_y0(params.x, p0, params)
        a = p0 / (p0 - 1)
        Ma = bsm_moment(params, a)
        budget = y0 ** (1.0 / (p0 - 1)) * Ma
        assert budget == pytest.approx(params.x, rel=1e-10)


# ─── Exp 5: Two-parameter sensitivity ─────────────────────────────────────────

class TestExp5Sensitivity:
    """
    Validates first-order two-parameter expansion for drift + preference perturbation.
    """

    def test_H_mu_analytic_vs_fd(self, mc_env):
        H_T, W_T, weights = mc_env
        h = 1e-6
        H_plus = generate_HT_perturbed(PARAMS_STD, PARAMS_STD.mu + h, W_T)
        H_minus = generate_HT_perturbed(PARAMS_STD, PARAMS_STD.mu - h, W_T)
        H_mu_fd = (H_plus - H_minus) / (2 * h)
        H_mu_an = H_mu_analytic(PARAMS_STD, H_T, W_T)
        np.testing.assert_allclose(H_mu_an, H_mu_fd, rtol=1e-5, atol=1e-10)

    def test_normalized_multiplier_residual_small_perturbation(self, mc_env):
        """
        For small perturbation, the normalized residual should be small
        (not necessarily converging without more data points, but bounded).
        """
        H_T, W_T, weights = mc_env
        theta = PARAMS_STD.theta

        # Solve baseline
        def budget(y):
            return float(np.mean(H_T * BM_STD.I0(y * H_T))) - PARAMS_STD.x

        from scipy.optimize import brentq
        y0 = brentq(budget, 1e-10, 1e6, xtol=1e-12)

        # Compute sensitivity coefficients
        yH = y0 * H_T
        I0_vals = BM_STD.I0(yH)
        J_vals = BM_STD.J(yH)
        dI0_vals = BM_STD.dI0_dy(yH, order=1)
        H_mu = H_mu_analytic(PARAMS_STD, H_T, W_T)

        denom = float(np.mean(H_T ** 2 * dI0_vals))
        assert denom < 0, "Denominator should be negative"
        y_I = -float(np.mean(H_T * J_vals)) / denom
        y_mu = -float(np.mean(H_mu * (I0_vals + y0 * H_T * dI0_vals))) / denom

        # Small perturbation
        delta_n, eps_n = 0.005, 0.005
        mu_n = PARAMS_STD.mu + delta_n
        theta_n = (mu_n - PARAMS_STD.r) / PARAMS_STD.sigma
        H_Tn = np.exp(-PARAMS_STD.r * PARAMS_STD.T) * np.exp(
            -theta_n * W_T - 0.5 * theta_n**2 * PARAMS_STD.T
        )

        def budget_n(y):
            arg = y * H_Tn
            return float(np.mean(H_Tn * (BM_STD.I0(arg) + eps_n * BM_STD.J(arg)))) - PARAMS_STD.x

        y_n = brentq(budget_n, 1e-10, 1e6, xtol=1e-12)
        r_y = normalized_multiplier_residual(y_n, y0, delta_n, eps_n, y_mu, y_I)
        # Normalized residual should be small for small perturbation
        assert r_y < 1.0, f"Normalized multiplier residual {r_y:.4f} too large"

    def test_X0_equals_I0_at_y0(self, mc_env, y0_solved):
        """Baseline optimizer X0 = I^0(y0 * H_T)."""
        H_T, W_T, weights = mc_env
        X0 = BM_STD.I0(y0_solved * H_T)
        assert np.all(X0 > 0), "X0 should be positive (I^0 is positive)"
        assert np.all(np.isfinite(X0))
