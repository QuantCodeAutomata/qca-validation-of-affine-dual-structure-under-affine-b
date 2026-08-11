"""
Tests for BSM model utilities: state price density generation,
quadrature accuracy, Monte Carlo properties, and moment formulas.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.model import BSMParams, generate_HT_quadrature, generate_HT_monte_carlo
from src.model import generate_HT_perturbed, H_mu_analytic, bsm_moment


@pytest.fixture()
def params_baseline() -> BSMParams:
    return BSMParams(x=1.0, T=1.0, r=0.02, mu=0.08, sigma=0.20)


class TestBSMParams:
    def test_theta_computation(self):
        p = BSMParams(x=1.0, T=1.0, r=0.02, mu=0.12, sigma=0.20)
        assert p.theta == pytest.approx((0.12 - 0.02) / 0.20, rel=1e-12)

    def test_zero_sigma_raises(self):
        with pytest.raises(AssertionError):
            BSMParams(x=1.0, T=1.0, r=0.0, mu=0.05, sigma=0.0)

    def test_negative_sigma_raises(self):
        with pytest.raises(AssertionError):
            BSMParams(x=1.0, T=1.0, r=0.0, mu=0.05, sigma=-0.1)

    def test_zero_T_raises(self):
        with pytest.raises(AssertionError):
            BSMParams(x=1.0, T=0.0, r=0.0, mu=0.05, sigma=0.2)

    def test_zero_x_raises(self):
        with pytest.raises(AssertionError):
            BSMParams(x=0.0, T=1.0, r=0.0, mu=0.05, sigma=0.2)

    def test_negative_theta_is_allowed(self):
        # mu < r is valid (negative Sharpe ratio)
        p = BSMParams(x=1.0, T=1.0, r=0.10, mu=0.05, sigma=0.20)
        assert p.theta < 0


class TestQuadrature:
    def test_H_T_is_positive(self, params_baseline):
        H_T, weights = generate_HT_quadrature(params_baseline, n_points=100)
        assert np.all(H_T > 0)

    def test_weights_sum_to_one(self, params_baseline):
        H_T, weights = generate_HT_quadrature(params_baseline, n_points=100)
        assert np.sum(weights) == pytest.approx(1.0, abs=1e-12)

    def test_weights_are_positive(self, params_baseline):
        H_T, weights = generate_HT_quadrature(params_baseline, n_points=100)
        assert np.all(weights > 0)

    def test_E_H_T_equals_1(self, params_baseline):
        """E[H_T] should equal e^{-rT} * E[exp(-theta W_T - 0.5 theta^2 T)].
        
        For BSM H_T = exp(-rT) * exp(-theta W_T - 0.5 theta^2 T), the expectation:
        E[H_T] = exp(-rT) * exp(0.5*theta^2*T - 0.5*theta^2*T) = exp(-rT).
        """
        p = params_baseline
        H_T, weights = generate_HT_quadrature(p, n_points=200)
        E_H_T = np.dot(weights, H_T)
        expected = np.exp(-p.r * p.T)
        assert E_H_T == pytest.approx(expected, rel=1e-6)

    def test_H_T_finite(self, params_baseline):
        H_T, weights = generate_HT_quadrature(params_baseline, n_points=100)
        assert np.all(np.isfinite(H_T))

    def test_moment_via_quadrature_matches_closed_form(self, params_baseline):
        """E[H_T^alpha] via quadrature should match bsm_moment."""
        H_T, weights = generate_HT_quadrature(params_baseline, n_points=300)
        for alpha in [0.5, 1.0, 1.5, 2.0, -0.5]:
            quad_moment = np.dot(weights, H_T ** alpha)
            closed_form = bsm_moment(params_baseline, alpha)
            assert quad_moment == pytest.approx(closed_form, rel=1e-4), \
                f"Moment mismatch for alpha={alpha}"


class TestMonteCarlo:
    def test_H_T_positive(self, params_baseline):
        H_T, W_T, weights = generate_HT_monte_carlo(params_baseline, n_samples=1000, seed=0)
        assert np.all(H_T > 0)

    def test_weights_uniform_and_sum_to_one(self, params_baseline):
        n = 5000
        H_T, W_T, weights = generate_HT_monte_carlo(params_baseline, n_samples=n, seed=1)
        assert weights.shape == (n,)
        assert np.sum(weights) == pytest.approx(1.0, rel=1e-12)
        assert np.all(weights == weights[0])  # uniform

    def test_W_T_is_normal(self, params_baseline):
        """W_T should have mean~0, std~sqrt(T) for large samples."""
        H_T, W_T, weights = generate_HT_monte_carlo(
            params_baseline, n_samples=200_000, seed=99
        )
        assert W_T.mean() == pytest.approx(0.0, abs=0.01)
        assert W_T.std() == pytest.approx(np.sqrt(params_baseline.T), rel=0.01)

    def test_reproducibility_with_seed(self, params_baseline):
        H1, W1, _ = generate_HT_monte_carlo(params_baseline, n_samples=100, seed=42)
        H2, W2, _ = generate_HT_monte_carlo(params_baseline, n_samples=100, seed=42)
        np.testing.assert_array_equal(H1, H2)
        np.testing.assert_array_equal(W1, W2)

    def test_E_H_T_approximately_exp_neg_rT(self, params_baseline):
        H_T, _, weights = generate_HT_monte_carlo(params_baseline, n_samples=500_000, seed=7)
        E_H_T = np.dot(weights, H_T)
        expected = np.exp(-params_baseline.r * params_baseline.T)
        assert E_H_T == pytest.approx(expected, rel=1e-2)


class TestPerturbedHT:
    def test_same_W_T_used(self, params_baseline):
        """Perturbed H_T^n should use same W_T draws."""
        H_T, W_T, _ = generate_HT_monte_carlo(params_baseline, n_samples=100, seed=0)
        mu_n = params_baseline.mu + 0.01
        H_Tn = generate_HT_perturbed(params_baseline, mu_n, W_T)
        assert H_Tn.shape == H_T.shape
        assert np.all(H_Tn > 0)

    def test_same_mu_gives_same_H_T(self, params_baseline):
        H_T, W_T, _ = generate_HT_monte_carlo(params_baseline, n_samples=100, seed=0)
        H_Tn = generate_HT_perturbed(params_baseline, params_baseline.mu, W_T)
        np.testing.assert_allclose(H_Tn, H_T, rtol=1e-12)


class TestHMuAnalytic:
    def test_H_mu_vs_finite_difference(self, params_baseline):
        """Analytic H_mu should match [H(mu+h) - H(mu-h)] / (2h)."""
        H_T, W_T, _ = generate_HT_monte_carlo(params_baseline, n_samples=5000, seed=2)
        h = 1e-6
        H_plus = generate_HT_perturbed(params_baseline, params_baseline.mu + h, W_T)
        H_minus = generate_HT_perturbed(params_baseline, params_baseline.mu - h, W_T)
        H_mu_fd = (H_plus - H_minus) / (2 * h)
        H_mu_analytic_val = H_mu_analytic(params_baseline, H_T, W_T)
        np.testing.assert_allclose(H_mu_analytic_val, H_mu_fd, rtol=1e-5, atol=1e-10)

    def test_H_mu_sign_depends_on_W_T(self, params_baseline):
        """H_mu = -(1/sigma)*H_T*(W_T + theta*T): sign changes with W_T."""
        H_T, W_T, _ = generate_HT_monte_carlo(params_baseline, n_samples=10000, seed=3)
        H_mu = H_mu_analytic(params_baseline, H_T, W_T)
        # Should have both positive and negative values (W_T is N(0,T) centered)
        assert np.any(H_mu > 0) and np.any(H_mu < 0)


class TestBSMMoment:
    def test_moment_at_zero_is_one(self, params_baseline):
        """M(0) = E[H_T^0] = 1."""
        assert bsm_moment(params_baseline, 0.0) == pytest.approx(1.0, rel=1e-12)

    def test_moment_at_one_is_exp_neg_rT(self, params_baseline):
        """M(1) = E[H_T] = exp(-rT)."""
        p = params_baseline
        expected = np.exp(-p.r * p.T)
        assert bsm_moment(p, 1.0) == pytest.approx(expected, rel=1e-12)

    def test_moment_positive(self, params_baseline):
        for alpha in [-1.0, 0.0, 0.5, 1.0, 2.0]:
            assert bsm_moment(params_baseline, alpha) > 0

    def test_moment_formula_explicit(self):
        """M(alpha) = exp(-alpha*r*T + 0.5*(alpha^2-alpha)*theta^2*T)."""
        p = BSMParams(x=1.0, T=2.0, r=0.03, mu=0.09, sigma=0.25)
        theta = p.theta
        alpha = 1.5
        expected = np.exp(-alpha * p.r * p.T + 0.5 * (alpha ** 2 - alpha) * theta ** 2 * p.T)
        assert bsm_moment(p, alpha) == pytest.approx(expected, rel=1e-12)
