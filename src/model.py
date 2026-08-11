"""
Black-Scholes-Merton model setup and state price density generation.

Provides quadrature and Monte Carlo representations of H_T, the
distinguished dual optimizer / state price density.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class BSMParams:
    """Parameters for the Black-Scholes-Merton model."""
    x: float      # initial wealth
    T: float      # terminal horizon
    r: float      # risk-free rate
    mu: float     # drift of risky asset
    sigma: float  # volatility

    def __post_init__(self) -> None:
        assert self.sigma > 0, "sigma must be positive"
        assert self.T > 0, "T must be positive"
        assert self.x > 0, "initial wealth x must be positive"

    @property
    def theta(self) -> float:
        """Market price of risk (Sharpe ratio): theta = (mu - r) / sigma."""
        return (self.mu - self.r) / self.sigma


def generate_HT_quadrature(
    params: BSMParams, n_points: int = 200
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate H_T values via Gauss-Hermite quadrature over W_T ~ N(0, T).

    H_T = exp(-rT) * exp(-theta * W_T - 0.5 * theta^2 * T)

    Substituting W_T = sqrt(2T) * t, the Gaussian integral becomes:
        E[f(W_T)] = (1/sqrt(pi)) * sum_i w_i * f(sqrt(2T) * t_i)

    Returns
    -------
    H_T : array of shape (n_points,)
        State price density samples.
    weights : array of shape (n_points,)
        Probability weights summing to 1.
    """
    theta = params.theta
    t_nodes, w_raw = np.polynomial.hermite.hermgauss(n_points)
    # W_T = sqrt(2T) * t_nodes
    W_T = np.sqrt(2.0 * params.T) * t_nodes
    weights = w_raw / np.sqrt(np.pi)  # normalized weights (sum to 1)
    H_T = np.exp(-params.r * params.T) * np.exp(
        -theta * W_T - 0.5 * theta ** 2 * params.T
    )
    assert np.all(np.isfinite(H_T)), "H_T contains non-finite values"
    return H_T, weights


def generate_HT_monte_carlo(
    params: BSMParams, n_samples: int = 200_000, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate H_T values via Monte Carlo with a fixed seed.

    Returns
    -------
    H_T : array of shape (n_samples,)
    W_T : array of shape (n_samples,)  – stored for H_mu computation
    weights : array of shape (n_samples,)  – uniform 1/n_samples
    """
    rng = np.random.default_rng(seed)
    W_T = rng.normal(0.0, np.sqrt(params.T), n_samples)
    theta = params.theta
    H_T = np.exp(-params.r * params.T) * np.exp(
        -theta * W_T - 0.5 * theta ** 2 * params.T
    )
    weights = np.full(n_samples, 1.0 / n_samples)
    return H_T, W_T, weights


def generate_HT_perturbed(
    params: BSMParams, mu_n: float, W_T: np.ndarray
) -> np.ndarray:
    """
    Generate perturbed H_T^n for drift mu_n, using the same W_T draws.

    H_T^n = exp(-rT) * exp(-theta_n * W_T - 0.5 * theta_n^2 * T)
    theta_n = (mu_n - r) / sigma
    """
    theta_n = (mu_n - params.r) / params.sigma
    H_T_n = np.exp(-params.r * params.T) * np.exp(
        -theta_n * W_T - 0.5 * theta_n ** 2 * params.T
    )
    return H_T_n


def H_mu_analytic(params: BSMParams, H_T: np.ndarray, W_T: np.ndarray) -> np.ndarray:
    """
    Analytic derivative dH_T/dmu = -(1/sigma) * H_T * (W_T + theta * T).

    This is obtained by differentiating H_T = exp(-rT)*exp(-theta*W_T - 0.5*theta^2*T)
    with respect to mu, using dtheta/dmu = 1/sigma.
    """
    theta = params.theta
    return -(1.0 / params.sigma) * H_T * (W_T + theta * params.T)


def bsm_moment(params: BSMParams, alpha: float) -> float:
    """
    Compute E[H_T^alpha] analytically for lognormal H_T in BSM.

    H_T = exp(-rT - theta*W_T - 0.5*theta^2*T)
    ln H_T ~ N(m_H, s_H^2) where:
        m_H = -rT - 0.5*theta^2*T
        s_H^2 = theta^2*T

    E[H_T^alpha] = exp(alpha*m_H + 0.5*alpha^2*s_H^2)
                 = exp(-alpha*rT + 0.5*(alpha^2 - alpha)*theta^2*T)
    """
    theta = params.theta
    return float(
        np.exp(-alpha * params.r * params.T
               + 0.5 * (alpha ** 2 - alpha) * theta ** 2 * params.T)
    )
