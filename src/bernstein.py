"""
Finite atomic Bernstein measures and associated CMIM inverse marginal utilities.

A Bernstein measure mu on (0, inf) defines the inverse marginal utility:
    I(y) = integral_0^inf exp(-y*z) mu(dz)

For a finite atomic measure mu = sum_i c_i * delta_{z_i}:
    I(y) = sum_i c_i * exp(-y * z_i)

This module supports:
  - Baseline measure mu^0 with coefficients a_i
  - Perturbation measure nu with coefficients b_i
  - Perturbed measure mu^eps = mu^0 + eps * nu
  - Associated inverse marginals I^0, J, I^eps
  - Normalized dual potential V^eps(y) using a fixed normalization point y_star > 0
  - Derivatives of I^0 and J of arbitrary order
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Union

ArrayLike = Union[float, np.ndarray]


class AtomicBernsteinSetup:
    """
    Finite atomic Bernstein measure setup.

    mu^0 = sum_i a_i * delta_{z_i}   (baseline, a_i > 0)
    nu   = sum_i b_i * delta_{z_i}   (perturbation, b_i in R)

    y_star: normalization point > 0 for the dual potential V^eps.
    """

    def __init__(
        self,
        z: ArrayLike,
        a: ArrayLike,
        b: ArrayLike,
        y_star: float = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        z : positive support points
        a : nonnegative baseline coefficients
        b : real-valued perturbation coefficients
        y_star : normalization reference point > 0
        """
        self.z = np.asarray(z, dtype=float)
        self.a = np.asarray(a, dtype=float)
        self.b = np.asarray(b, dtype=float)
        self.y_star = float(y_star)

        assert np.all(self.z > 0), "Support points z must be strictly positive"
        assert np.all(self.a >= 0), "Baseline coefficients a must be nonneg"
        assert self.z.shape == self.a.shape == self.b.shape, (
            "z, a, b must have the same length"
        )
        assert self.y_star > 0, "y_star must be strictly positive"

    def _broadcast(self, y: ArrayLike) -> Tuple[np.ndarray, bool]:
        """Return y as at least 1-d array and scalar flag."""
        y_arr = np.atleast_1d(np.asarray(y, dtype=float))
        scalar = y_arr.ndim == 1 and y_arr.size == 1 and np.ndim(y) == 0
        return y_arr, scalar

    def _exp_kernel(self, y: np.ndarray, coefs: np.ndarray) -> np.ndarray:
        """
        Compute sum_i coefs_i * exp(-y * z_i).
        y : array of shape (m,)
        Returns : array of shape (m,)
        """
        # Shape (m, n_atoms) via broadcasting
        exps = np.exp(-y[:, None] * self.z[None, :])  # (m, n_atoms)
        return exps @ coefs  # (m,)

    def I0(self, y: ArrayLike) -> np.ndarray:
        """Baseline inverse marginal I^0(y) = sum_i a_i * exp(-y z_i)."""
        y_arr, scalar = self._broadcast(y)
        result = self._exp_kernel(y_arr, self.a)
        return float(result[0]) if scalar else result

    def J(self, y: ArrayLike) -> np.ndarray:
        """Perturbation component J(y) = sum_i b_i * exp(-y z_i)."""
        y_arr, scalar = self._broadcast(y)
        result = self._exp_kernel(y_arr, self.b)
        return float(result[0]) if scalar else result

    def I_eps(self, y: ArrayLike, eps: float) -> np.ndarray:
        """Perturbed inverse marginal I^eps(y) = I^0(y) + eps * J(y)."""
        return self.I0(y) + eps * self.J(y)

    def dI0_dy(self, y: ArrayLike, order: int = 1) -> np.ndarray:
        """
        r-th derivative of I^0 w.r.t. y:
            d^r/dy^r I^0(y) = sum_i a_i * (-z_i)^r * exp(-y z_i)
        """
        y_arr, scalar = self._broadcast(y)
        coefs = self.a * (-self.z) ** order
        result = self._exp_kernel(y_arr, coefs)
        return float(result[0]) if scalar else result

    def dJ_dy(self, y: ArrayLike, order: int = 1) -> np.ndarray:
        """
        r-th derivative of J w.r.t. y:
            d^r/dy^r J(y) = sum_i b_i * (-z_i)^r * exp(-y z_i)
        """
        y_arr, scalar = self._broadcast(y)
        coefs = self.b * (-self.z) ** order
        result = self._exp_kernel(y_arr, coefs)
        return float(result[0]) if scalar else result

    def V_eps(self, y: ArrayLike, eps: float) -> np.ndarray:
        """
        Normalized dual potential:
            V^eps(y) = sum_i ((exp(-y z_i) - exp(-y_star z_i)) / z_i) * (a_i + eps b_i)

        This is the Bernstein-measure representation of the conjugate, normalized
        so that V^eps(y_star) = 0.
        """
        y_arr, scalar = self._broadcast(y)
        c = self.a + eps * self.b
        exp_y = np.exp(-y_arr[:, None] * self.z[None, :])      # (m, n_atoms)
        exp_ys = np.exp(-self.y_star * self.z)                   # (n_atoms,)
        kernel = (exp_y - exp_ys[None, :]) / self.z[None, :]    # (m, n_atoms)
        result = kernel @ c                                       # (m,)
        return float(result[0]) if scalar else result

    def V0(self, y: ArrayLike) -> np.ndarray:
        """Baseline dual potential V^0(y) = V_eps(y, 0)."""
        return self.V_eps(y, 0.0)

    def V1(self, y: ArrayLike) -> np.ndarray:
        """
        First perturbation of the dual potential:
            V^1(y) = sum_i ((exp(-y z_i) - exp(-y_star z_i)) / z_i) * b_i
        """
        return self.V_eps(y, 1.0) - self.V_eps(y, 0.0)

    def dV0_dy(self, y: ArrayLike, order: int = 1) -> np.ndarray:
        """
        r-th derivative of V^0 w.r.t. y:
            d^r/dy^r V^0(y) = (-1)^r * sum_i a_i * z_i^{r-1} * exp(-y z_i)
        Derived from d/dy [(exp(-yz) - exp(-y*z))/z] = -exp(-yz), so each
        differentiation brings down a factor of (-z_i), giving exponent r on
        (-1) and exponent r-1 on z_i.
        # Custom — Context7 found no library equivalent (paper Eq: V^0 derivative)
        """
        y_arr, scalar = self._broadcast(y)
        # d^r V^0/dy^r = (-1)^r * sum_i a_i * z_i^{r-1} * exp(-y z_i)
        coefs = self.a * ((-1) ** order) * (self.z ** (order - 1))
        result = self._exp_kernel(y_arr, coefs)
        return float(result[0]) if scalar else result

    def dV1_dy(self, y: ArrayLike, order: int = 1) -> np.ndarray:
        """
        r-th derivative of V^1 w.r.t. y (same structure as dV0 but with b instead of a):
            d^r/dy^r V^1(y) = (-1)^r * sum_i b_i * z_i^{r-1} * exp(-y z_i)
        """
        y_arr, scalar = self._broadcast(y)
        coefs = self.b * ((-1) ** order) * (self.z ** (order - 1))
        result = self._exp_kernel(y_arr, coefs)
        return float(result[0]) if scalar else result

    def check_positivity(self, eps_values: np.ndarray) -> bool:
        """
        Check that mu^eps = mu^0 + eps * nu is nonneg for each epsilon in eps_values.
        For atomic measures this means a_i + eps * b_i >= 0 for all i.
        """
        for eps in eps_values:
            if np.any(self.a + eps * self.b < 0):
                return False
        return True



