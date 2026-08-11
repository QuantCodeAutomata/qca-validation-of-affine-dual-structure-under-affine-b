"""
Tests for AtomicBernsteinSetup: inverse marginals, dual potentials,
normalization, derivatives, and positivity checking.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.bernstein import AtomicBernsteinSetup


@pytest.fixture()
def bm_simple() -> AtomicBernsteinSetup:
    """A three-atom setup with positive a and mixed-sign b."""
    return AtomicBernsteinSetup(
        z=[1.0, 2.0, 3.0],
        a=[0.5, 0.3, 0.2],
        b=[0.1, -0.05, 0.02],
        y_star=1.0,
    )


@pytest.fixture()
def bm_two_atom() -> AtomicBernsteinSetup:
    return AtomicBernsteinSetup(
        z=[0.5, 2.0],
        a=[1.0, 0.5],
        b=[0.2, -0.1],
        y_star=0.5,
    )


class TestAtomicBernsteinSetupValidation:
    def test_negative_z_raises(self):
        with pytest.raises(AssertionError):
            AtomicBernsteinSetup(z=[-1.0, 2.0], a=[0.5, 0.5], b=[0.0, 0.0])

    def test_zero_z_raises(self):
        with pytest.raises(AssertionError):
            AtomicBernsteinSetup(z=[0.0, 1.0], a=[0.5, 0.5], b=[0.0, 0.0])

    def test_negative_a_raises(self):
        with pytest.raises(AssertionError):
            AtomicBernsteinSetup(z=[1.0, 2.0], a=[-0.1, 0.5], b=[0.0, 0.0])

    def test_shape_mismatch_raises(self):
        with pytest.raises(AssertionError):
            AtomicBernsteinSetup(z=[1.0, 2.0], a=[0.5], b=[0.0, 0.0])

    def test_zero_y_star_raises(self):
        with pytest.raises(AssertionError):
            AtomicBernsteinSetup(z=[1.0, 2.0], a=[0.5, 0.5], b=[0.0, 0.0], y_star=0.0)


class TestInverseMarginals:
    def test_I0_scalar_positive(self, bm_simple):
        """I^0(y) must be positive for any y > 0."""
        for y in [0.1, 1.0, 5.0, 10.0]:
            assert bm_simple.I0(y) > 0

    def test_I0_explicit_value(self):
        """One atom: I^0(y) = a * exp(-y * z)."""
        bm = AtomicBernsteinSetup(z=[2.0], a=[1.0], b=[0.0], y_star=1.0)
        y = 1.5
        expected = np.exp(-y * 2.0)
        assert bm.I0(y) == pytest.approx(expected, rel=1e-12)

    def test_I0_array_input(self, bm_simple):
        y_arr = np.array([0.5, 1.0, 2.0])
        result = bm_simple.I0(y_arr)
        assert result.shape == (3,)
        assert np.all(result > 0)

    def test_I0_decreasing_in_y(self, bm_simple):
        """I^0 should be decreasing (a_i > 0, z_i > 0 → derivative is negative)."""
        y_vals = np.linspace(0.1, 5.0, 50)
        I0_vals = bm_simple.I0(y_vals)
        assert np.all(np.diff(I0_vals) < 0)

    def test_J_explicit_value(self):
        bm = AtomicBernsteinSetup(z=[1.0, 2.0], a=[0.5, 0.5], b=[0.3, -0.1], y_star=1.0)
        y = 1.0
        expected = 0.3 * np.exp(-y) + (-0.1) * np.exp(-2 * y)
        assert bm.J(y) == pytest.approx(expected, rel=1e-12)

    def test_I_eps_at_zero_eps_equals_I0(self, bm_simple):
        for y in [0.5, 1.0, 3.0]:
            assert bm_simple.I_eps(y, 0.0) == pytest.approx(bm_simple.I0(y), rel=1e-12)

    def test_I_eps_linear_in_eps(self, bm_simple):
        y, e1, e2 = 1.5, 0.1, 0.2
        val1 = bm_simple.I_eps(y, e1)
        val2 = bm_simple.I_eps(y, e2)
        val_mid = bm_simple.I_eps(y, 0.15)
        assert val_mid == pytest.approx(0.5 * (val1 + val2), rel=1e-12)


class TestDualPotential:
    def test_V_eps_normalization_at_y_star(self, bm_simple):
        """V^eps(y_star) must be zero for any eps (normalization)."""
        for eps in [-0.3, 0.0, 0.1, 0.3]:
            v = bm_simple.V_eps(bm_simple.y_star, eps)
            assert abs(v) < 1e-14, f"V^eps(y_star) = {v} ≠ 0 at eps={eps}"

    def test_V0_normalization_at_y_star(self, bm_simple):
        assert abs(bm_simple.V0(bm_simple.y_star)) < 1e-14

    def test_V1_normalization_at_y_star(self, bm_simple):
        assert abs(bm_simple.V1(bm_simple.y_star)) < 1e-14

    def test_V_eps_affine_in_eps(self, bm_simple):
        """V^eps(y) = V^0(y) + eps * V^1(y) exactly (algebraic identity)."""
        y_vals = np.array([0.2, 0.5, 1.0, 2.0, 5.0])
        for eps in [-0.2, 0.0, 0.1, 0.25]:
            if not bm_simple.check_positivity([eps]):
                continue
            V_full = bm_simple.V_eps(y_vals, eps)
            V_linear = bm_simple.V0(y_vals) + eps * bm_simple.V1(y_vals)
            np.testing.assert_allclose(V_full, V_linear, atol=1e-14,
                                       err_msg=f"Affine identity failed at eps={eps}")

    def test_V_eps_explicit_one_atom(self):
        """Single atom: V^eps(y) = (e^{-yz} - e^{-y*z}) / z * (a + eps*b)."""
        z, a, b, y_star = 2.0, 0.6, 0.1, 1.0
        bm = AtomicBernsteinSetup(z=[z], a=[a], b=[b], y_star=y_star)
        eps = 0.15
        y = 0.8
        expected = (np.exp(-y * z) - np.exp(-y_star * z)) / z * (a + eps * b)
        assert bm.V_eps(y, eps) == pytest.approx(expected, rel=1e-12)

    def test_V_eps_array_input(self, bm_simple):
        y_arr = np.array([0.5, 1.0, 2.0, 3.0])
        result = bm_simple.V_eps(y_arr, 0.1)
        assert result.shape == (4,)
        assert np.all(np.isfinite(result))


class TestDerivatives:
    def test_dI0_order1_explicit(self):
        """d/dy I^0(y) = sum_i a_i * (-z_i) * exp(-y z_i)."""
        bm = AtomicBernsteinSetup(z=[1.0, 2.0], a=[0.5, 0.3], b=[0.0, 0.0], y_star=1.0)
        y = 1.5
        expected = 0.5 * (-1.0) * np.exp(-1.5) + 0.3 * (-2.0) * np.exp(-3.0)
        assert bm.dI0_dy(y, order=1) == pytest.approx(expected, rel=1e-12)

    def test_dI0_order1_is_negative(self, bm_simple):
        """Since a_i, z_i > 0, first derivative must be negative."""
        for y in [0.5, 1.0, 3.0]:
            assert bm_simple.dI0_dy(y, order=1) < 0

    def test_dI0_order2_is_positive(self, bm_simple):
        """Second derivative sum a_i z_i^2 exp(-y z_i) > 0."""
        for y in [0.5, 1.0, 3.0]:
            assert bm_simple.dI0_dy(y, order=2) > 0

    def test_dJ_explicit(self):
        bm = AtomicBernsteinSetup(z=[1.0, 3.0], a=[0.5, 0.5], b=[0.4, -0.1], y_star=1.0)
        y = 2.0
        expected = 0.4 * (-1.0) * np.exp(-2.0) + (-0.1) * (-3.0) * np.exp(-6.0)
        assert bm.dJ_dy(y, order=1) == pytest.approx(expected, rel=1e-12)

    def test_dV0_dy_matches_finite_difference(self, bm_simple):
        """dV0/dy(y) ≈ [V0(y+h) - V0(y-h)]/(2h)."""
        y, h = 1.5, 1e-6
        fd = (bm_simple.V0(y + h) - bm_simple.V0(y - h)) / (2 * h)
        analytic = bm_simple.dV0_dy(y, order=1)
        assert analytic == pytest.approx(fd, rel=1e-5)

    def test_dV0_dy_equals_neg_I0(self, bm_simple):
        """dV0/dy(y) = -I0(y).

        d/dy V^0(y) = d/dy [sum a_i (e^{-yz_i} - e^{-y*z_i})/z_i]
                    = sum a_i * (-z_i)/z_i * e^{-yz_i}
                    = -sum a_i * e^{-yz_i} = -I^0(y)

        In the implementation:
          coefs = a * (-1)^order * z^{order-1}  →  for order=1: coefs = a * (-1) * 1 = -a
          dV0_dy(y, 1) = sum(-a_i * exp(-y z_i)) = -I^0(y)
        """
        y = 1.5
        assert bm_simple.dV0_dy(y, order=1) == pytest.approx(-bm_simple.I0(y), rel=1e-12)


class TestPositivityCheck:
    def test_positive_b_always_admissible(self):
        bm = AtomicBernsteinSetup(z=[1.0, 2.0], a=[0.5, 0.5], b=[0.1, 0.1], y_star=1.0)
        assert bm.check_positivity(np.array([0.0, 0.1, 0.5, 1.0]))

    def test_large_negative_eps_not_admissible(self):
        bm = AtomicBernsteinSetup(z=[1.0, 2.0], a=[0.5, 0.5], b=[-0.6, 0.1], y_star=1.0)
        # a[0] + eps*b[0] = 0.5 - 0.6*eps; negative for eps > 5/6
        assert not bm.check_positivity(np.array([0.9]))

    def test_boundary_admissibility(self):
        bm = AtomicBernsteinSetup(z=[1.0], a=[0.3], b=[0.3], y_star=1.0)
        # a + eps*b = 0.3 + 0.3*eps >= 0 for eps >= -1
        assert bm.check_positivity(np.array([-1.0 + 1e-10]))
        assert not bm.check_positivity(np.array([-1.0 - 1e-10]))
