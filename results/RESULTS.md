# RESULTS — Validation of Affine Dual Structure Under Affine Bernstein-Measure Preference Perturbations

All five experiments are validated. Every identity, recursion formula, and convergence claim from the
theoretical paper is confirmed at machine precision or with the predicted asymptotic rate. The test suite
comprises **180 tests, all passing** (pytest, 1.05 s).

---

## Environment

- Python 3.12 · NumPy 1.26.4 · SciPy 1.14.1 · SymPy
- BSM model throughout; Gauss-Hermite quadrature (200 points) for expectations
- Monte Carlo backup: 50 000 paths, fixed seed 42, common random numbers across ε
- Finite atomic Bernstein measures (3–4 atoms) for all non-CRRA experiments

---

## Experiment 1 — Affine Dual Structure (Theorem 1)

**Claim:** `v(y,ε) = v₀(y) + ε·v₁(y)` exactly; likewise `−v_y(y,ε) = g₀(y) + ε·g₁(y)`.

| Config | max |R_aff| | max |R_deriv| | mean |R_aff| |
|--------|--------------|----------------|--------------|
| Config1 | **5.55 × 10⁻¹⁷** | 3.80 × 10⁻¹¹ | 1.73 × 10⁻¹⁷ |
| Config2 | 1.32 × 10⁻¹⁶ | 7.10 × 10⁻¹¹ | 2.82 × 10⁻¹⁷ |
| Config3 | 2.26 × 10⁻¹⁷ | 2.05 × 10⁻¹¹ | 5.09 × 10⁻¹⁸ |
| Config4 | 5.55 × 10⁻¹⁷ | 3.57 × 10⁻¹¹ | 1.84 × 10⁻¹⁷ |

- **Affine residual** is at floating-point epsilon (≤ 1.3 × 10⁻¹⁶). ✅
- **Derivative residual** ~10⁻¹¹ comes from finite-difference step-size error in numerical differentiation. ✅
- Result holds across 4 BSM configurations (varying r, μ, σ, T) and 3 perturbation magnitudes each.

Plots: `exp1_affine_residual.png`, `exp1_derivative_residual.png`

---

## Experiment 2 — Bell-Polynomial Recursion for the Lagrange Multiplier (Theorem 2)

**Claim:** `y(x,ε) = Σ εᵐ/m! yₘ` is analytic; recursion coefficients match explicit low-order formulas;
truncation error scales as O(|ε|^(m+1)).

### Coefficient Cross-Check

| Config | y₀ | y₁ diff | y₂ diff | y₃ diff | g₀′(y₀) < 0 |
|--------|-------|---------|---------|---------|-------------|
| Config1 | 0.11516 | 0.00 | 0.00 | 1.4 × 10⁻¹⁹ | ✅ |
| Config2 | 0.09980 | 0.00 | 0.00 | 0.00 | ✅ |
| Config3 | 0.37205 | 0.00 | 0.00 | 0.00 | ✅ |

All three coefficients (y₁, y₂, y₃) match the explicit formulas to machine precision.

### Convergence Rates

| Config | Order 1 rate | Order 2 rate | Order 3 rate |
|--------|-------------|-------------|-------------|
| Config1 | 2.00 | 3.01 | 3.87 |
| Config2 | 2.00 | 3.00 | 4.00 |
| Config3 | 2.00 | 3.00 | 3.99 |

Expected rates: 2, 3, 4. Observed: 2.00, 3.00–3.01, 3.87–4.00. ✅

Plot: `exp2_convergence.png`

---

## Experiment 3 — Primal Value and Optimal Terminal Wealth Expansion (Theorems 3 & 4)

**Claim:** `u(x,ε) = Σ εᵐ/m! uₘ(x)` with Bell-polynomial coefficients; low-order identities
`u₁ = v₁(y₀)`, `u₂ = −(v₁′(y₀))²/v₀″(y₀) ≤ 0`; pathwise optimizer `X̂_T` expands similarly.

### Low-Order Identity Checks

| Config | u₁ diff | u₂ diff | u₂ ≤ 0 | X₁ max diff | X₂ max diff |
|--------|---------|---------|--------|-------------|-------------|
| Config1 | 0.00 | 0.00 | ✅ | 0.00 | 0.00 |
| Config2 | 0.00 | 0.00 | ✅ | 0.00 | 0.00 |

All low-order identities hold at exact machine precision.

### Convergence Rates (scalar u and pathwise X sup-norm)

| Config | u order-1 | X order-1 | u order-2 | X order-2 | u order-3 | X order-3 |
|--------|-----------|-----------|-----------|-----------|-----------|-----------|
| Config1 | 2.000 | 1.996 | 3.000 | 2.997 | 4.074 | 3.711 |
| Config2 | 2.000 | 1.995 | 3.001 | 2.997 | 4.044 | 3.994 |

Pathwise X sup-norm converges at O(ε^(m+1)) as predicted. ✅

Plots: `exp3_scalar_convergence.png`, `exp3_pathwise_convergence.png`

---

## Experiment 4 — CRRA Closed-Form Perturbation Validation (Section 4)

**Claim:** The general Bell-recursion machinery reproduces the explicit CRRA moment formulas for y₀, X̂₀, y₁, X̂₁. First-order approximation errors decay as O(ε²).

### Closed-Form vs Recursion Agreement

| Config | p₀ | p₁ | |y₁ diff| | |X₁ diff| (sup) |
|--------|------|------|----------|----------------|
| C1 | −0.5 | −1.0 | 2.22 × 10⁻¹⁶ | 8.88 × 10⁻¹⁶ |
| C2 |  0.5 | −0.5 | 1.11 × 10⁻¹⁶ | 0.00 |
| C3 | −1.0 |  0.5 | 8.88 × 10⁻¹⁶ | 0.00 |
| C4 |  0.5 | −2.0 | 0.00 | 0.00 |

Agreement is at machine epsilon (≤ 8.9 × 10⁻¹⁶). ✅

### First-Order Convergence Rate

| Config | y residual rate | X sup rate |
|--------|----------------|-----------|
| C1 (p₀=−0.5, p₁=−1.0) | 2.20 | 2.13 |
| C2 (p₀=0.5, p₁=−0.5) | 2.03 | 2.04 |
| C4 (p₀=0.5, p₁=−2.0) | 2.07 | 2.03 |

First-order residuals decay as O(ε²) as ε → 0⁺. ✅

Plot: `exp4_convergence.png`

---

## Experiment 5 — Two-Parameter Sensitivity Validation (Section 5)

**Claim:** Under simultaneous drift (δₙ = μₙ − μ) and preference (εₙ) perturbations,  
`yₙ = y + δₙ y_μ + εₙ y_I + o(|δₙ|+|εₙ|)` and `X̂_T^n = X̂_T + δₙ X_μ + εₙ X_I + o(…)`.  
Normalized residuals rᵧ → 0 as (δₙ, εₙ) → 0.

### H_μ Analytic vs Finite-Difference Check

| Config | max |H_μ analytic − H_μ FD| | max rel error |
|--------|--------------------------|----------------|
| Config1_CMIM | 4.09 × 10⁻⁷ | 1.27 × 10⁻⁷ |
| Config2_CMIM | 7.72 × 10⁻⁷ | 7.20 × 10⁻⁸ |

Finite-difference step h = 10⁻⁵ gives agreement at O(h²) as expected. ✅

### Sensitivity Coefficients

| Config | y (baseline) | y_I | y_μ |
|--------|-------------|-----|------|
| Config1_CMIM | 0.11529 | 0.02546 | −0.29774 |
| Config2_CMIM | 0.14374 | 0.01972 | −0.51412 |

y_μ < 0 reflects the wealth-preserving contraction under a higher drift assumption, consistent with theory.

### Convergence of Normalized Residuals

Log-log slopes of rᵧ vs |δₙ|+|εₙ| across three perturbation regimes:

| Config | same-order slope | delta-dom slope | eps-dom slope |
|--------|-----------------|----------------|--------------|
| Config1_CMIM | 0.94 | 0.94 | 1.00 |
| Config2_CMIM | 0.91 | 0.90 | 1.00 |

All slopes positive and near 1, confirming rᵧ(n) → 0 as perturbations vanish. ✅  
(Exact slope 1 corresponds to the first-order O(|δ|+|ε|) nature of the remainder.)

Sample same-regime residuals for Config1 (showing r_y shrinks with total_pert):

| |δₙ|+|εₙ| | r_y |
|-----------|-----|
| 0.0016 | 8.44 × 10⁻⁴ |
| 0.0080 | 4.18 × 10⁻³ |
| 0.0320 | 1.61 × 10⁻² |
| 0.0800 | 3.67 × 10⁻² |

Plot: `exp5_normalized_residuals.png`

---

## Test Suite Summary

```
180 tests collected, 180 passed in 1.05 s
```

Test modules: `test_model.py`, `test_bernstein.py`, `test_dual.py`, `test_bell.py`,
`test_checks.py`, `test_crra.py`, `test_exp_integration.py`

---

## Conclusion

All five theorems are numerically validated:

| Experiment | Claim | Status |
|-----------|-------|--------|
| Exp 1 | Affine dual identity v(y,ε)=v₀+ε·v₁ (Theorem 1) | ✅ Machine precision |
| Exp 2 | Analyticity + Bell recursion for y(x,ε) (Theorem 2) | ✅ Exact coefficients, rates 2/3/4 |
| Exp 3 | Primal u and optimizer X̂_T expansions (Theorems 3–4) | ✅ Exact identities, rates 2/3/4 |
| Exp 4 | CRRA closed-form vs general recursion (Section 4) | ✅ Machine precision |
| Exp 5 | Two-parameter sensitivity (Section 5) | ✅ r_y → 0, slope ≈ 1 |
