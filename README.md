# Validation of Affine Dual Structure Under Affine Bernstein-Measure Preference Perturbations

Numerical and symbolic validation of five theorem-level claims from a CMIM (Complete-Market Inverse
Marginal) portfolio-optimisation framework, covering:

1. **Exp 1** — Affine dual identity `v(y,ε) = v₀(y) + ε·v₁(y)` (Theorem 1)
2. **Exp 2** — Bell-polynomial recursion for the Lagrange multiplier `y(x,ε)` (Theorem 2)
3. **Exp 3** — Analytic expansion of primal value `u(x,ε)` and optimal wealth `X̂_T` (Theorems 3–4)
4. **Exp 4** — CRRA moment-formula cross-check against the general recursion (Section 4)
5. **Exp 5** — First-order sensitivity under simultaneous drift + preference perturbations (Section 5)

All experiments operate in a Black-Scholes-Merton complete-market setting with a lognormal
state-price density. No empirical market data are required.

---

## Repository Structure

```
.
├── data/                   # Reserved; no empirical data needed
├── src/
│   ├── model.py            # BSM model + H_T quadrature/Monte-Carlo generation
│   ├── bernstein.py        # Atomic Bernstein measures, I⁰, J, V
│   ├── dual.py             # Dual value functions v, g and derivatives
│   ├── bell.py             # Partial Bell polynomials + coefficient recursions
│   ├── checks.py           # Residual validation utilities
│   └── crra.py             # CRRA inverse marginals + moment formulas
├── exp/
│   ├── exp_1_affine_dual.py
│   ├── exp_2_bell_recursion.py
│   ├── exp_3_primal_expansion.py
│   ├── exp_4_crra_validation.py
│   └── exp_5_sensitivity.py
├── tests/                  # 180 pytest tests (all passing)
├── results/
│   ├── RESULTS.md          # Summary tables and plots
│   ├── exp1_*.{png,csv}
│   ├── exp2_*.{png,csv}
│   ├── exp3_*.{png,csv}
│   ├── exp4_*.{png,csv}
│   └── exp5_*.{png,csv}
└── README.md
```

---

## Setup

```bash
pip install numpy scipy sympy pandas matplotlib
```

All other dependencies (mpmath, pytest) are either in stdlib or already available in the container.

---

## Running Experiments

```bash
# Individual experiment
python exp/exp_1_affine_dual.py
python exp/exp_2_bell_recursion.py
python exp/exp_3_primal_expansion.py
python exp/exp_4_crra_validation.py
python exp/exp_5_sensitivity.py

# Full test suite
pytest tests/ -v
```

Results (CSV tables and PNG plots) are written to `results/`.

---

## Key Results

| Experiment | Validated Claim | Max Residual |
|-----------|----------------|-------------|
| Exp 1 | Affine dual identity | ≤ 1.3 × 10⁻¹⁶ (machine ε) |
| Exp 2 | Bell recursion coefficients y₁,y₂,y₃ | 0.0 (exact) |
| Exp 3 | u₁,u₂ identities; X₁,X₂ identities | 0.0 (exact) |
| Exp 4 | CRRA closed-form vs recursion (y₁,X₁) | ≤ 8.9 × 10⁻¹⁶ |
| Exp 5 | Normalized sensitivity residual → 0 | slope ≈ 0.94–1.00 |

See [`results/RESULTS.md`](results/RESULTS.md) for full tables and convergence plots.
