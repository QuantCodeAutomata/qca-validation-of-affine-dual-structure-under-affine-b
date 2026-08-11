"""
Experiment 4: Closed-Form CRRA Perturbation Validation via Moment Formulas.

Reproduces Section 4: CRRA baseline I^{p0} perturbed by I^{p1}.

I^epsilon(y) = I^{p0}(y) + eps * I^{p1}(y) = y^{1/(p0-1)} + eps * y^{1/(p1-1)}

Methodology (from paper):
  1. BSM model with explicit moments M(alpha) = E[H_T^alpha].
  2. CRRA parameters p0, p1 with p0<1, p1<1, p0!=p1.
  3. Closed-form y0, X0, y1, X1 via moment formulas.
  4. General Bell recursion route for y1 cross-check.
  5. Exact perturbed budget solutions via analytic moment equation.
  6. First-order residuals scale as O(eps^2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from math import factorial

from src.model import BSMParams, generate_HT_monte_carlo, bsm_moment
from src.crra import (
    crra_inverse_marginal, crra_inverse_marginal_deriv,
    crra_baseline_y0, crra_g0, crra_g0_deriv, crra_g1, crra_g1_deriv,
    crra_y1_closed_form, crra_solve_perturbed_y
)
from src.checks import convergence_rate

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_MC = 100_000


def run_single_config(
    params: BSMParams,
    p0: float,
    p1: float,
    eps_seq: np.ndarray,
    label: str = "",
) -> dict:
    """
    Run exp_4 for one CRRA parameter pair.
    eps_seq should be positive (right-sided perturbation).
    """
    assert p0 < 1 and p0 != 0 and p1 < 1 and p1 != 0 and p0 != p1, \
        "Invalid CRRA parameters"
    assert np.all(eps_seq >= 0), "Use non-negative epsilon for CRRA positivity"

    x = params.x
    a = p0 / (p0 - 1)
    b = p1 / (p1 - 1)
    Ma = bsm_moment(params, a)
    Mb = bsm_moment(params, b)

    # Step 5: Closed-form baseline
    y0_closed = crra_baseline_y0(x, p0, params)
    print(f"  [{label}] y0 closed-form = {y0_closed:.8f}")

    # Step 6: Baseline optimizer (pathwise, using MC)
    H_T_mc, W_T_mc, weights_mc = generate_HT_monte_carlo(params, n_samples=N_MC)
    Z0 = y0_closed * H_T_mc
    X0 = crra_inverse_marginal(Z0, p0)

    # Step 7a: General recursion for y1
    g0_val = crra_g0(y0_closed, p0, params)   # should ≈ x
    g0_prime = crra_g0_deriv(y0_closed, p0, params, order=1)
    g1_val = crra_g1(y0_closed, p1, params)
    y1_rec = -g1_val / g0_prime
    print(f"  [{label}] g0(y0)={g0_val:.6f} (should be x={x}), g0'(y0)={g0_prime:.6f}")

    # Step 7b: Closed-form y1
    y1_closed = crra_y1_closed_form(y0_closed, p0, p1, params)
    y1_diff = abs(y1_rec - y1_closed)
    print(f"  [{label}] y1: recursion={y1_rec:.8f}, closed={y1_closed:.8f}, diff={y1_diff:.2e}")

    # Step 8: First-order optimizer X1
    Z1 = y1_closed * H_T_mc
    # General formula: X1 = (I^{p0})'(Z0) * Z1 + I^{p1}(Z0)
    X1_general = crra_inverse_marginal_deriv(Z0, p0, order=1) * Z1 + crra_inverse_marginal(Z0, p1)
    # Closed form from paper: X1 = (y0 H_T)^{1/(p1-1)} + (y1/(p0-1)) * H_T * (y0 H_T)^{1/(p0-1)-1}
    X1_closed = (y0_closed * H_T_mc) ** (1.0 / (p1 - 1)) + \
                (y1_closed / (p0 - 1)) * H_T_mc * (y0_closed * H_T_mc) ** (1.0 / (p0 - 1) - 1)
    X1_diff_max = float(np.max(np.abs(X1_general - X1_closed)))
    print(f"  [{label}] X1 max diff (general vs closed): {X1_diff_max:.2e}")

    # Steps 9-10: Exact perturbed budget and first-order residuals
    scalar_rows = []
    for eps in eps_seq:
        y_exact = crra_solve_perturbed_y(x, eps, p0, p1, params)
        y_first_order = y0_closed + eps * y1_closed

        # Exact pathwise optimizer
        X_exact = crra_inverse_marginal(y_exact * H_T_mc, p0) + \
                  eps * crra_inverse_marginal(y_exact * H_T_mc, p1)
        X_first_order = X0 + eps * X1_closed

        y_err = abs(y_exact - y_first_order)
        X_sup = float(np.max(np.abs(X_exact - X_first_order)))
        X_L2 = float(np.sqrt(np.mean((X_exact - X_first_order) ** 2)))

        # Normalized (by eps, since first-order residual ~ O(eps^2))
        y_norm_err = y_err / (eps + 1e-15)
        X_norm_sup = X_sup / (eps + 1e-15)

        scalar_rows.append({
            "label": label, "p0": p0, "p1": p1, "eps": eps,
            "y_exact": y_exact, "y_first_order": y_first_order,
            "y_abs_err": y_err, "y_norm_err": y_norm_err,
            "X_sup": X_sup, "X_L2": X_L2,
            "X_norm_sup": X_norm_sup,
        })

    result_df = pd.DataFrame(scalar_rows)

    # Convergence rate for y
    if len(result_df) >= 2:
        mask = result_df["y_abs_err"] > 0
        rate_y = convergence_rate(result_df["eps"][mask].values,
                                  result_df["y_abs_err"][mask].values)
        rate_X = convergence_rate(result_df["eps"][mask].values,
                                  result_df["X_sup"][mask].values)
        print(f"  [{label}] y residual convergence rate: {rate_y:.3f} (expected ~2)")
        print(f"  [{label}] X sup residual convergence rate: {rate_X:.3f} (expected ~2)")

    return {
        "label": label, "p0": p0, "p1": p1,
        "y0": y0_closed, "y1_rec": y1_rec, "y1_closed": y1_closed,
        "y1_diff": y1_diff, "X1_diff_max": X1_diff_max,
        "result_df": result_df,
    }


def run() -> dict:
    """Main entry point for Experiment 4."""
    print("=" * 60)
    print("Experiment 4: CRRA Perturbation Validation via Moment Formulas")
    print("=" * 60)

    # Right-sided positive epsilon sequence
    eps_seq = np.array([0.30, 0.20, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02, 0.01, 0.005])

    configs = [
        dict(label="C1_p0=-0.5_p1=-1.0",
             x=1.0, T=1.0, r=0.02, mu=0.08, sigma=0.20,
             p0=-0.5, p1=-1.0),
        dict(label="C2_p0=0.5_p1=-0.5",
             x=1.0, T=1.0, r=0.02, mu=0.08, sigma=0.20,
             p0=0.5, p1=-0.5),
        dict(label="C3_p0=-1.0_p1=0.5",
             x=2.0, T=2.0, r=0.03, mu=0.10, sigma=0.25,
             p0=-1.0, p1=0.5),
        dict(label="C4_p0=0.5_p1=-2.0",
             x=0.5, T=1.5, r=0.01, mu=0.06, sigma=0.18,
             p0=0.5, p1=-2.0),
    ]

    all_results = []
    all_dfs = []

    for cfg in configs:
        params = BSMParams(x=cfg["x"], T=cfg["T"], r=cfg["r"],
                           mu=cfg["mu"], sigma=cfg["sigma"])
        result = run_single_config(params, cfg["p0"], cfg["p1"], eps_seq, label=cfg["label"])
        all_results.append(result)
        all_dfs.append(result["result_df"])

    full_df = pd.concat(all_dfs, ignore_index=True)

    # Summary table
    summary = pd.DataFrame([{
        "label": r["label"], "p0": r["p0"], "p1": r["p1"],
        "y1_diff": r["y1_diff"], "X1_diff_max": r["X1_diff_max"],
    } for r in all_results])
    print("\nClosed-Form vs Recursion Comparison:")
    print(summary.to_string(index=False))

    full_df.to_csv(RESULTS_DIR / "exp4_crra_results.csv", index=False)
    summary.to_csv(RESULTS_DIR / "exp4_summary.csv", index=False)

    _plot_crra_convergence(full_df, RESULTS_DIR / "exp4_convergence.png")

    print(f"\nResults saved to {RESULTS_DIR}/")
    return {"results": all_results, "full_df": full_df, "summary": summary}


def _plot_crra_convergence(df: pd.DataFrame, save_path: Path) -> None:
    labels = df["label"].unique()
    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for ax, lbl in zip(axes[0], labels):
        sub = df[df["label"] == lbl].sort_values("eps")
        mask = sub["y_abs_err"] > 0
        if mask.sum() >= 2:
            ax.loglog(sub["eps"][mask], sub["y_abs_err"][mask], "o-", label="|y_err|")
            eps_ref = sub["eps"][mask].values
            c = sub["y_abs_err"][mask].values[-1] / eps_ref[-1] ** 2
            ax.loglog(eps_ref, c * eps_ref ** 2, "--", alpha=0.5, label="O(ε²)")
        mask_X = sub["X_sup"] > 0
        if mask_X.sum() >= 2:
            ax.loglog(sub["eps"][mask_X], sub["X_sup"][mask_X], "s--", label="|X_sup|")
        ax.set_xlabel("ε")
        ax.set_ylabel("First-order error")
        ax.set_title(f"Exp4 CRRA Residuals\n{lbl}")
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


if __name__ == "__main__":
    run()
