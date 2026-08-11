"""
Experiment 2: Validation of Analyticity and Bell-Polynomial Recursion
for the Lagrange Multiplier.

Reproduces Theorem 2: y(x, epsilon) = sum_{m>=0} eps^m/m! * y_m is analytic
near epsilon=0, with y_m given by the partial-Bell-polynomial recursion.

Methodology (from paper):
  1. BSM model with explicit H_T.
  2. Atomic CMIM I^0 and J.
  3. Solve baseline budget equation g0(y0) = x.
  4. Compute derivatives g0^{(r)}(y0) and g1^{(r)}(y0).
  5. Implement Bell-polynomial recursion for y_m, m = 0,...,M.
  6. Cross-check explicit low-order formulas y1, y2, y3.
  7. Solve exact perturbed budget equation for y_exact(eps).
  8. Verify O(|eps|^{m+1}) truncation error for order-m approximations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from math import factorial

from src.model import BSMParams, generate_HT_quadrature
from src.bernstein import AtomicBernsteinSetup
from src.dual import (
    solve_baseline_budget, solve_perturbed_budget,
    g0_fn, g0_deriv, g1_fn, g1_deriv, collect_derivatives
)
from src.bell import lagrange_multiplier_recursion, partial_bell
from src.checks import truncation_error, convergence_rate, build_truncated_series

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MAX_ORDER = 3  # compute coefficients up to y_3


def run_single_config(
    params: BSMParams,
    bm: AtomicBernsteinSetup,
    eps_seq: np.ndarray,
    n_quad: int = 200,
    label: str = "",
) -> dict:
    """
    Run exp_2 for one parameter configuration.

    Returns dict with coefficient tables, low-order checks, and error tables.
    """
    H_T, weights = generate_HT_quadrature(params, n_quad)
    x = params.x

    # Step 4: Solve baseline budget
    y0 = solve_baseline_budget(x, bm, H_T, weights)
    print(f"  [{label}] y0 = {y0:.8f}")

    # Step 5: Collect derivatives
    g0_derivs, g1_derivs, v0_derivs, v1_derivs = collect_derivatives(
        y0, MAX_ORDER, bm, H_T, weights
    )
    g0_prime = g0_derivs[1]
    print(f"  [{label}] g0'(y0) = {g0_prime:.8f}  (must be < 0)")
    assert g0_prime < 0, "g0'(y0) must be negative"

    # Step 6: Bell-polynomial recursion for y_m
    y_coeffs_rec = lagrange_multiplier_recursion(g0_derivs, g1_derivs, MAX_ORDER)
    y_coeffs_rec[0] = y0  # inject baseline root

    # Step 7: Cross-check low-order formulas
    g0_val = g0_derivs[0]
    g1_val = g1_derivs[0]
    g0pp = g0_derivs[2]  # g0''
    g0ppp = g0_derivs[3] if MAX_ORDER >= 3 else None
    g1p = g1_derivs[1] if len(g1_derivs) > 1 else None
    g1pp = g1_derivs[2] if len(g1_derivs) > 2 else None

    y1_explicit = -g1_val / g0_prime
    y1_rec = y_coeffs_rec[1]

    y2_explicit = -(g0pp * y1_explicit ** 2 + 2 * g1p * y1_explicit) / g0_prime
    y2_rec = y_coeffs_rec[2]

    if MAX_ORDER >= 3 and g1pp is not None:
        y3_explicit = -(
            g0ppp * y1_explicit ** 3
            + 3 * g0pp * y1_explicit * y2_explicit
            + 3 * g1pp * y1_explicit ** 2
            + 3 * g1p * y2_explicit
        ) / g0_prime
        y3_rec = y_coeffs_rec[3]
    else:
        y3_explicit = float("nan")
        y3_rec = float("nan")

    coeff_checks = {
        "y0": y0,
        "y1_explicit": y1_explicit,
        "y1_recursion": y1_rec,
        "y1_abs_diff": abs(y1_explicit - y1_rec),
        "y2_explicit": y2_explicit,
        "y2_recursion": y2_rec,
        "y2_abs_diff": abs(y2_explicit - y2_rec),
        "y3_explicit": y3_explicit,
        "y3_recursion": y3_rec,
        "y3_abs_diff": abs(y3_explicit - y3_rec),
        "g0_prime_sign_ok": g0_prime < 0,
    }
    print(f"  [{label}] y1: explicit={y1_explicit:.8f}, recursion={y1_rec:.8f}, "
          f"diff={abs(y1_explicit - y1_rec):.2e}")
    print(f"  [{label}] y2: explicit={y2_explicit:.8f}, recursion={y2_rec:.8f}, "
          f"diff={abs(y2_explicit - y2_rec):.2e}")

    # Step 9-11: Exact perturbed budget and truncation errors
    error_rows = []
    for eps in eps_seq:
        if not bm.check_positivity([eps]):
            continue
        y_exact = solve_perturbed_budget(x, eps, bm, H_T, weights)
        for m in range(1, MAX_ORDER + 1):
            err = truncation_error(y_exact, y_coeffs_rec, eps, m)
            error_rows.append({
                "label": label,
                "eps": eps,
                "order": m,
                "y_exact": y_exact,
                "y_approx": build_truncated_series(y_coeffs_rec, eps, m),
                "abs_error": err,
            })

    error_df = pd.DataFrame(error_rows)

    # Compute convergence rates per order
    rates = {}
    for m in range(1, MAX_ORDER + 1):
        sub = error_df[error_df["order"] == m]
        if len(sub) >= 2:
            rate = convergence_rate(sub["eps"].values, sub["abs_error"].values)
            rates[m] = rate
            print(f"  [{label}] Order-{m} convergence rate: {rate:.3f} (expected ~{m+1})")

    return {
        "label": label,
        "y0": y0,
        "g0_prime": g0_prime,
        "y_coeffs": y_coeffs_rec,
        "coeff_checks": coeff_checks,
        "error_df": error_df,
        "convergence_rates": rates,
    }


def run() -> dict:
    """Main entry point for Experiment 2."""
    print("=" * 60)
    print("Experiment 2: Bell-Polynomial Recursion Validation")
    print("=" * 60)

    # Shrinking positive epsilon sequence
    eps_seq = np.array([0.4, 0.3, 0.2, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02, 0.01])

    configs = [
        dict(
            label="Config1",
            # g0(0) = e^{-0.02} * 1.0 ≈ 0.980 → x must be below this
            x=0.80, T=1.0, r=0.02, mu=0.08, sigma=0.20,
            z=[1.0, 2.0, 3.0], a=[0.5, 0.3, 0.2], b=[0.1, -0.05, 0.02], y_star=1.0,
        ),
        dict(
            label="Config2",
            # g0(0) = e^{-0.06} * 1.0 ≈ 0.942 → x must be below this
            x=0.80, T=2.0, r=0.03, mu=0.10, sigma=0.25,
            z=[0.5, 1.5, 4.0], a=[0.4, 0.4, 0.2], b=[0.05, -0.03, 0.01], y_star=0.5,
        ),
        dict(
            label="Config3",
            x=0.5, T=1.5, r=0.01, mu=0.06, sigma=0.18,
            z=[1.0, 3.0, 6.0], a=[0.6, 0.3, 0.1], b=[0.04, -0.02, 0.005], y_star=1.0,
        ),
    ]

    all_results = []
    all_error_dfs = []

    for cfg in configs:
        params = BSMParams(x=cfg["x"], T=cfg["T"], r=cfg["r"],
                           mu=cfg["mu"], sigma=cfg["sigma"])
        bm = AtomicBernsteinSetup(
            z=cfg["z"], a=cfg["a"], b=cfg["b"], y_star=cfg["y_star"]
        )
        # Use only positive eps where positivity holds
        admissible_eps = np.array([e for e in eps_seq if bm.check_positivity([e])])
        result = run_single_config(params, bm, admissible_eps, label=cfg["label"])
        all_results.append(result)
        all_error_dfs.append(result["error_df"])

    # Compile coefficient check table
    coeff_rows = []
    for r in all_results:
        row = {"label": r["label"]}
        row.update(r["coeff_checks"])
        coeff_rows.append(row)
    coeff_df = pd.DataFrame(coeff_rows)
    print("\nCoefficient Cross-Check Table:")
    print(coeff_df[["label", "y1_abs_diff", "y2_abs_diff", "y3_abs_diff"]].to_string(index=False))

    # Save results
    full_error_df = pd.concat(all_error_dfs, ignore_index=True)
    coeff_df.to_csv(RESULTS_DIR / "exp2_coeff_checks.csv", index=False)
    full_error_df.to_csv(RESULTS_DIR / "exp2_truncation_errors.csv", index=False)

    # Plot convergence
    _plot_convergence(full_error_df, RESULTS_DIR / "exp2_convergence.png")

    print(f"\nResults saved to {RESULTS_DIR}/")
    return {"results": all_results, "coeff_df": coeff_df, "error_df": full_error_df}


def _plot_convergence(error_df: pd.DataFrame, save_path: Path) -> None:
    """Log-log convergence plot of truncation errors vs eps."""
    labels = error_df["label"].unique()
    fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 4), squeeze=False)
    colors = {1: "blue", 2: "orange", 3: "green"}

    for ax, lbl in zip(axes[0], labels):
        sub = error_df[error_df["label"] == lbl]
        for order in sorted(sub["order"].unique()):
            grp = sub[sub["order"] == order].sort_values("eps")
            mask = grp["abs_error"] > 0
            if mask.sum() < 2:
                continue
            ax.loglog(grp["eps"][mask], grp["abs_error"][mask], "o-",
                      color=colors.get(order, "gray"),
                      label=f"Order {order}")
            # Reference line
            eps_ref = grp["eps"][mask].values
            c = grp["abs_error"][mask].values[-1] / eps_ref[-1] ** (order + 1)
            ax.loglog(eps_ref, c * eps_ref ** (order + 1), "--",
                      color=colors.get(order, "gray"), alpha=0.4, label=f"O(ε^{order+1})")

        ax.set_xlabel("ε")
        ax.set_ylabel("|Error|")
        ax.set_title(f"Exp2 Truncation Errors\n{lbl}")
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


if __name__ == "__main__":
    run()
