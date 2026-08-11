"""
Experiment 3: Validation of Analytic Expansions for the Primal Value Function
and Optimal Terminal Wealth Map.

Reproduces Theorems 3 and 4: expansion of u(x,epsilon) and X_hat_T(x,epsilon).

Methodology (from paper):
  1. BSM model + fixed MC sample of H_T (common random numbers).
  2. Finite atomic Bernstein setup for I^0, J, V^0, V^1.
  3. Compute v0 and v1 and their derivatives at y0.
  4. Bell-polynomial recursion: y_m -> u_m, X_m.
  5. Explicit low-order checks: u1=v1(y0), u2<=0 and -(v1')^2/v0''.
  6. Exact perturbed budget solve and exact optimizer X_exact.
  7. Scalar and pathwise residual decay as O(|eps|^{m+1}).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from math import factorial

from src.model import BSMParams, generate_HT_quadrature, generate_HT_monte_carlo
from src.bernstein import AtomicBernsteinSetup
from src.dual import (
    solve_baseline_budget, solve_perturbed_budget, v_eps as v_eps_fn,
    v0_fn, v1_fn, g0_fn, g1_fn, collect_derivatives
)
from src.bell import (
    lagrange_multiplier_recursion, primal_value_recursion,
    optimizer_recursion_pathwise
)
from src.checks import build_truncated_series, convergence_rate

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MAX_ORDER = 3
N_MC = 50_000


def make_dI0_fns(bm: AtomicBernsteinSetup):
    """Return list of callables for (I^0)^{(r)}(y), r = 0, 1, ..., MAX_ORDER."""
    fns = []
    fns.append(lambda y: bm.I0(y))
    for r in range(1, MAX_ORDER + 1):
        r_ = r
        fns.append(lambda y, r=r_: bm.dI0_dy(y, order=r))
    return fns


def make_dJ_fns(bm: AtomicBernsteinSetup):
    """Return list of callables for J^{(r)}(y), r = 0, 1, ..., MAX_ORDER."""
    fns = []
    fns.append(lambda y: bm.J(y))
    for r in range(1, MAX_ORDER + 1):
        r_ = r
        fns.append(lambda y, r=r_: bm.dJ_dy(y, order=r))
    return fns


def exact_u(
    y_exact: float,
    eps: float,
    x: float,
    bm: AtomicBernsteinSetup,
    H_T_quad: np.ndarray,
    weights_quad: np.ndarray,
) -> float:
    """
    Exact primal value via duality:
        u(x, eps) = v(y_exact, eps) + x * y_exact
    """
    v_val = v_eps_fn(y_exact, eps, bm, H_T_quad, weights_quad)
    return v_val + x * y_exact


def run_single_config(
    params: BSMParams,
    bm: AtomicBernsteinSetup,
    eps_seq: np.ndarray,
    label: str = "",
) -> dict:
    """Run exp_3 for one configuration."""
    x = params.x

    # Quadrature for scalar expectations
    H_T_quad, weights_quad = generate_HT_quadrature(params, n_points=200)
    # MC for pathwise comparisons (common random numbers across eps)
    H_T_mc, W_T_mc, weights_mc = generate_HT_monte_carlo(params, n_samples=N_MC)

    # Step 1: Solve baseline
    y0 = solve_baseline_budget(x, bm, H_T_quad, weights_quad)
    print(f"  [{label}] y0 = {y0:.8f}")

    # Step 2-4: Collect derivatives and run recursions
    g0_derivs, g1_derivs, v0_derivs, v1_derivs = collect_derivatives(
        y0, MAX_ORDER, bm, H_T_quad, weights_quad
    )
    y_coeffs = lagrange_multiplier_recursion(g0_derivs, g1_derivs, MAX_ORDER)
    y_coeffs[0] = y0

    # Primal value coefficients
    u_coeffs = primal_value_recursion(v0_derivs, v1_derivs, y_coeffs, x, MAX_ORDER)

    # Optimizer coefficients (pathwise, using MC sample)
    dI0_fns = make_dI0_fns(bm)
    dJ_fns = make_dJ_fns(bm)
    X_coeffs = optimizer_recursion_pathwise(dI0_fns, dJ_fns, y_coeffs, H_T_mc, MAX_ORDER)

    # Step 5: Low-order identity checks
    v1_at_y0 = v1_fn(y0, bm, H_T_quad, weights_quad)
    v0pp = v0_derivs[2]  # v0''(y0)
    v1p = v1_derivs[1]   # v1'(y0) = -g1(y0)

    u1_rec = u_coeffs[1]
    u1_expected = v1_at_y0  # should equal v1(y0)
    u1_diff = abs(u1_rec - u1_expected)

    u2_rec = u_coeffs[2]
    u2_formula = -(v1p ** 2) / v0pp  # = -(v1'(y0))^2 / v0''(y0)
    u2_diff = abs(u2_rec - u2_formula)

    print(f"  [{label}] u1: recursion={u1_rec:.8f}, v1(y0)={u1_expected:.8f}, diff={u1_diff:.2e}")
    print(f"  [{label}] u2: recursion={u2_rec:.8f}, formula={u2_formula:.8f}, "
          f"diff={u2_diff:.2e}, u2<=0: {u2_rec <= 0}")

    # X1 check: X1 = (I^0)'(Z0) * Z1 + J(Z0), Z_j = y_j * H_T
    Z0 = y_coeffs[0] * H_T_mc
    Z1 = y_coeffs[1] * H_T_mc
    X1_explicit = bm.dI0_dy(Z0, order=1) * Z1 + bm.J(Z0)
    X1_rec = X_coeffs[1]
    X1_diff = np.max(np.abs(X1_explicit - X1_rec))

    X2_explicit = (
        bm.dI0_dy(Z0, order=2) * Z1 ** 2
        + bm.dI0_dy(Z0, order=1) * (y_coeffs[2] * H_T_mc)
        + 2 * bm.dJ_dy(Z0, order=1) * Z1
    )
    X2_rec = X_coeffs[2]
    X2_diff = np.max(np.abs(X2_explicit - X2_rec))
    print(f"  [{label}] X1 max diff: {X1_diff:.2e}")
    print(f"  [{label}] X2 max diff: {X2_diff:.2e}")

    # Step 9-11: Exact perturbed solutions and residuals
    scalar_rows = []
    pathwise_rows = []

    for eps in eps_seq:
        if not bm.check_positivity([eps]):
            continue

        y_exact = solve_perturbed_budget(x, eps, bm, H_T_quad, weights_quad)
        u_exact = exact_u(y_exact, eps, x, bm, H_T_quad, weights_quad)

        # Pathwise exact optimizer
        X_exact = bm.I_eps(y_exact * H_T_mc, eps)

        for m in range(1, MAX_ORDER + 1):
            u_approx = build_truncated_series(u_coeffs, eps, m)
            u_err = abs(u_exact - u_approx)

            X_approx = sum(
                X_coeffs[j] * eps ** j / factorial(j) for j in range(m + 1)
            )
            X_resid = np.abs(X_exact - X_approx)
            X_sup = float(np.max(X_resid))
            X_L2 = float(np.sqrt(np.mean(X_resid ** 2)))
            X_mean = float(np.mean(X_resid))

            scalar_rows.append({
                "label": label, "eps": eps, "order": m,
                "u_exact": u_exact, "u_approx": u_approx, "u_abs_err": u_err,
            })
            pathwise_rows.append({
                "label": label, "eps": eps, "order": m,
                "X_sup": X_sup, "X_L2": X_L2, "X_mean": X_mean,
            })

    scalar_df = pd.DataFrame(scalar_rows)
    pathwise_df = pd.DataFrame(pathwise_rows)

    # Convergence rates
    for m in range(1, MAX_ORDER + 1):
        sub = scalar_df[scalar_df["order"] == m]
        if len(sub) >= 2:
            rate = convergence_rate(sub["eps"].values, sub["u_abs_err"].values)
            print(f"  [{label}] Scalar u order-{m} rate: {rate:.3f} (expected ~{m+1})")
        sub_pw = pathwise_df[pathwise_df["order"] == m]
        if len(sub_pw) >= 2:
            rate_X = convergence_rate(sub_pw["eps"].values, sub_pw["X_sup"].values)
            print(f"  [{label}] Pathwise X sup order-{m} rate: {rate_X:.3f} (expected ~{m+1})")

    return {
        "label": label,
        "y0": y0, "y_coeffs": y_coeffs, "u_coeffs": u_coeffs,
        "low_order_checks": {
            "u1_diff": u1_diff, "u2_diff": u2_diff,
            "u2_leq_0": bool(u2_rec <= 0),
            "X1_max_diff": X1_diff, "X2_max_diff": X2_diff,
        },
        "scalar_df": scalar_df,
        "pathwise_df": pathwise_df,
    }


def run() -> dict:
    """Main entry point for Experiment 3."""
    print("=" * 60)
    print("Experiment 3: Primal Value and Optimizer Expansion Validation")
    print("=" * 60)

    eps_seq = np.array([0.3, 0.2, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02, 0.01, 0.005])

    # x must be < e^{-rT}*sum(a) (the supremum of g0 over y>0).
    # Config1: e^{-0.02}*1.0 ≈ 0.9802  → use x=0.80
    # Config2: e^{-0.06}*1.0 ≈ 0.9418  → use x=0.75
    configs = [
        dict(
            label="Config1",
            x=0.80, T=1.0, r=0.02, mu=0.08, sigma=0.20,
            z=[1.0, 2.0, 3.0], a=[0.5, 0.3, 0.2], b=[0.05, -0.02, 0.01], y_star=1.0,
        ),
        dict(
            label="Config2",
            x=0.75, T=2.0, r=0.03, mu=0.10, sigma=0.25,
            z=[0.5, 1.5, 4.0], a=[0.4, 0.4, 0.2], b=[0.03, -0.01, 0.005], y_star=0.5,
        ),
    ]

    all_results = []
    all_scalar_dfs = []
    all_pathwise_dfs = []

    for cfg in configs:
        params = BSMParams(x=cfg["x"], T=cfg["T"], r=cfg["r"],
                           mu=cfg["mu"], sigma=cfg["sigma"])
        bm = AtomicBernsteinSetup(
            z=cfg["z"], a=cfg["a"], b=cfg["b"], y_star=cfg["y_star"]
        )
        admissible_eps = np.array([e for e in eps_seq if bm.check_positivity([e])])
        result = run_single_config(params, bm, admissible_eps, label=cfg["label"])
        all_results.append(result)
        all_scalar_dfs.append(result["scalar_df"])
        all_pathwise_dfs.append(result["pathwise_df"])

    scalar_df = pd.concat(all_scalar_dfs, ignore_index=True)
    pathwise_df = pd.concat(all_pathwise_dfs, ignore_index=True)

    # Low-order check summary
    check_rows = [
        {"label": r["label"], **r["low_order_checks"]}
        for r in all_results
    ]
    check_df = pd.DataFrame(check_rows)
    print("\nLow-Order Identity Checks:")
    print(check_df.to_string(index=False))

    # Save
    scalar_df.to_csv(RESULTS_DIR / "exp3_scalar_errors.csv", index=False)
    pathwise_df.to_csv(RESULTS_DIR / "exp3_pathwise_errors.csv", index=False)
    check_df.to_csv(RESULTS_DIR / "exp3_low_order_checks.csv", index=False)

    _plot_primal_convergence(scalar_df, RESULTS_DIR / "exp3_scalar_convergence.png")
    _plot_pathwise_convergence(pathwise_df, RESULTS_DIR / "exp3_pathwise_convergence.png")

    print(f"\nResults saved to {RESULTS_DIR}/")
    return {
        "results": all_results,
        "scalar_df": scalar_df,
        "pathwise_df": pathwise_df,
    }


def _plot_primal_convergence(df: pd.DataFrame, save_path: Path) -> None:
    labels = df["label"].unique()
    fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 4), squeeze=False)
    for ax, lbl in zip(axes[0], labels):
        sub = df[df["label"] == lbl]
        for m in sorted(sub["order"].unique()):
            grp = sub[sub["order"] == m].sort_values("eps")
            mask = grp["u_abs_err"] > 0
            if mask.sum() < 2:
                continue
            ax.loglog(grp["eps"][mask], grp["u_abs_err"][mask], "o-",
                      label=f"u order {m}")
            eps_ref = grp["eps"][mask].values
            c = grp["u_abs_err"][mask].values[-1] / eps_ref[-1] ** (m + 1)
            ax.loglog(eps_ref, c * eps_ref ** (m + 1), "--", alpha=0.4,
                      label=f"O(ε^{m+1})")
        ax.set_xlabel("ε")
        ax.set_ylabel("|u_exact - u_approx|")
        ax.set_title(f"Exp3 Scalar Primal Errors\n{lbl}")
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def _plot_pathwise_convergence(df: pd.DataFrame, save_path: Path) -> None:
    labels = df["label"].unique()
    fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 4), squeeze=False)
    for ax, lbl in zip(axes[0], labels):
        sub = df[df["label"] == lbl]
        for m in sorted(sub["order"].unique()):
            grp = sub[sub["order"] == m].sort_values("eps")
            mask = grp["X_sup"] > 0
            if mask.sum() < 2:
                continue
            ax.loglog(grp["eps"][mask], grp["X_sup"][mask], "s-",
                      label=f"X sup order {m}")
            eps_ref = grp["eps"][mask].values
            c = grp["X_sup"][mask].values[-1] / eps_ref[-1] ** (m + 1)
            ax.loglog(eps_ref, c * eps_ref ** (m + 1), "--", alpha=0.4,
                      label=f"O(ε^{m+1})")
        ax.set_xlabel("ε")
        ax.set_ylabel("Pathwise sup |X_exact - X_approx|")
        ax.set_title(f"Exp3 Pathwise Optimizer Errors\n{lbl}")
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


if __name__ == "__main__":
    run()
