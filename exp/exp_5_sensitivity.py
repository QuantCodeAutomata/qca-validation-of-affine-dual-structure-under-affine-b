"""
Experiment 5: First-Order Sensitivity Validation for Simultaneous Drift
and Preference Perturbations in Black-Scholes-Merton.

Reproduces Section 5: two-parameter first-order expansion.

Methodology (from paper):
  1. Baseline BSM: H_T = e^{-rT} exp(-theta W_T - 0.5 theta^2 T).
  2. Perturbed SPD: H_T^n = e^{-rT} exp(-theta_n W_T - 0.5 theta_n^2 T).
  3. H_mu = -(1/sigma) * H_T * (W_T + theta*T) — drift derivative.
  4. Preference perturbation I^n = I^0 + eps_n * I^1.
  5. First-order sensitivities y_I, y_mu; optimizer increments X_I, X_mu.
  6. Normalized residuals -> 0 as (delta_n, eps_n) -> 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import brentq

from src.model import BSMParams, generate_HT_monte_carlo
from src.bernstein import AtomicBernsteinSetup
from src.checks import normalized_multiplier_residual, normalized_optimizer_residual

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_MC = 50_000


def compute_expectation(f_vals: np.ndarray) -> float:
    """Monte Carlo expectation (equal weights)."""
    return float(np.mean(f_vals))


def solve_perturbed_budget_general(
    x: float,
    H_Tn: np.ndarray,
    eps_n: float,
    bm: AtomicBernsteinSetup,
    bracket: tuple = (1e-14, 1e6),
) -> float:
    """
    Solve E[H_T^n * (I^0(y H_T^n) + eps_n * I^1(y H_T^n))] = x for y.
    """
    def f(y: float) -> float:
        arg = y * H_Tn
        integrand = H_Tn * (bm.I0(arg) + eps_n * bm.J(arg))
        return compute_expectation(integrand) - x

    fa, fb = f(bracket[0]), f(bracket[1])
    if fa * fb >= 0:
        for lo, hi in [(1e-12, 1e-4), (1e-4, 1e2), (1e2, 1e8)]:
            fa2, fb2 = f(lo), f(hi)
            if fa2 * fb2 < 0:
                bracket = (lo, hi)
                fa, fb = fa2, fb2
                break

    assert fa * fb < 0, (
        f"Cannot bracket root for eps_n={eps_n:.3g}: "
        f"f({bracket[0]:.2e})={fa:.4g}, f({bracket[1]:.2e})={fb:.4g}"
    )
    return float(brentq(f, bracket[0], bracket[1], xtol=1e-12, rtol=1e-12))


def compute_sensitivity_coefficients(
    y: float,
    H_T: np.ndarray,
    W_T: np.ndarray,
    theta: float,
    sigma: float,
    T: float,
    bm: AtomicBernsteinSetup,
) -> dict:
    """
    Compute first-order sensitivity coefficients:
        y_I = -E[H_T I^1(y H_T)] / E[H_T^2 (I^0)'(y H_T)]
        y_mu = -E[H_mu (I^0(y H_T) + y H_T (I^0)'(y H_T))] / E[H_T^2 (I^0)'(y H_T)]
        H_mu = -(1/sigma) H_T (W_T + theta*T)
        X_I = I^1(yH_T) + y_I H_T (I^0)'(yH_T)
        X_mu = (I^0)'(yH_T) (y_mu H_T + y H_mu)
    """
    yHT = y * H_T
    I0_vals = bm.I0(yHT)
    J_vals = bm.J(yHT)
    dI0_vals = bm.dI0_dy(yHT, order=1)

    H_mu = -(1.0 / sigma) * H_T * (W_T + theta * T)
    denom = compute_expectation(H_T ** 2 * dI0_vals)

    assert abs(denom) > 1e-15, f"Denominator near zero: {denom:.3e}"
    assert denom < 0, f"Denominator should be negative (I^0 is decreasing): {denom:.4g}"

    # Preference sensitivity
    numer_I = -compute_expectation(H_T * J_vals)
    y_I = numer_I / denom

    # Drift sensitivity
    numer_mu = -compute_expectation(H_mu * (I0_vals + y * H_T * dI0_vals))
    y_mu = numer_mu / denom

    # Optimizer increments
    X0 = bm.I0(yHT)
    X_I = J_vals + y_I * H_T * dI0_vals
    X_mu = dI0_vals * (y_mu * H_T + y * H_mu)

    return {
        "y_I": y_I, "y_mu": y_mu,
        "X0": X0, "X_I": X_I, "X_mu": X_mu,
        "H_mu": H_mu, "denom": denom,
    }


def verify_H_mu_finite_diff(
    H_T: np.ndarray,
    W_T: np.ndarray,
    params: BSMParams,
    h: float = 1e-5,
) -> dict:
    """Validate H_mu analytically vs finite difference."""
    theta = (params.mu - params.r) / params.sigma
    mu_plus = params.mu + h
    mu_minus = params.mu - h
    theta_plus = (mu_plus - params.r) / params.sigma
    theta_minus = (mu_minus - params.r) / params.sigma

    H_plus = np.exp(-params.r * params.T) * np.exp(-theta_plus * W_T - 0.5 * theta_plus ** 2 * params.T)
    H_minus = np.exp(-params.r * params.T) * np.exp(-theta_minus * W_T - 0.5 * theta_minus ** 2 * params.T)
    H_mu_fd = (H_plus - H_minus) / (2 * h)

    H_mu_analytic = -(1.0 / params.sigma) * H_T * (W_T + theta * params.T)
    max_diff = float(np.max(np.abs(H_mu_fd - H_mu_analytic)))
    rel_max = float(np.max(np.abs((H_mu_fd - H_mu_analytic) / (np.abs(H_mu_analytic) + 1e-15))))
    return {
        "H_mu_max_abs_diff": max_diff,
        "H_mu_max_rel_diff": rel_max,
    }


def run_perturbation_sequence(
    params: BSMParams,
    bm: AtomicBernsteinSetup,
    n_seq: list,
    label: str = "",
) -> pd.DataFrame:
    """
    Run exp_5 over a sequence of perturbation pairs (delta_n, eps_n).

    Each element of n_seq is a dict with keys 'delta', 'eps'.
    """
    theta = (params.mu - params.r) / params.sigma
    x = params.x

    H_T_mc, W_T_mc, _ = generate_HT_monte_carlo(params, n_samples=N_MC)

    # Validate H_mu formula
    h_mu_check = verify_H_mu_finite_diff(H_T_mc, W_T_mc, params)
    print(f"  [{label}] H_mu fd check: max_abs={h_mu_check['H_mu_max_abs_diff']:.3e}, "
          f"max_rel={h_mu_check['H_mu_max_rel_diff']:.3e}")

    # Baseline budget solve
    def budget_baseline(y: float) -> float:
        return compute_expectation(H_T_mc * bm.I0(y * H_T_mc)) - x

    y_base = brentq(budget_baseline, 1e-12, 1e6, xtol=1e-12)
    print(f"  [{label}] y_baseline = {y_base:.8f}")

    # Sensitivity coefficients
    sens = compute_sensitivity_coefficients(y_base, H_T_mc, W_T_mc, theta,
                                            params.sigma, params.T, bm)
    y_I = sens["y_I"]
    y_mu = sens["y_mu"]
    X0 = sens["X0"]
    X_I = sens["X_I"]
    X_mu = sens["X_mu"]
    print(f"  [{label}] y_I = {y_I:.6f}, y_mu = {y_mu:.6f}")

    rows = []
    for pt in n_seq:
        delta_n = pt["delta"]
        eps_n = pt["eps"]
        mu_n = params.mu + delta_n
        theta_n = (mu_n - params.r) / params.sigma

        # Perturbed H_T^n (same W_T)
        H_Tn = np.exp(-params.r * params.T) * \
               np.exp(-theta_n * W_T_mc - 0.5 * theta_n ** 2 * params.T)

        # Check positivity of I^n for eps_n
        if not bm.check_positivity([eps_n]):
            continue

        # Solve exact perturbed budget
        y_n = solve_perturbed_budget_general(x, H_Tn, eps_n, bm)

        # Exact perturbed optimizer
        yHn = y_n * H_Tn
        X_n = bm.I0(yHn) + eps_n * bm.J(yHn)

        # Normalized residuals
        r_y = normalized_multiplier_residual(y_n, y_base, delta_n, eps_n, y_mu, y_I)
        r_X = normalized_optimizer_residual(X_n, X0, X_mu, X_I, delta_n, eps_n)

        total_pert = abs(delta_n) + abs(eps_n)
        rows.append({
            "label": label,
            "regime": pt.get("regime", "unknown"),
            "delta_n": delta_n, "eps_n": eps_n, "total_pert": total_pert,
            "y_n": y_n, "y_approx": y_base + delta_n * y_mu + eps_n * y_I,
            "r_y": r_y,
            "r_X_sup": r_X["sup"], "r_X_mean": r_X["mean"], "r_X_L2": r_X["L2"],
            "H_mu_max_abs_diff": h_mu_check["H_mu_max_abs_diff"],
        })

    return pd.DataFrame(rows)


def build_perturbation_sequences(scale: float = 0.1) -> list:
    """Build three perturbation regimes, tagged by 'regime'."""
    scales = [1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]
    same_order = [{"delta": s * scale, "eps": s * scale, "regime": "same"}
                  for s in scales]
    delta_dom = [{"delta": s * scale, "eps": s * 0.01 * scale, "regime": "delta_dom"}
                 for s in scales]
    eps_dom = [{"delta": s * 0.01 * scale, "eps": s * scale, "regime": "eps_dom"}
               for s in scales]
    return same_order + delta_dom + eps_dom


def run() -> dict:
    """Main entry point for Experiment 5."""
    print("=" * 60)
    print("Experiment 5: Two-Parameter Sensitivity Validation (BSM)")
    print("=" * 60)

    # x must be < e^{-rT}*sum(a) (the supremum of g0 over y>0).
    # Config1: e^{-0.02}*1.0 ≈ 0.9802  → use x=0.80
    # Config2: e^{-0.06}*1.0 ≈ 0.9418  → use x=0.75
    configs = [
        dict(
            label="Config1_CMIM",
            x=0.80, T=1.0, r=0.02, mu=0.08, sigma=0.20,
            z=[1.0, 2.0, 3.0], a=[0.5, 0.3, 0.2], b=[0.05, -0.02, 0.01], y_star=1.0,
        ),
        dict(
            label="Config2_CMIM",
            x=0.75, T=2.0, r=0.03, mu=0.10, sigma=0.25,
            z=[0.5, 1.5, 4.0], a=[0.4, 0.4, 0.2], b=[0.03, -0.01, 0.005], y_star=0.5,
        ),
    ]

    all_dfs = []
    for cfg in configs:
        params = BSMParams(x=cfg["x"], T=cfg["T"], r=cfg["r"],
                           mu=cfg["mu"], sigma=cfg["sigma"])
        bm = AtomicBernsteinSetup(
            z=cfg["z"], a=cfg["a"], b=cfg["b"], y_star=cfg["y_star"]
        )
        pert_seq = build_perturbation_sequences(scale=0.08)
        df = run_perturbation_sequence(params, bm, pert_seq, label=cfg["label"])
        all_dfs.append(df)

    full_df = pd.concat(all_dfs, ignore_index=True)

    # Summary: show convergence per regime for each label
    for lbl in full_df["label"].unique():
        sub = full_df[full_df["label"] == lbl]
        print(f"\n  [{lbl}] Per-regime convergence (r_y as perturbation shrinks):")
        for regime in ["same", "delta_dom", "eps_dom"]:
            rdf = sub[sub["regime"] == regime].sort_values("total_pert")
            if len(rdf) < 2:
                continue
            # Display every-other row for brevity
            disp = rdf[["total_pert", "r_y", "r_X_sup"]].iloc[::2]
            print(f"    regime={regime}:")
            print(disp.to_string(index=False))
            # Log-log slope
            log_tp = np.log(rdf["total_pert"].values)
            log_ry = np.log(rdf["r_y"].values + 1e-20)
            slope = np.polyfit(log_tp, log_ry, 1)[0]
            print(f"    log-log slope r_y: {slope:.2f} (expect ≈1, confirms r_y→0)")

    full_df.to_csv(RESULTS_DIR / "exp5_sensitivity_results.csv", index=False)
    _plot_sensitivity_convergence(full_df, RESULTS_DIR / "exp5_normalized_residuals.png")

    print(f"\nResults saved to {RESULTS_DIR}/")
    return {"full_df": full_df}


def _plot_sensitivity_convergence(df: pd.DataFrame, save_path: Path) -> None:
    labels = df["label"].unique()
    regimes = ["same", "delta_dom", "eps_dom"]
    colors = {"same": "steelblue", "delta_dom": "tomato", "eps_dom": "seagreen"}
    markers = {"same": "o", "delta_dom": "s", "eps_dom": "^"}

    fig, axes = plt.subplots(len(labels), 2, figsize=(11, 4 * len(labels)), squeeze=False)

    for row_i, lbl in enumerate(labels):
        sub = df[df["label"] == lbl]

        ax0 = axes[row_i][0]
        ax1 = axes[row_i][1]

        for reg in regimes:
            rdf = sub[sub["regime"] == reg].sort_values("total_pert")
            if rdf.empty:
                continue
            tp = rdf["total_pert"].values
            ax0.loglog(tp, rdf["r_y"].values, f"{markers[reg]}-",
                       color=colors[reg], label=f"r_y ({reg})")
            ax1.loglog(tp, rdf["r_X_sup"].values, f"{markers[reg]}-",
                       color=colors[reg], label=f"r_X sup ({reg})")

        # Reference slope-1 line
        tp_ref = np.array([sub["total_pert"].min(), sub["total_pert"].max()])
        mid_y = sub["r_y"].median()
        ax0.loglog(tp_ref, mid_y * (tp_ref / tp_ref[0]), "k--", alpha=0.5, label="slope 1")
        ax0.set_xlabel("|δ_n| + |ε_n|")
        ax0.set_ylabel("Normalized |y_n - y_approx| / total_pert")
        ax0.set_title(f"Exp5 Multiplier Residual\n{lbl}")
        ax0.legend(fontsize=7)

        ax1.loglog(tp_ref, mid_y * (tp_ref / tp_ref[0]) * 10, "k--", alpha=0.5, label="slope 1")
        ax1.set_xlabel("|δ_n| + |ε_n|")
        ax1.set_ylabel("Normalized |X_n - X_approx|_sup / total_pert")
        ax1.set_title(f"Exp5 Optimizer Residual\n{lbl}")
        ax1.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


if __name__ == "__main__":
    run()
