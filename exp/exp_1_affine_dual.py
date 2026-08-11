"""
Experiment 1: Validation of Affine Dual Structure Under Affine Bernstein-Measure
Preference Perturbations.

Reproduces Theorem 1: v(y, epsilon) = v0(y) + epsilon * v1(y) exactly.

Methodology (from paper):
  1. BSM complete-market setting with explicit H_T.
  2. Finite atomic Bernstein measures mu^0 and nu.
  3. Dual potential V^eps(y) = sum_i ((e^{-yz_i} - e^{-y*z_i})/z_i)(a_i + eps b_i).
  4. Compute v(y,eps)=E[V^eps(yH_T)], v0(y)=E[V^0(yH_T)], v1(y)=E[V^1(yH_T)].
  5. Verify affine identity residual R = v - v0 - eps*v1 ≈ 0.
  6. Verify derivative identity -v_y ≈ g0 + eps*g1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product as iterproduct

from src.model import BSMParams, generate_HT_quadrature
from src.bernstein import AtomicBernsteinSetup
from src.dual import (
    v_eps, v0_fn, v1_fn, g0_fn, g1_fn, g0_deriv
)
from src.checks import affine_residual

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_config(
    params: BSMParams,
    bm: AtomicBernsteinSetup,
    y_grid: np.ndarray,
    eps_grid: np.ndarray,
    n_quad: int = 200,
    label: str = "",
) -> pd.DataFrame:
    """
    Run exp_1 validation for one parameter configuration.

    Returns a DataFrame with columns:
        y, eps, v_full, v0, v1, R_aff, neg_vy_fd, g0, g1, R_deriv
    """
    assert bm.check_positivity(eps_grid), (
        "mu^eps is not nonneg for some eps in eps_grid! Adjust parameters."
    )

    H_T, weights = generate_HT_quadrature(params, n_quad)

    rows = []
    h_fd = 1e-6  # finite-difference step for derivative check

    for y, eps in iterproduct(y_grid, eps_grid):
        # Affine identity check
        v_full_val = v_eps(y, eps, bm, H_T, weights)
        v0_val = v0_fn(y, bm, H_T, weights)
        v1_val = v1_fn(y, bm, H_T, weights)
        R_aff = affine_residual(v_full_val, v0_val, eps * v1_val + v0_val - v0_val, 0.0)
        # Correct formula: R_aff = v(y,eps) - v0(y) - eps*v1(y)
        R_aff = v_full_val - v0_val - eps * v1_val

        # Derivative identity: -dv/dy(y,eps) = g0(y) + eps*g1(y)
        # Finite-difference approximation
        v_plus = v_eps(y + h_fd, eps, bm, H_T, weights)
        v_minus = v_eps(y - h_fd, eps, bm, H_T, weights)
        neg_vy_fd = -(v_plus - v_minus) / (2 * h_fd)

        g0_val = g0_fn(y, bm, H_T, weights)
        g1_val = g1_fn(y, bm, H_T, weights)
        R_deriv = neg_vy_fd - (g0_val + eps * g1_val)

        rows.append({
            "label": label,
            "y": y,
            "eps": eps,
            "v_full": v_full_val,
            "v0": v0_val,
            "v1": v1_val,
            "R_aff": R_aff,
            "neg_vy_fd": neg_vy_fd,
            "g0": g0_val,
            "g1": g1_val,
            "R_deriv": R_deriv,
        })

    return pd.DataFrame(rows)


def run_multiple_configs() -> pd.DataFrame:
    """
    Run exp_1 across multiple BSM parameters and Bernstein measure specifications.
    """
    all_results = []

    # Configuration 1: baseline parameters
    configs = [
        dict(
            label="Config1",
            x=1.0, T=1.0, r=0.02, mu=0.08, sigma=0.20,
            z=[1.0, 2.0, 3.0],
            a=[0.5, 0.3, 0.2],
            b=[0.1, -0.05, 0.02],
            y_star=1.0,
        ),
        dict(
            label="Config2",
            x=2.0, T=2.0, r=0.03, mu=0.10, sigma=0.25,
            z=[0.5, 1.5, 4.0],
            a=[0.4, 0.4, 0.2],
            b=[0.05, -0.03, 0.01],
            y_star=0.5,
        ),
        dict(
            label="Config3",
            x=1.0, T=0.5, r=0.0, mu=0.12, sigma=0.30,
            z=[2.0, 5.0],
            a=[0.6, 0.4],
            b=[-0.1, 0.08],
            y_star=2.0,
        ),
        dict(
            label="Config4",
            x=1.5, T=3.0, r=0.05, mu=0.15, sigma=0.35,
            z=[0.8, 1.2, 2.5, 5.0],
            a=[0.25, 0.25, 0.25, 0.25],
            b=[0.02, -0.01, 0.015, -0.025],
            y_star=1.5,
        ),
    ]

    y_grid = np.array([0.2, 0.5, 1.0, 2.0, 5.0])
    eps_grid = np.array([-0.3, -0.1, 0.0, 0.1, 0.3])

    for cfg in configs:
        params = BSMParams(
            x=cfg["x"], T=cfg["T"], r=cfg["r"],
            mu=cfg["mu"], sigma=cfg["sigma"]
        )
        bm = AtomicBernsteinSetup(
            z=cfg["z"], a=cfg["a"], b=cfg["b"], y_star=cfg["y_star"]
        )
        # Restrict eps_grid to admissible values
        admissible_eps = [e for e in eps_grid if bm.check_positivity([e])]
        df = run_config(
            params, bm, y_grid, np.array(admissible_eps), label=cfg["label"]
        )
        all_results.append(df)

    return pd.concat(all_results, ignore_index=True)


def summarize_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute max absolute residuals per configuration."""
    summary = df.groupby("label").agg(
        max_abs_R_aff=("R_aff", lambda x: x.abs().max()),
        max_abs_R_deriv=("R_deriv", lambda x: x.abs().max()),
        mean_abs_R_aff=("R_aff", lambda x: x.abs().mean()),
        mean_abs_R_deriv=("R_deriv", lambda x: x.abs().mean()),
    ).reset_index()
    return summary


def plot_affine_residual(df: pd.DataFrame, save_path: Path) -> None:
    """Plot the affine residual |R_aff| across y and eps for each config."""
    configs = df["label"].unique()
    fig, axes = plt.subplots(1, len(configs), figsize=(5 * len(configs), 4), squeeze=False)
    for ax, cfg_label in zip(axes[0], configs):
        sub = df[df["label"] == cfg_label]
        for eps_val, grp in sub.groupby("eps"):
            ax.plot(grp["y"], np.abs(grp["R_aff"]), "o-", label=f"eps={eps_val:.2g}")
        ax.set_yscale("log")
        ax.set_xlabel("y")
        ax.set_ylabel("|Affine Residual|")
        ax.set_title(f"Exp1 Affine Residual\n{cfg_label}")
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def plot_derivative_residual(df: pd.DataFrame, save_path: Path) -> None:
    """Plot the derivative residual |R_deriv| across y and eps."""
    configs = df["label"].unique()
    fig, axes = plt.subplots(1, len(configs), figsize=(5 * len(configs), 4), squeeze=False)
    for ax, cfg_label in zip(axes[0], configs):
        sub = df[df["label"] == cfg_label]
        for eps_val, grp in sub.groupby("eps"):
            ax.plot(grp["y"], np.abs(grp["R_deriv"]), "s--", label=f"eps={eps_val:.2g}")
        ax.set_yscale("log")
        ax.set_xlabel("y")
        ax.set_ylabel("|Derivative Residual|")
        ax.set_title(f"Exp1 Derivative Residual\n{cfg_label}")
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def run() -> None:
    """Main entry point for Experiment 1."""
    print("=" * 60)
    print("Experiment 1: Affine Dual Structure Validation")
    print("=" * 60)

    df = run_multiple_configs()
    summary = summarize_residuals(df)

    print("\nAffine Identity Residual Summary:")
    print(summary.to_string(index=False))

    # Verify theorem: residuals should be near machine precision
    max_aff = summary["max_abs_R_aff"].max()
    max_deriv = summary["max_abs_R_deriv"].max()
    print(f"\nMax |R_aff|   across all configs: {max_aff:.3e}")
    print(f"Max |R_deriv| across all configs: {max_deriv:.3e}")
    if max_aff < 1e-10:
        print("✓ Affine identity confirmed at machine precision")
    else:
        print(f"✗ Affine residual {max_aff:.3e} exceeds threshold (check quadrature accuracy)")

    # Save results
    df.to_csv(RESULTS_DIR / "exp1_full_results.csv", index=False)
    summary.to_csv(RESULTS_DIR / "exp1_summary.csv", index=False)

    plot_affine_residual(df, RESULTS_DIR / "exp1_affine_residual.png")
    plot_derivative_residual(df, RESULTS_DIR / "exp1_derivative_residual.png")

    print(f"\nResults saved to {RESULTS_DIR}/")
    return df, summary


if __name__ == "__main__":
    run()
