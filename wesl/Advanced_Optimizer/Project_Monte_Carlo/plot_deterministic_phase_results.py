"""
Risk Research -- Deterministic Phase: risk-distribution outputs
(methodology Section 6).

Reads results/deterministic_phase_216_combinations.csv (produced by
deterministic_phase.py) and produces 4 separate figures:
  - deterministic_phase_risk_histogram.png    : loss_pct distribution, P10/P50/P90
  - deterministic_phase_by_wake_model.png     : grouped by wake model (epistemic)
  - deterministic_phase_by_southcoast_turbine.png : grouped by neighbor turbine
  - deterministic_phase_by_climate_year.png   : grouped by climate year
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_histogram(df):
    p10, p50, p90 = np.percentile(df["loss_pct"], [10, 50, 90])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(df["loss_pct"], bins=30, color="steelblue", edgecolor="white")
    for p, label in [(p10, "P10"), (p50, "P50"), (p90, "P90")]:
        ax.axvline(p, color="crimson", linestyle="--", linewidth=1.5)
        ax.text(p, ax.get_ylim()[1] * 0.95, f"{label}\n{p:.2f}%", color="crimson", ha="center", fontsize=9)
    ax.set_xlabel("Vineyard AEP loss to neighbor wakes (%)")
    ax.set_ylabel("Count (out of 216 combinations)")
    ax.set_title("Risk Research -- Deterministic Phase\nRisk distribution of Vineyard AEP loss")
    fig.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "deterministic_phase_risk_histogram.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_by_wake_model(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    models = sorted(df["wake_model"].unique())
    ax.boxplot([df[df["wake_model"] == m]["loss_pct"] for m in models], tick_labels=models,
               patch_artist=True, boxprops=dict(facecolor="lightsteelblue", alpha=0.7))
    ax.set_ylabel("loss_pct (%)")
    ax.set_title("Risk Research -- Deterministic Phase\nGrouped by wake model (epistemic uncertainty)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "deterministic_phase_by_wake_model.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_by_southcoast_turbine(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    turbines = sorted(df["southcoast_turbine"].unique())
    ax.boxplot([df[df["southcoast_turbine"] == t]["loss_pct"] for t in turbines], tick_labels=turbines,
               patch_artist=True, boxprops=dict(facecolor="lightsteelblue", alpha=0.7))
    ax.set_ylabel("loss_pct (%)")
    ax.set_title("Risk Research -- Deterministic Phase\nGrouped by SouthCoast turbine (neighbor design uncertainty)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "deterministic_phase_by_southcoast_turbine.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_by_climate_year(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    years = sorted(df["year"].unique())
    ax.boxplot([df[df["year"] == y]["loss_pct"] for y in years], tick_labels=years,
               patch_artist=True, boxprops=dict(facecolor="lightsteelblue", alpha=0.7))
    ax.set_ylabel("loss_pct (%)")
    ax.set_title("Risk Research -- Deterministic Phase\nGrouped by climate year (interannual variability)")
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    fig.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "deterministic_phase_by_climate_year.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(RESULTS_DIR, "deterministic_phase_216_combinations.csv"))
    plot_histogram(df)
    plot_by_wake_model(df)
    plot_by_southcoast_turbine(df)
    plot_by_climate_year(df)
