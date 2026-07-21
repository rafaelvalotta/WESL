"""
Risk Research -- Deterministic Phase: risk distribution with wake model
held fixed, one at a time.

Reuses results/deterministic_phase_216_combinations.csv (no new PyWake
runs needed) -- filters to the 72 rows (24 years x 3 SouthCoast turbines)
for each wake model separately, to isolate climate + neighbor-design
variability from the much larger wake-model effect that makes the pooled
216-row distribution bimodal (see deterministic_phase_by_wake_model.png).

Output: figures/deterministic_phase_risk_histogram_{model}.png, one per model
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_model_histogram(df, model_name):
    sub = df[df["wake_model"] == model_name]["loss_pct"]
    p10, p50, p90 = np.percentile(sub, [10, 50, 90])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(sub, bins=20, color="steelblue", edgecolor="white")
    for p, label in [(p10, "P10"), (p50, "P50"), (p90, "P90")]:
        ax.axvline(p, color="crimson", linestyle="--", linewidth=1.5)
        ax.text(p, ax.get_ylim()[1] * 0.95, f"{label}\n{p:.3f}%", color="crimson", ha="center", fontsize=9)
    ax.set_xlabel("Vineyard AEP loss to neighbor wakes (%)")
    ax.set_ylabel("Count (out of 72: 24 years x 3 SouthCoast turbines)")
    ax.set_title(f"{model_name} only -- climate + turbine variability")
    fig.tight_layout()

    safe_name = model_name.lower()
    out_path = os.path.join(FIGURES_DIR, f"deterministic_phase_risk_histogram_{safe_name}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[{model_name}] P10={p10:.3f}%  P50={p50:.3f}%  P90={p90:.3f}%  "
          f"mean={sub.mean():.3f}%  std={sub.std():.3f}%  -> {out_path}")


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(RESULTS_DIR, "deterministic_phase_216_combinations.csv"))
    for model_name in df["wake_model"].unique():
        plot_model_histogram(df, model_name)
