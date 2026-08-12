"""A simple, illustrative curve (not the noisy simulation output): the gap between one
new farm and the next shrinking smoothly over the 25 years, and the resulting farm count
building up. Anchored on two real numbers already in hand: the average number of farms
each tier actually delivered in the production run, and the fixed 25-year horizon.

    conda run -n Wind_2200 python plot_expected_buildout.py
"""
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt

T = 25.0
START_YEAR = 2026
N_AVG = {"low": 4.4, "medium": 8.23, "high": 15.63}  # measured average farms placed, 150 trials/tier
TITLE = {"low": "Low scenario", "medium": "Medium scenario", "high": "High scenario"}
COLOR = {"low": "#1B6E86", "medium": "#C17A1F", "high": "#8B1E3F"}


def solve_tau(m, n_avg):
    f = lambda tau: (tau / (2 * m)) * np.expm1(T / tau) - n_avg
    return brentq(f, 0.5, 200)


def main():
    fig_dir = Path(__file__).resolve().parent / "figures"
    fig_dir.mkdir(exist_ok=True)

    for tier, n_avg in N_AVG.items():
        m = T / n_avg
        tau = solve_tau(m, n_avg)
        g0 = 2 * m

        t = np.linspace(0.001, T, 400)
        gap_t = g0 * np.exp(-t / tau)
        count_t = (tau / (2 * m)) * np.expm1(t / tau)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        axes[0].plot(START_YEAR + t, gap_t, color=COLOR[tier], linewidth=2.5)
        axes[0].set_title("Gap over time")
        axes[0].set_xlabel("Year")
        axes[0].set_ylabel("Years between one farm and the next")
        axes[0].set_ylim(0, g0 * 1.1)

        axes[1].plot(START_YEAR + t, count_t, color=COLOR[tier], linewidth=2.5)
        axes[1].set_title("Number of farms built over time")
        axes[1].set_xlabel("Year")
        axes[1].set_ylabel("Farms built")
        axes[1].set_ylim(0, n_avg * 1.1)

        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlim(START_YEAR, START_YEAR + T)
        fig.suptitle(f"{TITLE[tier]} -- {n_avg:.1f} farms on average by {int(START_YEAR + T)}", fontsize=13)
        fig.tight_layout()
        out = fig_dir / f"expected_gap_and_buildout_{tier}.png"
        fig.savefig(out, dpi=170, bbox_inches="tight")
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
