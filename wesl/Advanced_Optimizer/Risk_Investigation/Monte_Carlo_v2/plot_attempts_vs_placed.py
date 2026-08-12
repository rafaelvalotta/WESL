"""Tests a specific hypothesis: the raw Poisson ATTEMPTS should hug the theoretical
Exponential(lambda) curve tightly (they ARE unfiltered draws from it), while the
PLACED farms (attempts that survived the geography/eligibility filter) may drift away
from that clean shape, since which attempts fail depends on how crowded the site
already is -- not a fixed, memoryless probability.

Both series come from the SAME populate_scenario() calls (log_rows carries every
attempt, successful or not), so attempts and placed are directly comparable, not two
separately-drawn samples.

    conda run -n Wind_2200 python plot_attempts_vs_placed.py
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import settings
import layout

N_DRAWS = 30
TIERS = ["low", "medium", "high"]
TITLE = {"low": "Low scenario", "medium": "Medium scenario", "high": "High scenario"}
COLOR = {"low": "#1B6E86", "medium": "#C17A1F", "high": "#8B1E3F"}


def expected_curve(n, lam):
    ascending, running = [], 0.0
    for k in range(1, n + 1):
        running += (1 / lam) / (n - k + 1)
        ascending.append(running)
    return ascending[::-1]


def collect_gaps(tier, n_draws):
    attempt_gaps, placed_gaps = [], []
    for seed in range(n_draws):
        rng = np.random.default_rng(seed)
        log = []
        _, _, _, new_farms = layout.populate_scenario(tier, rng, log_rows=log)

        attempt_years = sorted(row["arrival_year"] for row in log)
        attempt_gaps.extend(np.diff([0.0] + attempt_years))

        placed_years = sorted(f["arrival_year"] for f in new_farms)
        placed_gaps.extend(np.diff([0.0] + placed_years))
    return np.array(attempt_gaps), np.array(placed_gaps)


def plot_panel(ax, gaps, lam, color, title, lam_effective=None):
    gaps = sorted(gaps, reverse=True)
    n = len(gaps)
    curve = expected_curve(n, lam)
    ax.bar(range(1, n + 1), gaps, color=color, alpha=0.5, edgecolor="none", width=1.0)
    label = f"Same rate as attempts (1/{1/lam:.1f} yr)"
    ax.plot(range(1, n + 1), curve, color="black", linewidth=1.6, linestyle="--", label=label)
    if lam_effective is not None:
        curve_eff = expected_curve(n, lam_effective)
        ax.plot(range(1, n + 1), curve_eff, color="black", linewidth=1.8,
                 label=f"Rate fit to placed farms (1/{1/lam_effective:.1f} yr)")
        ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey="row")

    for col, tier in enumerate(TIERS):
        lam = settings.TIER_LAMBDA[tier]
        attempt_gaps, placed_gaps = collect_gaps(tier, N_DRAWS)
        lam_effective = 1 / placed_gaps.mean()

        plot_panel(axes[0, col], attempt_gaps, lam, COLOR[tier], TITLE[tier])
        plot_panel(axes[1, col], placed_gaps, lam, COLOR[tier], "", lam_effective=lam_effective)

    axes[0, 0].set_ylabel("Attempts\nyears between one try and the next")
    axes[1, 0].set_ylabel("Placed farms\nyears between one farm and the next")
    for ax in axes[1, :]:
        ax.set_xlabel("Gaps, longest to shortest")

    fig.suptitle("Attempts vs. placed farms -- gap pattern, 30 draws pooled per scenario", fontsize=13)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "figures" / "attempts_vs_placed_gaps.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
