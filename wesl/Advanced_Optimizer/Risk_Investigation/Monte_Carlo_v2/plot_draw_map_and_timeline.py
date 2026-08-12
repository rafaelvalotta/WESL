"""One high-tier draw, two separate plots (never combined in one axes):

  1. Map -- self highlighted, each new speculative farm its own color, numbered by
     arrival order (1 = first to arrive, ... N = last).
  2. Timeline -- the same numbered arrivals, plotted against real calendar years, one
     row, no shapes -- just where the exponential process actually put them in time.

Geometry is regenerated locally from the same (tier, seed) already run in production --
instrumentation.py deliberately does not persist farm shapes (see its docstring), so this
is the one-time local re-run needed to get something plottable.

    conda run -n Wind_2200 python plot_draw_map_and_timeline.py --tier high --seed 0
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

import settings
import layout

START_YEAR = 2026  # calendar year self's horizon (arrival_year=0) starts at
SELF_COLOR = "#F2B705"
CLUSTER_COLOR = "#B0B0B0"
FARM_COLORS = [plt.cm.tab20(i) for i in range(20)]


def to_km(shape_or_arr):
    if hasattr(shape_or_arr, "exterior"):
        x, y = shape_or_arr.exterior.xy
        return np.array(x) / 1000, np.array(y) / 1000
    arr = np.asarray(shape_or_arr)
    return arr[:, 0] / 1000, arr[:, 1] / 1000


def plot_map(cluster_farms, new_farms, out_path):
    ordered = sorted(new_farms, key=lambda f: f["arrival_year"])
    fig, ax = plt.subplots(figsize=(9, 9))

    for name, poly in layout.LEASE_TEMPLATES.items():
        if name == settings.FOCUS_FARM:
            continue
        x, y = to_km(poly)
        ax.fill(x, y, color=CLUSTER_COLOR, alpha=0.25, edgecolor=CLUSTER_COLOR, linewidth=0.8)

    self_x, self_y = to_km(layout.LEASE_TEMPLATES[settings.FOCUS_FARM])
    ax.fill(self_x, self_y, color=SELF_COLOR, alpha=0.85, edgecolor="black", linewidth=1.5, label="Self (South Coast Wind)")

    for i, farm in enumerate(ordered):
        color = FARM_COLORS[i % len(FARM_COLORS)]
        x, y = to_km(farm["shape"])
        ax.fill(x, y, color=color, alpha=0.75, edgecolor="black", linewidth=0.6)
        cx, cy = farm["shape"].centroid.x / 1000, farm["shape"].centroid.y / 1000
        ax.annotate(str(i + 1), (cx, cy), ha="center", va="center", fontsize=10, fontweight="bold")

    ax.fill([], [], color=FARM_COLORS[0], alpha=0.75, edgecolor="black", label="New farm (numbered by arrival order)")
    ax.fill([], [], color=CLUSTER_COLOR, alpha=0.25, label="Existing cluster")

    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_title("High tier -- one draw")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    print(f"Saved: {out_path}")


def plot_timeline(new_farms, out_path):
    ordered = sorted(new_farms, key=lambda f: f["arrival_year"])
    years = [START_YEAR + f["arrival_year"] for f in ordered]

    # Arrivals close enough together to collide in the label alternate stem height,
    # so back-to-back numbers stay readable instead of merging into one blob.
    gap_threshold = settings.HORIZON_YEARS * 0.018
    levels, level = [], 0
    for i, yr in enumerate(years):
        level = 1 - level if (i > 0 and yr - years[i - 1] < gap_threshold) else 0
        levels.append(level)

    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.hlines(0, START_YEAR, START_YEAR + settings.HORIZON_YEARS, color="black", linewidth=1)
    for i, (yr, level) in enumerate(zip(years, levels)):
        color = FARM_COLORS[i % len(FARM_COLORS)]
        top = 1.0 + 0.4 * level
        ax.vlines(yr, 0, top, color=color, linewidth=2)
        ax.plot(yr, top, "o", color=color, markersize=9, markeredgecolor="black")
        ax.annotate(str(i + 1), (yr, top + 0.15), ha="center", fontsize=9, fontweight="bold")

    ax.set_xlim(START_YEAR, START_YEAR + settings.HORIZON_YEARS)
    ax.set_xticks(range(START_YEAR, int(START_YEAR + settings.HORIZON_YEARS) + 1))
    ax.set_xticklabels(range(START_YEAR, int(START_YEAR + settings.HORIZON_YEARS) + 1), rotation=45, ha="right", fontsize=8)
    ax.set_ylim(-0.3, 2.0)
    ax.set_yticks([])
    ax.set_title("High tier -- same draw -- arrival timeline")
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    print(f"Saved: {out_path}")


def _expected_gap_curve(n, lam):
    """Expected order statistics of n iid Exponential(lambda) draws -- closed form
    E[X_(k)] = (1/lambda) * sum_{j=n-k+1}^{n} 1/j -- ascending, then reversed to match
    "largest to smallest" bar order."""
    ascending, running = [], 0.0
    for k in range(1, n + 1):
        running += (1 / lam) / (n - k + 1)
        ascending.append(running)
    return ascending[::-1]


TIER_PACE_LABEL = {"low": "every 2 years", "medium": "every 1 year", "high": "every 6 months"}
TIER_TITLE = {"low": "Low scenario", "medium": "Medium scenario", "high": "High scenario"}


def plot_gap_comparison(gaps_by_tier, lambdas, out_path):
    """One panel per tier, same y-axis, so the three growth scenarios read side by side.
    Plain-language labels throughout -- no 'lambda' or 'exponential' on screen."""
    tiers = ["low", "medium", "high"]
    y_max = max(max(g) for g in gaps_by_tier.values()) * 1.1
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)

    for ax, tier in zip(axes, tiers):
        gaps = sorted(gaps_by_tier[tier], reverse=True)
        n = len(gaps)
        expected = _expected_gap_curve(n, lambdas[tier])
        ax.bar(range(1, n + 1), gaps, color="#1B6E86", alpha=0.85, edgecolor="white", label="Actual gaps in this draw")
        ax.plot(range(1, n + 1), expected, color="#8B1E3F", linewidth=2.5, marker="o", markersize=5,
                label="Typical pattern for this pace")
        ax.set_title(f"{TIER_TITLE[tier]}\n(average: a new farm {TIER_PACE_LABEL[tier]})", fontsize=11)
        ax.set_xlabel("Gaps in this draw, longest to shortest")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Years until the next farm")
    axes[0].set_ylim(0, y_max)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.06), fontsize=10)
    fig.suptitle("Time between one new farm and the next", y=1.14, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", default="high")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    cluster_x, cluster_y, cluster_farms, new_farms = layout.populate_scenario(args.tier, rng)

    fig_dir = _THIS_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)
    plot_map(cluster_farms, new_farms, fig_dir / f"draw_map_{args.tier}_seed{args.seed}.png")
    plot_timeline(new_farms, fig_dir / f"draw_timeline_{args.tier}_seed{args.seed}.png")

    gaps_by_tier = {}
    for tier in ["low", "medium", "high"]:
        rng_t = np.random.default_rng(args.seed)
        _, _, _, farms_t = layout.populate_scenario(tier, rng_t)
        years_t = sorted(f["arrival_year"] for f in farms_t)
        gaps_by_tier[tier] = np.diff([0.0] + years_t)
    plot_gap_comparison(gaps_by_tier, settings.TIER_LAMBDA, fig_dir / f"draw_gaps_by_tier_seed{args.seed}.png")


if __name__ == "__main__":
    main()
