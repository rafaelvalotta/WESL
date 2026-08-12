"""AEP-free look at the time axis: runs layout.populate_scenario for a handful of seeds
per tier (layout generation only -- no climate, no PyWake) and plots (1) cumulative
new-farm count vs. year and (2) cumulative added MW vs. year, both per tier over self's
horizon. Answers "what does the arrival process look like" without paying for a single
AEP call -- useful before committing to a real (expensive) campaign.

    conda run -n Wind_2200 python plot_timeline_demo.py --n-draws 15
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

# Same qualitative palette already used elsewhere in this project (wind_resource_emulator.ipynb,
# analyze_campaign.py) -- 3 distinct hues, not a light->dark ramp of one color, so the 3
# tiers stay visually separable even with many overlapping draws.
TIER_COLOR = {"low": "#1B6E86", "medium": "#C17A1F", "high": "#8B1E3F"}


def one_draw(tier, seed):
    """Returns (arrival_years, capacities) for one draw -- arrival_years are the RAW
    Poisson event times, not yet turned into a cumulative curve."""
    rng = np.random.default_rng(seed)
    _, _, _, new_farms = layout.populate_scenario(tier, rng)
    ordered = sorted(new_farms, key=lambda f: f["arrival_year"])
    return [f["arrival_year"] for f in ordered], [f["capacity_mw"] for f in ordered]


def cumulative_curve(arrival_years, capacities, value_per_arrival):
    years = [0.0] + list(arrival_years) + [settings.HORIZON_YEARS]
    cum = np.concatenate([[0.0], np.cumsum(value_per_arrival)])
    cum = np.append(cum, cum[-1])
    return np.array(years), cum[:len(years)]


def plot_timelines(n_draws, seed_start, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    all_gaps = {tier: [] for tier in settings.TIER_LAMBDA}

    for tier in settings.TIER_LAMBDA:
        for j in range(n_draws):
            arrival_years, capacities = one_draw(tier, seed_start + j)
            label = tier.capitalize() if j == 0 else None

            years, cum_count = cumulative_curve(arrival_years, capacities, [1] * len(arrival_years))
            _, cum_mw = cumulative_curve(arrival_years, capacities, capacities)
            axes[0].step(years, cum_count, where="post", color=TIER_COLOR[tier], alpha=0.6, label=label)
            axes[1].step(years, cum_mw, where="post", color=TIER_COLOR[tier], alpha=0.6, label=label)

            # The exponential shape lives HERE -- in the gaps between consecutive
            # arrivals -- not in the cumulative curves above (those are ~linear on
            # average for a Poisson process: E[count(t)] = lambda*t).
            gaps = np.diff([0.0] + list(arrival_years))
            all_gaps[tier].extend(gaps)

    axes[0].set_xlabel("Year"); axes[0].set_ylabel("Cumulative new farms")
    axes[0].set_title("New-farm arrivals over self's horizon")
    axes[1].set_xlabel("Year"); axes[1].set_ylabel("Cumulative added capacity (MW)")
    axes[1].set_title("Added capacity over self's horizon")

    # Panel 3: pooled inter-arrival gaps per tier, as a density histogram, with the
    # theoretical Exponential(lambda) PDF overlaid -- this is where "exponential" is
    # actually visible.
    max_gap = max(max(g) for g in all_gaps.values() if len(g))
    x = np.linspace(0, max_gap, 200)
    for tier, lam in settings.TIER_LAMBDA.items():
        gaps = all_gaps[tier]
        axes[2].hist(gaps, bins=30, density=True, color=TIER_COLOR[tier], alpha=0.35,
                     label=f"{tier.capitalize()} (n={len(gaps)})")
        axes[2].plot(x, lam * np.exp(-lam * x), color=TIER_COLOR[tier], linewidth=2)
    axes[2].set_xlabel("Gap between consecutive arrivals (years)")
    axes[2].set_ylabel("Density")
    axes[2].set_title("Inter-arrival gaps -- histogram vs. Exponential(λ) (solid line)")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-draws", type=int, default=15)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--out", type=str, default="figures/timeline_demo.png")
    args = parser.parse_args()
    out_path = _THIS_DIR / args.out
    out_path.parent.mkdir(exist_ok=True)
    plot_timelines(args.n_draws, args.seed_start, out_path)


if __name__ == "__main__":
    main()
