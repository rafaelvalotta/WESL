"""Reads a campaign's merged results, writes one clean, organized copy to
Results_HPC/<campaign>/ (combined CSVs + figures).

Deliberately talks in farm counts and added-turbine counts, not MW delivered vs.
requested -- the delivered/requested gap is a placement-engine detail (see
Layout_Generation/layout_generation_v2.ipynb, section 7) still being tuned, not a
scenario-population fact ready to present. "Turbines added" = turbines belonging to
NEW speculative farms only (name prefix "Speculative_", from scenario_bridge.py) --
excludes Vineyard and the rest of the already-existing 13-lease cluster, which the
Monte Carlo does not add, it starts from them.

    python analyze_campaign.py --campaign mc_20260807_0627
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from postprocess_report import merge_if_needed

RESULTS_HPC_DIR = _THIS_DIR.parent / "Results_HPC"

MODEL_COLOR = {"Nygaard_TurboGaussian": "#1B6E86", "SuperGaussian": "#C17A1F"}
LEVEL_ORDER = ["low", "medium", "high"]
LEVEL_COLOR = {"low": "#AED4E0", "medium": "#5B9BB5", "high": "#1B6E86"}  # one hue, light->dark: ordinal

IMPACT_INPUTS = {
    "n_new_farms": "New farms",
    "n_turbines_added": "Turbines added",
    "climate_mean_speed": "Mean wind speed",
    "climate_dominant_sector_freq": "Dominant-sector freq.",
}


# --------------------------------------------------------------------------- loading

def load_run(run_id):
    run_dir = _THIS_DIR / "results" / run_id
    merged_dir = merge_if_needed(run_dir)
    trials = pd.read_csv(merged_dir / "trials.csv")
    aep = pd.read_csv(merged_dir / "aep_results.csv")
    layout_attempts = pd.read_csv(merged_dir / "layout_attempts.csv") if (merged_dir / "layout_attempts.csv").exists() else pd.DataFrame()
    turbines = pd.read_csv(merged_dir / "turbine_assignments.csv")
    return trials, aep, layout_attempts, turbines


def load_campaign(run_ids):
    trials_l, aep_l, attempts_l, turbines_l = [], [], [], []
    for r in run_ids:
        t, a, la, tu = load_run(r)
        trials_l.append(t); aep_l.append(a); attempts_l.append(la); turbines_l.append(tu)
    trials = pd.concat(trials_l, ignore_index=True)
    aep = pd.concat(aep_l, ignore_index=True).merge(trials[["trial_id", "pd_level"]], on="trial_id")
    layout_attempts = pd.concat(attempts_l, ignore_index=True) if attempts_l and len(pd.concat(attempts_l)) else pd.DataFrame()
    turbines = pd.concat(turbines_l, ignore_index=True)

    # turbines added by NEW farms only (Speculative_XX), per trial -- 0 where none (e.g. "low")
    new_farm_turbines = turbines[turbines.farm_name.str.startswith("Speculative_")]
    added = new_farm_turbines.groupby("trial_id")["n_turbines"].sum().rename("n_turbines_added")
    trials = trials.merge(added, on="trial_id", how="left")
    trials["n_turbines_added"] = trials["n_turbines_added"].fillna(0).astype(int)

    levels_present = [l for l in LEVEL_ORDER if l in trials["pd_level"].unique()]
    return trials, aep, layout_attempts, levels_present


def _clean_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


# --------------------------------------------------------------------------- plots

def plot_loss_distribution(aep, levels, out_path):
    """Violin, not boxplot -- shows the actual shape (multimodal, skew), not just
    quartiles. Same data as before, richer view."""
    models = [m for m in MODEL_COLOR if m in aep["wake_model"].unique()]
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        data = [aep.loc[(aep.pd_level == lvl) & (aep.wake_model == model), "full_cluster_loss_pct"]
                for lvl in levels]
        parts = ax.violinplot(data, showmedians=True, showextrema=True)
        for body in parts["bodies"]:
            body.set(facecolor=MODEL_COLOR[model], alpha=0.7, edgecolor="none")
        for key in ("cbars", "cmins", "cmaxes", "cmedians"):
            parts[key].set(color="#333333", linewidth=1)
        ax.set_xticks(range(1, len(levels) + 1))
        ax.set_xticklabels([l.capitalize() for l in levels])
        ax.set_title(model, fontsize=11)
        _clean_ax(ax)
    axes[0].set_ylabel("Vineyard Wind AEP loss (%)")
    fig.suptitle("Wake loss distribution by growth scenario", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_new_farm_sizes(layout_attempts, out_path):
    """Distribution of INDIVIDUAL accepted new-farm sizes (MW) -- a population
    description (what sizes came out of the generator), not a target-vs-actual
    comparison."""
    if layout_attempts.empty:
        print("No layout_attempts data -- skipping new_farm_sizes.png")
        return
    accepted = layout_attempts[layout_attempts.success == True]["delivered_mw"]
    if accepted.empty:
        print("No accepted new farms in this campaign -- skipping new_farm_sizes.png")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(accepted, bins=20, color="#3B7EA1", alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("New farm capacity (MW)")
    ax.set_ylabel("Count")
    ax.set_title(f"Size distribution of new farms placed (n={len(accepted)})")
    _clean_ax(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_input_independence(trials, out_path):
    """Pairwise correlation among the Monte Carlo's own inputs (new farms, turbines
    added, wind-rose concentration) -- a sanity check that they're sampled
    independently by construction, not a hidden confound. High correlation between
    n_new_farms and n_turbines_added is expected (more farms -> more turbines, nearly
    by definition); climate_dominant_sector_freq should sit near zero against both,
    since climate and layout are unrelated draws."""
    cols = {"n_new_farms": "New farms", "n_turbines_added": "Turbines added",
            "climate_dominant_sector_freq": "Wind rose\nconcentration"}
    mat = trials[list(cols)].corr().values
    labels = list(cols.values())

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                     color="white" if abs(mat[i, j]) > 0.6 else "#333333", fontsize=11)
    ax.set_title("Are the MC's own inputs independent?")
    fig.colorbar(im, ax=ax, shrink=0.8, label="correlation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_loss_vs_growth(trials, aep, out_path, model="Nygaard_TurboGaussian"):
    """Loss risk vs. how much neighbor growth actually happened, in counts (farms,
    added turbines) instead of MW -- one model shown (Nygaard) to keep it readable."""
    merged = aep[aep.wake_model == model].merge(
        trials[["trial_id", "n_new_farms", "n_turbines_added"]], on="trial_id")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for lvl in trials["pd_level"].unique():
        sub = merged[merged.pd_level == lvl]
        axes[0].scatter(sub["n_new_farms"], sub["full_cluster_loss_pct"], s=22,
                         color=LEVEL_COLOR[lvl], alpha=0.75, edgecolor="none", label=lvl.capitalize())
        axes[1].scatter(sub["n_turbines_added"], sub["full_cluster_loss_pct"], s=22,
                         color=LEVEL_COLOR[lvl], alpha=0.75, edgecolor="none", label=lvl.capitalize())

    axes[0].set_xlabel("Number of new farms")
    axes[1].set_xlabel("Turbines added (new farms only)")
    for ax in axes:
        ax.set_ylabel(f"AEP loss (%) -- {model}")
        _clean_ax(ax)
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle("Loss risk vs. neighbor growth, per trial")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_input_impact(trials, aep, out_path):
    models = [m for m in MODEL_COLOR if m in aep["wake_model"].unique()]
    merged = aep.merge(trials[list(IMPACT_INPUTS) + ["trial_id"]], on="trial_id")
    corr_by_model = {}
    for model in models:
        sub = merged[merged.wake_model == model]
        corr_by_model[model] = {label: sub[col].corr(sub["full_cluster_loss_pct"])
                                 for col, label in IMPACT_INPUTS.items()}
    labels = sorted(IMPACT_INPUTS.values(),
                     key=lambda lb: -np.mean([abs(corr_by_model[m][lb]) for m in models]))

    fig, ax = plt.subplots(figsize=(7, 5))
    y = np.arange(len(labels))
    height = 0.35
    for i, model in enumerate(models):
        vals = [corr_by_model[model][lb] for lb in labels]
        offset = (i - (len(models) - 1) / 2) * height
        ax.barh([yi + offset for yi in y], vals, height=height * 0.9, color=MODEL_COLOR[model], alpha=0.85)
    ax.axvline(0, color="#888888", linewidth=1)
    ax.set_yticks(list(y)); ax.set_yticklabels(labels); ax.invert_yaxis()
    ax.set_xlabel("Correlation with AEP loss (%)")
    ax.set_title("Which input moves loss risk the most")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=MODEL_COLOR[m], alpha=0.85) for m in models]
    ax.legend(handles, models, loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_aep_exceedance(trials, aep, levels, out_path, model="Nygaard_TurboGaussian"):
    merged = aep[aep.wake_model == model]
    fig, ax = plt.subplots(figsize=(8, 6))
    for lvl in levels:
        values = np.sort(merged.loc[merged.pd_level == lvl, "full_cluster_aep_gwh"].values)[::-1]
        n = len(values)
        exceedance = np.arange(1, n + 1) / (n + 1) * 100
        ax.plot(exceedance, values, color=LEVEL_COLOR[lvl], linewidth=1.8, label=lvl.capitalize())
        p50, p90 = np.percentile(values, 50), np.percentile(values, 10)
        ax.axhline(p50, color=LEVEL_COLOR[lvl], linestyle=":", linewidth=0.8, alpha=0.6)
        ax.axhline(p90, color=LEVEL_COLOR[lvl], linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(50, color="#888888", linewidth=0.6)
    ax.axvline(90, color="#888888", linewidth=0.6)
    ax.text(50, 1.01, "P50", transform=ax.get_xaxis_transform(), ha="center", fontsize=8, color="#666666")
    ax.text(90, 1.01, "P90", transform=ax.get_xaxis_transform(), ha="center", fontsize=8, color="#666666")
    ax.set_xlabel("Exceedance probability (%)")
    ax.set_ylabel(f"Vineyard Wind AEP (GWh/yr) -- {model}")
    ax.set_title("AEP exceedance curve, by growth scenario", pad=20)
    _clean_ax(ax)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def save_summary_stats(trials, aep, levels, out_path):
    """Columns named q10/q50/q90 -- plain statistical percentiles of the value, NOT
    the wind-industry P10/P90 exceedance convention plot_aep_exceedance() uses (where
    P90 means the value exceeded 90% of the time -- the low end, i.e. this table's
    q10). Kept separate on purpose so the same word doesn't mean two different numbers
    in two different files."""
    models = [m for m in MODEL_COLOR if m in aep["wake_model"].unique()]
    rows = []
    for lvl in levels:
        t = trials[trials.pd_level == lvl]
        for col in ["n_new_farms", "n_turbines_added"]:
            rows.append(dict(pd_level=lvl, metric=col, q10=t[col].quantile(.1),
                              q50=t[col].median(), q90=t[col].quantile(.9)))
        for model in models:
            a = aep[(aep.pd_level == lvl) & (aep.wake_model == model)]
            rows.append(dict(pd_level=lvl, metric=f"aep_gwh_{model}", q10=a.full_cluster_aep_gwh.quantile(.1),
                              q50=a.full_cluster_aep_gwh.median(), q90=a.full_cluster_aep_gwh.quantile(.9)))
            rows.append(dict(pd_level=lvl, metric=f"loss_pct_{model}", q10=a.full_cluster_loss_pct.quantile(.1),
                              q50=a.full_cluster_loss_pct.median(), q90=a.full_cluster_loss_pct.quantile(.9)))
    pd.DataFrame(rows).round(2).to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", help="prefix -- expects <campaign>_low/_medium/_high run_ids")
    parser.add_argument("--run-ids", nargs="+", help="explicit run_ids instead of --campaign")
    args = parser.parse_args()

    if args.run_ids:
        run_ids = args.run_ids
        campaign_name = args.run_ids[0]
    elif args.campaign:
        run_ids = [f"{args.campaign}_{lvl}" for lvl in LEVEL_ORDER
                   if (_THIS_DIR / "results" / f"{args.campaign}_{lvl}").exists()]
        campaign_name = args.campaign
    else:
        parser.error("need --campaign or --run-ids")

    print(f"Loading: {run_ids}")
    trials, aep, layout_attempts, levels = load_campaign(run_ids)
    print(f"{len(trials)} trials total, scenarios present: {levels}")

    out_dir = RESULTS_HPC_DIR / campaign_name
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    trials.to_csv(out_dir / "all_trials.csv", index=False)
    aep.to_csv(out_dir / "all_aep_results.csv", index=False)
    print(f"Combined CSVs -> {out_dir}")

    plot_loss_distribution(aep, levels, fig_dir / "loss_distribution.png")
    plot_new_farm_sizes(layout_attempts, fig_dir / "new_farm_sizes.png")
    plot_input_independence(trials, fig_dir / "input_independence.png")
    plot_loss_vs_growth(trials, aep, fig_dir / "loss_vs_growth.png")
    plot_input_impact(trials, aep, fig_dir / "input_impact.png")
    plot_aep_exceedance(trials, aep, levels, fig_dir / "aep_exceedance.png")
    save_summary_stats(trials, aep, levels, out_dir / "summary_stats.csv")

    print(f"\nAll set: {out_dir}")


if __name__ == "__main__":
    main()
