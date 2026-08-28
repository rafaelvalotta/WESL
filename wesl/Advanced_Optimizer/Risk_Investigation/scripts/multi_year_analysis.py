"""
24-year sweep (ERA5 2000-2023): Vineyard Wind AEP loss, real neighbors
vs. full populated cluster. No flow maps -- see run_pipeline.py for that.

    conda run -n Wind_2200 python multi_year_analysis.py
"""
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import ERA5_DIR, ALL_YEARS, RESULTS_DIR, FIGURES_DIR, FOCUS_FARM
from site_builder import build_site
from cluster_layout import load_cluster
from aep_simulation import build_turbine_types, run_scenarios


def _summarize(series, label):
    p10, p50, p90 = np.percentile(series, [10, 50, 90])
    print(f"  {label}: P10={p10:.2f}%  P50={p50:.2f}%  P90={p90:.2f}%  mean={np.mean(series):.2f}%")


def _focus_capacity_mw(farms, focus_farm_name):
    focus = next(f for f in farms if f["name"] == focus_farm_name)
    rated_w = focus["turbine_node"]["performance"]["rated_power"]
    return focus["n_turbines"] * rated_w / 1e6


def _cf_pct(aep_gwh, capacity_mw, hours=8760):
    return aep_gwh / (capacity_mw / 1000 * hours) * 100


def plot_contrast(df, out_name="multi_year_loss_contrast"):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(df["year"], df["real_neighbors_loss_pct"], "o-", color="teal", label="Real neighbors only")
    ax.plot(df["year"], df["full_cluster_loss_pct"], "o-", color="darkorange", label="Full cluster")
    ax.set_xlabel("Year (ERA5 climate)")
    ax.set_ylabel("Vineyard Wind AEP loss (%)")
    ax.set_title("Wake loss: real neighbors vs. full cluster (2000-2023)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    out_path = FIGURES_DIR / f"{out_name}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_distributions(df, out_name="multi_year_loss_boxplot"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.boxplot(
        [df["real_neighbors_loss_pct"], df["full_cluster_loss_pct"]],
        tick_labels=["Real neighbors", "Full cluster"],
    )
    ax.set_ylabel("Vineyard Wind AEP loss (%)")
    ax.set_title("Wake loss distribution (24 ERA5 years)")
    ax.grid(alpha=0.3, axis="y")

    out_path = FIGURES_DIR / f"{out_name}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_aep_exceedance(df, out_name="aep_exceedance_curve"):
    """P50/P90 AEP exceedance curve per scenario -- standard wind-assessment chart."""
    # "internal only" drawn last, dashed -- otherwise it's hidden exactly
    # behind "real neighbors" (their AEP is nearly identical, loss ~0%)
    scenarios = [
        ("real_neighbors_aep_gwh", "Real neighbors", "teal", "-"),
        ("full_cluster_aep_gwh", "Full cluster", "darkorange", "-"),
        ("isolated_aep_gwh", "Internal only", "black", "--"),
    ]

    fig, ax = plt.subplots(figsize=(9, 6))
    for col, label, color, style in scenarios:
        values = np.sort(df[col].values)[::-1]
        n = len(values)
        exceedance = np.arange(1, n + 1) / (n + 1) * 100
        ax.plot(exceedance, values, style, marker="o", color=color, label=label, markersize=4, linewidth=1.5)

        p50 = np.percentile(df[col], 50)
        p90 = np.percentile(df[col], 10)  # P90 = exceeded 90% of years = 10th percentile
        ax.axhline(p50, color=color, linestyle=":", linewidth=0.8, alpha=0.5)
        ax.axhline(p90, color=color, linestyle="--", linewidth=0.8, alpha=0.5)

    ax.axvline(50, color="black", linewidth=0.6, alpha=0.4)
    ax.axvline(90, color="black", linewidth=0.6, alpha=0.4)
    ax.text(50, 1.02, "P50", transform=ax.get_xaxis_transform(), ha="center", fontsize=8)
    ax.text(90, 1.02, "P90", transform=ax.get_xaxis_transform(), ha="center", fontsize=8)

    ax.set_xlabel("Exceedance probability (%)")
    ax.set_ylabel("AEP (GWh)")
    ax.set_title("Vineyard Wind AEP exceedance curve (24 ERA5 years)", pad=20)
    ax.legend()
    ax.grid(alpha=0.3)

    out_path = FIGURES_DIR / f"{out_name}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_summary_table(df, out_name="scenario_summary_table"):
    """AEP / CF / wake loss by scenario, mean and range across years -- paper-style summary table."""
    def stats(col):
        return df[col].mean(), df[col].min(), df[col].max()

    rows = [
        ("Internal only", "isolated_aep_gwh", "isolated_cf_pct", None),
        ("+ Real neighbors", "real_neighbors_aep_gwh", "real_neighbors_cf_pct", "real_neighbors_loss_pct"),
        ("+ Full cluster", "full_cluster_aep_gwh", "full_cluster_cf_pct", "full_cluster_loss_pct"),
    ]

    table_data = []
    for label, aep_col, cf_col, loss_col in rows:
        aep_m, aep_lo, aep_hi = stats(aep_col)
        cf_m, cf_lo, cf_hi = stats(cf_col)
        loss_str = "0.0 (ref)" if loss_col is None else f"{stats(loss_col)[0]:.1f} ({stats(loss_col)[1]:.1f} to {stats(loss_col)[2]:.1f})"
        table_data.append([
            label,
            f"{aep_m:.0f} ({aep_lo:.0f}-{aep_hi:.0f})",
            f"{cf_m:.1f} ({cf_lo:.1f}-{cf_hi:.1f})",
            loss_str,
        ])

    fig, ax = plt.subplots(figsize=(9, 2.2))
    ax.axis("off")
    tbl = ax.table(
        cellText=table_data,
        colLabels=["Scenario", "AEP (GWh/yr)", "CF (%)", "Wake loss (%)"],
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.2)
    for (row, _col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#333333")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#f2f2f2" if row % 2 == 0 else "white")

    ax.set_title("Vineyard Wind: AEP, CF, wake loss by scenario (mean and range, 24 years)", fontsize=11, pad=16)

    out_path = FIGURES_DIR / f"{out_name}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def main():
    t0 = time.time()

    print("Assembling cluster layout...")
    farms = load_cluster()
    multi_turbines, type_by_farm = build_turbine_types(farms)
    capacity_mw = _focus_capacity_mw(farms, FOCUS_FARM)
    for f in farms:
        kind = "synthetic" if f["is_synthetic"] else "real"
        print(f"  {f['name']}: {f['n_turbines']} turbines ({kind})")
    print()

    rows = []
    for i, year in enumerate(ALL_YEARS, 1):
        grib_path = ERA5_DIR / f"era5_{year}.grib"
        site, summary = build_site(grib_path)
        result = run_scenarios(site, farms, multi_turbines, type_by_farm)

        rows.append(dict(
            year=year,
            mean_speed=summary["mean_speed"],
            isolated_aep_gwh=result["isolated_aep_gwh"],
            isolated_cf_pct=_cf_pct(result["isolated_aep_gwh"], capacity_mw),
            real_neighbors_aep_gwh=result["real_neighbors_aep_gwh"],
            real_neighbors_cf_pct=_cf_pct(result["real_neighbors_aep_gwh"], capacity_mw),
            real_neighbors_loss_pct=result["real_neighbors_loss_pct"],
            full_cluster_aep_gwh=result["full_cluster_aep_gwh"],
            full_cluster_cf_pct=_cf_pct(result["full_cluster_aep_gwh"], capacity_mw),
            full_cluster_loss_pct=result["full_cluster_loss_pct"],
        ))
        print(f"[{i}/{len(ALL_YEARS)}] {year}: "
              f"real={result['real_neighbors_loss_pct']:.2f}%  "
              f"full={result['full_cluster_loss_pct']:.2f}%")

    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "multi_year_scenario_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    print(f"\n--- Loss distribution across {len(ALL_YEARS)} years ({FOCUS_FARM}) ---")
    _summarize(df["real_neighbors_loss_pct"], "Real neighbors")
    _summarize(df["full_cluster_loss_pct"], "Full cluster ")

    plot_contrast(df)
    plot_distributions(df)
    plot_aep_exceedance(df)
    plot_summary_table(df)
    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
