"""One presentation figure per trial: wind rose + layout map (gray turbines, Vineyard
in red, federal boundary, faint bathymetry) + compact key numbers. Reuses the AEP
numbers already logged for a run instead of recomputing them -- only climate + layout
are regenerated (reproducible from the same seed), which is the cheap part.

    conda run -n Wind_2200 python make_trial_figures.py --run-id 20260807_033623
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

import climate
import layout

FIGURES_DIR = _THIS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

ROSE_COLOR = "#3B7EA1"
GRAY = "#9AA0A6"
RED = "#C0392B"
SLA_COLOR = "#444444"
BATHY_CMAP = "Blues"
BATHY_ALPHA = 0.18

MAP_HALF_WIDTH_KM = 170  # fixed across all trial figures, so they're directly comparable


def read_csv_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def load_trial_and_aep(run_dir, seed):
    trials = {r["seed"]: r for r in read_csv_rows(run_dir / "trials.csv")}
    aep_rows = [r for r in read_csv_rows(run_dir / "aep_results.csv") if r["seed"] == str(seed)]
    aep_by_model = {r["wake_model"]: r for r in aep_rows}
    return trials[str(seed)], aep_by_model


def plot_bathymetry(ax, xlim_km, ylim_km, resolution=250):
    gx = np.linspace(xlim_km[0] * 1000, xlim_km[1] * 1000, resolution)
    gy = np.linspace(ylim_km[0] * 1000, ylim_km[1] * 1000, resolution)
    GX, GY = np.meshgrid(gx, gy)
    GZ = layout.depth_at(GX.ravel(), GY.ravel()).reshape(GX.shape)
    ax.contourf(GX / 1000, GY / 1000, GZ, levels=8, cmap=BATHY_CMAP, alpha=BATHY_ALPHA, antialiased=True)


def plot_wind_rose(ax, freq, A, k):
    n_sectors = len(freq)
    sector_deg = np.linspace(0, 360, n_sectors, endpoint=False)
    theta = np.radians(sector_deg)
    ax.bar(theta, freq, width=2 * np.pi / n_sectors * 0.9, color=ROSE_COLOR, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["N", "E", "S", "W"], fontsize=8)
    ax.set_yticklabels([])
    ax.spines["polar"].set_visible(False)
    ax.set_title("Wind rose", fontsize=9, pad=2)


def make_trial_figure(seed, pd_level, run_dir, out_path):
    rng = np.random.default_rng(seed)

    # Climate: same draw the trial used (site consumed first from this rng stream).
    site = climate.sample_synthetic_site(rng, scale=1.0)
    row = climate.sample_synthetic_row(np.random.default_rng(seed), scale=1.0)
    freq, A, k = row[:12], row[12:24], row[24:36]

    # Layout: continues the same rng stream, exactly like the original trial.
    cluster_x, cluster_y, cluster_farms, new_farms = layout.populate_scenario(pd_level, rng)

    trial_row, aep_by_model = load_trial_and_aep(run_dir, seed)

    vineyard = next(f for f in cluster_farms if f["name"] == "Vineyard Wind")
    other_cluster_x = np.concatenate([f["x"] for f in cluster_farms if f["name"] != "Vineyard Wind" and f["n_turbines"] > 0])
    other_cluster_y = np.concatenate([f["y"] for f in cluster_farms if f["name"] != "Vineyard Wind" and f["n_turbines"] > 0])
    new_x = np.concatenate([f["pts"][:, 0] for f in new_farms]) if new_farms else np.array([])
    new_y = np.concatenate([f["pts"][:, 1] for f in new_farms]) if new_farms else np.array([])
    gray_x = np.concatenate([other_cluster_x, new_x])
    gray_y = np.concatenate([other_cluster_y, new_y])

    fig = plt.figure(figsize=(11, 8.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[3, 1, 0.05, 0.05])
    ax_map = fig.add_subplot(gs[0, 0])
    ax_rose = fig.add_subplot(gs[0, 1], projection="polar")

    vc_km = layout.VINEYARD_CENTER / 1000
    xlim = (vc_km[0] - MAP_HALF_WIDTH_KM, vc_km[0] + MAP_HALF_WIDTH_KM)
    ylim = (vc_km[1] - MAP_HALF_WIDTH_KM, vc_km[1] + MAP_HALF_WIDTH_KM)

    plot_bathymetry(ax_map, xlim, ylim)
    for line in layout.sla_lines_utm:
        ax_map.plot(line[:, 0] / 1000, line[:, 1] / 1000, color=SLA_COLOR, linewidth=0.8, alpha=0.7)

    ax_map.scatter(gray_x / 1000, gray_y / 1000, s=3, color=GRAY, alpha=0.7, label="Other turbines")
    ax_map.scatter(vineyard["x"] / 1000, vineyard["y"] / 1000, s=8, color=RED, label="Vineyard Wind")

    ax_map.set_xlim(*xlim)
    ax_map.set_ylim(*ylim)
    ax_map.set_aspect("equal")
    ax_map.set_xlabel("km", fontsize=8)
    ax_map.set_ylabel("km", fontsize=8)
    ax_map.tick_params(labelsize=7)
    ax_map.legend(loc="upper left", fontsize=8, frameon=False, markerscale=2.5)

    plot_wind_rose(ax_rose, freq, A, k)

    n_turbines = int(trial_row["n_turbines_total"])
    n_new_farms = int(trial_row["n_new_farms"])
    delivered_mw = float(trial_row["delivered_mw"])
    nyg = aep_by_model.get("Nygaard_TurboGaussian", {})
    sg = aep_by_model.get("SuperGaussian", {})

    title = f"Trial {seed} -- {pd_level} density scenario"
    subtitle = (f"AEP Vineyard: {float(nyg.get('full_cluster_aep_gwh', 0)):.0f} GWh (Nygaard)  /  "
                f"{float(sg.get('full_cluster_aep_gwh', 0)):.0f} GWh (SuperGaussian)")
    subtitle2 = f"{n_turbines} turbines  ·  {delivered_mw:.0f} MW new capacity  ·  {n_new_farms} new farms"

    fig.suptitle(f"{title}\n{subtitle}\n{subtitle2}", fontsize=11, y=0.99, linespacing=1.6)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--pd-level", default="high")
    args = parser.parse_args()

    run_dir = _THIS_DIR / "results" / args.run_id
    paths = []
    for seed in args.seeds:
        out_path = FIGURES_DIR / f"trial_{seed}_{args.run_id}.png"
        make_trial_figure(seed, args.pd_level, run_dir, out_path)
        paths.append(out_path)
    print("\n".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
