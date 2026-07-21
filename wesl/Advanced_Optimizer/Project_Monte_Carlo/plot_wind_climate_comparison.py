"""
Compares the single long-term wind climatology used today by the
deterministic optimizer (Data/energy_resource_us.yaml) against the 24
individual ERA5 years (2000-2023) that feed the Risk Research --
Deterministic Phase -- the input-side justification for why a single
averaged wind rose misses real interannual variability.

Outputs (in figures/ and results/):
  - wind_climate_windrose.png       : sector frequency, climatology vs. 24 years
  - wind_climate_annual_mean_speed.png : annual mean speed vs. climatology
  - wind_climate_weibull_a_by_sector.png : Weibull A spread vs. climatology
  - wind_climate_by_year.csv        : underlying per-year numbers
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import yaml
from scipy.special import gamma as gamma_fn

from download_era5_vineyard import load_wind_series, LAT_CENTER, LON_CENTER
from build_pywake_sites import fit_sector_weibull, YEARS, N_SECTORS

# Absolute paths, anchored to this file's location, so the script works
# regardless of the caller's current working directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "Data")
ERA5_DIR = os.path.join(SCRIPT_DIR, "era5_vineyard")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

SECTOR_CENTERS = np.arange(N_SECTORS) * (360 / N_SECTORS)  # 0, 30, ..., 330


def load_climatology():
    """The single wind rose currently used by the deterministic optimizer."""
    with open(os.path.join(DATA_DIR, "energy_resource_us.yaml")) as f:
        d = yaml.safe_load(f)["wind_resource"]
    freq = np.array(d["sector_probability"]["data"])
    A = np.array(d["weibull_a"]["data"])
    k = np.array(d["weibull_k"]["data"])
    mean_speed = float(np.sum(freq * A * gamma_fn(1 + 1 / k)))
    return freq, A, k, mean_speed


def fit_all_years(years=YEARS, era5_dir=ERA5_DIR):
    """Per-year sector Weibull fit + overall mean speed, reusing the deterministic-phase fit function."""
    by_year = {}
    for year in years:
        path = os.path.join(era5_dir, f"era5_{year}.grib")
        _t, speed, direction = load_wind_series(path, LAT_CENTER, LON_CENTER)
        freq, A, k = fit_sector_weibull(speed, direction)
        by_year[year] = dict(freq=np.array(freq), A=np.array(A), k=np.array(k),
                              mean_speed=float(speed.mean()))
        print(f"[{year}] mean speed = {by_year[year]['mean_speed']:.2f} m/s")
    return by_year


def plot_windrose(clim_freq, by_year):
    years = sorted(by_year)
    mean_of_years_freq = np.mean([by_year[y]["freq"] for y in years], axis=0)

    fig = plt.figure(figsize=(9, 9.5))
    ax = fig.add_subplot(projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    theta = np.append(np.radians(SECTOR_CENTERS), np.radians(SECTOR_CENTERS[0]))

    for y in years:
        r = np.append(by_year[y]["freq"], by_year[y]["freq"][0]) * 100
        ax.plot(theta, r, color="steelblue", alpha=0.35, linewidth=1)
    ax.plot(theta, np.append(mean_of_years_freq, mean_of_years_freq[0]) * 100,
            color="black", linestyle="--", linewidth=1.8, label="Mean of 24 ERA5 years")
    ax.plot(theta, np.append(clim_freq, clim_freq[0]) * 100,
            color="crimson", linewidth=2.5, label="Used in optimization\n(long-term climatology)")
    fig.suptitle("Risk Research -- Deterministic Phase\nWind direction frequency by sector: "
                 "24 individual years vs. climatology used today", fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), fontsize=9, ncol=2)

    out_path = os.path.join(FIGURES_DIR, "wind_climate_windrose.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_annual_mean_speed(clim_mean, by_year):
    years = sorted(by_year)
    annual_means = [by_year[y]["mean_speed"] for y in years]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["indianred" if v < clim_mean else "steelblue" for v in annual_means]
    ax.bar(years, annual_means, color=colors)
    ax.axhline(clim_mean, color="crimson", linewidth=2, label=f"Climatology used in optimization ({clim_mean:.2f} m/s)")
    ax.set_ylabel("Mean wind speed (m/s)")
    ax.set_title("Risk Research -- Deterministic Phase\nAnnual mean wind speed, ERA5 2000-2023 "
                 "vs. the single value implied by today's climatology")
    ax.tick_params(axis="x", rotation=60)
    ax.legend(fontsize=9)
    fig.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "wind_climate_annual_mean_speed.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_weibull_a_by_sector(clim_A, by_year):
    years = sorted(by_year)

    fig, ax = plt.subplots(figsize=(10, 6))
    A_by_sector = [[by_year[y]["A"][i] for y in years] for i in range(N_SECTORS)]
    bp = ax.boxplot(A_by_sector, positions=SECTOR_CENTERS, widths=15, patch_artist=True)
    for box in bp["boxes"]:
        box.set(facecolor="lightsteelblue", alpha=0.7)
    ax.scatter(SECTOR_CENTERS, clim_A, color="crimson", marker="D", zorder=5, s=50,
               label="Used in optimization")
    ax.set_xticks(SECTOR_CENTERS)
    ax.set_xticklabels([f"{int(d)}°" for d in SECTOR_CENTERS])
    ax.set_xlabel("Wind direction sector")
    ax.set_ylabel("Weibull A (m/s)")
    ax.set_title("Risk Research -- Deterministic Phase\nWeibull scale parameter (A) per sector: "
                 "spread across 24 real years vs. the fixed value used today")
    ax.legend(fontsize=9)
    fig.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "wind_climate_weibull_a_by_sector.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def save_csv(by_year, clim_mean):
    import csv
    out_path = os.path.join(RESULTS_DIR, "wind_climate_by_year.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["year", "mean_speed_ms"] + [f"freq_{int(d)}" for d in SECTOR_CENTERS] \
            + [f"A_{int(d)}" for d in SECTOR_CENTERS] + [f"k_{int(d)}" for d in SECTOR_CENTERS]
        w.writerow(header)
        for y in sorted(by_year):
            d = by_year[y]
            w.writerow([y, f"{d['mean_speed']:.3f}"] + list(d["freq"]) + list(d["A"]) + list(d["k"]))
        w.writerow(["climatology_used_in_optimization", f"{clim_mean:.3f}"])
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    clim_freq, clim_A, clim_k, clim_mean = load_climatology()
    print(f"Climatology used in optimization -- implied mean speed: {clim_mean:.2f} m/s\n")

    by_year = fit_all_years()
    plot_windrose(clim_freq, by_year)
    plot_annual_mean_speed(clim_mean, by_year)
    plot_weibull_a_by_sector(clim_A, by_year)
    save_csv(by_year, clim_mean)
