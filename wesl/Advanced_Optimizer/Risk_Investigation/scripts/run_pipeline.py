"""
Single-year run: 1 ERA5 year, SuperGaussian wake model, layout QA + flow maps.
For the 24-year sweep see multi_year_analysis.py.

    conda run -n Wind_2200 python run_pipeline.py
"""
import time

import pandas as pd

from config import ERA5_DIR, SITE_YEAR, RESULTS_DIR, FOCUS_FARM
from site_builder import build_site
from cluster_layout import load_cluster
from aep_simulation import build_turbine_types, run_scenarios
from visualize_cluster import plot_layout, plot_flow_maps


def main():
    t0 = time.time()

    print(f"Building site from ERA5 {SITE_YEAR}...")
    grib_path = ERA5_DIR / f"era5_{SITE_YEAR}.grib"
    site, summary = build_site(grib_path)
    print(f"  {summary['n_hours']} hours, mean speed={summary['mean_speed']:.2f} m/s\n")

    print("Assembling cluster layout...")
    farms = load_cluster()
    multi_turbines, type_by_farm = build_turbine_types(farms)
    for f in farms:
        kind = "synthetic" if f["is_synthetic"] else "real"
        print(f"  {f['name']}: {f['n_turbines']} turbines ({kind})")
    print()

    print("Layout QA plot...")
    plot_layout(farms)

    print(f"\nRunning AEP scenarios ({FOCUS_FARM})...")
    result = run_scenarios(site, farms, multi_turbines, type_by_farm)
    print(f"  Isolated:      {result['isolated_aep_gwh']:.1f} GWh")
    print(f"  Real neighbors: {result['real_neighbors_aep_gwh']:.1f} GWh "
          f"(loss {result['real_neighbors_loss_pct']:.2f}%, {result['n_real_turbines']} turbines)")
    print(f"  Full cluster:   {result['full_cluster_aep_gwh']:.1f} GWh "
          f"(loss {result['full_cluster_loss_pct']:.2f}%, {result['n_full_turbines']} turbines)")

    print("\nWake flow-map snapshots...")
    plot_flow_maps(result, farms)

    df = pd.DataFrame([{
        "year": SITE_YEAR,
        "wake_model": "SuperGaussian",
        "focus_farm": FOCUS_FARM,
        "isolated_aep_gwh": result["isolated_aep_gwh"],
        "n_real_turbines": result["n_real_turbines"],
        "real_neighbors_aep_gwh": result["real_neighbors_aep_gwh"],
        "real_neighbors_loss_pct": result["real_neighbors_loss_pct"],
        "n_full_turbines": result["n_full_turbines"],
        "full_cluster_aep_gwh": result["full_cluster_aep_gwh"],
        "full_cluster_loss_pct": result["full_cluster_loss_pct"],
    }])
    csv_path = RESULTS_DIR / "cluster_aep_single_scenario.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
