"""
Wake-effect flow maps for one representative scenario from the Risk
Research -- Deterministic Phase: Nygaard_TurboGaussian wake model, year
2013 (closest-to-median loss_pct for this model), SouthCoast populated
with the Haliade-X 13MW option -- all 3 farms in a single snapshot.

Renders 3 directions:
  - dominant   : cluster's most frequent real wind direction (240 deg)
  - rev_to_vin : wind aligned so Revolution's wake travels onto Vineyard
  - sc_to_vin  : wind aligned so SouthCoast's wake travels onto Vineyard

Reuses the already-validated objects from deterministic_phase.py instead
of rebuilding anything -- these are exactly 3 directional slices of one
of the 216 combinations already computed.

Output: figures/wake_flow_map_{scenario}.png
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from build_pywake_sites import build_site_for_year
from generate_southcoast_layouts import generate_all_layouts
from deterministic_phase import (
    VINEYARD_X, VINEYARD_Y, REVOLUTION_X, REVOLUTION_Y,
    N_VINEYARD, N_REVOLUTION,
    VINEYARD_TURBINE, REVOLUTION_TURBINE, SOUTHCOAST_TURBINES,
    WAKE_MODELS, SUPERPOSITION_BY_MODEL,
)
from py_wake.wind_turbines import WindTurbines
from py_wake.wind_farm_models import PropagateDownwind
from py_wake.turbulence_models import CrespoHernandez
from py_wake.flow_map import HorizontalGrid

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

YEAR = "2013"
MODEL_NAME = "Nygaard_TurboGaussian"
SOUTHCOAST_OPTION = "option1_haliadex_13mw"
WS_SNAPSHOT = 10
GRID_RESOLUTION = 500

# wd computed from actual farm centroids: bearing such that wake travel
# direction (source -> Vineyard) matches the meteorological "wind from" angle.
SCENARIOS = [
    ("dominant", 240, "Dominant wind (240°)"),
    ("rev_to_vin", 282, "Revolution -> Vineyard (282°)"),
    ("sc_to_vin", 173, "SouthCoast -> Vineyard (173°)"),
]


def render(site, sc_layout, sc_turbine, wd, title, out_name):
    multi_type_turbines = WindTurbines.from_WindTurbine_lst(
        [VINEYARD_TURBINE, REVOLUTION_TURBINE, sc_turbine]
    )
    wfm = PropagateDownwind(site, multi_type_turbines, wake_deficitModel=WAKE_MODELS[MODEL_NAME],
                            superpositionModel=SUPERPOSITION_BY_MODEL[MODEL_NAME],
                            turbulenceModel=CrespoHernandez())

    x_full = VINEYARD_X + REVOLUTION_X + list(sc_layout["x"])
    y_full = VINEYARD_Y + REVOLUTION_Y + list(sc_layout["y"])
    type_full = [0] * N_VINEYARD + [1] * N_REVOLUTION + [2] * sc_layout["n_turbines"]

    print(f"[{out_name}] wd={wd}, ws={WS_SNAPSHOT}, {len(x_full)} turbines...")
    sim_res = wfm(x_full, y_full, type=type_full, wd=[wd], ws=[WS_SNAPSHOT])
    fmap = sim_res.flow_map(grid=HorizontalGrid(resolution=GRID_RESOLUTION))

    ws_eff = fmap.WS_eff.squeeze().values
    vmin = max(np.nanpercentile(ws_eff, 1), WS_SNAPSHOT * 0.5)
    levels = np.linspace(vmin, WS_SNAPSHOT * 1.02, 100)

    fig, ax = plt.subplots(figsize=(13, 11))
    fmap.plot_wake_map(ax=ax, cmap="Blues_r", normalize_with=1000, levels=levels, plot_windturbines=False)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")

    for label, x_arr, y_arr, color in [
        ("Vineyard", VINEYARD_X, VINEYARD_Y, "black"),
        ("Revolution", REVOLUTION_X, REVOLUTION_Y, "teal"),
        ("SouthCoast", sc_layout["x"], sc_layout["y"], "darkorange"),
    ]:
        xk, yk = np.array(x_arr) / 1000, np.array(y_arr) / 1000
        ax.scatter(xk, yk, s=12, color=color, edgecolor="white", linewidth=0.4, zorder=5)

    for label, x, y, dy in [
        ("Vineyard", sum(VINEYARD_X) / len(VINEYARD_X), max(VINEYARD_Y), 10),
        ("Revolution", sum(REVOLUTION_X) / len(REVOLUTION_X), max(REVOLUTION_Y), 10),
        ("SouthCoast", sum(sc_layout["x"]) / len(sc_layout["x"]), min(sc_layout["y"]), -22),
    ]:
        ax.annotate(label, (x / 1000, y / 1000), textcoords="offset points", xytext=(0, dy),
                    ha="center", fontsize=10, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="none"))

    ax.set_title(title, fontsize=12)

    out_path = os.path.join(FIGURES_DIR, f"wake_flow_map_{out_name}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    grib_path = os.path.join(SCRIPT_DIR, "era5_vineyard", f"era5_{YEAR}.grib")
    site, _ = build_site_for_year(grib_path)
    sc_layout = generate_all_layouts(verbose=False)[SOUTHCOAST_OPTION]
    sc_turbine = SOUTHCOAST_TURBINES[SOUTHCOAST_OPTION]

    for out_name, wd, title in SCENARIOS:
        render(site, sc_layout, sc_turbine, wd, title, out_name)


if __name__ == "__main__":
    main()
