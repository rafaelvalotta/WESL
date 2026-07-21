"""
Risk Research -- Deterministic Phase: main loop (methodology Section 5.2).

Combines the already-validated building blocks:
  - build_pywake_sites.py           -> 24 Site objects (1 per ERA5 year)
  - generate_southcoast_layouts.py  -> 3 candidate layouts (1 per turbine)
  - Data/*.yaml (WindIO)            -> real Vineyard/Revolution turbines +
                                        the 3 SouthCoast candidates

Enumerates the 216 combinations (24 years x 3 wake models x 3 SouthCoast
turbines) exhaustively -- deterministic full enumeration, not random
sampling. Computes for each:
  - Vineyard AEP isolated (no neighbors)
  - Vineyard AEP with Revolution (fixed) + SouthCoast (variable) present
  - loss_pct = (isolated - with_neighbors) / isolated * 100

Isolated AEP only depends on (year, wake_model), not on the SouthCoast
turbine, so it's computed once per (year, model) and reused across the
3 SouthCoast variations (288 simulations instead of 432).

Output: results/deterministic_phase_216_combinations.csv
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import windIO

from build_pywake_sites import build_all_sites, YEARS
from generate_southcoast_layouts import generate_all_layouts

from py_wake.wind_turbines import WindTurbines
from py_wake.wind_turbines._wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.wind_farm_models import PropagateDownwind
from py_wake.deficit_models import NOJDeficit, TurboGaussianDeficit
from py_wake.deficit_models.gaussian import BlondelSuperGaussianDeficit2023
from py_wake.turbulence_models import CrespoHernandez
from py_wake.superposition_models import SquaredSum, LinearSum

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Wind resolution used on every wfm() call. PyWake's default (360 wd x 23
# ws) produces a deficit_ijlk matrix too large for memory with ~350+
# simultaneous turbines (Vineyard+Revolution+SouthCoast). Since the sector
# Weibull fit only has 30 deg resolution anyway, sampling wd every 30 deg
# loses no real information -- just cuts compute ~30x.
SIM_WD = np.arange(0, 360, 30)   # 12 directions -- 1 per fitted sector
SIM_WS = np.arange(3, 26, 1)     # 23 speeds (3-25 m/s)


# ---------------------------------------------------------------------
# Vineyard Wind 1 and Revolution Wind: layout + turbine read straight from
# the WindIO ontology (Data/vineyard_revolution_system.yaml), single
# source of truth -- no coordinates duplicated here.
# ---------------------------------------------------------------------

_system_data = windIO.load_yaml(DATA_DIR / "vineyard_revolution_system.yaml")
_vineyard_farm = _system_data["wind_farm"][0]
_revolution_farm = _system_data["wind_farm"][1]
assert _vineyard_farm["name"] == "Vineyard Wind" and _revolution_farm["name"] == "Revolution Wind", \
    "Unexpected wind_farm order in vineyard_revolution_system.yaml"

VINEYARD_X = list(_vineyard_farm["layouts"][0]["coordinates"]["x"])
VINEYARD_Y = list(_vineyard_farm["layouts"][0]["coordinates"]["y"])
REVOLUTION_X = list(_revolution_farm["layouts"][0]["coordinates"]["x"])
REVOLUTION_Y = list(_revolution_farm["layouts"][0]["coordinates"]["y"])

N_VINEYARD = len(VINEYARD_X)
N_REVOLUTION = len(REVOLUTION_X)


# ---------------------------------------------------------------------
# Turbines. Vineyard/Revolution come resolved from the system yaml's own
# !include. The 3 SouthCoast candidates have no real farm entry in
# wind_farm_us.yaml (only the lease boundary is real -- Section 3.2), so
# they're loaded straight from their individual files in Data/.
# ---------------------------------------------------------------------

def build_pywake_turbine(turbine_data):
    """Converts a resolved WindIO turbine dict into a PyWake WindTurbine."""
    pc = turbine_data["performance"]["power_curve"]
    ctc = turbine_data["performance"]["Ct_curve"]
    return WindTurbine(
        name=turbine_data["name"],
        diameter=turbine_data["rotor_diameter"],
        hub_height=turbine_data["hub_height"],
        powerCtFunction=PowerCtTabular(
            ws=pc["power_wind_speeds"], power=pc["power_values"],
            power_unit="W", ct=ctc["Ct_values"],
        ),
    )


def load_turbine_file(path):
    with open(path) as f:
        return build_pywake_turbine(yaml.safe_load(f))


VINEYARD_TURBINE = build_pywake_turbine(_vineyard_farm["turbines"])
REVOLUTION_TURBINE = build_pywake_turbine(_revolution_farm["turbines"])
SOUTHCOAST_TURBINES = {
    "option1_haliadex_13mw": load_turbine_file(DATA_DIR / "haliadex_13mw.yaml"),
    "option2_haliadex_250": load_turbine_file(DATA_DIR / "haliadex_15.5.yaml"),
    "option3_gwh300_22mw": load_turbine_file(DATA_DIR / "gwh300_22mw.yaml"),
}


# ---------------------------------------------------------------------
# Wake models -- discrete set from the methodology (Section 3.2)
# ---------------------------------------------------------------------

WAKE_MODELS = {
    "Nygaard_TurboGaussian": TurboGaussianDeficit(),
    "SuperGaussian": BlondelSuperGaussianDeficit2023(),
    "NOJ": NOJDeficit(),
}

# Each deficit model has a literature-calibrated superposition recipe
# (py_wake.literature), not just the deficit class on its own.
# PropagateDownwind()'s default is LinearSum, which only matches
# SuperGaussian's recipe -- Nygaard_2022 and Jensen_1983 (NOJ) are
# calibrated with SquaredSum, matching Data/analysis.yaml's
# deterministic pipeline (ws_superposition: Squared).
SUPERPOSITION_BY_MODEL = {
    "Nygaard_TurboGaussian": SquaredSum(),  # py_wake.literature.noj.Nygaard_2022
    "NOJ": SquaredSum(),                    # py_wake.literature.noj.Jensen_1983
    "SuperGaussian": LinearSum(),           # py_wake.literature.gaussian_models.Blondel_Cathelain_2020
}


# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------

def run_deterministic_phase(years=YEARS, save_csv=True):
    print("Building the 24 Site objects (once)...")
    sites_by_year = build_all_sites(years=years, verbose=False)
    print(f"  {len(sites_by_year)} sites ready.\n")

    print("Generating the 3 SouthCoast candidate layouts (once)...")
    southcoast_layouts = generate_all_layouts(verbose=False)
    for key, data in southcoast_layouts.items():
        print(f"  {key}: {data['n_turbines']} turbines")
    print()

    rows = []
    isolated_aep_cache = {}  # (year, model) -> Vineyard's isolated AEP (GWh)

    n_total = len(years) * len(WAKE_MODELS) * len(SOUTHCOAST_TURBINES)
    count = 0
    start = time.time()

    for year in years:
        site = sites_by_year[year]

        for model_name, wake_model in WAKE_MODELS.items():

            # multi-type turbines: Vineyard(0) + Revolution(1) + SouthCoast(2)
            for sc_key, sc_layout in southcoast_layouts.items():
                count += 1

                sc_turbine = SOUTHCOAST_TURBINES[sc_key]
                multi_type_turbines = WindTurbines.from_WindTurbine_lst(
                    [VINEYARD_TURBINE, REVOLUTION_TURBINE, sc_turbine]
                )
                wfm = PropagateDownwind(site, multi_type_turbines, wake_deficitModel=wake_model,
                                        superpositionModel=SUPERPOSITION_BY_MODEL[model_name],
                                        turbulenceModel=CrespoHernandez())

                cache_key = (year, model_name)
                if cache_key not in isolated_aep_cache:
                    isolated_sim = wfm(VINEYARD_X, VINEYARD_Y, type=[0] * N_VINEYARD,
                                        wd=SIM_WD, ws=SIM_WS)
                    isolated_aep_cache[cache_key] = isolated_sim.aep().sum().values
                isolated_aep = isolated_aep_cache[cache_key]

                x_full = VINEYARD_X + REVOLUTION_X + list(sc_layout["x"])
                y_full = VINEYARD_Y + REVOLUTION_Y + list(sc_layout["y"])
                type_full = ([0] * N_VINEYARD + [1] * N_REVOLUTION
                             + [2] * sc_layout["n_turbines"])

                sim_with_neighbors = wfm(x_full, y_full, type=type_full, wd=SIM_WD, ws=SIM_WS)
                aep_with_neighbors = (sim_with_neighbors.aep()
                                      .isel(wt=slice(0, N_VINEYARD))
                                      .sum().values)

                loss_pct = float((isolated_aep - aep_with_neighbors) / isolated_aep * 100)

                rows.append(dict(
                    year=year,
                    wake_model=model_name,
                    southcoast_turbine=sc_key,
                    n_southcoast_turbines=sc_layout["n_turbines"],
                    isolated_aep_gwh=float(isolated_aep),
                    aep_with_neighbors_gwh=float(aep_with_neighbors),
                    loss_pct=loss_pct,
                ))

                if count % 20 == 0 or count == n_total:
                    elapsed = time.time() - start
                    print(f"  [{count}/{n_total}] {year} | {model_name} | {sc_key} "
                          f"-> loss={loss_pct:.2f}%  ({elapsed:.0f}s elapsed)")

    df = pd.DataFrame(rows)

    if save_csv:
        csv_path = os.path.join(RESULTS_DIR, "deterministic_phase_216_combinations.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}  ({len(df)} rows)")

    return df


if __name__ == "__main__":
    df = run_deterministic_phase()

    print("\n--- loss_pct distribution summary (methodology Section 6) ---")
    print(f"P10: {np.percentile(df['loss_pct'], 10):.2f}%")
    print(f"P50: {np.percentile(df['loss_pct'], 50):.2f}%")
    print(f"P90: {np.percentile(df['loss_pct'], 90):.2f}%")
    print(f"\nMean by wake model:")
    print(df.groupby("wake_model")["loss_pct"].mean())
    print(f"\nMean by SouthCoast turbine:")
    print(df.groupby("southcoast_turbine")["loss_pct"].mean())
