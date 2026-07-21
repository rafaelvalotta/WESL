"""
Deterministic 2-farm benchmark (Vineyard + Revolution only), under the
single long-term climatology (Data/energy_resource_us.yaml) -- reproduces
the setup of the Moura & Rodrigues (2025) benchmark this project extends,
as a standalone cross-check independent of the ERA5-based Risk Research
pipeline (deterministic_phase.py). Does not import from or affect it.

Varies 2 factors, combined:
  - wake_model: the same 3 discrete models used elsewhere
    (Nygaard_TurboGaussian, SuperGaussian, NOJ)
  - turbine_config: real turbines (Vineyard Haliade-X 13MW / Revolution
    SG11-200DD) vs. 2 uniform-turbine variants, to separate how much of
    loss_pct comes from turbine choice vs. wake model choice

Uses the single climatology directly (not the 24 ERA5 years) -- the same
long-term wind rose already found to run higher than any real ERA5 year.

Output: results/benchmark_wake_turbine.csv, figures/benchmark_wake_turbine.png
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import windIO
import matplotlib.pyplot as plt

from py_wake.site import UniformWeibullSite
from py_wake.wind_turbines import WindTurbines
from py_wake.wind_turbines._wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.wind_farm_models import PropagateDownwind
from py_wake.deficit_models import NOJDeficit, TurboGaussianDeficit
from py_wake.deficit_models.gaussian import BlondelSuperGaussianDeficit2020
from py_wake.turbulence_models import CrespoHernandez
from py_wake.superposition_models import SquaredSum, LinearSum

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = Path(SCRIPT_DIR).parent / "Data"
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# Only 2 farms / ~143 turbines here (vs. 355 in the 3-farm pipeline), so a
# much finer wd/ws grid is still cheap -- this also removes the numerical
# noise a coarse 12x23 grid left in the SuperGaussian result (a pure
# deficit model, with no blockage/ground model attached, can never
# physically produce a *gain* in AEP; a negative loss_pct is discretization
# noise, not a real effect).
SIM_WD = np.arange(0, 360, 5)
SIM_WS = np.arange(3, 25.5, 0.5)

WAKE_MODELS = {
    "Nygaard_TurboGaussian": (TurboGaussianDeficit(), SquaredSum()),
    # 2020 calibration, not 2023 -- matches the PI's benchmark magnitude much
    # more closely (2023's recalibrated constants under-predict here by ~5x)
    "SuperGaussian": (BlondelSuperGaussianDeficit2020(), LinearSum()),
    "NOJ": (NOJDeficit(), SquaredSum()),
}

DISTANCE_KM = 22  # Vineyard <-> Revolution separation

# Keys double as the turbine-MW legend label -- that's what matters here,
# not which yaml file backs each option.
TURBINE_CONFIGS = {
    "13+11 MW (real)": ("haliadex_13mw.yaml", "sg11_200dd.yaml"),
    "13+13 MW": ("haliadex_13mw.yaml", "haliadex_13mw.yaml"),
    "11+11 MW": ("sg11_200dd.yaml", "sg11_200dd.yaml"),
}


def load_climatology_site():
    with open(DATA_DIR / "energy_resource_us.yaml") as f:
        d = yaml.safe_load(f)["wind_resource"]
    freq = np.array(d["sector_probability"]["data"])
    freq = freq / freq.sum()
    A = d["weibull_a"]["data"]
    k = d["weibull_k"]["data"]
    ti = d["turbulence_intensity"]["data"][0]
    return UniformWeibullSite(p_wd=freq, a=A, k=k, ti=ti)


def load_layouts():
    system_data = windIO.load_yaml(DATA_DIR / "vineyard_revolution_system.yaml")
    vineyard, revolution = system_data["wind_farm"][0], system_data["wind_farm"][1]
    vx = list(vineyard["layouts"][0]["coordinates"]["x"])
    vy = list(vineyard["layouts"][0]["coordinates"]["y"])
    rx = list(revolution["layouts"][0]["coordinates"]["x"])
    ry = list(revolution["layouts"][0]["coordinates"]["y"])
    return vx, vy, rx, ry


def build_turbine(yaml_path):
    with open(yaml_path) as f:
        d = yaml.safe_load(f)
    pc = d["performance"]["power_curve"]
    ctc = d["performance"]["Ct_curve"]
    return WindTurbine(name=d["name"], diameter=d["rotor_diameter"], hub_height=d["hub_height"],
                        powerCtFunction=PowerCtTabular(ws=pc["power_wind_speeds"], power=pc["power_values"],
                                                       power_unit="W", ct=ctc["Ct_values"]))


def main():
    site = load_climatology_site()
    vx, vy, rx, ry = load_layouts()
    n_v, n_r = len(vx), len(rx)

    rows = []
    for config_name, (v_file, r_file) in TURBINE_CONFIGS.items():
        v_turbine = build_turbine(DATA_DIR / v_file)
        r_turbine = build_turbine(DATA_DIR / r_file)
        turbines = WindTurbines.from_WindTurbine_lst([v_turbine, r_turbine])

        for model_name, (deficit_model, superposition) in WAKE_MODELS.items():
            wfm = PropagateDownwind(site, turbines, wake_deficitModel=deficit_model,
                                    superpositionModel=superposition, turbulenceModel=CrespoHernandez())

            isolated = wfm(vx, vy, type=[0] * n_v, wd=SIM_WD, ws=SIM_WS).aep().sum().values
            sim = wfm(vx + rx, vy + ry, type=[0] * n_v + [1] * n_r, wd=SIM_WD, ws=SIM_WS)
            with_neighbor = sim.aep().isel(wt=slice(0, n_v)).sum().values

            loss_pct = float((isolated - with_neighbor) / isolated * 100)
            note = ""
            if loss_pct < 0:
                # A pure wake-deficit model (no blockage/ground model here)
                # cannot physically produce an AEP gain -- clip to 0 and
                # flag it as residual discretization noise.
                note = f"  (raw {loss_pct:.4f}%, clipped -- deficit-only model can't yield a gain)"
                loss_pct = 0.0
            rows.append(dict(turbine_config=config_name, wake_model=model_name,
                              isolated_aep_gwh=float(isolated), aep_with_neighbor_gwh=float(with_neighbor),
                              loss_pct=loss_pct))
            print(f"[{config_name} | {model_name}] loss = {loss_pct:.3f}%{note}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "benchmark_wake_turbine.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    plot_benchmark(df)


def plot_benchmark(df):
    models = list(WAKE_MODELS.keys())
    configs = list(TURBINE_CONFIGS.keys())
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, config in enumerate(configs):
        vals = [df[(df.wake_model == m) & (df.turbine_config == config)]["loss_pct"].values[0] for m in models]
        ax.bar(x + (i - 1) * width, vals, width, label=config)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Vineyard AEP loss (%)")
    ax.set_title(f"Vineyard <-> Revolution, {DISTANCE_KM} km apart")
    ax.legend(fontsize=8, title="turbines")
    fig.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "benchmark_wake_turbine.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
