"""Focus farm AEP: isolated vs. real neighbors vs. full cluster."""
from py_wake.wind_turbines import WindTurbines

from config import (
    FOCUS_FARM, SYNTHETIC_TURBINE_SPEC,
    WFM_MODEL, TURBULENCE_MODEL,
    SIM_WD, SIM_WS,
)
from turbines import build_pywake_turbine, load_turbine_file


def build_turbine_types(farms):
    """One PyWake type per real farm + one shared type for all synthetic farms. Year-independent."""
    wt_objects = []
    type_by_farm = {}

    for farm in farms:
        if farm["is_synthetic"]:
            continue
        wt_objects.append(build_pywake_turbine(farm["turbine_node"]))
        type_by_farm[farm["name"]] = len(wt_objects) - 1

    synthetic_turbine = load_turbine_file(SYNTHETIC_TURBINE_SPEC)
    wt_objects.append(synthetic_turbine)
    synthetic_type = len(wt_objects) - 1
    for farm in farms:
        if farm["is_synthetic"]:
            type_by_farm[farm["name"]] = synthetic_type

    return WindTurbines.from_WindTurbine_lst(wt_objects), type_by_farm


def construct_wfm(site, multi_turbines, wfm_builder=None):
    """wfm_builder: optional (site, multi_turbines) -> wfm callable, e.g. one of
    Monte_Carlo/wake_models.py's WFM_BUILDERS, to run under a specific literature model
    instead of the config.py default. Existing callers that don't pass it are unaffected."""
    if wfm_builder is not None:
        return wfm_builder(site, multi_turbines)
    return WFM_MODEL(site, multi_turbines, turbulenceModel=TURBULENCE_MODEL())


def _combined_arrays(farms, type_by_farm, focus_farm_name):
    """Focus farm first, so its AEP can be sliced back out with isel(wt=slice(0, n))."""
    focus = next(f for f in farms if f["name"] == focus_farm_name)
    others = [f for f in farms if f["name"] != focus_farm_name and f["n_turbines"] > 0]

    x_full = list(focus["x"])
    y_full = list(focus["y"])
    type_full = [type_by_farm[focus_farm_name]] * focus["n_turbines"]
    for farm in others:
        x_full += list(farm["x"])
        y_full += list(farm["y"])
        type_full += [type_by_farm[farm["name"]]] * farm["n_turbines"]

    return focus, x_full, y_full, type_full


def compute_isolated_aep(wfm, focus, focus_type, n_cpu=1):
    sim = wfm(list(focus["x"]), list(focus["y"]),
              type=[focus_type] * focus["n_turbines"], wd=SIM_WD, ws=SIM_WS, n_cpu=n_cpu)
    return float(sim.aep().sum().values)


def compute_neighbor_aep(wfm, farms, type_by_farm, focus_farm_name, n_cpu=1):
    focus, x_full, y_full, type_full = _combined_arrays(farms, type_by_farm, focus_farm_name)
    sim = wfm(x_full, y_full, type=type_full, wd=SIM_WD, ws=SIM_WS, n_cpu=n_cpu)
    focus_aep = float(sim.aep().isel(wt=slice(0, focus["n_turbines"])).sum().values)
    return focus_aep, x_full, y_full, type_full


def run_scenarios(site, farms, multi_turbines, type_by_farm, focus_farm_name=FOCUS_FARM, wfm_builder=None, n_cpu=1):
    """Isolated, real-neighbors, and full-cluster AEP for one year's site.

    n_cpu: PyWake's own flow-case parallelism WITHIN each of these 3 calls (default 1,
    matching every existing caller's prior behavior). For the Monte Carlo, leave at 1
    and parallelize ACROSS trials instead -- see Monte_Carlo_HPC/PROFILE_RESULTS.md for
    why nested flow-case parallelism loses to embarrassingly-parallel trials here."""
    wfm = construct_wfm(site, multi_turbines, wfm_builder=wfm_builder)
    focus = next(f for f in farms if f["name"] == focus_farm_name)
    focus_type = type_by_farm[focus_farm_name]

    isolated_aep = compute_isolated_aep(wfm, focus, focus_type, n_cpu=n_cpu)

    real_farms = [f for f in farms if not f["is_synthetic"]]
    real_aep, _, _, _ = compute_neighbor_aep(wfm, real_farms, type_by_farm, focus_farm_name, n_cpu=n_cpu)

    full_aep, x_full, y_full, type_full = compute_neighbor_aep(wfm, farms, type_by_farm, focus_farm_name, n_cpu=n_cpu)

    return dict(
        focus_farm=focus_farm_name,
        isolated_aep_gwh=isolated_aep,
        real_neighbors_aep_gwh=real_aep,
        real_neighbors_loss_pct=(isolated_aep - real_aep) / isolated_aep * 100,
        n_real_turbines=sum(f["n_turbines"] for f in real_farms),
        full_cluster_aep_gwh=full_aep,
        full_cluster_loss_pct=(isolated_aep - full_aep) / isolated_aep * 100,
        n_full_turbines=len(x_full),
        wfm=wfm,
        x_full=x_full, y_full=y_full, type_full=type_full,
    )
