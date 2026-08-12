"""Honest local profile: PyWake's native n_cpu flow-case parallelization vs. running
several independent AEP calls concurrently (multiprocessing across trials) -- to check,
on real hardware, whether the paper's finding (top-level parallelization beats
flow-case parallelization) holds here too before designing the Chimera SLURM strategy.

Test case: Vineyard Wind + its 3 real neighbors (fixed, no stochastic layout draw --
isolates AEP compute time from layout-generation time).

    conda run -n Wind_2200 python profile_parallelization.py
"""
import multiprocessing as mp
import sys
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent))  # Risk_Investigation/

from config import ERA5_DIR, SITE_YEAR, SIM_WD, SIM_WS, FOCUS_FARM
from site_builder import build_site
from cluster_layout import load_cluster
from aep_simulation import build_turbine_types, _combined_arrays
import wake_models
sys.path.insert(0, str(_THIS_DIR))


def build_case():
    site, _ = build_site(ERA5_DIR / f"era5_{SITE_YEAR}.grib")
    farms = load_cluster()
    real_farms = [f for f in farms if not f["is_synthetic"]]
    multi_turbines, type_by_farm = build_turbine_types(real_farms)
    _, x_full, y_full, type_full = _combined_arrays(real_farms, type_by_farm, FOCUS_FARM)
    return site, multi_turbines, x_full, y_full, type_full


def run_aep(n_cpu):
    site, multi_turbines, x, y, type_full = build_case()
    wfm = wake_models.WFM_BUILDERS["Nygaard_TurboGaussian"](site, multi_turbines)
    t0 = time.time()
    sim = wfm(x, y, type=type_full, wd=SIM_WD, ws=SIM_WS, n_cpu=n_cpu)
    aep = float(sim.aep().sum().values)
    return time.time() - t0, aep, len(x)


def _worker(_):
    site, multi_turbines, x, y, type_full = build_case()
    wfm = wake_models.WFM_BUILDERS["Nygaard_TurboGaussian"](site, multi_turbines)
    t0 = time.time()
    sim = wfm(x, y, type=type_full, wd=SIM_WD, ws=SIM_WS, n_cpu=1)
    aep = float(sim.aep().sum().values)
    return time.time() - t0, aep


if __name__ == "__main__":
    print("=== Part 1: PyWake native n_cpu (flow-case parallelization within ONE AEP call) ===")
    for n_cpu in [1, 2, 4, 8, 10]:
        dt, aep, n_turb = run_aep(n_cpu)
        print(f"n_cpu={n_cpu:2d}: {dt:6.2f}s  (AEP={aep:.0f} GWh, {n_turb} turbines)")

    print("\n=== Part 2: N independent single-core AEP calls, run CONCURRENTLY (multiprocessing.Pool) ===")
    dt_single, _, _ = run_aep(1)
    for n_parallel in [2, 4, 8]:
        t0 = time.time()
        with mp.Pool(n_parallel) as pool:
            results = pool.map(_worker, range(n_parallel))
        wall = time.time() - t0
        serial_equivalent = dt_single * n_parallel
        print(f"{n_parallel} trials in parallel: wall={wall:6.2f}s  "
              f"(vs. {serial_equivalent:.2f}s if run one-by-one -- speedup {serial_equivalent/wall:.2f}x, "
              f"efficiency {serial_equivalent/wall/n_parallel*100:.0f}%)")
