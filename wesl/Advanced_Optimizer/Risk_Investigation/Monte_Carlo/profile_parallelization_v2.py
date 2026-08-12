"""Honest profile, v2 -- fixed to use the actual production path (climate.py's cached
PCA sampler, not a fresh GRIB read) so the parallel-trials test isn't penalized by I/O
that doesn't exist in the real pipeline. Same question as v1: PyWake's native n_cpu
flow-case parallelization vs. running independent trials concurrently.
"""
import multiprocessing as mp
import sys
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_THIS_DIR.parent))

import numpy as np
from config import SIM_WD, SIM_WS, FOCUS_FARM
from cluster_layout import load_cluster
from aep_simulation import build_turbine_types, _combined_arrays
import climate
import wake_models


def build_case(seed=0):
    site = climate.sample_synthetic_site(np.random.default_rng(seed), scale=1.0)
    farms = load_cluster()
    real_farms = [f for f in farms if not f["is_synthetic"]]
    multi_turbines, type_by_farm = build_turbine_types(real_farms)
    _, x_full, y_full, type_full = _combined_arrays(real_farms, type_by_farm, FOCUS_FARM)
    return site, multi_turbines, x_full, y_full, type_full


def run_aep(n_cpu, seed=0):
    site, multi_turbines, x, y, type_full = build_case(seed)
    wfm = wake_models.WFM_BUILDERS["Nygaard_TurboGaussian"](site, multi_turbines)
    t0 = time.time()
    sim = wfm(x, y, type=type_full, wd=SIM_WD, ws=SIM_WS, n_cpu=n_cpu)
    aep = float(sim.aep().sum().values)
    return time.time() - t0, aep


def _worker(seed):
    """One full 'trial' -- climate draw + AEP, single-core, exactly what one MC trial
    on one core looks like in production (minus the layout-generation step, isolated
    out here to profile AEP specifically, as asked)."""
    t0 = time.time()
    site, multi_turbines, x, y, type_full = build_case(seed)
    wfm = wake_models.WFM_BUILDERS["Nygaard_TurboGaussian"](site, multi_turbines)
    sim = wfm(x, y, type=type_full, wd=SIM_WD, ws=SIM_WS, n_cpu=1)
    aep = float(sim.aep().sum().values)
    return time.time() - t0, aep


if __name__ == "__main__":
    print("=== Warm-up (import + first climate draw, excluded from timings below) ===")
    run_aep(1)

    print("\n=== Part 1: PyWake native n_cpu (flow-case parallelization within ONE AEP call) ===")
    for n_cpu in [1, 2, 4, 8, 10]:
        dt, aep = run_aep(n_cpu)
        print(f"n_cpu={n_cpu:2d}: {dt:6.2f}s  (AEP={aep:.0f} GWh)")

    print("\n=== Part 2: N independent single-core trials, run CONCURRENTLY (multiprocessing.Pool) ===")
    dt_single, _ = run_aep(1)
    print(f"(single trial baseline: {dt_single:.2f}s)")
    for n_parallel in [2, 4, 8, 10]:
        t0 = time.time()
        with mp.Pool(n_parallel) as pool:
            results = pool.map(_worker, range(n_parallel))
        wall = time.time() - t0
        serial_equivalent = dt_single * n_parallel
        print(f"{n_parallel:2d} trials in parallel: wall={wall:6.2f}s  "
              f"(vs. {serial_equivalent:.2f}s serial -- speedup {serial_equivalent/wall:.2f}x, "
              f"efficiency {serial_equivalent/wall/n_parallel*100:.0f}%)")
