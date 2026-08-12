"""Honest profile, v3 -- realistic trial size (a real populate_scenario draw, ~1000+
turbines, matching actual production trials) and a PERSISTENT worker pool (workers
created once, reused across many trials -- not respawned per trial, which is what a
real multi-trial SLURM job would do) instead of v1/v2's one-shot Pool per test.
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
import climate
import layout
import scenario_bridge
import wake_models
import aep_simulation


def run_one_trial(seed, n_cpu=1):
    rng = np.random.default_rng(seed)
    site = climate.sample_synthetic_site(rng, scale=1.0)
    cluster_x, cluster_y, cluster_farms, new_farms = layout.populate_scenario("high", rng)
    aep_farms = scenario_bridge.build_aep_farms(cluster_farms, new_farms)
    multi_turbines, type_by_farm = scenario_bridge.build_turbine_types(aep_farms)

    from aep_simulation import _combined_arrays
    _, x_full, y_full, type_full = _combined_arrays(aep_farms, type_by_farm, FOCUS_FARM)
    wfm = wake_models.WFM_BUILDERS["Nygaard_TurboGaussian"](site, multi_turbines)

    t_setup_done = time.time()
    sim = wfm(x_full, y_full, type=type_full, wd=SIM_WD, ws=SIM_WS, n_cpu=n_cpu)
    aep = float(sim.aep().sum().values)
    t_aep_done = time.time()

    return dict(seed=seed, n_turbines=len(x_full), aep_gwh=aep,
                t_setup=t_setup_done, t_aep_done=t_aep_done)


def _worker(seed):
    t0 = time.time()
    r = run_one_trial(seed, n_cpu=1)
    return time.time() - t0, r["n_turbines"]


if __name__ == "__main__":
    print("=== Baseline: one real trial, single core, layout+AEP ===")
    t0 = time.time()
    r = run_one_trial(seed=100, n_cpu=1)
    dt = time.time() - t0
    print(f"seed=100: {r['n_turbines']} turbines, {dt:.1f}s total")

    print("\n=== PyWake native n_cpu on that SAME realistic-size scenario ===")
    rng = np.random.default_rng(100)
    site = climate.sample_synthetic_site(rng, scale=1.0)
    _, _, cluster_farms, new_farms = layout.populate_scenario("high", rng)
    aep_farms = scenario_bridge.build_aep_farms(cluster_farms, new_farms)
    multi_turbines, type_by_farm = scenario_bridge.build_turbine_types(aep_farms)
    from aep_simulation import _combined_arrays
    _, x_full, y_full, type_full = _combined_arrays(aep_farms, type_by_farm, FOCUS_FARM)
    wfm = wake_models.WFM_BUILDERS["Nygaard_TurboGaussian"](site, multi_turbines)
    for n_cpu in [1, 2, 4, 8, 10]:
        t0 = time.time()
        sim = wfm(x_full, y_full, type=type_full, wd=SIM_WD, ws=SIM_WS, n_cpu=n_cpu)
        aep = float(sim.aep().sum().values)
        print(f"n_cpu={n_cpu:2d}: {time.time()-t0:6.2f}s ({len(x_full)} turbines)")

    print("\n=== Many trials, PERSISTENT pool (workers created once, reused) vs. serial ===")
    N_TRIALS = 8
    seeds = list(range(200, 200 + N_TRIALS))

    t0 = time.time()
    serial_times = [_worker(s)[0] for s in seeds]
    t_serial = time.time() - t0
    print(f"Serial, {N_TRIALS} trials: {t_serial:.1f}s total ({np.mean(serial_times):.1f}s/trial avg)")

    for n_workers in [2, 4]:
        t0 = time.time()
        with mp.Pool(n_workers) as pool:  # workers persist for the whole pool.map call
            results = pool.map(_worker, seeds)
        t_parallel = time.time() - t0
        print(f"{n_workers} persistent workers, {N_TRIALS} trials: {t_parallel:.1f}s total -- "
              f"speedup {t_serial/t_parallel:.2f}x, efficiency {t_serial/t_parallel/n_workers*100:.0f}%")
