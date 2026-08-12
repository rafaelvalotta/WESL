"""Final, decisive test: same total core budget (4), two ways to spend it, realistic
trial size. A: 4 processes x n_cpu=1 each (embarrassingly parallel across trials,
no internal PyWake parallelism). B: 2 processes x n_cpu=2 each (a mix). Compared
against C, derived analytically from profile_v3's already-measured n_cpu=4 single-call
time (8.3s) x N trials serially -- also spends exactly 4 cores, just one trial at a
time instead of several concurrently.
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
from aep_simulation import _combined_arrays

N_TRIALS = 8
CORE_BUDGET = 4


def _worker(args):
    seed, n_cpu = args
    t0 = time.time()
    rng = np.random.default_rng(seed)
    site = climate.sample_synthetic_site(rng, scale=1.0)
    _, _, cluster_farms, new_farms = layout.populate_scenario("high", rng)
    aep_farms = scenario_bridge.build_aep_farms(cluster_farms, new_farms)
    multi_turbines, type_by_farm = scenario_bridge.build_turbine_types(aep_farms)
    _, x_full, y_full, type_full = _combined_arrays(aep_farms, type_by_farm, FOCUS_FARM)
    wfm = wake_models.WFM_BUILDERS["Nygaard_TurboGaussian"](site, multi_turbines)
    sim = wfm(x_full, y_full, type=type_full, wd=SIM_WD, ws=SIM_WS, n_cpu=n_cpu)
    aep = float(sim.aep().sum().values)
    return time.time() - t0, len(x_full)


if __name__ == "__main__":
    seeds = list(range(300, 300 + N_TRIALS))

    print(f"=== Config A: {CORE_BUDGET} processes x n_cpu=1 (embarrassingly parallel across trials) ===")
    t0 = time.time()
    with mp.Pool(CORE_BUDGET) as pool:
        results_a = pool.map(_worker, [(s, 1) for s in seeds])
    wall_a = time.time() - t0
    print(f"{N_TRIALS} trials, wall={wall_a:.1f}s ({wall_a/N_TRIALS:.1f}s/trial average throughput)")

    n_procs_b = CORE_BUDGET // 2
    print(f"\n=== Config B: {n_procs_b} processes x n_cpu=2 (mixed) ===")
    t0 = time.time()
    with mp.Pool(n_procs_b) as pool:
        results_b = pool.map(_worker, [(s, 2) for s in seeds])
    wall_b = time.time() - t0
    print(f"{N_TRIALS} trials, wall={wall_b:.1f}s ({wall_b/N_TRIALS:.1f}s/trial average throughput)")

    # Config C derived: 1 process x n_cpu=4, serial (from profile_v3: 8.3s/trial at n_cpu=4)
    wall_c_estimated = 8.3 * N_TRIALS
    print(f"\n=== Config C (from profile_v3 data, not rerun): 1 process x n_cpu=4, serial ===")
    print(f"{N_TRIALS} trials, wall~={wall_c_estimated:.1f}s (8.3s/trial x {N_TRIALS}, same 4-core budget)")

    print(f"\n=== Summary, {CORE_BUDGET} cores spent either way ===")
    print(f"A (4x n_cpu=1):  {wall_a:.1f}s")
    print(f"B (2x n_cpu=2):  {wall_b:.1f}s")
    print(f"C (1x n_cpu=4):  {wall_c_estimated:.1f}s")
    best = min([("A", wall_a), ("B", wall_b), ("C", wall_c_estimated)], key=lambda kv: kv[1])
    print(f"Best: {best[0]}")
