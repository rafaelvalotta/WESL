"""Main Monte Carlo loop. Per trial: draw climate, draw layout scenario, bridge to
turbine types, run AEP under BOTH wake models (see wake_models.py for why not a random
pick), log everything (instrumentation.py) with absolute timestamps at every stage
boundary.

    conda run -n Wind_2200 python orchestrator.py --n-trials 10 --pd-level high
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_THIS_DIR.parent))

import climate
import layout
import scenario_bridge
import wake_models
import aep_simulation
from config import FOCUS_FARM
from instrumentation import MonteCarloLogger, StageTimer


def run_one_trial(trial_id, seed, pd_level, logger, climate_scale=1.0, climate_extreme_modes=None, n_cpu=1):
    timer = StageTimer()
    rng = np.random.default_rng(seed)  # single stream: climate, then layout, draw from it in order
                                        # -> the whole trial is exactly reproducible from `seed` alone

    site = climate.sample_synthetic_site(rng, scale=climate_scale, extreme_modes=climate_extreme_modes)
    climate_info = climate.climate_summary(np.random.default_rng(seed), scale=climate_scale,
                                            extreme_modes=climate_extreme_modes)
    timer.mark("climate_done")

    layout_log = []
    cluster_x, cluster_y, cluster_farms, new_farms = layout.populate_scenario(pd_level, rng, log_rows=layout_log)
    timer.mark("layout_done")

    aep_farms = scenario_bridge.build_aep_farms(cluster_farms, new_farms)
    turbine_log = []
    multi_turbines, type_by_farm = scenario_bridge.build_turbine_types(aep_farms, log_rows=turbine_log)
    timer.mark("bridge_done")

    aep_summaries = {}
    for model_name, wfm_builder in wake_models.WFM_BUILDERS.items():
        timer.mark(f"aep_{model_name}_start")
        result = aep_simulation.run_scenarios(
            site, aep_farms, multi_turbines, type_by_farm,
            focus_farm_name=FOCUS_FARM, wfm_builder=wfm_builder, n_cpu=n_cpu,
        )
        timer.mark(f"aep_{model_name}_done")
        dt_aep = timer.elapsed(f"aep_{model_name}_start", f"aep_{model_name}_done")
        aep_summaries[model_name] = result

        logger.log_aep_result(dict(
            trial_id=trial_id, run_id=logger.run_id, seed=seed, wake_model=model_name,
            dt_aep_s=round(dt_aep, 2),
            isolated_aep_gwh=result["isolated_aep_gwh"],
            real_neighbors_aep_gwh=result["real_neighbors_aep_gwh"],
            real_neighbors_loss_pct=round(result["real_neighbors_loss_pct"], 3),
            full_cluster_aep_gwh=result["full_cluster_aep_gwh"],
            full_cluster_loss_pct=round(result["full_cluster_loss_pct"], 3),
        ))

    timer.mark("trial_done")

    n_new_farms = len(new_farms)
    requested_mw = sum(r["target_mw"] for r in layout_log)
    delivered_mw = sum(r["delivered_mw"] for r in layout_log)
    n_turbines_total = sum(f["n_turbines"] for f in aep_farms)

    logger.log_trial(dict(
        trial_id=trial_id, run_id=logger.run_id, seed=seed, pd_level=pd_level,
        t_start_iso=timer.iso("trial_start"), t_end_iso=timer.iso("trial_done"),
        t_climate_done_iso=timer.iso("climate_done"), t_layout_done_iso=timer.iso("layout_done"),
        t_bridge_done_iso=timer.iso("bridge_done"),
        dt_climate_s=round(timer.elapsed("trial_start", "climate_done"), 2),
        dt_layout_s=round(timer.elapsed("climate_done", "layout_done"), 2),
        dt_bridge_s=round(timer.elapsed("layout_done", "bridge_done"), 2),
        dt_aep_total_s=round(timer.elapsed("bridge_done", "trial_done"), 2),
        dt_total_s=round(timer.elapsed("trial_start", "trial_done"), 2),
        n_farms_total=len(aep_farms), n_new_farms=n_new_farms,
        n_turbines_total=n_turbines_total,
        requested_mw=round(requested_mw, 1), delivered_mw=round(delivered_mw, 1),
        delivered_pct=round(delivered_mw / requested_mw * 100, 2) if requested_mw > 0 else None,
        climate_scale=climate_scale, climate_extreme_modes=climate_extreme_modes,
        climate_mean_speed=round(climate_info["overall_mean_speed"], 3),
        climate_dominant_sector=climate_info["dominant_sector"],
        climate_dominant_sector_freq=round(climate_info["dominant_sector_freq"], 4),
    ))

    layout_rows = []
    for row in layout_log:
        band_lo, band_hi = row["band"]
        layout_rows.append(dict(
            trial_id=trial_id, run_id=logger.run_id, seed=seed,
            band_lo_km=band_lo, band_hi_km=band_hi, angle_deg=row["angle"],
            target_mw=row["target_mw"], delivered_mw=row["delivered_mw"],
            fraction_used=row["fraction_used"], n_tries=row["n_tries"],
            success=row["success"], template=row["template"], turbine_attempted=row.get("turbine"),
        ))
    logger.log_layout_attempts(layout_rows)

    turbine_rows = [
        dict(trial_id=trial_id, run_id=logger.run_id, seed=seed, **row)
        for row in turbine_log
    ]
    logger.log_turbine_assignments(turbine_rows)

    print(f"[trial {trial_id}] seed={seed} pd={pd_level} -- {n_new_farms} new farms, "
          f"{delivered_mw:.0f}/{requested_mw:.0f} MW ({delivered_mw/requested_mw*100 if requested_mw else 0:.0f}%), "
          + " ".join(f"{m}={r['full_cluster_loss_pct']:.1f}%" for m, r in
                      ((m, dict(full_cluster_loss_pct=aep_summaries[m]["full_cluster_loss_pct"])) for m in aep_summaries))
          + f" -- {timer.elapsed('trial_start', 'trial_done'):.0f}s")

    return aep_summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--pd-level", type=str, default="high", choices=["low", "medium", "high"])
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--climate-scale", type=float, default=1.0)
    parser.add_argument("--climate-extreme-modes", type=int, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    logger = MonteCarloLogger(run_id=args.run_id)
    t0 = time.time()
    for i in range(args.n_trials):
        seed = args.seed_start + i
        trial_id = f"{logger.run_id}_{seed:06d}"
        run_one_trial(trial_id, seed, args.pd_level, logger,
                      climate_scale=args.climate_scale, climate_extreme_modes=args.climate_extreme_modes)
    logger.close()
    print(f"\nDone: {args.n_trials} trials in {time.time()-t0:.0f}s -> {logger.run_dir}")


if __name__ == "__main__":
    main()
