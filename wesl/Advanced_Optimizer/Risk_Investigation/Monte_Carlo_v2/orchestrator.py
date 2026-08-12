"""Main Monte Carlo loop, v2. Per trial: draw climate, draw a full committed cluster +
self + time-phased speculative neighbors (layout.py + timeline.py), then walk self's
25-year history segment by segment -- recomputing full_cluster AEP once per new arrival,
under all 3 wake models. See PIPELINE_DESIGN_v2.md for the full design.

    conda run -n Wind_2200 python orchestrator.py --n-trials 10 --tier high
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent))  # Risk_Investigation/ -- for aep_simulation
sys.path.insert(0, str(_THIS_DIR))         # this package -- must win over the line above

import settings
import climate
import layout
import timeline
import scenario_bridge
import wake_models
import aep_simulation
from instrumentation import MonteCarloLogger, StageTimer


def run_one_trial(trial_id, seed, tier, logger, horizon_years=settings.HORIZON_YEARS, n_cpu=1):
    timer = StageTimer()
    rng = np.random.default_rng(seed)  # single stream: climate, then layout, same as v1

    site = climate.sample_synthetic_site(rng)
    climate_info = climate.climate_summary(np.random.default_rng(seed))
    timer.mark("climate_done")

    arrival_log = []
    cluster_x, cluster_y, cluster_farms, new_farms = layout.populate_scenario(
        tier, rng, horizon_years=horizon_years, log_rows=arrival_log)
    timer.mark("layout_done")

    # Built once against EVERY farm that could appear this trial, even late arrivals, so
    # turbine-type indices stay stable across segments -- only which (x, y) rows go into
    # a given PyWake call changes per segment, not the type mapping.
    aep_farms_full = scenario_bridge.build_aep_farms(cluster_farms, new_farms)
    turbine_log = []
    multi_turbines, type_by_farm = scenario_bridge.build_turbine_types(aep_farms_full, log_rows=turbine_log)
    timer.mark("bridge_done")

    committed_farms = [f for f in aep_farms_full if not f["name"].startswith("Speculative_")]
    speculative_by_name = {f["name"]: f for f in aep_farms_full if f["name"].startswith("Speculative_")}
    ordered_new = sorted(new_farms, key=lambda f: f["arrival_year"])
    speculative_names_in_order = [f"Speculative_{i:02d}" for i in range(len(new_farms))]
    segments = timeline.build_segments(new_farms, horizon_years)  # same for every wake model
    focus = next(f for f in aep_farms_full if f["name"] == settings.FOCUS_FARM)
    focus_type = type_by_farm[settings.FOCUS_FARM]

    segment_rows_all = []
    for model_name, wfm_builder in wake_models.WFM_BUILDERS.items():
        timer.mark(f"aep_{model_name}_start")
        wfm = aep_simulation.construct_wfm(site, multi_turbines, wfm_builder=wfm_builder)

        isolated_aep = aep_simulation.compute_isolated_aep(wfm, focus, focus_type, n_cpu=n_cpu)
        real_neighbors_aep, _, _, _ = aep_simulation.compute_neighbor_aep(
            wfm, committed_farms, type_by_farm, settings.FOCUS_FARM, n_cpu=n_cpu)

        rows = []
        for seg_idx, seg in enumerate(segments):
            if seg_idx == 0:
                full_aep = real_neighbors_aep  # no speculative farm active yet -- reuse, no extra PyWake call
            else:
                active_names = speculative_names_in_order[:len(seg["active_farms"])]
                active_farms = committed_farms + [speculative_by_name[n] for n in active_names]
                full_aep, _, _, _ = aep_simulation.compute_neighbor_aep(
                    wfm, active_farms, type_by_farm, settings.FOCUS_FARM, n_cpu=n_cpu)
            rows.append(dict(
                trial_id=trial_id, run_id=logger.run_id, seed=seed, tier=tier, wake_model=model_name,
                segment_idx=seg_idx, start_year=round(seg["start_year"], 3), end_year=round(seg["end_year"], 3),
                duration_years=round(seg["duration_years"], 3), n_active_new_farms=len(seg["active_farms"]),
                isolated_aep_gwh=isolated_aep, real_neighbors_aep_gwh=real_neighbors_aep,
                full_cluster_aep_gwh=full_aep,
                full_cluster_loss_pct=round((isolated_aep - full_aep) / isolated_aep * 100, 3),
            ))
        timer.mark(f"aep_{model_name}_done")
        logger.log_segments(rows)
        segment_rows_all.extend(rows)

        time_weighted_aep = sum(r["full_cluster_aep_gwh"] * r["duration_years"] for r in rows) / horizon_years
        logger.log_aep_summary(dict(
            trial_id=trial_id, run_id=logger.run_id, seed=seed, tier=tier, wake_model=model_name,
            isolated_aep_gwh=isolated_aep, real_neighbors_aep_gwh=real_neighbors_aep,
            real_neighbors_loss_pct=round((isolated_aep - real_neighbors_aep) / isolated_aep * 100, 3),
            time_weighted_aep_gwh=round(time_weighted_aep, 2),
            time_weighted_loss_pct=round((isolated_aep - time_weighted_aep) / isolated_aep * 100, 3),
            n_segments=len(rows),
        ))

    timer.mark("trial_done")

    delivered_mw_total = sum(f["capacity_mw"] for f in new_farms)
    logger.log_trial(dict(
        trial_id=trial_id, run_id=logger.run_id, seed=seed, tier=tier,
        lam=settings.TIER_LAMBDA[tier], horizon_years=horizon_years,
        t_start_iso=timer.iso("trial_start"), t_end_iso=timer.iso("trial_done"),
        dt_climate_s=round(timer.elapsed("trial_start", "climate_done"), 2),
        dt_layout_s=round(timer.elapsed("climate_done", "layout_done"), 2),
        dt_bridge_s=round(timer.elapsed("layout_done", "bridge_done"), 2),
        dt_aep_total_s=round(timer.elapsed("bridge_done", "trial_done"), 2),
        dt_total_s=round(timer.elapsed("trial_start", "trial_done"), 2),
        n_farms_committed=len(cluster_farms), n_arrivals_attempted=len(arrival_log),
        n_new_farms_placed=len(new_farms), n_turbines_total=sum(f["n_turbines"] for f in aep_farms_full),
        delivered_mw_total=round(delivered_mw_total, 1),
        climate_mean_speed=round(climate_info["overall_mean_speed"], 3),
        climate_dominant_sector=climate_info["dominant_sector"],
        climate_dominant_sector_freq=round(climate_info["dominant_sector_freq"], 4),
    ))
    logger.log_arrivals([dict(trial_id=trial_id, run_id=logger.run_id, seed=seed, tier=tier, **row)
                          for row in arrival_log])
    logger.log_turbine_assignments([dict(trial_id=trial_id, run_id=logger.run_id, seed=seed, **row)
                                     for row in turbine_log])

    print(f"[trial {trial_id}] seed={seed} tier={tier} -- {len(new_farms)}/{len(arrival_log)} farms placed, "
          f"{delivered_mw_total:.0f} MW, {len(segments)} segments -- "
          f"{timer.elapsed('trial_start', 'trial_done'):.0f}s")

    return segment_rows_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--tier", type=str, default="high", choices=["low", "medium", "high"])
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--horizon-years", type=float, default=settings.HORIZON_YEARS)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--n-cpu", type=int, default=1)
    args = parser.parse_args()

    logger = MonteCarloLogger(run_id=args.run_id)
    t0 = time.time()
    for i in range(args.n_trials):
        seed = args.seed_start + i
        trial_id = f"{logger.run_id}_{seed:06d}"
        run_one_trial(trial_id, seed, args.tier, logger, horizon_years=args.horizon_years, n_cpu=args.n_cpu)
    logger.close()
    print(f"\nDone: {args.n_trials} trials in {time.time()-t0:.0f}s -> {logger.run_dir}")


if __name__ == "__main__":
    main()
