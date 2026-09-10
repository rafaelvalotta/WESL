import time

import numpy as np

import settings
import cluster
import layout
import climate
import grid_fill
import turbines
import aep
import wake_models
import scenarios
from instrumentation import Logger


def _tile(farm):
    if "x" in farm:
        return farm
    pts = grid_fill.fill(farm["polygon"], turbines.diameter(farm["turbine"]), farm["spacing_d"])
    farm["x"], farm["y"] = pts[:, 0], pts[:, 1]
    return farm


def _source(farm, scenario):
    if farm["is_self"]:
        return "self"
    if farm["has_real_layout"]:
        return "real_layout"
    if farm["name"] in scenario.fixed_farm_names:
        return "cluster_fixed"
    return "speculative"


def _checkpoints():
    # Fixed year boundaries, SEGMENT_CHECKPOINT_YEARS apart, same for every trial
    # regardless of how many farms/events it draws. Keeps segment_id comparable
    # across the whole batch (segment_id=2 is always ~2036 in every trial).
    years = list(range(settings.YEAR_START, settings.YEAR_END, settings.SEGMENT_CHECKPOINT_YEARS))
    if years[-1] != settings.YEAR_END:
        years.append(settings.YEAR_END)
    return years


def generate_world(seed, scenario):
    # Same rng-seeded steps run_trial() needs before its segment loop, pulled out
    # so a given seed's world can be rebuilt elsewhere (see hpc/backfill_final_segment.py)
    # without duplicating this logic or re-running the whole trial.
    rng = np.random.default_rng(seed)

    # Tile cluster_farms first so we know its capacity before calling
    # layout.generate_scenario -- the 30GW target needs to be this scenario's
    # TOTAL capacity, not speculative on top of a different fixed baseline per
    # scenario. cluster.generate_cluster doesn't draw any random numbers, so
    # doing this first doesn't affect the rest of the trial's rng sequence.
    cluster_farms = [_tile(f) for f in cluster.generate_cluster(rng, scenario)]
    fixed_capacity_mw = sum(len(f["x"]) * aep._rated_mw(f["turbine"]) for f in cluster_farms)
    speculative_farms = layout.generate_scenario(rng, scenario, fixed_capacity_mw=fixed_capacity_mw)
    site, climate_scenario = climate.sample_site(rng)
    return cluster_farms, speculative_farms, site, climate_scenario, fixed_capacity_mw


def run_trial(seed, logger, scenario, model_name=wake_models.ACTIVE_MODEL, n_cpu=1):
    t0 = time.time()
    cluster_farms, speculative_farms, site, climate_scenario, fixed_capacity_mw = generate_world(seed, scenario)
    t_generate = time.time() - t0

    t0 = time.time()
    all_farms = [_tile(f) for f in cluster_farms + speculative_farms]  # both already tiled -- _tile no-ops here
    t_tile = time.time() - t0

    self_farm = next(f for f in all_farms if f["is_self"])

    # Only the 3 climate knobs get logged -- climate_scenario also carries the
    # full derived weight/mu/kappa arrays climate.py needs internally, but those
    # are just derived from the 3 knobs so there's no point saving them per trial.
    climate_log = {
        "climate_rotation_deg": climate_scenario["rotation_deg"],
        "climate_speed_multiplier": climate_scenario["speed_multiplier"],
        "climate_spread_multiplier": climate_scenario["spread_multiplier"],
    }

    logger.log_trial(
        trial_id=seed, seed=seed, scenario=scenario.key, n_speculative=len(speculative_farms),
        fixed_capacity_mw=fixed_capacity_mw,
        self_turbine=self_farm["turbine"], self_spacing_d=self_farm["spacing_d"],
        wake_model=model_name, t_generate_s=t_generate, t_tile_s=t_tile,
        **climate_log,
    )
    for f in all_farms:
        logger.log_arrival(trial_id=seed, farm=f["name"], source=_source(f, scenario), arrival_year=f["arrival_year"],
                            turbine=f["turbine"], spacing_d=f.get("spacing_d"), n_turbines=len(f["x"]),
                            rho_target_mw_km2=f.get("rho_target_mw_km2"), spacing_clipped=f.get("spacing_clipped"))

    boundaries = _checkpoints()
    for seg_id in range(len(boundaries) - 1):
        start, end = boundaries[seg_id], boundaries[seg_id + 1]
        active = [f for f in all_farms if f["arrival_year"] <= start]
        result = aep.evaluate(site, active, scenario.self_farm, model_name=model_name, n_cpu=n_cpu)
        logger.log_segment(trial_id=seed, segment_id=seg_id, start_year=start, end_year=end,
                            duration_years=end - start, n_active_farms=len(active), **result)

    # Final snapshot: full build-out at YEAR_END. The loop above only ever evaluates
    # the state at each segment's START, so YEAR_END itself is never reached as an
    # evaluation point -- every farm's arrival_year is always < YEAR_END (see
    # timeline.draw_arrival_years), so the full build-out is simply all_farms, no filter.
    final_year = boundaries[-1]
    result = aep.evaluate(site, all_farms, scenario.self_farm, model_name=model_name, n_cpu=n_cpu)
    logger.log_segment(trial_id=seed, segment_id=len(boundaries) - 1, start_year=final_year, end_year=final_year,
                        duration_years=0, n_active_farms=len(all_farms), **result)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="smoke_test")
    parser.add_argument("--scenario", default="scenario_1", choices=list(scenarios.SCENARIOS))
    args = parser.parse_args()

    scenario = scenarios.SCENARIOS[args.scenario]
    logger = Logger()
    t0 = time.time()
    run_trial(seed=args.seed, logger=logger, scenario=scenario)
    elapsed = time.time() - t0
    n_farms = len(logger.arrivals)
    out = logger.save(f"{scenario.key}/{args.out}_seed{args.seed}")
    print(f"scenario={scenario.key} seed={args.seed}: {n_farms} farms, {len(logger.segments)} segments, "
          f"{elapsed:.1f}s -- saved to {out}")
