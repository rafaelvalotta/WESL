# Backfills the missing full-build-out (2050) segment for campaigns that already
# ran before orchestrator.py's fix (segments 0-4 there are correct and untouched --
# this only adds the segment_id=5 row that was never computed).
#
# Reuses orchestrator.generate_world() so a given seed rebuilds the exact same world
# the original trial produced -- same rng, same farms, same climate. Only the AEP
# evaluation at YEAR_END (never run originally) is new work here, not a full re-run.
#
#   sbatch --array=0-999 slurm/submit_backfill_array.sh <scenario> <run_id> <model> [trials_per_task=10]
#   # or run one task by hand for testing:
#   python backfill_final_segment.py --scenario scenario_1 --run-id production_10k_s1_supergaussian --model supergaussian --task-id 0 --trials-per-task 2
import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_MC_V4_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_MC_V4_DIR.parent))  # Risk_Investigation/
sys.path.insert(0, str(_MC_V4_DIR))         # Monte_Carlo_v4/ -- must win over the line above

from orchestrator import generate_world, _checkpoints
import aep
import wake_models
import scenarios

RESULTS_DIR = _MC_V4_DIR / "results"


def backfill_trial(seed, scenario, model_name, n_cpu=1):
    cluster_farms, speculative_farms, site, _climate_scenario, _fixed_capacity_mw = generate_world(seed, scenario)
    all_farms = cluster_farms + speculative_farms  # both already tiled inside generate_world
    final_year = _checkpoints()[-1]
    result = aep.evaluate(site, all_farms, scenario.self_farm, model_name=model_name, n_cpu=n_cpu)
    return dict(trial_id=seed, segment_id=5, start_year=final_year, end_year=final_year,
                duration_years=0, n_active_farms=len(all_farms), **result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=list(scenarios.SCENARIOS))
    parser.add_argument("--run-id", required=True,
                         help="Must match the existing campaign folder being backfilled, "
                              "e.g. production_10k_s1_supergaussian.")
    parser.add_argument("--model", required=True, choices=list(wake_models.MODELS),
                         help="Must match the wake model that run_id campaign used.")
    parser.add_argument("--task-id", type=int, default=None,
                         help="Defaults to $SLURM_ARRAY_TASK_ID if set, else 0.")
    parser.add_argument("--trials-per-task", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-cpu", type=int, default=1)
    args = parser.parse_args()

    scenario = scenarios.SCENARIOS[args.scenario]
    task_id = args.task_id if args.task_id is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    task_seed_start = args.seed_start + task_id * args.trials_per_task

    out_dir = RESULTS_DIR / args.scenario / args.run_id / "backfill_final" / f"task_{task_id:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "segments_final.parquet"

    print(f"[task {task_id}] scenario={scenario.key} model={args.model} run_id={args.run_id} "
          f"seeds {task_seed_start}..{task_seed_start + args.trials_per_task - 1}")

    rows = []
    t0 = time.time()
    for i in range(args.trials_per_task):
        seed = task_seed_start + i
        t_trial = time.time()
        rows.append(backfill_trial(seed, scenario, args.model, n_cpu=args.n_cpu))
        pd.DataFrame(rows).to_parquet(out_path, index=False)  # incremental, same reason as run_trial_range.py
        print(f"[task {task_id}] seed={seed} done in {time.time() - t_trial:.1f}s "
              f"({i + 1}/{args.trials_per_task} saved)")

    print(f"[task {task_id}] {args.trials_per_task} trials, {time.time() - t0:.1f}s total -> {out_path}")


if __name__ == "__main__":
    main()
