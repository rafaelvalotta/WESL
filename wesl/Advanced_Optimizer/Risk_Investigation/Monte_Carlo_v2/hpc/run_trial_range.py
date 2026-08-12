"""SLURM-array-aware entry point, v2 -- same embarrassingly-parallel-across-trials
pattern as Monte_Carlo_HPC/run_trial_range.py (see ../../Monte_Carlo_HPC/PROFILE_RESULTS.md
for why). Imports the trial logic directly from ../ (this package doesn't duplicate it).

Each array task writes its own log files (results/<run_id>/task_<id>/); merge_results.py
consolidates everything after the array job finishes.

    sbatch slurm/submit_array.sh
    # or run one task by hand for testing:
    python run_trial_range.py --run-id test001 --task-id 0 --trials-per-task 5 --tier high
"""
import argparse
import os
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_MC_V2_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_MC_V2_DIR.parent))  # Risk_Investigation/ -- for aep_simulation
sys.path.insert(0, str(_MC_V2_DIR))         # Monte_Carlo_v2/ -- must win over the line above

from orchestrator import run_one_trial
from instrumentation import MonteCarloLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task-id", type=int, default=None,
                        help="Defaults to $SLURM_ARRAY_TASK_ID if set, else 0.")
    parser.add_argument("--trials-per-task", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--tier", type=str, default="high", choices=["low", "medium", "high"])
    parser.add_argument("--n-cpu", type=int, default=1,
                        help="PyWake flow-case parallelism WITHIN one trial's AEP call -- safe to "
                             "raise here (submit_array.sh sets $SLURM_CPUS_PER_TASK) since this "
                             "entry point is a plain SLURM array task, not a multiprocessing.Pool "
                             "worker. See ../../Monte_Carlo_HPC/PROFILE_RESULTS.md.")
    args = parser.parse_args()

    task_id = args.task_id if args.task_id is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    task_seed_start = args.seed_start + task_id * args.trials_per_task

    logger = MonteCarloLogger(run_id=f"{args.run_id}/task_{task_id:04d}")
    print(f"[task {task_id}] seeds {task_seed_start}..{task_seed_start + args.trials_per_task - 1}, "
          f"tier={args.tier}, n_cpu={args.n_cpu}")

    for i in range(args.trials_per_task):
        seed = task_seed_start + i
        trial_id = f"{args.run_id}_{seed:06d}"
        run_one_trial(trial_id, seed, args.tier, logger, n_cpu=args.n_cpu)

    logger.close()
    print(f"[task {task_id}] done -> {logger.run_dir}")


if __name__ == "__main__":
    main()
