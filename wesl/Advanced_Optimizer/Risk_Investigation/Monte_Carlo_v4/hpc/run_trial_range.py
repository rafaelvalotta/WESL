# SLURM-array entry point: each task runs a range of trials and writes its own
# results (results/<scenario>/<run_id>/task_<id>/*.parquet); merge_results.py
# combines everything after the array job finishes. Saves after every trial, not
# just at the end, so a walltime kill doesn't lose a task's already-finished trials.
#
#   sbatch slurm/submit_array.sh <run_id> <scenario> [trials_per_task=5] [model=supergaussian]
#   # or run one task by hand for testing:
#   python run_trial_range.py --run-id test001 --scenario scenario_1 --task-id 0 --trials-per-task 2
import argparse
import os
import sys
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_MC_V4_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_MC_V4_DIR.parent))  # Risk_Investigation/
sys.path.insert(0, str(_MC_V4_DIR))         # Monte_Carlo_v4/ -- must win over the line above

from orchestrator import run_trial
from instrumentation import Logger
import wake_models
import scenarios


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scenario", required=True, choices=list(scenarios.SCENARIOS),
                        help="scenario_1 (open buildout) or scenario_2 (locked cluster) -- see SCENARIOS.md.")
    parser.add_argument("--task-id", type=int, default=None,
                        help="Defaults to $SLURM_ARRAY_TASK_ID if set, else 0.")
    parser.add_argument("--trials-per-task", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--model", type=str, default=wake_models.ACTIVE_MODEL, choices=list(wake_models.MODELS))
    parser.add_argument("--n-cpu", type=int, default=1,
                        help="PyWake parallelism within one trial's AEP call. KEEP AT 1 -- raising it "
                             "deadlocked every task on Chimera (BLAS/multiprocessing lock issue after "
                             "forking). See hpc/README.md. Parallelism should happen across trials, "
                             "not inside one.")
    args = parser.parse_args()

    if args.model == "turbopark":
        print("[warning] model='turbopark' works (aep.evaluate() is the all-merged path, supports "
              "any model) but costs ~4x a supergaussian trial (534.1s vs 133.8s, same seed=0 "
              "scenario -- see ../TRIAL_RECORD.md). Not disallowed, just expensive -- size the "
              "array accordingly.")

    scenario = scenarios.SCENARIOS[args.scenario]
    task_id = args.task_id if args.task_id is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    task_seed_start = args.seed_start + task_id * args.trials_per_task

    logger = Logger()
    print(f"[task {task_id}] scenario={scenario.key} seeds {task_seed_start}..{task_seed_start + args.trials_per_task - 1}, "
          f"model={args.model}, n_cpu={args.n_cpu}")

    out_name = f"{scenario.key}/{args.run_id}/task_{task_id:04d}"
    t0 = time.time()
    for i in range(args.trials_per_task):
        seed = task_seed_start + i
        t_trial = time.time()
        run_trial(seed=seed, logger=logger, scenario=scenario, model_name=args.model, n_cpu=args.n_cpu)
        out = logger.save(out_name, fmt="parquet")  # saved after every trial, see top of file
        print(f"[task {task_id}] seed={seed} done in {time.time() - t_trial:.1f}s "
              f"({i + 1}/{args.trials_per_task} saved)")

    print(f"[task {task_id}] {args.trials_per_task} trials, {time.time() - t0:.1f}s total -> {out}")


if __name__ == "__main__":
    main()
