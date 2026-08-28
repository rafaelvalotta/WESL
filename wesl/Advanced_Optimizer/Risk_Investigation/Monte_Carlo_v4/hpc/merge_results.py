# After a SLURM array job finishes, concatenate every task's parquet files into
# one merged file per table.
#
#   python merge_results.py --scenario scenario_1 --run-id 20260820_120000
import argparse

import pandas as pd

from pathlib import Path

_MC_V4_RESULTS = Path(__file__).resolve().parent.parent / "results"

TABLES = ["trials", "arrivals", "segments"]


def merge(scenario, run_id):
    run_root = _MC_V4_RESULTS / scenario / run_id
    task_dirs = sorted(run_root.glob("task_*"))
    if not task_dirs:
        raise FileNotFoundError(f"No task_* directories under {run_root}")

    out_dir = run_root / "merged"
    out_dir.mkdir(exist_ok=True)

    for table in TABLES:
        frames = []
        for task_dir in task_dirs:
            path = task_dir / f"{table}.parquet"
            if path.exists():
                frames.append(pd.read_parquet(path))

        if not frames:
            print(f"{table}: no files found across {len(task_dirs)} tasks, skipping")
            continue

        df = pd.concat(frames, ignore_index=True)
        out_path = out_dir / f"{table}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"{table}: {len(df)} rows from {len(task_dirs)} tasks -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=["scenario_1", "scenario_2"])
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    merge(args.scenario, args.run_id)
