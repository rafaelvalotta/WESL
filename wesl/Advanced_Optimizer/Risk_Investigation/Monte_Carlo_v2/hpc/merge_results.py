"""After a SLURM array job finishes: concatenate every task's per-task CSVs
(../results/<run_id>/task_XXXX/*.csv) into one consolidated set, same schema as a
single-machine run.

    python merge_results.py --run-id 20260810_120000
"""
import argparse
import csv
from pathlib import Path

_MC_V2_RESULTS = Path(__file__).resolve().parent.parent / "results"

TABLES = ["trials", "arrivals", "segments", "aep_summary", "turbine_assignments"]


def merge(run_id):
    run_root = _MC_V2_RESULTS / run_id
    task_dirs = sorted(run_root.glob("task_*"))
    if not task_dirs:
        raise FileNotFoundError(f"No task_* directories under {run_root}")

    out_dir = run_root / "merged"
    out_dir.mkdir(exist_ok=True)

    for table in TABLES:
        rows, fieldnames = [], None
        for task_dir in task_dirs:
            path = task_dir / f"{table}.csv"
            if not path.exists():
                continue
            with open(path) as f:
                reader = csv.DictReader(f)
                if fieldnames is None:
                    fieldnames = reader.fieldnames
                rows.extend(reader)

        if fieldnames is None:
            print(f"{table}: no files found across {len(task_dirs)} tasks, skipping")
            continue

        out_path = out_dir / f"{table}.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"{table}: {len(rows)} rows from {len(task_dirs)} tasks -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    merge(args.run_id)
