from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results"


class Logger:
    def __init__(self):
        self.trials = []
        self.arrivals = []
        self.segments = []

    def log_trial(self, **row):
        self.trials.append(row)

    def log_arrival(self, **row):
        self.arrivals.append(row)

    def log_segment(self, **row):
        self.segments.append(row)

    def save(self, run_name, fmt="csv"):
        # "csv" for single-machine runs (human-readable). "parquet" for HPC batch
        # runs, where hpc/merge_results.py later combines every task's files.
        out = RESULTS_DIR / run_name
        out.mkdir(parents=True, exist_ok=True)
        for name, rows in [("trials", self.trials), ("arrivals", self.arrivals), ("segments", self.segments)]:
            df = pd.DataFrame(rows)
            if fmt == "csv":
                df.to_csv(out / f"{name}.csv", index=False)
            elif fmt == "parquet":
                df.to_parquet(out / f"{name}.parquet", index=False)
            else:
                raise ValueError(f"unknown fmt {fmt!r}, expected 'csv' or 'parquet'")
        return out
