"""Logging for the v2 pipeline -- same one-run-id-per-invocation, append-per-row design
as v1 (see Monte_Carlo/instrumentation.py's docstring for the full rationale: absolute
timestamps at every stage, append instead of batch-in-memory so a crash loses at most
one trial). Five tables now instead of four:

  trials.csv               -- 1 row/trial, wide summary
  arrivals.csv              -- 1 row/arrival ATTEMPT (replaces layout_attempts.csv --
                                triggered by a Poisson arrival event now, not a budget item)
  segments.csv              -- 1 row/(trial, wake_model, segment) -- the primary artifact
                                of the time axis, PIPELINE_DESIGN_v2.md §4/§8
  aep_summary.csv           -- 1 row/(trial, wake_model), duration-weighted rollup of segments
  turbine_assignments.csv   -- 1 row/farm actually in the scenario, unchanged from v1
"""
import csv
import time
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def new_run_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class StageTimer:
    def __init__(self):
        self.marks = {}
        self.mark("trial_start")

    def mark(self, stage):
        self.marks[stage] = time.time()
        return self.marks[stage]

    def elapsed(self, from_stage, to_stage):
        return self.marks[to_stage] - self.marks[from_stage]

    def iso(self, stage):
        return datetime.fromtimestamp(self.marks[stage]).isoformat(timespec="milliseconds")


class MonteCarloLogger:
    def __init__(self, run_id=None):
        self.run_id = run_id or new_run_id()
        self.run_dir = RESULTS_DIR / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._writers = {}
        self._files = {}
        print(f"[instrumentation] run_id={self.run_id} -> {self.run_dir}")

    def _writer_for(self, table_name, fieldnames):
        if table_name not in self._writers:
            path = self.run_dir / f"{table_name}.csv"
            is_new = not path.exists()
            f = open(path, "a", newline="")
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if is_new:
                w.writeheader()
            self._files[table_name] = f
            self._writers[table_name] = w
        return self._writers[table_name]

    def _write_row(self, table_name, row):
        w = self._writer_for(table_name, list(row.keys()))
        w.writerow(row)
        self._files[table_name].flush()

    def log_trial(self, row):
        self._write_row("trials", row)

    def log_arrivals(self, rows):
        for row in rows:
            self._write_row("arrivals", row)

    def log_segments(self, rows):
        for row in rows:
            self._write_row("segments", row)

    def log_aep_summary(self, row):
        self._write_row("aep_summary", row)

    def log_turbine_assignments(self, rows):
        for row in rows:
            self._write_row("turbine_assignments", row)

    def close(self):
        for f in self._files.values():
            f.close()
