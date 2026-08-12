"""Logging + profiling for the Monte Carlo pipeline.

Design: one `run_id` per pipeline invocation (timestamped, so runs never clobber each
other), writing FOUR tables under results/<run_id>/, all joinable on `trial_id`:

  trials.csv             -- 1 row/trial, wide summary (what a quick analysis needs)
  aep_results.csv        -- 1 row/(trial, wake_model), long (extensible to a 3rd model)
  layout_attempts.csv    -- 1 row/farm PLACEMENT ATTEMPT (success or not -- see the
                             earlier discussion on why this must include failures)
  turbine_assignments.csv -- 1 row/farm actually in the accepted scenario

Every table is appended to (not batched in memory and written at the end) so a crash
mid-run loses at most the in-flight trial, not the whole campaign.

Reproducibility over storage: the full sampled climate vector (36 numbers) and exact
farm geometries are NOT persisted -- given the same `seed`, `pd_level` and code
version, `orchestrator.run_one_trial` regenerates them exactly (climate and layout
both draw from one continuous rng seeded once per trial). Logs record the INPUTS
(seed, pd_level) and SUMMARY outputs, not full intermediate state -- smaller files,
same information reachable by re-running one seed if ever needed.

Absolute timing: `StageTimer` records a wall-clock timestamp at the *entrance* of
every stage (not just a duration), so a full run's timeline can be reconstructed
after the fact -- e.g. to see if a particular hour of the run was slower, or to
correlate wall-clock drift with trial index over a long unattended run.
"""
import csv
import time
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def new_run_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class StageTimer:
    """Records an absolute wall-clock timestamp every time `.mark(stage)` is called --
    the *entrance* of `stage`. `elapsed(a, b)` reads back the duration between any two
    marks. `.marks` holds everything, in order, for full-timeline reconstruction."""

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

    def log_aep_result(self, row):
        self._write_row("aep_results", row)

    def log_layout_attempts(self, rows):
        for row in rows:
            self._write_row("layout_attempts", row)

    def log_turbine_assignments(self, rows):
        for row in rows:
            self._write_row("turbine_assignments", row)

    def close(self):
        for f in self._files.values():
            f.close()
