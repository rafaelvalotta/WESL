# After the backfill array job finishes: concatenate every task's segments_final.parquet,
# then append those rows onto the campaign's existing merged/segments.parquet (segments
# 0-4 untouched, this only adds segment_id=5). Writes to a NEW file, segments_with_final.parquet,
# instead of overwriting the original -- check it, then rename it over segments.parquet yourself.
#
#   python merge_backfill_final_segment.py --scenario scenario_1 --run-id production_10k_s1_supergaussian
import argparse
from pathlib import Path

import pandas as pd

_MC_V4_RESULTS = Path(__file__).resolve().parent.parent / "results"


def merge(scenario, run_id):
    campaign_dir = _MC_V4_RESULTS / scenario / run_id
    backfill_root = campaign_dir / "backfill_final"
    task_dirs = sorted(backfill_root.glob("task_*"))
    if not task_dirs:
        raise FileNotFoundError(f"No task_* directories under {backfill_root}")

    frames = [pd.read_parquet(d / "segments_final.parquet") for d in task_dirs
              if (d / "segments_final.parquet").exists()]
    final_rows = pd.concat(frames, ignore_index=True)
    print(f"backfill: {len(final_rows)} rows from {len(task_dirs)} tasks")

    existing_path = campaign_dir / "merged" / "segments.parquet"
    existing = pd.read_parquet(existing_path)
    if (existing.segment_id == 5).any():
        raise ValueError(f"{existing_path} already has a segment_id=5 -- already backfilled?")

    combined = pd.concat([existing, final_rows], ignore_index=True).sort_values(["trial_id", "segment_id"])
    out_path = campaign_dir / "merged" / "segments_with_final.parquet"
    combined.to_parquet(out_path, index=False)
    print(f"{existing_path}: {len(existing)} rows -> {out_path}: {len(combined)} rows")
    print(f"Check it, then replace segments.parquet with it once you're happy: "
          f"mv {out_path} {existing_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=["scenario_1", "scenario_2"])
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    merge(args.scenario, args.run_id)
