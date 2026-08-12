"""Run this LOCALLY after rsync-ing a Chimera run's results/<run_id>/ folder back
(see Monte_Carlo_HPC/sync_from_chimera.sh). Two jobs:

1. Completeness check -- an overnight unattended array job can have individual tasks
   time out, OOM, or hit a bad seed. This diffs the expected array range against what
   actually produced a trials.csv row, so a gap is caught before it silently biases
   any statistic computed on "whatever happened to finish."
2. A first-look summary (P10/P50/P90 AEP loss per wake model, delivered-MW spread,
   timing) -- the same instinct as multi_year_analysis.py's _summarize, generalized
   to a multi-axis Monte Carlo run instead of a 24-year loop.

    conda run -n Wind_2200 python postprocess_report.py --run-id 20260810_120000 --expected-range 0 199
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent / "Monte_Carlo_HPC"))


def merge_if_needed(run_dir):
    """Handles both shapes: a plain single-machine run (trials.csv directly under
    run_dir, from orchestrator.py) and an HPC array-job run (task_0000/, task_0001/,
    ... subfolders, from Monte_Carlo_HPC/run_trial_range.py) -- merges the latter,
    passes the former through untouched."""
    if (run_dir / "trials.csv").exists():
        return run_dir  # plain single-machine run, nothing to merge

    merged_dir = run_dir / "merged"
    if not merged_dir.exists() or not (merged_dir / "trials.csv").exists():
        print(f"No trials.csv directly under {run_dir}, but task_* subfolders found "
              f"-- this looks like an HPC run, running merge_results.py first...")
        from merge_results import merge
        merge(run_dir.name)
    return merged_dir


def completeness_check(merged_dir, expected_range):
    trials = pd.read_csv(merged_dir / "trials.csv")
    got_seeds = set(trials["seed"].astype(int))
    expected_seeds = set(range(expected_range[0], expected_range[1] + 1))
    missing = sorted(expected_seeds - got_seeds)

    print(f"\n=== Completeness ===")
    print(f"Expected: {len(expected_seeds)} trials (seeds {expected_range[0]}-{expected_range[1]})")
    print(f"Got:      {len(got_seeds)} trials")
    if missing:
        print(f"MISSING {len(missing)} seeds -- check slurm_out/*.err for these task IDs before "
              f"trusting any aggregate stat below: {missing[:20]}{' ...' if len(missing) > 20 else ''}")
    else:
        print("No gaps -- every expected trial produced a result.")
    return trials, missing


def summarize(merged_dir):
    """Labeled q10/q50/q90 -- plain statistical percentiles of the VALUE (10th/50th/90th
    percentile), deliberately NOT called "P10/P50/P90" here. Wind-industry P90 means
    something with the opposite shape (the value exceeded 90% of the time -- the LOW,
    conservative end, i.e. the statistical 10th percentile) -- see aep_exceedance.png
    and plot_aep_exceedance() in analyze_campaign.py for that convention. Using "P90"
    for both would silently mean two different numbers in two different outputs."""
    trials = pd.read_csv(merged_dir / "trials.csv")
    aep = pd.read_csv(merged_dir / "aep_results.csv")

    print(f"\n=== Delivered capacity ({len(trials)} trials) ===")
    print(f"delivered_mw:  q10={trials['delivered_mw'].quantile(.1):.0f}  "
          f"q50={trials['delivered_mw'].median():.0f}  q90={trials['delivered_mw'].quantile(.9):.0f}")
    print(f"delivered_pct: q10={trials['delivered_pct'].quantile(.1):.1f}%  "
          f"q50={trials['delivered_pct'].median():.1f}%  q90={trials['delivered_pct'].quantile(.9):.1f}%")
    print(f"n_new_farms:   q10={trials['n_new_farms'].quantile(.1):.0f}  "
          f"q50={trials['n_new_farms'].median():.0f}  q90={trials['n_new_farms'].quantile(.9):.0f}")

    print(f"\n=== AEP / loss, by wake model ===")
    for model, g in aep.groupby("wake_model"):
        print(f"[{model}]")
        print(f"  full_cluster_aep_gwh:   q10={g['full_cluster_aep_gwh'].quantile(.1):.0f}  "
              f"q50={g['full_cluster_aep_gwh'].median():.0f}  q90={g['full_cluster_aep_gwh'].quantile(.9):.0f}")
        print(f"  full_cluster_loss_pct:  q10={g['full_cluster_loss_pct'].quantile(.1):.2f}%  "
              f"q50={g['full_cluster_loss_pct'].median():.2f}%  q90={g['full_cluster_loss_pct'].quantile(.9):.2f}%")

    print(f"\n=== Timing (sanity, not science) ===")
    print(f"dt_total_s:  mean={trials['dt_total_s'].mean():.1f}s  "
          f"q90={trials['dt_total_s'].quantile(.9):.1f}s  max={trials['dt_total_s'].max():.1f}s")
    print(f"dt_layout_s: mean={trials['dt_layout_s'].mean():.1f}s")
    print(f"dt_aep_total_s: mean={trials['dt_aep_total_s'].mean():.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-range", type=int, nargs=2, metavar=("MIN_SEED", "MAX_SEED"),
                        help="e.g. --expected-range 0 199 for an --array=0-199 job. "
                             "Omit to skip the completeness check.")
    args = parser.parse_args()

    run_dir = _THIS_DIR / "results" / args.run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"{run_dir} not found -- did you rsync it back from Chimera yet? "
                                 f"See Monte_Carlo_HPC/sync_from_chimera.sh")

    merged_dir = merge_if_needed(run_dir)

    if args.expected_range:
        completeness_check(merged_dir, tuple(args.expected_range))

    summarize(merged_dir)


if __name__ == "__main__":
    main()
