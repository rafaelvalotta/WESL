#!/bin/bash
# Chimera array job, v4. Partition/env confirmed from Bruno's prior working SLURM
# script -- see ../README.md.
#
# --mem=64G / --time=06:00:00 below are CARRIED OVER FROM v3's cost model, NOT
# re-validated against v4's redesigned pipeline (density-first fill, 5yr-checkpoint
# segments, 3-knob climate -- see ../../SCENARIOS.md). Local single-machine timing
# post-redesign (2026-08-27, not a valid proxy for cluster scaling, but a real data
# point): scenario_1 ~11-21s/trial, scenario_2 ~57-83s/trial (starts from a bigger
# fixed baseline: whole 13-lease cluster vs. scenario_1's 5) -- both far below v3's
# 133.8-534s/trial that justified the current defaults. Turbine counts (the memory
# driver) are similar order of magnitude to before (~2-4k), so --mem likely still
# needed as-is; --time is probably now generous, not tight -- but RUN A REAL v4
# SHAKEDOWN (per each scenario, they cost differently) before trusting either number
# for a full production array, exactly like v3's own shakedown-first lesson below.
#
# --n-cpu is hardcoded to 1 below, NOT $SLURM_CPUS_PER_TASK -- unlike v2's entry point
# (see run_trial_range.py's --n-cpu help text for why: v3's aep.py has its OWN
# established convention of n_cpu=1, PyWake's flow-case parallelism deliberately left
# off; v2's "safe to raise, it's a plain OS process not a Pool worker" reasoning does
# NOT transfer here). CONFIRMED 2026-08-20 on Chimera: raising it to
# $SLURM_CPUS_PER_TASK=4 deadlocked every task in the shakedown array (all worker
# processes alive, 'Sl' sleep state, near-zero %CPU, stuck 18+ minutes, 124/144 cores
# on the node sitting idle the whole time) -- classic post-fork BLAS/multiprocessing
# lock-inheritance hazard, made worse by OMP_NUM_THREADS/MKL_NUM_THREADS also being set
# to 4 in the same process. `--cpus-per-task=4` below is kept anyway -- with --n-cpu=1
# there's no PyWake-level fork, so BLAS's own internal multi-threading (still governed
# by OMP_NUM_THREADS etc.) can safely use the extra cores within the single process.
#
# v3 trials are much cheaper per call than v2's (one wake-model call per segment, not
# k arrivals x 3 models) -- but each trial still runs ~18 segments through PyWake, so
# still run a small shakedown array first (--array=0-4, trials_per_task=2) and read the
# real wall-clock time in slurm_out/ before sizing a full production array. See
# TRIAL_RECORD.md / PIPELINE.md for local single-core timing (not a valid proxy for
# array-job scaling on dedicated cores -- see Open Items there).
#
# Model: defaults to wake_models.ACTIVE_MODEL (supergaussian as of 2026-08-20, chosen for
# campaign throughput -- a turbopark trial costs ~4x a supergaussian one, see
# ../TRIAL_RECORD.md). turbopark still works fine here (aep.evaluate() is the all-merged
# path, model-agnostic) -- just size the array for the higher per-trial cost if you use it.
#
# Usage (run from Monte_Carlo_v4/hpc/, NOT from inside slurm/):
#   sbatch --array=0-19 slurm/submit_array.sh <run_id> <scenario> [trials_per_task=5] [model=supergaussian]
#   <scenario> is required -- scenario_1 (open buildout) or scenario_2 (locked cluster),
#   see ../../SCENARIOS.md. No default: an accidental wrong-scenario production campaign
#   is expensive enough (hours of Chimera time) that this should never silently fall back.
#
# --mem=64G (was 32G, was 8G originally): 32G still wasn't enough -- the production_10k
# campaign's Round 2 hit 60 OUT_OF_MEMORY tasks right at the 32G ceiling (sacct MaxRSS
# ~33553xxxK, i.e. capped exactly at the limit, so the real requirement was never
# observed, only that it's >32G) and needed a Round 3 repair pass at --mem=64G that was
# never actually launched (see ../CAMPAIGN_LOG.md). Raised to 64G here as the default
# going forward instead of repeating that repair cycle every campaign. Root cause
# unchanged: aep.evaluate() (merged) builds a pairwise turbine-i x turbine-j deficit
# array per (wd,ws) -- O(n_turbines^2 x 12 x 23 x 8 bytes); at ~2000 turbines (the
# biggest seed=0 segment) that's already ~8.8GB for that one array alone, and N_RANGE
# draws up to 16 speculative farms (more than seed=0's 9), so bigger seeds can need much
# more. Node has 773GB RAM, nearly idle -- no reason to be stingy. Re-tune down only
# after a shakedown with real MaxRSS numbers across many seeds (sacct --format=MaxRSS).
#
# --time=06:00:00 (was 02:00:00): sized for trials_per_task=10 (needed to fit the
# 10,000-trial production array under MaxArraySize=1001 -- see PIPELINE.md/TRIAL_RECORD.md
# for per-trial timing). 2h was carried over unchanged from the original trials_per_task=2
# shakedown default and NOT rescaled when trials_per_task went up 5x -- found the hard
# way mid-production-run: with real seed variance (N_RANGE draws up to 16 farms), some
# 10-trial tasks reached 1:48-1:49 elapsed, close enough to the old 2h cap to risk a
# walltime kill (which, pre-incremental-save-fix in run_trial_range.py, would have lost
# the WHOLE task's trials, not just the slow one). If you raise trials_per_task further
# than 10, raise this proportionally too (`sbatch --time=HH:MM:SS ...` overrides this
# default the same way `--partition=`/`--array=` already do on the command line).

#SBATCH --job-name=wesl_mc_v4
#SBATCH --partition=EPYC9565
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=slurm_out/mc_%A_%a.out
#SBATCH --error=slurm_out/mc_%A_%a.err

set -euo pipefail

RUN_ID="${1:?usage: sbatch --array=0-N slurm/submit_array.sh <run_id> <scenario> [trials_per_task] [model]}"
SCENARIO="${2:?usage: sbatch --array=0-N slurm/submit_array.sh <run_id> <scenario> [trials_per_task] [model] -- scenario_1 or scenario_2}"
TRIALS_PER_TASK="${3:-5}"
MODEL="${4:-supergaussian}"

# sbatch stages this script in a spool dir -- use $SLURM_SUBMIT_DIR, not $BASH_SOURCE,
# to find the project folder (same gotcha v1/v2 hit -- see Monte_Carlo_HPC/slurm/submit_array.sh).
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    HPC_DIR="${SLURM_SUBMIT_DIR}"
else
    HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
SCRIPT_DIR="${HPC_DIR}/slurm"

mkdir -p "${HPC_DIR}/slurm_out"
source "${SCRIPT_DIR}/env_setup.sh"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
# Unbuffered stdout -- otherwise Python fully buffers output redirected to a file (our
# .out logs), so `tail -f`/live progress checks can sit showing nothing for many minutes
# even though the task is progressing normally (seen 2026-08-20: a 12+min-silent task
# turned out to just be a large seed draw, not stuck -- this would have shown live
# "seed=N done" lines instead of a blank file).
export PYTHONUNBUFFERED=1

cd "${HPC_DIR}"

python run_trial_range.py \
    --run-id "${RUN_ID}" \
    --scenario "${SCENARIO}" \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --trials-per-task "${TRIALS_PER_TASK}" \
    --seed-start 0 \
    --model "${MODEL}" \
    --n-cpu 1
