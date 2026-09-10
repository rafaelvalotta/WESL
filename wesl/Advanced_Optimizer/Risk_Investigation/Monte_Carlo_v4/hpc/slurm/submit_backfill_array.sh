#!/bin/bash
# Backfill array job: computes the missing segment_id=5 (full build-out, 2050) for
# an already-run campaign. One evaluation per trial, not five, but it's the biggest
# evaluation of the whole trial (more turbines active than any of the original
# segments) -- run a small shakedown array first (--array=0-4) before sizing the
# full 10k-trial job, same as submit_array.sh.
#
# Chimera's MaxArraySize is 1001 (see ../CAMPAIGN_LOG.md) -- max array index is 1000,
# so trials_per_task=10 to cover all 10k seeds in 1000 tasks, same as the original
# campaign. Do NOT raise the array past 0-1000 to compensate for a smaller
# trials_per_task -- sbatch will reject it.
#
# cpus-per-task=1, not 4: shakedown jobs 992651 (cpu=1) vs 992654 (cpu=4), same node
# (chimera11, Intel6248), same trials -- cpu=1 averaged 52.2s/trial, cpu=4 averaged
# 64.4s/trial. More cores made this workload SLOWER, not faster (aep.evaluate() isn't
# BLAS-threaded enough to benefit, extra threads just add contention) -- and cpu=1
# also means ~4x more tasks can run at once under the same CPU budget.
#
# mem=48G, not 64G: no OOM in the shakedown, kept below the original 64G as a margin
# without real MaxRSS numbers for this specific (bigger, final-year) snapshot -- if a
# rare large seed OOMs, treat it like any other repair pass (see ../CAMPAIGN_LOG.md).
#
# Usage (run from Monte_Carlo_v4/hpc/, NOT from inside slurm/):
#   sbatch --array=0-999 slurm/submit_backfill_array.sh <scenario> <run_id> <model> [trials_per_task=10]
#   <scenario>: scenario_1 or scenario_2
#   <run_id>: must match the existing campaign folder, e.g. production_10k_s1_supergaussian
#   <model>: must match the wake model that campaign used

#SBATCH --job-name=wesl_mc_v4_backfill
#SBATCH --partition=EPYC9565
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --output=slurm_out/backfill_%A_%a.out
#SBATCH --error=slurm_out/backfill_%A_%a.err

set -euo pipefail

SCENARIO="${1:?usage: sbatch --array=0-N slurm/submit_backfill_array.sh <scenario> <run_id> <model> [trials_per_task]}"
RUN_ID="${2:?usage: sbatch --array=0-N slurm/submit_backfill_array.sh <scenario> <run_id> <model> [trials_per_task]}"
MODEL="${3:?usage: sbatch --array=0-N slurm/submit_backfill_array.sh <scenario> <run_id> <model> [trials_per_task]}"
TRIALS_PER_TASK="${4:-10}"

# sbatch stages this script in a spool dir -- use $SLURM_SUBMIT_DIR, not $BASH_SOURCE,
# to find the project folder.
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
export PYTHONUNBUFFERED=1

cd "${HPC_DIR}"

python backfill_final_segment.py \
    --scenario "${SCENARIO}" \
    --run-id "${RUN_ID}" \
    --model "${MODEL}" \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --trials-per-task "${TRIALS_PER_TASK}" \
    --seed-start 0 \
    --n-cpu 1
