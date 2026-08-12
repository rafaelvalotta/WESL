#!/bin/bash
# Chimera array job. Partition/env confirmed from Bruno's prior working SLURM script --
# see ../README.md. n_cpu=4 inside each trial is safe here specifically because a SLURM
# array task is a plain OS process, not a Python multiprocessing.Pool worker (see
# ../../../Monte_Carlo_HPC/PROFILE_RESULTS.md for the crash that ONLY happens in the
# latter case).
#
# v2 trials cost more than v1's -- each one now runs k arrivals x 3 wake models of
# PyWake calls instead of 1 call x 2 models. Run a small shakedown array first
# (--array=0-4, trials_per_task=2) and read the real wall-clock time before sizing a
# full production array.
#
# Usage (run from Monte_Carlo_v2/hpc/, NOT from inside slurm/):
#   sbatch --array=0-19 slurm/submit_array.sh <run_id> <tier> [trials_per_task=5]

#SBATCH --job-name=wesl_mc_v2
#SBATCH --partition=EPYC9565
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=slurm_out/mc_%A_%a.out
#SBATCH --error=slurm_out/mc_%A_%a.err

set -euo pipefail

RUN_ID="${1:?usage: sbatch --array=0-N slurm/submit_array.sh <run_id> <tier> [trials_per_task]}"
TIER="${2:-high}"
TRIALS_PER_TASK="${3:-5}"

# sbatch stages this script in a spool dir -- use $SLURM_SUBMIT_DIR, not $BASH_SOURCE,
# to find the project folder (same gotcha v1 hit -- see Monte_Carlo_HPC/slurm/submit_array.sh).
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

cd "${HPC_DIR}"

python run_trial_range.py \
    --run-id "${RUN_ID}" \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --trials-per-task "${TRIALS_PER_TASK}" \
    --seed-start 0 \
    --tier "${TIER}" \
    --n-cpu "${SLURM_CPUS_PER_TASK}"
