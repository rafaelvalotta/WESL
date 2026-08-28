#!/bin/bash
# One-off sanity check, NOT part of the production pipeline: confirms each partition
# actually accepts our jobs and the Advanced_env / py_wake import works there, before
# committing a big production array to hardware we've never actually run on
# (EPYC9565 is the only partition validated so far -- see ../TRIAL_RECORD.md).
#
# Submits ONE tiny job per partition explicitly (not a comma-list array) so each
# partition is individually proven, not just "whichever had a free slot first".
#
# Usage (run from Monte_Carlo_v3/hpc/):
#   bash slurm/test_partitions.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."   # Monte_Carlo_v3/hpc/
mkdir -p slurm_out

for PART in EPYC9565 Intel6126 Intel6240 Intel6248; do
    sbatch --job-name="test_${PART}" \
           --partition="${PART}" \
           --time=00:05:00 \
           --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G \
           --output="slurm_out/test_${PART}_%j.out" \
           --error="slurm_out/test_${PART}_%j.err" \
           --wrap="source slurm/env_setup.sh && echo \"partition=${PART} node=\$(hostname) date=\$(date)\" && python -c \"import sys, py_wake; print('python', sys.version.split()[0]); print('py_wake', py_wake.__version__); print('OK')\""
done

echo "4 test jobs submitted (one per partition). Check with: squeue -u \$USER"
echo "Read results once done with: cat slurm_out/test_*.out"
echo "Check for failures with:     cat slurm_out/test_*.err"
