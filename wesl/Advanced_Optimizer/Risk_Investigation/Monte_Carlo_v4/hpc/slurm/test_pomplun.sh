#!/bin/bash
# One-off sanity check for the pomplun partition (chimera21, cs672 class account --
# authorized by Bruno for this project 2026-08-20, see hpc/README.md). Two things this
# checks that test_partitions.sh didn't need to for the other 4 partitions:
#   1. access actually works with --account=cs672 --qos=pomplun (both required --
#      AllowAccounts=cs / AllowQos=pomplun per `scontrol show partition pomplun`)
#   2. REAL per-trial timing on this hardware -- not just "does it import py_wake".
#      Lesson learned 2026-08-20 production run: "accepts jobs" != "same per-core speed
#      as EPYC9565" (84 Intel-partition tasks timed out from exactly this gap). Runs one
#      actual orchestrator trial (seed=0, same scenario as every other timing number in
#      TRIAL_RECORD.md) so the result is directly comparable.
#
# Usage (run from Monte_Carlo_v3/hpc/):
#   sbatch slurm/test_pomplun.sh

#SBATCH --job-name=test_pomplun
#SBATCH --account=cs672
#SBATCH --qos=pomplun
#SBATCH --partition=pomplun
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_out/test_pomplun_%j.out
#SBATCH --error=slurm_out/test_pomplun_%j.err

set -euo pipefail

# sbatch stages this script in a spool dir -- use $SLURM_SUBMIT_DIR, not $BASH_SOURCE,
# to find the project folder (same gotcha submit_array.sh already works around).
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    HPC_DIR="${SLURM_SUBMIT_DIR}"
else
    HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${HPC_DIR}"
mkdir -p slurm_out
source slurm/env_setup.sh

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

echo "partition=pomplun node=$(hostname) date=$(date)"
python -c "import sys, py_wake; print('python', sys.version.split()[0]); print('py_wake', py_wake.__version__)"

cd ..
python -c "
import time, numpy as np
import aep, cluster, layout, climate, grid_fill, turbines, settings, wake_models

def tile(f):
    if 'x' in f: return f
    pts = grid_fill.fill(f['polygon'], turbines.diameter(f['turbine']), f['spacing_d'])
    f['x'], f['y'] = pts[:, 0], pts[:, 1]
    return f

rng = np.random.default_rng(0)
farms = [tile(f) for f in cluster.generate_cluster(rng) + layout.generate_scenario(rng)]
site, _ = climate.sample_site(rng)
active = farms  # full seed=0 scenario, largest/all-farms case -- same as every other benchmark in this repo

t0 = time.time()
r = aep.evaluate(site, active, cluster.FOCUS_FARM, model_name='supergaussian', n_cpu=1)
dt = time.time() - t0
print(f'RESULT: full-scenario evaluate() on pomplun: {dt:.1f}s (active_aep={r[\"active_aep_gwh\"]:.1f} GWh)')
print('Reference: same call on EPYC9565 took 20.64s (see aep.py evaluate_background_points docstring / TRIAL_RECORD.md)')
"
