#!/bin/bash
# Run LOCALLY to push the whole Advanced_Optimizer tree (code + climate cache, incl.
# both Monte_Carlo_v3 -- frozen reference -- and Monte_Carlo_v4 -- active) to Chimera.
# Excludes raw ERA5 GRIB (climate.py only reads the cached .npz), notebooks/figures/
# results, __pycache__. Same layout convention as Monte_Carlo_v2/hpc/ sync_to_chimera.sh
# -- hpc/ sits one level deeper inside Monte_Carlo_v4/, so results always land under
# Monte_Carlo_v4/results/<scenario>/ regardless of whether a run was launched via
# orchestrator.py directly or via hpc/run_trial_range.py.
CHIMERA_HOST="bruno.boer001@chimerahead.umb.edu"
CHIMERA_DEST="/home/bruno.boer001/Summer_26/August/Advanced_Optimizer/"

set -euo pipefail
LOCAL_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/"   # .../Advanced_Optimizer/

echo "Syncing ${LOCAL_SRC} -> ${CHIMERA_HOST}:${CHIMERA_DEST}"
rsync -avz --progress \
    --exclude='Project_Monte_Carlo/era5_vineyard/' \
    --exclude='__pycache__/' \
    --exclude='.ipynb_checkpoints/' \
    --exclude='Risk_Investigation/Monte_Carlo/results/' \
    --exclude='Risk_Investigation/Monte_Carlo_v2/results/' \
    --exclude='Risk_Investigation/Monte_Carlo_v2/figures/' \
    --exclude='Risk_Investigation/Monte_Carlo_v3/results/' \
    --exclude='Risk_Investigation/Monte_Carlo_v3/hpc/slurm_out/' \
    --exclude='Risk_Investigation/Monte_Carlo_v3/dev_scratch/' \
    --exclude='Risk_Investigation/Monte_Carlo_v4/results/' \
    --exclude='Risk_Investigation/Monte_Carlo_v4/hpc/slurm_out/' \
    --exclude='Risk_Investigation/ExternalWindFarm/' \
    --exclude='Risk_Investigation/Layout_Generation/dev_scratch/' \
    --exclude='Risk_Investigation/Layout_Generation/figures/' \
    --exclude='Risk_Investigation/Layout_Generation/figures_check/' \
    --exclude='*.ipynb' \
    "${LOCAL_SRC}" "${CHIMERA_HOST}:${CHIMERA_DEST}"

echo "Done. The climate cache (Monte_Carlo_v4/cache/wind_climate_basis.npz) AND the"
echo "scenario_2 fixed-cluster cache (Monte_Carlo_v4/cache/scenario2_fixed_cluster.json)"
echo "are included -- neither climate_fit.py nor dev_scratch/build_scenario2_fixed_cluster.py"
echo "need to be re-run on Chimera."
