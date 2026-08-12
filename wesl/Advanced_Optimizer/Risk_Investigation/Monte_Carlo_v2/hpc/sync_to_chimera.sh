#!/bin/bash
# Run LOCALLY to push the whole Advanced_Optimizer tree (code + climate cache) to
# Chimera. Excludes raw ERA5 GRIB (climate.py only reads the cached .npz),
# notebooks/figures/results, __pycache__. hpc/ sits one level deeper than v1's
# Monte_Carlo_HPC/ (it's nested inside Monte_Carlo_v2/ now, not a sibling folder), so
# results always land under Monte_Carlo_v2/results/ regardless of whether a run was
# launched via orchestrator.py directly or via hpc/run_trial_range.py -- no more of the
# path-flattening confusion v1's sync scripts had to work around.
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
    --exclude='Risk_Investigation/Layout_Generation/dev_scratch/' \
    --exclude='Risk_Investigation/Layout_Generation/figures/' \
    --exclude='Risk_Investigation/Layout_Generation/figures_check/' \
    --exclude='*.ipynb' \
    "${LOCAL_SRC}" "${CHIMERA_HOST}:${CHIMERA_DEST}"

echo "Done. The climate cache (Monte_Carlo_v2/cache/wind_climate_pca_basis.npz) is included --"
echo "climate_fit.py does NOT need to be re-run on Chimera."
