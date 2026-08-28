#!/bin/bash
# Run LOCALLY to pull one run's results back from Chimera, after the array job has
# finished (check with `squeue -u bruno.boer001` on Chimera first).
#
#   ./sync_from_chimera.sh scenario_1 20260820_120000
CHIMERA_HOST="bruno.boer001@chimerahead.umb.edu"
CHIMERA_SRC_BASE="/home/bruno.boer001/Summer_26/August/Advanced_Optimizer/Risk_Investigation/Monte_Carlo_v4/results"

set -euo pipefail
SCENARIO="${1:?usage: ./sync_from_chimera.sh <scenario> <run_id> -- scenario_1 or scenario_2}"
RUN_ID="${2:?usage: ./sync_from_chimera.sh <scenario> <run_id> -- scenario_1 or scenario_2}"
LOCAL_DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/../results" && pwd)/${SCENARIO}/"

echo "Pulling ${CHIMERA_HOST}:${CHIMERA_SRC_BASE}/${SCENARIO}/${RUN_ID}/ -> ${LOCAL_DEST}${RUN_ID}/"
mkdir -p "${LOCAL_DEST}"
rsync -avz --progress \
    "${CHIMERA_HOST}:${CHIMERA_SRC_BASE}/${SCENARIO}/${RUN_ID}/" \
    "${LOCAL_DEST}${RUN_ID}/"

echo ""
echo "Next: cd .. && python -c \"from hpc.merge_results import merge; merge('${SCENARIO}', '${RUN_ID}')\""
