#!/bin/bash
# Run LOCALLY to pull one run's results back from Chimera, after the array job has
# finished (check with `squeue -u bruno.boer001` on Chimera first).
#
#   ./sync_from_chimera.sh 20260810_120000
CHIMERA_HOST="bruno.boer001@chimerahead.umb.edu"
CHIMERA_SRC_BASE="/home/bruno.boer001/Summer_26/August/Advanced_Optimizer/Risk_Investigation/Monte_Carlo_v2/results"

set -euo pipefail
RUN_ID="${1:?usage: ./sync_from_chimera.sh <run_id>}"
LOCAL_DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/../results" && pwd)/"

echo "Pulling ${CHIMERA_HOST}:${CHIMERA_SRC_BASE}/${RUN_ID}/ -> ${LOCAL_DEST}${RUN_ID}/"
rsync -avz --progress \
    "${CHIMERA_HOST}:${CHIMERA_SRC_BASE}/${RUN_ID}/" \
    "${LOCAL_DEST}${RUN_ID}/"

echo ""
echo "Next: cd .. && python -c \"from hpc.merge_results import merge; merge('${RUN_ID}')\""
