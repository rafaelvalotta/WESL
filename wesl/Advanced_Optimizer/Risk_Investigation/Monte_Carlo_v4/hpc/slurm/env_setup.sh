#!/bin/bash
# Advanced_env -- same dedicated conda env as Monte_Carlo_v2/hpc (not the shared
# pywake_topfarm_2025 env from Bruno's other project). Reused as-is for v3 -- same
# py_wake==2.6.20, same Chimera account. Only new addition for v3 is pyarrow (parquet
# output) -- see ../requirements_advanced_env.txt. If Advanced_env doesn't have it yet:
#   conda activate Advanced_env && pip install pyarrow

source /home/bruno.boer001/miniforge3/etc/profile.d/conda.sh
conda activate Advanced_env
