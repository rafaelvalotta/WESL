#!/bin/bash
# Advanced_env -- dedicated conda env for this pipeline (not the shared
# pywake_topfarm_2025 env from Bruno's other project). Created once via:
#   conda create -n Advanced_env python=3.11 -y
#   conda activate Advanced_env
#   pip install -r requirements_advanced_env.txt
# (see requirements_advanced_env.txt in this same folder for exact pinned versions).

source /home/bruno.boer001/miniforge3/etc/profile.d/conda.sh
conda activate Advanced_env
