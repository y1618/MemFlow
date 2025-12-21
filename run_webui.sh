#!/bin/bash
# MemFlow WebUI Launcher

# Activate virtual environment
source .venv/bin/activate

# Set CUDA environment
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Launch WebUI
# --host 0.0.0.0 allows access from VPN/LAN
# --autoload automatically loads the model on startup
python webui.py --host 0.0.0.0 --port 7860 "$@"
