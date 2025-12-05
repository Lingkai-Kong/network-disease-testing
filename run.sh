#!/bin/bash
# Run Disease Graph experiments
# Usage: ./run.sh [THRESHOLD] [BUDGET] [SEED] [N_EVAL]
# Example: ./run.sh 300 5 0 10

set -e

# ============================================================================
# CONFIGURATION - Adjust these paths for your system
# ============================================================================
AFEG_ENV_PATH="${AFEG_ENV_PATH:-/n/netscratch/tambe_lab/Lab/msong300/.conda/envs/AFEG}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ============================================================================
# Setup environment
# ============================================================================
# Initialize conda (if not already in PATH)
if ! command -v conda &> /dev/null; then
  for conda_path in "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh"; do
    [ -f "$conda_path" ] && source "$conda_path" && break
  done
fi
eval "$(conda shell.bash hook)"

# Activate environment
conda activate "$AFEG_ENV_PATH"

# Set CUDA
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Set Gurobi license (priority: license server > workspace file > home file)
if [ -n "$GUROBI_LICENSE_SERVER" ]; then
  export GRB_LICENSE_FILE="$GUROBI_LICENSE_SERVER"
elif [ -f "$SCRIPT_DIR/gurobi.lic" ]; then
  export GRB_LICENSE_FILE="$SCRIPT_DIR/gurobi.lic"
elif [ -f "$HOME/gurobi.lic" ]; then
  export GRB_LICENSE_FILE="$HOME/gurobi.lic"
fi

# ============================================================================
# Experiment parameters
# ============================================================================
DISEASE="HIV"
THRESHOLD=${1:-300}
BUDGET=${2:-5}
SEED=${3:-0}
N_EVAL=${4:-10}
MODELS="null,random,sampling,iter_myopic,iter_dqn,dqn_mip"

# Quick test mode (set DQN_QUICK_TEST=0 for full training)
export DQN_QUICK_TEST=${DQN_QUICK_TEST:-0}

# ============================================================================
# Run experiment
# ============================================================================
python driver_disease.py \
  -s "$SEED" \
  -D "$DISEASE" \
  -T "$THRESHOLD" \
  -B "$BUDGET" \
  -V "$N_EVAL" \
  -M "$MODELS"
