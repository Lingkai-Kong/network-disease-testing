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
# Load CUDA module (required for libnvJitLink.so.12 and other CUDA libraries)
# PyTorch 2.5.1+cu121 is compatible with CUDA 12.4.1
if command -v module &> /dev/null; then
  module load cuda/12.4.1-fasrc01 2>/dev/null || true
fi

# Initialize conda (if not already in PATH)
if ! command -v conda &> /dev/null; then
  for conda_path in "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh"; do
    [ -f "$conda_path" ] && source "$conda_path" && break
  done
fi
eval "$(conda shell.bash hook)"

# Activate environment
conda activate "$AFEG_ENV_PATH"

# Set up LD_LIBRARY_PATH for CUDA libraries (required for PyTorch)
# Priority: System CUDA (from module) > PyTorch bundled > Conda env > CUDA targets

# Start with system CUDA libraries (from module, contains libnvJitLink.so.12)
if [ -n "$CUDA_HOME" ] && [ -d "$CUDA_HOME/targets/x86_64-linux/lib" ]; then
  export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
fi

# Add PyTorch bundled CUDA libraries if they exist
TORCH_LIB_DIR="$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib"
if [ -d "$TORCH_LIB_DIR" ]; then
  export LD_LIBRARY_PATH="$TORCH_LIB_DIR:$LD_LIBRARY_PATH"
fi

# Add conda environment lib directory
if [ -d "$CONDA_PREFIX/lib" ]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
fi

# Add CUDA toolkit targets directory if it exists in conda env
CUDA_TARGETS_LIB="$CONDA_PREFIX/targets/x86_64-linux/lib"
if [ -d "$CUDA_TARGETS_LIB" ]; then
  export LD_LIBRARY_PATH="$CUDA_TARGETS_LIB:$LD_LIBRARY_PATH"
fi

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
MODELS="dqn_mip"  # Only run DQN-MIP
# MODELS="null, random, iter_myopic"

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
