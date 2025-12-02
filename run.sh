#!/bin/bash
#
# Run Disease Graph experiments (all 6 models) for a simple HIV instance.
# Usage:
#   ./run.sh [THRESHOLD] [BUDGET] [SEED]
#
# Examples:
#   ./run.sh                # default: THRESHOLD=300, BUDGET=5, SEED=0
#   ./run.sh 300 5 1
#

set -e

# -----------------------------
# 1. Activate AFEG environment
# -----------------------------
# Adjust this "conda.sh" path if your Anaconda/Miniconda is installed elsewhere.
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/n/sw/Miniforge3-24.11.3-0/etc/profile.d/conda.sh" ]; then
  source "/n/sw/Miniforge3-24.11.3-0/etc/profile.d/conda.sh"
fi

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Activate AFEG environment using the prefix path
AFEG_ENV_PATH="/n/netscratch/tambe_lab/Lab/msong300/.conda/envs/AFEG"
conda activate "$AFEG_ENV_PATH"

# -----------------------------
# Set library paths for CUDA
# -----------------------------
# Add conda environment lib directory to library path
export LD_LIBRARY_PATH="$AFEG_ENV_PATH/lib:$LD_LIBRARY_PATH"

# Add PyTorch CUDA library paths (if they exist in site-packages)
if [ -d "$AFEG_ENV_PATH/lib/python3.12/site-packages/torch/lib" ]; then
  export LD_LIBRARY_PATH="$AFEG_ENV_PATH/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH"
fi

# Add nvidia-cuda-* package library paths from conda environment
for cuda_pkg in "$AFEG_ENV_PATH"/lib/python3.12/site-packages/nvidia-*/lib; do
  if [ -d "$cuda_pkg" ]; then
    export LD_LIBRARY_PATH="$cuda_pkg:$LD_LIBRARY_PATH"
  fi
done

# Add nvidia CUDA library paths from user's local site-packages
# (These are installed via pip in the user's local directory)
USER_SITE_PACKAGES="$HOME/.local/lib/python3.12/site-packages"
if [ -d "$USER_SITE_PACKAGES/nvidia" ]; then
  for nvidia_lib in "$USER_SITE_PACKAGES"/nvidia/*/lib; do
    if [ -d "$nvidia_lib" ]; then
      export LD_LIBRARY_PATH="$nvidia_lib:$LD_LIBRARY_PATH"
    fi
  done
fi

# -----------------------------
# 2. Configure CUDA
# -----------------------------
# Set CUDA_VISIBLE_DEVICES if not already set (use first GPU)
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0
fi

# -----------------------------
# 3. Set Gurobi license (edit if your license is elsewhere)
# -----------------------------
# Always use the workspace license file as the primary source
# Override any existing GRB_LICENSE_FILE that might point to wrong location
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_LIC="$SCRIPT_DIR/gurobi.lic"

# Unset any existing GRB_LICENSE_FILE that points to wrong/invalid location
# We want to use the workspace license file, so unset if it's not a license server
if [ -n "$GRB_LICENSE_FILE" ]; then
    if [[ "$GRB_LICENSE_FILE" != *"@"* ]] && [ "$GRB_LICENSE_FILE" != "$WORKSPACE_LIC" ]; then
        # It's not a license server and not our workspace license, so unset it
        echo "⚠ Found GRB_LICENSE_FILE pointing to different location: $GRB_LICENSE_FILE"
        echo "  Overriding with workspace license file..."
        unset GRB_LICENSE_FILE
    elif [ ! -f "$GRB_LICENSE_FILE" ] && [[ "$GRB_LICENSE_FILE" != *"@"* ]]; then
        # File doesn't exist and it's not a license server
        echo "⚠ Found invalid GRB_LICENSE_FILE: $GRB_LICENSE_FILE (file does not exist)"
        echo "  Unsetting and using workspace license instead..."
        unset GRB_LICENSE_FILE
    fi
fi

# Detect if we're on a compute node
COMPUTE_NODE=$(hostname | grep -E "compute|node|gpu" || echo "")
if [ -n "$COMPUTE_NODE" ] || [ -n "$SLURM_JOB_NODELIST" ]; then
    if [ -z "$GUROBI_LICENSE_SERVER" ]; then
        echo "⚠ WARNING: Running on compute node without Gurobi license server configured"
        echo "  License files tied to host IDs will not work on compute nodes"
        echo "  Set GUROBI_LICENSE_SERVER='PORT@SERVER' before running"
    fi
fi

# Priority: 1) License server, 2) Workspace license file (FORCE), 3) Home license file
if [ -n "$GUROBI_LICENSE_SERVER" ]; then
    # Use license server if explicitly set (best for compute nodes)
    export GRB_LICENSE_FILE="$GUROBI_LICENSE_SERVER"
    echo "Using Gurobi license server: $GUROBI_LICENSE_SERVER"
elif [ -f "$WORKSPACE_LIC" ]; then
    # FORCE use of workspace license file (override any existing setting)
    export GRB_LICENSE_FILE="$WORKSPACE_LIC"
    echo "✓ Using Gurobi license from workspace: $GRB_LICENSE_FILE"
    if [ -n "$COMPUTE_NODE" ] || [ -n "$SLURM_JOB_NODELIST" ]; then
        echo "  ⚠ Warning: This license is tied to host ID 163b78fb"
        echo "  Compute nodes have different host IDs and may cause license errors"
    fi
elif [ -f "$HOME/gurobi.lic" ]; then
    # Fallback to home directory license
    export GRB_LICENSE_FILE="$HOME/gurobi.lic"
    if [ -n "$COMPUTE_NODE" ] || [ -n "$SLURM_JOB_NODELIST" ]; then
        echo "⚠ Warning: License file may fail on compute nodes (host ID mismatch)"
    fi
fi

# Verify license file exists and is readable
if [ -n "$GRB_LICENSE_FILE" ] && [ -f "$GRB_LICENSE_FILE" ]; then
    echo "✓ License file verified: $GRB_LICENSE_FILE"
elif [ -n "$GRB_LICENSE_FILE" ] && [[ "$GRB_LICENSE_FILE" == *"@"* ]]; then
    echo "✓ License server configured: $GRB_LICENSE_FILE"
fi

# -----------------------------
# 4. Experiment hyperparameters
# -----------------------------
DISEASE="HIV"
THRESHOLD=${1:-30}  # Minimal: 30 nodes for quick testing (default was 100)
BUDGET=${2:-3}       # Minimal: 3 nodes per step (default was 5)
SEED=${3:-0}
N_EVAL=${4:-2}       # Minimal: 2 evaluation episodes (default was 10)
PREFIX=""

# Enable quick test mode for DQN (reduces training episodes and MILP time limits)
export DQN_QUICK_TEST=1

MODELS="null,random,sampling,iter_myopic,iter_dqn,dqn_mip"
# MODELS="iter_dqn"
# MODELS="dqn_mip"


echo "=================================================="
echo "Disease Graph Experiment (HIV)"
echo "=================================================="
echo "Disease:     $DISEASE"
echo "Threshold:   $THRESHOLD"
echo "Budget:      $BUDGET"
echo "Seed:        $SEED"
echo "N Episodes:  $N_EVAL"
echo "Models:      $MODELS"
echo ""
echo "Note: Using MINIMAL settings for quick end-to-end test"
echo "      (THRESHOLD=30, BUDGET=3, N_EVAL=2, DQN_QUICK_TEST=1)"
echo "      DQN will train for 5 episodes with 5s MILP time limit"
echo "=================================================="

python driver_disease.py \
  -s "$SEED" \
  -D "$DISEASE" \
  -T "$THRESHOLD" \
  -B "$BUDGET" \
  -V "$N_EVAL" \
  -M "$MODELS" \
  -p "$PREFIX"

echo "=================================================="
echo "Run complete!"
echo "=================================================="
