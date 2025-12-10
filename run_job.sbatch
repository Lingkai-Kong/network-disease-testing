#!/bin/bash
#SBATCH --job-name=dqn_mip_disease
#SBATCH --output=logs/dqn_mip_disease_%j.out
#SBATCH --error=logs/dqn_mip_disease_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=seas_gpu

# Minimal setup: activate conda env, set LD_LIBRARY_PATH for PyTorch + CUDA,
# set Gurobi license, then run the experiment.

set -e

WORK_DIR="${WORK_DIR:-/n/netscratch/tambe_lab/Lab/msong300/network-disease-testing}"
AFEG_ENV_PATH="${AFEG_ENV_PATH:-/n/netscratch/tambe_lab/Lab/msong300/.conda/envs/AFEG}"
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-$WORK_DIR/gurobi.lic}"

# Load CUDA module (required for libnvJitLink.so.12)
if command -v module &> /dev/null; then
  module load cuda/12.4.1-fasrc01 2>/dev/null || true
fi

cd "$WORK_DIR"
mkdir -p logs

# Initialize conda (use site Miniforge)
if ! command -v conda &> /dev/null; then
  if [ -f "/n/sw/Miniforge3-24.11.3-0/etc/profile.d/conda.sh" ]; then
    source "/n/sw/Miniforge3-24.11.3-0/etc/profile.d/conda.sh"
  else
    for p in "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh"; do
      [ -f "$p" ] && source "$p" && break
    done
  fi
fi
eval "$(conda shell.bash hook)"
conda activate "$AFEG_ENV_PATH"

# CUDA visibility
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Build LD_LIBRARY_PATH cleanly (avoid duplication)
# Start fresh, then add paths that exist
LD_LIBRARY_PATH_NEW=""

# Add conda environment lib directory
if [ -d "$CONDA_PREFIX/lib" ]; then
  LD_LIBRARY_PATH_NEW="$CONDA_PREFIX/lib"
fi

# Add PyTorch bundled CUDA libraries (highest priority for torch)
TORCH_LIB_DIR="$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib"
if [ -d "$TORCH_LIB_DIR" ]; then
  if [ -n "$LD_LIBRARY_PATH_NEW" ]; then
    LD_LIBRARY_PATH_NEW="$TORCH_LIB_DIR:$LD_LIBRARY_PATH_NEW"
  else
    LD_LIBRARY_PATH_NEW="$TORCH_LIB_DIR"
  fi
fi

# Add CUDA toolkit targets directory (where cuda-toolkit installs libraries)
CUDA_TARGETS_LIB="$CONDA_PREFIX/targets/x86_64-linux/lib"
if [ -d "$CUDA_TARGETS_LIB" ]; then
  if [ -n "$LD_LIBRARY_PATH_NEW" ]; then
    LD_LIBRARY_PATH_NEW="$CUDA_TARGETS_LIB:$LD_LIBRARY_PATH_NEW"
  else
    LD_LIBRARY_PATH_NEW="$CUDA_TARGETS_LIB"
  fi
fi

# Also add the main lib directory from conda env (cuda-toolkit installs some libs here)
if [ -d "$CONDA_PREFIX/lib" ]; then
  # Check if libnvJitLink or other CUDA libs are here
  if find "$CONDA_PREFIX/lib" -name "libnvJitLink.so*" -o -name "libcusparseLt.so*" 2>/dev/null | grep -q .; then
    # Already added above, but ensure it's first
    if [[ ":$LD_LIBRARY_PATH_NEW:" != *":$CONDA_PREFIX/lib:"* ]]; then
      LD_LIBRARY_PATH_NEW="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH_NEW"
    fi
  fi
fi

# Add system CUDA from module if it exists
if [ -n "$CUDA_HOME" ] && [ -d "$CUDA_HOME/targets/x86_64-linux/lib" ]; then
  if [ -n "$LD_LIBRARY_PATH_NEW" ]; then
    LD_LIBRARY_PATH_NEW="$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH_NEW"
  else
    LD_LIBRARY_PATH_NEW="$CUDA_HOME/targets/x86_64-linux/lib"
  fi
fi

# Add system CUDA if it exists (fallback)
if [ -d "/usr/local/cuda/lib64" ]; then
  if [ -n "$LD_LIBRARY_PATH_NEW" ]; then
    LD_LIBRARY_PATH_NEW="$LD_LIBRARY_PATH_NEW:/usr/local/cuda/lib64"
  else
    LD_LIBRARY_PATH_NEW="/usr/local/cuda/lib64"
  fi
fi

# Preserve any existing LD_LIBRARY_PATH that was set before this script
# (e.g., from module system) but avoid duplication
if [ -n "$LD_LIBRARY_PATH" ]; then
  # Only add paths from existing LD_LIBRARY_PATH that aren't already included
  IFS=':' read -ra EXISTING_PATHS <<< "$LD_LIBRARY_PATH"
  for existing_path in "${EXISTING_PATHS[@]}"; do
    if [ -n "$existing_path" ] && [[ ":$LD_LIBRARY_PATH_NEW:" != *":$existing_path:"* ]]; then
      LD_LIBRARY_PATH_NEW="$LD_LIBRARY_PATH_NEW:$existing_path"
    fi
  done
fi

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH_NEW"

# Debug: Check for required CUDA libraries
echo "Checking for CUDA libraries..."
if find "$CONDA_PREFIX" -name "libnvJitLink.so*" 2>/dev/null | head -1; then
  echo "  ✓ Found libnvJitLink in conda env"
else
  echo "  ⚠ libnvJitLink not found in $CONDA_PREFIX"
fi
if find "$CONDA_PREFIX" -name "libcusparseLt.so*" 2>/dev/null | head -1; then
  echo "  ✓ Found libcusparseLt in conda env"
else
  echo "  ⚠ libcusparseLt not found in $CONDA_PREFIX"
fi
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""

# Quick sanity check for PyTorch/CUDA
python - <<'PY'
import sys
try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {getattr(torch.version, 'cuda', 'N/A')}")
except Exception as e:
    print("ERROR: PyTorch import failed. Most likely missing CUDA libs.")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Error: {e}")
    import os
    print(f"CONDA_PREFIX: {os.environ.get('CONDA_PREFIX', 'not set')}")
    raise
PY

# Set experiment parameters (can be overridden via command line)
# Default: THRESHOLD=300, BUDGET=5, SEED=0, N_EVAL=10
THRESHOLD="${1:-300}"
BUDGET="${2:-5}"
SEED="${3:-0}"
N_EVAL="${4:-10}"

# Enable quick test mode for faster execution
export DQN_QUICK_TEST=1

# Run experiment
./run.sh "$THRESHOLD" "$BUDGET" "$SEED" "$N_EVAL"



