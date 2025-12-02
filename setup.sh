#!/bin/bash
#
# First-time setup script for network-disease-testing
# This script creates the conda environment, installs dependencies, and sets up Gurobi license
#

set -e

echo "=================================================="
echo "First-time Setup for Network Disease Testing"
echo "=================================================="

# -----------------------------
# 1. Initialize conda
# -----------------------------
# Check if conda is already available
if ! command -v conda &> /dev/null; then
  # If CONDA_PREFIX is set, try to find conda.sh relative to it
  if [ -n "$CONDA_PREFIX" ]; then
    # Check if CONDA_PREFIX itself is the base (for base environment)
    if [ -f "$CONDA_PREFIX/etc/profile.d/conda.sh" ]; then
      source "$CONDA_PREFIX/etc/profile.d/conda.sh"
    # Otherwise, try parent directory (for named environments)
    elif [ -f "$(dirname "$CONDA_PREFIX")/etc/profile.d/conda.sh" ]; then
      source "$(dirname "$CONDA_PREFIX")/etc/profile.d/conda.sh"
    fi
  fi
  
  # If still not available, try common installation locations
  if ! command -v conda &> /dev/null; then
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
      source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
      source "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "/n/sw/Miniforge3-24.11.3-0/etc/profile.d/conda.sh" ]; then
      source "/n/sw/Miniforge3-24.11.3-0/etc/profile.d/conda.sh"
    fi
  fi
  
  # Final check
  if ! command -v conda &> /dev/null; then
    echo "Error: Could not find conda. Please install Anaconda or Miniconda first."
    exit 1
  fi
fi

# Initialize conda for bash (enables conda activate)
eval "$(conda shell.bash hook)"

# Set custom directories
WORKSPACE_DIR="/n/netscratch/tambe_lab/Lab/msong300"
CONDA_ENVS_DIR="$WORKSPACE_DIR/.conda/envs"
PIP_CACHE_DIR="$WORKSPACE_DIR/.cache/pip"

# Create directories if they don't exist
mkdir -p "$CONDA_ENVS_DIR"
mkdir -p "$PIP_CACHE_DIR"

# Configure pip to use custom cache directory
export PIP_CACHE_DIR="$PIP_CACHE_DIR"

echo "Using conda environments directory: $CONDA_ENVS_DIR"
echo "Using pip cache directory: $PIP_CACHE_DIR"

# -----------------------------
# 2. Create AFEG environment (if it doesn't exist)
# -----------------------------
AFEG_ENV_PATH="$CONDA_ENVS_DIR/AFEG"
if [ -d "$AFEG_ENV_PATH" ]; then
  echo "AFEG environment already exists at $AFEG_ENV_PATH. Skipping creation."
else
  echo "Creating AFEG conda environment with Python 3.12 at $AFEG_ENV_PATH..."
  conda create --prefix "$AFEG_ENV_PATH" python=3.12 -y
fi

# -----------------------------
# 3. Activate environment
# -----------------------------
echo "Activating AFEG environment..."
conda activate "$AFEG_ENV_PATH"

# -----------------------------
# 4. Install packages from requirements.txt
# -----------------------------
echo "Installing packages from requirements.txt..."
pip install --cache-dir "$PIP_CACHE_DIR" -r requirements.txt

# -----------------------------
# 5. Setup Gurobi license
# -----------------------------
if [ -f "gurobi.lic" ]; then
  if [ ! -f "$HOME/gurobi.lic" ]; then
    echo "Copying Gurobi license to home directory..."
    cp gurobi.lic "$HOME/gurobi.lic"
  else
    echo "Gurobi license already exists at $HOME/gurobi.lic"
  fi
else
  echo "Warning: gurobi.lic not found in current directory."
  echo "Please ensure your Gurobi license is set up correctly."
fi

# -----------------------------
# 6. Verify installation
# -----------------------------
echo ""
echo "Verifying installation..."
python -c "import torch; print('✓ PyTorch:', torch.__version__)" || echo "✗ PyTorch not found"
python -c "import torch_geometric; print('✓ PyTorch Geometric:', torch_geometric.__version__)" || echo "✗ PyTorch Geometric not found"
python -c "import gurobipy; print('✓ Gurobi:', gurobipy.gurobi.version())" || echo "✗ Gurobi not found"
python -c "import networkx; print('✓ NetworkX:', networkx.__version__)" || echo "✗ NetworkX not found"
python -c "import matplotlib; print('✓ Matplotlib:', matplotlib.__version__)" || echo "✗ Matplotlib not found"
python -c "import pandas; print('✓ Pandas:', pandas.__version__)" || echo "✗ Pandas not found"

echo ""
echo "=================================================="
echo "Setup complete!"
echo "=================================================="
echo ""
echo "To run experiments, use:"
echo "  ./run.sh"
echo ""
echo "Or manually:"
echo "  conda activate /n/netscratch/tambe_lab/Lab/msong300/.conda/envs/AFEG"
echo "  export GRB_LICENSE_FILE=\"\$HOME/gurobi.lic\""
echo "  ./run.sh"
echo ""

