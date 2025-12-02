#!/bin/bash
#SBATCH --job-name=disease_graph
#SBATCH --output=logs/disease_graph_%j.out
#SBATCH --error=logs/disease_graph_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Change to the script directory
cd /n/netscratch/tambe_lab/Lab/msong300/network-disease-testing

# -----------------------------
# Initialize conda
# -----------------------------
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/n/sw/Miniforge3-24.11.3-0/etc/profile.d/conda.sh" ]; then
  source "/n/sw/Miniforge3-24.11.3-0/etc/profile.d/conda.sh"
fi

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# -----------------------------
# Activate AFEG environment
# -----------------------------
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
# Set Gurobi license
# -----------------------------
# Always use the workspace license file as the primary source
# Override any existing GRB_LICENSE_FILE that might point to wrong location
WORKSPACE_LIC="/n/netscratch/tambe_lab/Lab/msong300/network-disease-testing/gurobi.lic"

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

# Detect if we're on a compute node (different host ID than login node)
COMPUTE_NODE=$(hostname | grep -E "compute|node|gpu" || echo "")
if [ -n "$COMPUTE_NODE" ] || [ -n "$SLURM_JOB_NODELIST" ]; then
    echo "Detected compute node. License server is required for Gurobi."
    echo "  Current hostname: $(hostname)"
    if [ -z "$GUROBI_LICENSE_SERVER" ]; then
        echo ""
        echo "⚠ WARNING: Gurobi license server not configured!"
        echo "  Compute nodes have different host IDs than login nodes."
        echo "  A license file tied to a specific host ID will not work."
        echo ""
        echo "  To fix this, set one of the following before running:"
        echo "    export GUROBI_LICENSE_SERVER='PORT@SERVER'"
        echo ""
        echo "  Example:"
        echo "    export GUROBI_LICENSE_SERVER='7010@gurobi-license-server.fas.harvard.edu'"
        echo ""
    fi
fi

# Priority: 1) License server, 2) Workspace license file (FORCE), 3) Home license file
if [ -n "$GUROBI_LICENSE_SERVER" ]; then
    # Use license server if explicitly set (best for compute nodes)
    export GRB_LICENSE_FILE="$GUROBI_LICENSE_SERVER"
    echo "✓ Using Gurobi license server: $GUROBI_LICENSE_SERVER"
elif [ -f "$WORKSPACE_LIC" ]; then
    # FORCE use of workspace license file (override any existing setting)
    export GRB_LICENSE_FILE="$WORKSPACE_LIC"
    echo "✓ Using Gurobi license from workspace: $GRB_LICENSE_FILE"
    if [ -n "$COMPUTE_NODE" ] || [ -n "$SLURM_JOB_NODELIST" ]; then
        echo "  ⚠ Warning: This license is tied to host ID 163b78fb"
        echo "  Compute nodes have different host IDs and may cause license errors"
        echo "  Consider using a license server for compute nodes"
    fi
elif [ -f "$HOME/gurobi.lic" ]; then
    # Fallback to home directory license
    export GRB_LICENSE_FILE="$HOME/gurobi.lic"
    echo "Using Gurobi license from: $HOME/gurobi.lic"
    if [ -n "$COMPUTE_NODE" ] || [ -n "$SLURM_JOB_NODELIST" ]; then
        echo "  ⚠ Warning: License file may not work on compute nodes due to host ID mismatch"
    fi
else
    echo "⚠ Warning: Gurobi license file not found."
    if [ -n "$COMPUTE_NODE" ] || [ -n "$SLURM_JOB_NODELIST" ]; then
        echo "  License server is REQUIRED for compute nodes."
    else
        echo "  Expected license at: $WORKSPACE_LIC"
        echo "  Set GRB_LICENSE_FILE or GUROBI_LICENSE_SERVER environment variable"
    fi
fi

# Verify license file exists and is readable
if [ -n "$GRB_LICENSE_FILE" ] && [ -f "$GRB_LICENSE_FILE" ]; then
    echo "✓ License file verified: $GRB_LICENSE_FILE"
    echo "  File exists and is readable"
elif [ -n "$GRB_LICENSE_FILE" ] && [[ "$GRB_LICENSE_FILE" == *"@"* ]]; then
    echo "✓ License server configured: $GRB_LICENSE_FILE"
else
    echo "⚠ Warning: License file not found or not readable: ${GRB_LICENSE_FILE:-Not set}"
fi

# -----------------------------
# Set pip cache directory
# -----------------------------
export PIP_CACHE_DIR="/n/netscratch/tambe_lab/Lab/msong300/.cache/pip"

# -----------------------------
# Run the experiment
# -----------------------------
# Ensure run.sh is executable
chmod +x run.sh

# -----------------------------
# Configure CUDA
# -----------------------------
# Set CUDA_VISIBLE_DEVICES if not already set (use first GPU)
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0
    echo "Set CUDA_VISIBLE_DEVICES=0"
fi

# Print environment info
echo "=================================================="
echo "Job Environment Information"
echo "=================================================="
echo "Working directory: $(pwd)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
CUDA_AVAIL=$(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')
echo "CUDA available: $CUDA_AVAIL"
if [ "$CUDA_AVAIL" = "True" ]; then
    echo "GPU info: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")' 2>/dev/null || echo 'N/A')"
fi
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "=================================================="
echo ""

# Run the experiment
# You can pass arguments: sbatch run_job.sbatch [THRESHOLD] [BUDGET] [SEED] [N_EVAL]
# Default: THRESHOLD=100, BUDGET=5, SEED=0, N_EVAL=10 (simplified for faster testing)
# Example: sbatch run_job.sbatch 100 5 0 10
# Example: sbatch run_job.sbatch 300 5 0 50  (full settings)
echo "Starting experiment with arguments: $@"
echo "  (Defaults: THRESHOLD=100, BUDGET=5, SEED=0, N_EVAL=10)"
echo ""

# Execute run.sh and capture exit code
./run.sh "$@"
EXIT_CODE=$?

# Check exit status
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "Job completed successfully!"
    echo "=================================================="
else
    echo ""
    echo "=================================================="
    echo "Job failed with exit code: $EXIT_CODE"
    echo "=================================================="
    exit $EXIT_CODE
fi

