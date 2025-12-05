#!/bin/bash
#SBATCH --job-name=disease_graph
#SBATCH --output=logs/disease_graph_%j.out
#SBATCH --error=logs/disease_graph_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# ============================================================================
# CONFIGURATION - Adjust these paths for your system
# ============================================================================
WORK_DIR="${WORK_DIR:-/n/netscratch/tambe_lab/Lab/msong300/network-disease-testing}"
AFEG_ENV_PATH="${AFEG_ENV_PATH:-/n/netscratch/tambe_lab/Lab/msong300/.conda/envs/AFEG}"

cd "$WORK_DIR"

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

# Set Gurobi license (for compute nodes, use license server if available)
if [ -n "$GUROBI_LICENSE_SERVER" ]; then
  export GRB_LICENSE_FILE="$GUROBI_LICENSE_SERVER"
elif [ -f "$WORK_DIR/gurobi.lic" ]; then
  export GRB_LICENSE_FILE="$WORK_DIR/gurobi.lic"
elif [ -f "$HOME/gurobi.lic" ]; then
  export GRB_LICENSE_FILE="$HOME/gurobi.lic"
fi

# ============================================================================
# Run experiment
# ============================================================================
# Usage: sbatch run_job.sbatch [THRESHOLD] [BUDGET] [SEED] [N_EVAL]
./run.sh "$@"
