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
fi

conda activate AFEG

# -----------------------------
# 2. Set Gurobi license (edit if your license is elsewhere)
# -----------------------------
export GRB_LICENSE_FILE="$HOME/gurobi.lic"

# -----------------------------
# 3. Experiment hyperparameters
# -----------------------------
DISEASE="HIV"
THRESHOLD=${1:-300}
BUDGET=${2:-5}
SEED=${3:-0}
N_EVAL=50
PREFIX=""

MODELS="null,random,sampling,iter_myopic,iter_dqn,dqn_mip"

echo "=================================================="
echo "Disease Graph Experiment (HIV)"
echo "=================================================="
echo "Disease:     $DISEASE"
echo "Threshold:   $THRESHOLD"
echo "Budget:      $BUDGET"
echo "Seed:        $SEED"
echo "N Episodes:  $N_EVAL"
echo "Models:      $MODELS"
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
