#!/bin/bash
# Run Disease Graph DQN Training
#
# Usage:
#   ./run_dqn.sh [DISEASE] [THRESHOLD] [BUDGET] [SEED]
#
# Examples:
#   ./run_dqn.sh HIV 100 5 0
#   ./run_dqn.sh Gonorrhea 300 10 42

# Set default parameters
DISEASE=${1:-HIV}
THRESHOLD=${2:-100}
BUDGET=${3:-5}
SEED=${4:-0}
HORIZON=20
N_EVAL=50
PREFIX=""

# Set Gurobi license file
export GRB_LICENSE_FILE=~/gurobi.lic

echo "=================================================="
echo "Disease Graph DQN Training"
echo "=================================================="
echo "Disease:     $DISEASE"
echo "Threshold:   $THRESHOLD"
echo "Budget:      $BUDGET"
echo "Seed:        $SEED"
echo "Horizon:     $HORIZON"
echo "N Episodes:  $N_EVAL"
echo "=================================================="

# Run the driver
python driver_disease.py \
    -s $SEED \
    -H $HORIZON \
    -D $DISEASE \
    -T $THRESHOLD \
    -B $BUDGET \
    -V $N_EVAL \
    -p $PREFIX

echo "=================================================="
echo "Training complete!"
echo "=================================================="
