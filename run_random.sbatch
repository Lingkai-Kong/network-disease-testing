#!/bin/bash
#SBATCH --job-name=random_disease
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1

#SBATCH --chdir=/n/netscratch/tambe_lab/Lab/anagha/network-disease-testing
#SBATCH --output=/n/netscratch/tambe_lab/Lab/anagha/network-disease-testing/logs/random_disease_%j.out
#SBATCH --error=/n/netscratch/tambe_lab/Lab/anagha/network-disease-testing/logs/random_disease_%j.err

set -euo pipefail

WORK_DIR="${WORK_DIR:-/n/netscratch/tambe_lab/Lab/anagha/network-disease-testing}"
AFEG_ENV_NAME="${AFEG_ENV_NAME:-AFEG}"

mkdir -p "$WORK_DIR/logs"
cd "$WORK_DIR"

module purge
# no CUDA needed (gpu:0), but harmless if present:
module load cuda/12.4.1-fasrc01 2>/dev/null || true

# Conda init (site Miniforge)
if [ -f "/n/sw/Miniforge3-25.3.1-0/etc/profile.d/conda.sh" ]; then
  source "/n/sw/Miniforge3-25.3.1-0/etc/profile.d/conda.sh"
elif [ -f "/n/sw/Miniforge3-24.11.3-0/etc/profile.d/conda.sh" ]; then
  source "/n/sw/Miniforge3-24.11.3-0/etc/profile.d/conda.sh"
else
  echo "ERROR: Could not find site Miniforge conda.sh" >&2
  exit 1
fi

conda activate "$AFEG_ENV_NAME"

echo "=== JOB DEBUG ==="
echo "HOST: $(hostname)"
echo "PWD:  $(pwd)"
echo "CONDA_PREFIX: ${CONDA_PREFIX:-<unset>}"
which python
python --version
echo "================="

THRESHOLD="${1:-300}"
BUDGET="${2:-1}"
SEED="${3:-0}"
N_EVAL="${4:-50}"

bash ./run_random.sh "$THRESHOLD" "$BUDGET" "$SEED" "$N_EVAL"
