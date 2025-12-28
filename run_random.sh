#!/bin/bash
# Usage: ./run_random.sh [THRESHOLD] [BUDGET] [SEED] [N_EVAL]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DISEASE="HIV"
THRESHOLD="${1:-300}"
BUDGET="${2:-1}"
SEED="${3:-0}"
N_EVAL="${4:-50}"

echo "=== RUN_RANDOM.SH DEBUG ==="
which python
python --version
python -c "import numpy, pandas, scipy, networkx, tqdm, matplotlib; print('core deps OK')"
echo "==========================="

python driver_disease_random.py \
  -s "$SEED" \
  -D "$DISEASE" \
  -T "$THRESHOLD" \
  -B "$BUDGET" \
  -V "$N_EVAL"
