#!/bin/bash -l
#SBATCH -J e5_ace1_fixed_full
#SBATCH -o e5_ace1_fixed_full_%j.out
#SBATCH -e e5_ace1_fixed_full_%j.err
#SBATCH -p <gpu-partition>
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 100000MB
#SBATCH -A <account>
#SBATCH -q <qos>
#SBATCH -t 1-06:00:00

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate hsc

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
[ -f "$REPO/.env" ] && { set -a; source "$REPO/.env"; set +a; }

OUT="$REPO/results/e5_mechanism_3seed_ace1_full"
mkdir -p "$OUT"
cd "$REPO/code"
set +e
python3 experiments/e5_mechanism.py \
    --teacher llama31-8b \
    --student llama32-1b \
    --arms full_fkl \
    --alpha-ce 1.0 \
    --k 256 \
    --probe-batches 32 \
    --steps 500 \
    --seeds 0 1 2 3 4 5 \
    --out "$OUT"
exit_code=$?
set -e
if [ "$exit_code" -eq 0 ]; then
    echo "Done: $OUT/e5.json (written after every cell)."
else
    echo "FAILED (exit $exit_code); completed cells are in $OUT/e5.json." >&2
fi
exit "$exit_code"
