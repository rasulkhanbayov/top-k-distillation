#!/bin/bash -l
#SBATCH -J hsc_e1_exactness
#SBATCH -o e1_exactness_%j.out
#SBATCH -e e1_exactness_%j.err
#SBATCH -p <gpu-partition>
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 100000MB
#SBATCH -A <account>
#SBATCH -q <qos>
#SBATCH -t 12:00:00

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate hsc

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
[ -f "$REPO/.env" ] && { set -a; source "$REPO/.env"; set +a; }

OUT="$REPO/results/e1_exactness"
mkdir -p "$OUT"

cd "$REPO/code"
set +e
python3 experiments/e1_exactness.py \
    --teacher qwen3-8b \
    --student qwen3-0.6b \
    --dtype bf16 \
    --steps 200 \
    --batch 4 \
    --seqlen 1024 \
    --chunk 16384 \
    --seed 0 \
    --out "$OUT"
exit_code=$?
set -e

if [ "$exit_code" -eq 0 ]; then
    echo "E1 done. Check first-step mismatch order in $OUT: should be ~1e-3 or"
    echo "tighter. Order-1 mismatch means the hidden-state capture point is"
fi
exit "$exit_code"
