#!/bin/bash -l
#SBATCH -J vs3_t4_345
#SBATCH -o t4_residual_plateau_seeds345_%j.out
#SBATCH -e t4_residual_plateau_seeds345_%j.err
#SBATCH -p <gpu-partition>
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 100000MB
#SBATCH -A <account>
#SBATCH -q <qos>
#SBATCH -t 2-00:00:00

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate hsc

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
[ -f "$REPO/.env" ] && { set -a; source "$REPO/.env"; set +a; }

OUT="$REPO/results/t4_residual_plateau/seeds345"
mkdir -p "$OUT"

cd "$REPO/code"
set +e
python3 experiments/e5_mechanism.py \
    --teacher llama31-8b \
    --student llama32-1b \
    --arms topk_fkl \
    --alpha-ce 0.0 \
    --k 64 \
    --select teacher \
    --probe-batches 8 \
    --measure-every 250 \
    --steps 5000 \
    --seeds 3 4 5 \
    --out "$OUT"
exit_code=$?
set -e

if [ "$exit_code" -eq 0 ]; then
    echo "T4 seeds 3-5 done. Combine with seeds012's e5.json and inspect"
    echo "grad_trunc_norm (should decay toward zero) against"
    echo "residual/identity_mismatch/tightness (should plateau)."
fi
exit "$exit_code"
