#!/bin/bash -l
#SBATCH -J hsc_e4_onpolicy
#SBATCH -o e4_onpolicy_%j.out
#SBATCH -e e4_onpolicy_%j.err
#SBATCH -p <gpu-partition>
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 100000MB
#SBATCH -A <account>
#SBATCH -q <qos>
#SBATCH -t 1-12:00:00

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate hsc

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
[ -f "$REPO/.env" ] && { set -a; source "$REPO/.env"; set +a; }

OUT="$REPO/results/e4_onpolicy_3seed"
mkdir -p "$OUT"

cd "$REPO/code"
set +e
python3 experiments/e4_onpolicy.py \
    --recipe gkd \
    --teacher qwen3-8b \
    --student qwen3-0.6b \
    --supervision topk exact \
    --select teacher student \
    --k 100 \
    --refresh 0 \
    --steps 500 \
    --batch 8 \
    --prompt-len 256 \
    --max-new 256 \
    --temp 1.0 \
    --chunk 16384 \
    --lr 1e-5 \
    --seeds 0 1 2 \
    --out "$OUT"
exit_code=$?
set -e

if [ "$exit_code" -eq 0 ]; then
    echo "not appear on-policy, the paper's reach is confined to off-policy"
    echo "distillation and Section 6's framing must change -- report this"
    echo "either way, do not bury a null result. Compare loss_low/high_entropy"
    echo "(the shared held-out exact-KL metric), NOT the raw per-arm training"
    echo "loss printed during the run."
fi
exit "$exit_code"
