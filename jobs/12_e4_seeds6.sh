#!/bin/bash -l
#SBATCH -J vs3_e4_seeds6
#SBATCH -o e4_seeds6_%j.out
#SBATCH -e e4_seeds6_%j.err
#SBATCH -p <gpu-partition>
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 100000MB
#SBATCH -A <account>
#SBATCH -q <qos>
#SBATCH -t 3-00:00:00

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate hsc

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
[ -f "$REPO/.env" ] && { set -a; source "$REPO/.env"; set +a; }

OUT="$REPO/results/e4_onpolicy_3seed_6seed"
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
    --seeds 0 1 2 3 4 5 \
    --out "$OUT"
exit_code=$?
set -e

if [ "$exit_code" -eq 0 ]; then
    echo "E4 (6 seeds) done. Run code/analysis/stats.py's compare()"
    echo "on $OUT/e4.json's rows -- loss_low_entropy/loss_high_entropy,"
    echo "matched by seed, for supervision=topk vs exact at selection=teacher"
    echo "(the case Theorems 2/3 cover). n=6's floor is 0.031, so this is the"
    echo "first run where the 74x gap could be reported with real"
    echo "significance rather than as a point estimate."
fi
exit "$exit_code"
