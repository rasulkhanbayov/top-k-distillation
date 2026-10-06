#!/bin/bash -l
#SBATCH -J vs3_e3_m1_smollm2
#SBATCH -o e3_m1_smollm2_%j.out
#SBATCH -e e3_m1_smollm2_%j.err
#SBATCH -p <gpu-partition>
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 100000MB
#SBATCH -A <account>
#SBATCH -q <qos>
#SBATCH -t 2-12:00:00

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate hsc

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
[ -f "$REPO/.env" ] && { set -a; source "$REPO/.env"; set +a; }

OUT="$REPO/results/e3_matched_smollm2-1.7b"
mkdir -p "$OUT"
cd "$REPO/code"

set +e
python3 experiments/e3_offpolicy.py \
    --teacher smollm2-1.7b \
    --student smollm2-360m \
    --arms full_fkl_cached topk_fkl feature \
    --match storage \
    --budget-bytes 6144 \
    --k 1024 \
    --cache-dtype bf16 \
    --chunk 16384 \
    --lr 2e-5 \
    --steps 500 \
    --batch 4 \
    --seqlen 512 \
    --cache-batches 500 \
    --seeds 0 1 2 3 4 5 \
    --out "$OUT"
exit_code=$?
set -e

if [ "$exit_code" -eq 0 ]; then
    echo "SmolLM2 M1 run done: $OUT/e3.json. Recompute cached-vs-topk and"
    echo "cached-vs-feature with code/analysis/stats.py compare() on"
    echo "per_seed_by_arm (exact test; the script's own p is Monte Carlo)."
    echo "Fold into main.tex's Off-policy paragraph either way, updating the"
    echo "teacher count and the condition-number/anisotropy comparison."
else
    echo "SmolLM2 M1 run FAILED (exit $exit_code)." >&2
fi
exit "$exit_code"
