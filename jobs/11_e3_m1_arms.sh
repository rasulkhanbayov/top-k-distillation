#!/bin/bash -l
#SBATCH -J vs3_e3_m1_arms
#SBATCH -o e3_m1_arms_%j.out
#SBATCH -e e3_m1_arms_%j.err
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

OUT="$REPO/results/e3_matched_qwen3-1.7b"
mkdir -p "$OUT"

cd "$REPO/code"
set +e
python3 experiments/e3_offpolicy.py \
    --teacher qwen3-1.7b \
    --student qwen3-0.6b \
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
    echo "M1 arms done. Inspect $OUT/e3.json:"
    echo "  - full_fkl_cached's bytes_per_position should be <= 6144 (below"
    echo "    top-1024), confirming the matched/beats-storage comparison."
    echo "  - Compare full_fkl_cached vs topk_fkl and vs feature on val_ppl"
    echo "    using code/analysis/stats.py's compare() on"
    echo "    per_seed_by_arm (n=6, real paired permutation test possible)."
    echo "    comparison that WAS run at mismatched storage already reads"
    echo "    against the paper; this is the one that settles it properly."
fi
exit "$exit_code"
