#!/bin/bash -l
#SBATCH -J vs3_e3_m1_qwen4b
#SBATCH -o e3_m1_qwen4b_%j.out
#SBATCH -e e3_m1_qwen4b_%j.err
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

OUT="$REPO/results/e3_matched_qwen3-4b"
mkdir -p "$OUT"

cd "$REPO/code"
set +e
python3 experiments/e3_offpolicy.py \
    --teacher qwen3-4b \
    --student qwen3-1.7b \
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
    echo "Qwen3-4B M1 third-teacher run done. Inspect $OUT/e3.json:"
    echo "  - full_fkl_cached's bytes_per_position should be near 5120"
    echo "    (2*d for d=2560, identical to Gemma-3-4B's cache size)."
    echo "  - Compare full_fkl_cached vs topk_fkl (top-1024) and vs feature"
    echo "    on val_ppl using code/analysis/stats.py's compare()"
    echo "    on per_seed_by_arm (n=6, real paired permutation test)."
    echo "  - If the cache wins here (matching Gemma-3-4B, same d, no"
    echo "    softcap): softcap is ruled out as the explanation, since an"
    echo "    unsoftcapped teacher at the same d also favors the cache."
    echo "  - If top-1024 wins here (matching Qwen3-1.7B, same family, no"
    echo "    softcap): consistent with a family-level or non-softcap"
    echo "    explanation instead."
    echo "  - Report whichever way it comes out plainly, per this"
    echo "    project's standing discipline; fold into main.tex's"
    echo "    Off-policy paragraph and the new candidate-confounds"
    echo "    sentence alongside it."
fi
exit "$exit_code"
