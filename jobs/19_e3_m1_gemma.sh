#!/bin/bash -l
#SBATCH -J vs3_e3_m1_gemma
#SBATCH -o e3_m1_gemma_%j.out
#SBATCH -e e3_m1_gemma_%j.err
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

OUT="$REPO/results/e3_matched_gemma3-4b"
mkdir -p "$OUT"

cd "$REPO/code"
set +e
python3 experiments/e3_offpolicy.py \
    --teacher gemma3-4b \
    --student gemma3-1b \
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
    echo "Gemma-3-4B M1 rerun done. Inspect $OUT/e3.json:"
    echo "  - full_fkl_cached's bytes_per_position should be <= 6144 (and"
    echo "    specifically near 5120, 2*d for d=2560), confirming this"
    echo "    teacher genuinely gives the cache a real storage advantage."
    echo "  - Compare full_fkl_cached vs topk_fkl (top-1024) and vs feature"
    echo "    on val_ppl using code/analysis/stats.py's compare()"
    echo "    on per_seed_by_arm (n=6, real paired permutation test)."
    echo "  - Report whichever way it comes out. If the cache wins here,"
    echo "    fold both results into main.tex's Off-policy paragraph as"
    echo "    teacher-dependent (report both pairs, don't cherry-pick); if"
    echo "    top-1024 wins again, the falsification generalizes across the"
    echo "    two teachers where the comparison is fair, which is itself a"
    echo "    stronger, more general negative result worth stating plainly."
fi
exit "$exit_code"
