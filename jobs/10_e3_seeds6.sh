#!/bin/bash -l
#SBATCH -J vs3_e3_seeds6
#SBATCH -o e3_seeds6_%j.out
#SBATCH -e e3_seeds6_%j.err
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

OUT="$REPO/results/e3_offpolicy_3seed_6seed"
mkdir -p "$OUT"

cd "$REPO/code"
set +e
python3 experiments/e3_offpolicy.py \
    --teacher llama31-8b \
    --student llama32-1b \
    --arms full_fkl_online full_fkl_cached topk_fkl \
    --match storage \
    --budget-bytes 8192 \
    --k 256 \
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
    echo "E3 (6 seeds) done. Run code/analysis/stats.py's compare()"
    echo "on $OUT/e3.json's per_seed_by_arm val_ppl arrays for"
    echo "full_fkl_cached vs full_fkl_online and full_fkl_cached vs topk_fkl."
    echo "Report the bootstrap CI, exact paired permutation p, and Cohen's dz"
    echo "-- n=6's floor is 0.031, so this is the first run where a"
    echo "significant result is even possible, unlike the original n=3 run."
fi
exit "$exit_code"
