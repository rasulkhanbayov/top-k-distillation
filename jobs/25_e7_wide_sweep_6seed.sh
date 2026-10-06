#!/bin/bash -l
#SBATCH -J vs3_e7_wide_6seed
#SBATCH -o e7_wide_6seed_%j.out
#SBATCH -e e7_wide_6seed_%j.err
#SBATCH -p <gpu-partition>
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 100000MB
#SBATCH -A <account>
#SBATCH -q <qos>
#SBATCH -t 08:00:00

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate hsc

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
[ -f "$REPO/.env" ] && { set -a; source "$REPO/.env"; set +a; }

OUT="$REPO/results/e7_disagreement_wide_6seed"
mkdir -p "$OUT"
cd "$REPO/code"

set +e
python3 experiments/e7_disagreement.py \
    --settings bild_style pretrain_style toolcall_style \
    --ks 1024 2048 4096 \
    --k-ref 256 \
    --predictor-batches 32 \
    --steps 100 \
    --eval-batches 16 \
    --batch 4 \
    --seqlen 512 \
    --lr 1e-5 \
    --seeds 0 1 2 3 4 5 \
    --out "$OUT"
exit_code=$?
set -e

if [ "$exit_code" -eq 0 ]; then
    echo "Done: $OUT/e7.json (54 rows). Next: (1) reproducibility check of"
    echo "seeds 0-2 against the earlier runs; (2) per setting, exact paired"
    echo "test k=1024 vs 4096 and 2048 vs 4096 with stats.py compare()."
else
    echo "FAILED (exit $exit_code); partial results are in $OUT/e7.json." >&2
fi
exit "$exit_code"
