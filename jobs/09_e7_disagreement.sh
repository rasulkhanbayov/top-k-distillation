#!/bin/bash -l
#SBATCH -J hsc_e7_disagreement
#SBATCH -o e7_disagreement_%j.out
#SBATCH -e e7_disagreement_%j.err
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

OUT="$REPO/results/e7_disagreement"
mkdir -p "$OUT"

cd "$REPO/code"
set +e
python3 experiments/e7_disagreement.py \
    --settings bild_style pretrain_style toolcall_style \
    --ks 8 64 256 1024 \
    --k-ref 256 \
    --predictor-batches 32 \
    --steps 100 \
    --eval-batches 16 \
    --batch 4 \
    --seqlen 512 \
    --seeds 0 1 2 \
    --out "$OUT"
exit_code=$?
set -e

if [ "$exit_code" -eq 0 ]; then
    echo "E7 done. $OUT/e7_predictors.json holds the REGISTERED prediction"
    echo "(tail_rank_corr per setting, committed before the sweep ran)."
    echo "$OUT/e7.json's printed verdict compares the settings ordered by"
    echo "measured tail_rank_corr against the settings ordered by measured"
    echo "best k. Report the match or mismatch plainly either way -- this is"
    echo "a hypothesis test of whether the trade-off predicts the published"
    echo "6). A mismatch does not touch Section 4's identifiability results."
fi
exit "$exit_code"
