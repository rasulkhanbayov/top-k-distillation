#!/bin/bash -l
#SBATCH -J vs3_e7_wide_sweep
#SBATCH -o e7_wide_sweep_%j.out
#SBATCH -e e7_wide_sweep_%j.err
#SBATCH -p <gpu-partition>
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 100000MB
#SBATCH -A <account>
#SBATCH -q <qos>
#SBATCH -t 1-00:00:00

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate hsc

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
[ -f "$REPO/.env" ] && { set -a; source "$REPO/.env"; set +a; }

OUT="$REPO/results/e7_disagreement_wide"
mkdir -p "$OUT"

cd "$REPO/code"
set +e
python3 experiments/e7_disagreement.py \
    --settings bild_style pretrain_style toolcall_style \
    --ks 2048 4096 \
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
    echo "E7 wide sweep done. $OUT/e7.json has 3 settings x 2 ks x 3 seeds"
    echo "= 18 new rows. Combine with the original run's 36 rows"
    echo "(results/e7_disagreement/e7.json) into one 6-point-per-setting"
    echo "picture: for each setting, check whether eval_kl keeps falling"
    echo "monotonically through k=4096 (still no interior optimum, but a"
    echo "much wider sweep) or reverses somewhere in {2048,4096} (a real"
    echo "interior optimum, testable against the pre-registered"
    echo "tail_reliability ordering in results/e7_disagreement/e7_predictors.json)."
    echo "Report whichever way it comes out -- this is still a hypothesis"
    echo "test, not a curve to fit after seeing the new points. This"
fi
exit "$exit_code"
