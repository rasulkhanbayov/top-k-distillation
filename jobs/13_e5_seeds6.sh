#!/bin/bash -l
#SBATCH -J vs3_e5_seeds6
#SBATCH -o e5_seeds6_%j.out
#SBATCH -e e5_seeds6_%j.err
#SBATCH -p <gpu-partition>
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 100000MB
#SBATCH -A <account>
#SBATCH -q <qos>
#SBATCH -t 2-00:00:00

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate hsc

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
[ -f "$REPO/.env" ] && { set -a; source "$REPO/.env"; set +a; }

OUT="$REPO/results/e5_mechanism_3seed_6seed"
mkdir -p "$OUT"

cd "$REPO/code"
set +e
python3 experiments/e5_mechanism.py \
    --teacher llama31-8b \
    --student llama32-1b \
    --arms topk_fkl full_fkl \
    --alpha-ce 0.0 1.0 \
    --k 256 \
    --probe-batches 32 \
    --steps 500 \
    --seeds 0 1 2 3 4 5 \
    --out "$OUT"
exit_code=$?
set -e

if [ "$exit_code" -eq 0 ]; then
    echo "E5 (6 seeds) done. Run code/analysis/stats.py's compare()"
    echo "on $OUT/e5.json's rows -- final-step tail_mass_student, matched by"
    echo "seed, for topk_fkl vs full_fkl at alpha_ce=0.0 (the decisive"
    echo "variant). n=6's floor is 0.031."
fi
exit "$exit_code"
