#!/bin/bash -l
#SBATCH -J hsc_e5_mechanism
#SBATCH -o e5_mechanism_%j.out
#SBATCH -e e5_mechanism_%j.err
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

OUT="$REPO/results/e5_mechanism_3seed"
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
    --seeds 0 1 2 \
    --out "$OUT"
exit_code=$?
set -e

if [ "$exit_code" -eq 0 ]; then
    echo "E5 done. Compare measured residual against Theorem 2's closed-form"
    echo "only valid from this run, not the earlier synthetic verify_theory.py"
    echo "configuration (46-73% vs 83-96%) -- do not quote the synthetic number."
fi
exit "$exit_code"
