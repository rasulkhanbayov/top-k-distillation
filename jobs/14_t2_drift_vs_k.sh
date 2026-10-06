#!/bin/bash -l
#SBATCH -J vs3_t2_drift_vs_k
#SBATCH -o t2_drift_vs_k_%j.out
#SBATCH -e t2_drift_vs_k_%j.err
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

OUT="$REPO/results/t2_drift_vs_k"
mkdir -p "$OUT"

cd "$REPO/code"
set +e
overall_exit=0

for k in 16 64 256 1024 4096; do
    echo "=== T2: k=$k, arm=topk_fkl ==="
    python3 experiments/e5_mechanism.py \
        --teacher llama31-8b \
        --student llama32-1b \
        --arms topk_fkl \
        --alpha-ce 0.0 \
        --k "$k" \
        --select teacher \
        --probe-batches 32 \
        --steps 500 \
        --seeds 0 1 2 3 4 5 \
        --out "$OUT/k$k"
    ec=$?
    if [ "$ec" -ne 0 ]; then overall_exit=$ec; fi
done

echo "=== T2: full-vocabulary negative control (full_fkl) ==="
python3 experiments/e5_mechanism.py \
    --teacher llama31-8b \
    --student llama32-1b \
    --arms full_fkl \
    --alpha-ce 0.0 \
    --k 256 \
    --select teacher \
    --probe-batches 32 \
    --steps 500 \
    --seeds 0 1 2 3 4 5 \
    --out "$OUT/full_vocab"
ec=$?
if [ "$ec" -ne 0 ]; then overall_exit=$ec; fi

if [ "$overall_exit" -eq 0 ]; then
    echo "T2 done. Compare final-step |tail_mass_student - tail_mass_teacher|"
    echo "across $OUT/k{16,64,256,1024,4096}/e5.json and $OUT/full_vocab/e5.json"
    echo "with code/analysis/stats.py. Prediction: monotone decrease"
    echo "in k, reaching seed noise at the full-vocabulary arm."
fi
exit "$overall_exit"
