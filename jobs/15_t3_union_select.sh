#!/bin/bash -l
#SBATCH -J vs3_t3_union
#SBATCH -o t3_union_select_%j.out
#SBATCH -e t3_union_select_%j.err
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

OUT="$REPO/results/t3_union_select"
mkdir -p "$OUT"

cd "$REPO/code"
set +e
overall_exit=0

for select in teacher union; do
    echo "=== T3: select=$select ==="
    python3 experiments/e5_mechanism.py \
        --teacher llama31-8b \
        --student llama32-1b \
        --arms topk_fkl \
        --alpha-ce 0.0 \
        --k 10 \
        --select "$select" \
        --probe-batches 32 \
        --steps 500 \
        --seeds 0 1 2 3 4 5 \
        --out "$OUT/$select"
    ec=$?
    if [ "$ec" -ne 0 ]; then overall_exit=$ec; fi
done

if [ "$overall_exit" -eq 0 ]; then
    echo "T3 done. Compare $OUT/teacher/e5.json and $OUT/union/e5.json --"
    echo "no prediction was registered, report whichever way it comes out."
fi
exit "$overall_exit"
