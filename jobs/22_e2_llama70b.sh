#!/bin/bash -l
#SBATCH -J vs3_e2_llama70b
#SBATCH -o e2_llama70b_%j.out
#SBATCH -e e2_llama70b_%j.err
#SBATCH -p <gpu-partition>
#SBATCH --gres=gpu:2
#SBATCH -c 16
#SBATCH --mem 200000MB
#SBATCH -A <account>
#SBATCH -q <qos>
#SBATCH -t 12:00:00

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate hsc

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
[ -f "$REPO/.env" ] && { set -a; source "$REPO/.env"; set +a; }

OUT="$REPO/results/e2_threshold"
mkdir -p "$OUT"

cd "$REPO/code"
set +e
python3 experiments/e2_threshold.py \
    --teachers llama31-70b \
    --device auto \
    --n-positions 64 \
    --seqlen 512 \
    --tol 1e-6 \
    --probe 50000 \
    --all-rules \
    --seed 0 \
    --out "$OUT"
exit_code=$?
set -e

if [ "$exit_code" -eq 0 ]; then
    echo "E2/llama31-70b done: $OUT/e2_llama31-70b.json."
    echo "Fold into the existing 6-teacher merge:"
    echo "  python3 $REPO/code/experiments/merge_e2_threshold.py --dir $OUT --out $OUT/e2.json"
    echo "This should now report 7/7 teachers, completing tab:e2 and closing"
    echo "tab:e2's caption (currently states 'Llama-3.1-70B was not run')."
else
    echo "E2/llama31-70b FAILED (exit $exit_code)." >&2
fi
exit "$exit_code"
