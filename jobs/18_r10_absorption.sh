#!/bin/bash -l
#SBATCH -J vs3_r10_absorb
#SBATCH -o r10_absorb_%j.out
#SBATCH -e r10_absorb_%j.err
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

OUT="$REPO/results/e5_token_absorption"
mkdir -p "$OUT"

cd "$REPO/code"
set +e
python3 experiments/e5_mechanism.py \
    --teacher llama31-8b \
    --student llama32-1b \
    --arms topk_fkl \
    --alpha-ce 0.0 \
    --k 256 \
    --absorb-report topk_fkl \
    --probe-batches 32 \
    --steps 500 \
    --seeds 0 \
    --out "$OUT"
exit_code=$?
set -e

if [ "$exit_code" -eq 0 ]; then
    echo "R10 absorption run done. $OUT/e5.json's final trajectory row"
    echo "(step=500) has absorption_top_ids (token ids, ranked) and"
    echo "absorption_top_mass (their accumulated tail probability mass)."
    echo "Decode the ids with the student's tokenizer"
    echo "(STUDENTS['llama32-1b'][0] in code/experiments/common.py) to see"
    echo "whether the top absorbers form a recognizable class (e.g."
    echo "punctuation, rare subwords, a specific script) -- if so, that is"
    echo "the qualitative finding for App. F; if the top ids look like an"
    echo "unstructured long tail, report that plainly instead. This closes"
fi
exit "$exit_code"
