#!/bin/bash -l
#SBATCH -J vs3_t1_blind
#SBATCH -o t1_blind_subspace_%j.out
#SBATCH -e t1_blind_subspace_%j.err
#SBATCH -p <gpu-partition>
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 100000MB
#SBATCH -A <account>
#SBATCH -q <qos>
#SBATCH -t 00:45:00

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate hsc

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
[ -f "$REPO/.env" ] && { set -a; source "$REPO/.env"; set +a; }

OUT="$REPO/results/t1_blind_subspace"
mkdir -p "$OUT"

PROMPTS="$OUT/prompts.jsonl"

cd "$REPO/code/experiments"
set +e

if [ -s "$PROMPTS" ]; then
    echo "Reusing existing $PROMPTS from a prior run."
else
    python3 make_t1_prompts.py --n 96 --seed 0 --corpus HuggingFaceFW/fineweb-edu \
        --out "$PROMPTS"
    prompts_exit=$?
    if [ "$prompts_exit" -ne 0 ]; then
        echo "Failed to build prompts.jsonl (exit $prompts_exit)." >&2
        exit "$prompts_exit"
    fi
fi

python3 t1_blind_subspace.py \
    --teacher Qwen/Qwen3-8B \
    --student Qwen/Qwen3-0.6B \
    --prompts "$PROMPTS" \
    --k 64 256 1024 151936 \
    --positions 256 \
    --seed 0 \
    --dtype float64 \
    --out "$OUT/t1.json"
exit_code=$?
set -e

if [ "$exit_code" -eq 0 ]; then
    echo "T1 done. Inspect $OUT/t1.json and the printed summary table:"
    echo "  ratio_L should be at float precision (~1e-16) at EVERY k -- this is"
    echo "  the claim that the truncated objective's gradient is exactly blind"
    echo "  to A-perp, regardless of k."
    echo "  ratio_m should be bounded well away from zero at small k (order"
    echo "  0.5-0.7) and fall toward 0 as k approaches 151936 (the full"
    echo "  vocabulary control) -- this is the claim that tail mass DOES move"
    echo "  along exactly the directions the objective cannot see."
    echo "If ratio_L is NOT at float precision, the paper's central mechanism"
    echo "anything else in the outstanding list -- report this plainly, do not"
    echo "average it away against the other k values."
fi
exit "$exit_code"
