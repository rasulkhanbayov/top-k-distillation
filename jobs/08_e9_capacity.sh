#!/bin/bash -l
#SBATCH -J hsc_e9_capacity
#SBATCH -o e9_capacity_%j.out
#SBATCH -e e9_capacity_%j.err
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

OUT="$REPO/results/e9_capacity"
mkdir -p "$OUT"

cd "$REPO/code"
set +e
python3 experiments/e9_capacity.py \
    --teachers qwen3-1.7b qwen3-8b \
    --students qwen3-0.6b qwen3-1.7b \
    --ks 64 256 1024 \
    --seeds 0 1 2 \
    --objectives topk_fkl full_fkl \
    --steps 150 \
    --batch 4 \
    --seqlen 512 \
    --chunk 16384 \
    --lr 1e-5 \
    --eval-batches 8 \
    --out "$OUT"
exit_code=$?
set -e

if [ "$exit_code" -eq 0 ]; then
    echo "E9 done. Read the two grouped-mean tables main() printed (and saved"
    echo "under $OUT/e9.json): eval KL vs k/(d_t+1) at fixed student"
    echo "(teacher-identifiability effect, Proposition 1) and eval KL vs"
    echo "k/(d_s+1) at fixed teacher (student-capacity effect, Theorem 3)."
    echo "Report these as partial effects, not a single regression on k --"
    echo "Remember qwen3-32b was deliberately dropped from this run (see"
    echo "header) -- note that scope limit in the writeup if this table is"
    echo "used to make a claim about the largest teacher scale."
fi
exit "$exit_code"
