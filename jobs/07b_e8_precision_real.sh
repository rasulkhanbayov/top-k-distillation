#!/bin/bash -l
#SBATCH -J hsc_e8_real
#SBATCH -o e8_real_%j.out
#SBATCH -e e8_real_%j.err
#SBATCH -p <gpu-partition>
#SBATCH -c 8
#SBATCH --mem 32000MB
#SBATCH -t 00:45:00

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate hsc

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
[ -f "$REPO/.env" ] && { set -a; source "$REPO/.env"; set +a; }

OUT="$REPO/results/e8_precision_real"
mkdir -p "$OUT"

cd "$REPO/code"
set +e
python3 experiments/e8_precision.py \
    --teacher qwen3-1.7b \
    --n 64 \
    --ks 64 256 1024 4096 \
    --seed 0 \
    --out "$OUT"
exit_code=$?
set -e

if [ "$exit_code" -eq 0 ]; then
    echo "E8 (real teacher) done. Compare the crossover point against"
    echo "07_e8_precision.sh's synthetic run and README.md's stated synthetic"
    echo "figure (int8 at 2052 B/pos, KL~4e-6, vs 6144 B for top-1024) -- if"
    echo "the real number differs meaningfully, update the paper's quoted"
    echo "figure rather than keep the synthetic one now that a real check"
    echo "exists."
fi
exit "$exit_code"
