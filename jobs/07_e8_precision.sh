#!/bin/bash -l
#SBATCH -J hsc_e8_precision
#SBATCH -o e8_precision_%j.out
#SBATCH -e e8_precision_%j.err
#SBATCH -p <gpu-partition>
#SBATCH -c 4
#SBATCH --mem 16000MB
#SBATCH -t 00:30:00

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate hsc

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT="$REPO/results/e8_precision_synthetic"
mkdir -p "$OUT"

cd "$REPO/code"
set +e
python3 experiments/e8_precision.py \
    --V 151936 \
    --d 4096 \
    --n 64 \
    --ks 64 256 1024 4096 \
    --seed 0 \
    --out "$OUT"
exit_code=$?
set -e

if [ "$exit_code" -eq 0 ]; then
    echo "E8 (synthetic) done. This is a synthetic-unembedding check only --"
    echo "(07b_e8_precision_real.sh) before quoting any crossover number in"
    echo "the paper."
fi
exit "$exit_code"
