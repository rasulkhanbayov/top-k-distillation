#!/bin/bash -l
#SBATCH -J hsc_e6_systems
#SBATCH -o e6_systems_%j.out
#SBATCH -e e6_systems_%j.err
#SBATCH -p <gpu-partition>
#SBATCH -c 4
#SBATCH --mem 16000MB
#SBATCH -t 00:30:00

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate hsc

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT="$REPO/results/e6_systems"
mkdir -p "$OUT"

cd "$REPO/code"
set +e
python3 experiments/e6_systems.py \
    --teachers qwen3-1.7b qwen3-8b qwen3-32b llama31-8b llama31-70b gemma3-4b gemma3-12b \
    --dtypes fp32 bf16 fp16 int8 \
    --ks 64 256 1024 \
    --positions 8192 \
    --out "$OUT"
exit_code=$?
set -e

if [ "$exit_code" -eq 0 ]; then
    echo "E6 done. Inspect $OUT/e6.json's frontier table: the paper's claim is"
    echo "the smallest EXACT cache (hidden-state, some dtype), not the smallest"
    echo "cache overall -- top-k at small k will usually be smaller but is lossy."
fi
exit "$exit_code"
