#!/bin/bash -l
#SBATCH -J vs3_28_tailmass
#SBATCH -o tail_mass_%j.out
#SBATCH -e tail_mass_%j.err
#SBATCH -p <gpu-partition>
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 100000MB
#SBATCH -A <account>
#SBATCH -q <qos>
#SBATCH -t 06:00:00

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate hsc
REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
[ -f "$REPO/.env" ] && { set -a; source "$REPO/.env"; set +a; }
OUT="$REPO/results/tail_mass"; mkdir -p "$OUT"
cd "$REPO/code"

status=0
for t in qwen3-1.7b qwen3-4b gemma3-4b llama32-3b smollm2-1.7b; do
    set +e
    python3 experiments/diag_tail_mass.py "$t" --batches 32 --out "$OUT"
    c=$?
    set -e
    echo "$t: exit $c"
    status=$(( status + c ))
done
[ "$status" -eq 0 ] && echo "Done: $OUT/tail_mass_<teacher>.json for all five teachers."
exit "$status"
