#!/bin/bash -l
#SBATCH -J vs3_t2_k4096
#SBATCH -o t2_k4096_%j.out
#SBATCH -e t2_k4096_%j.err
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

OUT="$REPO/results/t2_drift_vs_k/k4096"
mkdir -p "$OUT"

cd "$REPO/code"
set +e
python3 experiments/e5_mechanism.py \
    --teacher llama31-8b \
    --student llama32-1b \
    --arms topk_fkl \
    --alpha-ce 0.0 \
    --k 4096 \
    --select teacher \
    --probe-batches 32 \
    --steps 500 \
    --seeds 0 1 2 3 4 5 \
    --out "$OUT"
exit_code=$?
set -e

if [ "$exit_code" -eq 0 ]; then
    echo "T2 k=4096 done -- this completes the k-sweep. Compare"
    echo "$OUT/e5.json's final-step |tail_mass_student - tail_mass_teacher|"
    echo "against k={16,64,256,1024} and the full-vocabulary control already"
    echo "in results/t2_drift_vs_k/, and fold into the paper's T2 paragraph"
fi
exit "$exit_code"
