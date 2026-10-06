#!/bin/bash -l
#SBATCH -J hsc_e3_offpolicy
#SBATCH -o e3_offpolicy_%j.out
#SBATCH -e e3_offpolicy_%j.err
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

OUT="$REPO/results/e3_offpolicy_3seed"
mkdir -p "$OUT"

cd "$REPO/code"
set +e
python3 experiments/e3_offpolicy.py \
    --teacher llama31-8b \
    --student llama32-1b \
    --arms full_fkl_online full_fkl_cached topk_fkl \
    --match storage \
    --budget-bytes 8192 \
    --k 256 \
    --top-p 0.95 \
    --temperature 1.0 \
    --cache-dtype bf16 \
    --chunk 16384 \
    --lr 2e-5 \
    --steps 500 \
    --batch 4 \
    --seqlen 512 \
    --cache-batches 500 \
    --seeds 0 1 2 \
    --out "$OUT"
exit_code=$?
set -e

if [ "$exit_code" -eq 0 ]; then
    echo "E3 done. Confirm hidden-state caching matches full_fkl_online and"
    echo "beats topk_fkl at matched storage per the registered prediction;"
    echo "report bootstrap CIs and the paired permutation test, not a"
    echo "t-test, per README.md's reproducibility section. Remaining scope:"
    echo "topp_fkl/bild/full_rkl/full_jsd/full_skew/feature arms and the"
    echo "full lm-eval suite are still unimplemented, see e3_offpolicy.py's"
    echo "SCOPE NOTE."
fi
exit "$exit_code"
