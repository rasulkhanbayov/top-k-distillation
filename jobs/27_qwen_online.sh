#!/bin/bash -l
#SBATCH -J vs3_27_qwen_online
#SBATCH -o qwen_online_%j.out
#SBATCH -e qwen_online_%j.err
#SBATCH -p <gpu-partition>
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 100000MB
#SBATCH -A <account>
#SBATCH -q <qos>
#SBATCH -t 1-00:00:00

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate hsc
REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
[ -f "$REPO/.env" ] && { set -a; source "$REPO/.env"; set +a; }
cd "$REPO/code"

status=0
for pair in "qwen3-1.7b qwen3-0.6b e3_online_qwen3-1.7b" "qwen3-4b qwen3-1.7b e3_online_qwen3-4b"; do
    set -- $pair
    OUT="$REPO/results/$3"; mkdir -p "$OUT"
    set +e
    python3 experiments/e3_offpolicy.py \
        --teacher "$1" --student "$2" \
        --arms full_fkl_online --match storage --budget-bytes 6144 \
        --k 1024 --cache-dtype bf16 --chunk 16384 --lr 2e-5 \
        --steps 500 --batch 4 --seqlen 512 --cache-batches 500 \
        --seeds 0 1 2 3 4 5 --out "$OUT"
    c=$?
    set -e
    echo "$1 online: exit $c -> $OUT/e3.json"
    status=$(( status + c ))
done
if [ "$status" -eq 0 ]; then
    echo "Compare full_fkl_online with full_fkl_cached and topk_fkl from"
    echo "results/e3_matched_qwen3-1.7b (qwen3-1.7b) and results/e3_matched_qwen3-4b (qwen3-4b)"
    echo "with stats.py compare(), seed-matched."
fi
exit "$status"
