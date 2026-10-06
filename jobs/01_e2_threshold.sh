#!/bin/bash -l
#SBATCH -J hsc_e2_threshold
#SBATCH -o e2_threshold_%A_%a.out
#SBATCH -e e2_threshold_%A_%a.err
#SBATCH -p <gpu-partition>
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 80000MB
#SBATCH -A <account>
#SBATCH -q <qos>
#SBATCH -t 08:00:00
#SBATCH --array=0-5%1

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate hsc

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
[ -f "$REPO/.env" ] && { set -a; source "$REPO/.env"; set +a; }

TEACHERS=(qwen3-1.7b qwen3-8b qwen3-32b llama31-8b gemma3-4b gemma3-12b)
TEACHER="${TEACHERS[$SLURM_ARRAY_TASK_ID]}"

OUT="$REPO/results/e2_threshold"
mkdir -p "$OUT"

cd "$REPO/code"
set +e
python3 experiments/e2_threshold.py \
    --teachers "$TEACHER" \
    --n-positions 64 \
    --seqlen 512 \
    --tol 1e-6 \
    --probe 50000 \
    --all-rules \
    --seed 0 \
    --out "$OUT"
exit_code=$?
set -e

python3 - "$TEACHER" <<'PY'
import shutil, sys
sys.path.insert(0, "experiments")
from huggingface_hub import scan_cache_dir
from common import TEACHERS
hf_id = TEACHERS[sys.argv[1]][0]
for repo in scan_cache_dir().repos:
    if repo.repo_id == hf_id:
        print(f"removing cached weights for {hf_id} ({repo.size_on_disk / 1e9:.1f} GB)")
        shutil.rmtree(repo.repo_path, ignore_errors=True)
        break
else:
    print(f"no cache entry found for {hf_id} (nothing to clean up)")
PY

if [ "$exit_code" -eq 0 ]; then
    echo "E2/$TEACHER done: $OUT/e2_$TEACHER.json. Once all 6 array tasks"
    echo "finish, run:"
    echo "  python3 $REPO/code/experiments/merge_e2_threshold.py --dir $OUT --out $OUT/e2.json"
    echo "to combine them and print the required_k-vs-d scaling fit and"
else
    echo "E2/$TEACHER FAILED (exit $exit_code). Per README.md: stop and" >&2
    echo "reconsider before spending anything else if the d+1 transition" >&2
    echo "does not reproduce for this teacher." >&2
fi
exit "$exit_code"
