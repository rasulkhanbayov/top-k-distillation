#!/bin/bash -l
#SBATCH -J hsc_setup
#SBATCH -o setup_%j.out
#SBATCH -e setup_%j.err
#SBATCH -p <gpu-partition>
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 40000MB
#SBATCH -A <account>
#SBATCH -q <qos>
#SBATCH -t 01:00:00

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"

if [ ! -f "$REPO/.env" ]; then
    echo "Missing $REPO/.env with HF_TOKEN=... -- create it before running any" >&2
    exit 1
fi

conda create -y -n hsc python=3.11
conda activate hsc

pip install -r "$REPO/code/requirements.txt"

[ -f "$REPO/.env" ] && { set -a; source "$REPO/.env"; set +a; }

cd "$REPO/code"
python3 - <<'PY'
import shutil
import torch
from huggingface_hub import scan_cache_dir
from experiments.common import TEACHERS, load_teacher

SKIP_TEACHERS = {"llama31-70b"}

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

for name in TEACHERS:
    if name in SKIP_TEACHERS:
        print(f"--- {name} SKIPPED (disk budget, see SKIP_TEACHERS) ---")
        continue
    hf_id = TEACHERS[name][0]
    print(f"--- {name} ({hf_id}) ---")
    model, tok, U, bias, softcap = load_teacher(name, device="cuda", dtype="bfloat16")
    ids = tok("The quick brown fox", return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        out = model(ids)
    print(f"  logits shape: {tuple(out.logits.shape)}, OK")
    del model, tok, U, bias
    torch.cuda.empty_cache()

    cache = scan_cache_dir()
    for repo in cache.repos:
        if repo.repo_id == hf_id:
            print(f"  removing cached weights for {hf_id} "
                  f"({repo.size_on_disk / 1e9:.1f} GB) to stay under quota")
            shutil.rmtree(repo.repo_path, ignore_errors=True)
            break
    else:
        print(f"  WARNING: no cache entry found for {hf_id} to remove")
PY

echo "Setup and gated-model access verified for all teachers in TEACHERS."
