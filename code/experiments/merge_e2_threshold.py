#!/usr/bin/env python3
"""
Combine the per-teacher e2_<teacher>.json files that 01_e2_threshold.sh's
SLURM array produces (one file per array task, to avoid a shared-file race
between concurrently running tasks) into one e2.json with the required_k vs d
scaling fit, matching what a single non-array e2_threshold.py run over all
teachers would have produced directly.

Usage:
  python merge_e2_threshold.py --dir results/e2_threshold --out results/e2_threshold/e2.json
"""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from common import record


def main(a):
    teachers = {}
    for path in sorted(glob.glob(os.path.join(a.dir, "e2_*.json"))):
        if os.path.basename(path) == "e2.json":
            continue
        with open(path) as f:
            payload = json.load(f)
        for name, entry in payload.get("teachers", {}).items():
            if name in teachers:
                print(f"WARNING: {name} present in more than one file "
                      f"(latest: {path}); keeping the first one seen.",
                      file=sys.stderr)
                continue
            teachers[name] = entry

    if not teachers:
        print(f"No e2_*.json files with teacher results found under {a.dir}",
              file=sys.stderr)
        sys.exit(1)

    out = {"teachers": teachers}
    xs = [(v["d"], v["required_k"]) for v in teachers.values()
          if v["required_k"] is not None]
    if len(xs) >= 3:
        d_, k_ = np.array(xs).T
        slope, icept = np.polyfit(d_, k_, 1)
        out["scaling"] = {"slope": float(slope), "intercept": float(icept),
                          "r2": float(np.corrcoef(d_, k_)[0, 1] ** 2)}
        print(f"required_k ~ {slope:.2f} * d + {icept:.0f}   "
              f"R^2 = {out['scaling']['r2']:.3f}")
        print("Prediction (b) holds if the slope is close to 1 and R^2 is high.")
    else:
        print(f"Only {len(xs)} teacher(s) with a required_k so far; need at "
              f"least 3 for the scaling fit.")

    for name, v in teachers.items():
        print(f"  {name}: d={v['d']} required_k={v['required_k']} "
              f"(d+1={v['d']+1})  ones_residual={v['assumptions']['ones_residual']:.4f}")

    record(a.out, out)
    print(f"\nWrote merged result to {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="results/e2_threshold")
    p.add_argument("--out", default="results/e2_threshold/e2.json")
    main(p.parse_args())
