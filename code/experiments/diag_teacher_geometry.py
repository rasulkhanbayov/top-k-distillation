#!/usr/bin/env python3
"""
Standalone diagnostic, no training: loads one registered teacher and reports
identifiability_report() (condition number, anisotropy, ones-residual) for
its unembedding U, optionally writing it to JSON.

Usage: diag_teacher_geometry.py <teacher> [--out DIR] [--probe N] [--device D]
--probe and --seed default to the Threshold experiment's settings (50000, 0),
so values are comparable with the assumptions block of e2_<teacher>.json.
"""
import argparse, sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common import load_teacher, record
from hsc.reconstruct import identifiability_report

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("teacher")
    p.add_argument("--out", default=None)
    p.add_argument("--probe", type=int, default=50000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    model, tok, U_t, bias_t, softcap = load_teacher(a.teacher, device=a.device)
    print(f"{a.teacher}: loaded OK, U shape {tuple(U_t.shape)}, dtype {U_t.dtype}", flush=True)
    U = U_t.float().cpu().numpy()
    b = bias_t.float().cpu().numpy() if bias_t is not None else None
    rep = identifiability_report(U, b, n_probe=a.probe, seed=a.seed)
    print(json.dumps(rep, indent=2))
    if a.out:
        record(os.path.join(a.out, f"geometry_{a.teacher}.json"),
               {"config": vars(a), "teacher": a.teacher, "report": rep})
