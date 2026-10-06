#!/usr/bin/env python3
"""decode the token ids in e5.json's absorption_top_ids
into readable strings, using the same student tokenizer the run used. No
GPU needed -- this only loads a tokenizer, not a model.

    python code/analysis/decode_absorption.py \
        results/e5_token_absorption/e5.json --student llama32-1b
"""
from __future__ import annotations
import argparse, json, sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "code"))
from experiments.common import STUDENTS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("json_path")
    p.add_argument("--student", default="llama32-1b")
    p.add_argument("--top", type=int, default=40)
    a = p.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(STUDENTS[a.student][0])

    d = json.load(open(a.json_path))
    for row in d["rows"]:
        for point in row["trajectory"]:
            if "absorption_top_ids" not in point:
                continue
            ids = point["absorption_top_ids"][: a.top]
            mass = point["absorption_top_mass"][: a.top]
            total = point.get("absorption_total_tail_mass")
            print(f"=== arm={row['arm']} alpha_ce={row['alpha_ce']} "
                  f"seed={row['seed']} step={point['step']} "
                  f"total_tail_mass={total} ===")
            for rank, (tid, m) in enumerate(zip(ids, mass), start=1):
                piece = tok.convert_ids_to_tokens([tid])[0]
                print(f"  {rank:3d}. id={tid:>7} mass={m:.6f} token={piece!r}")


if __name__ == "__main__":
    main()
