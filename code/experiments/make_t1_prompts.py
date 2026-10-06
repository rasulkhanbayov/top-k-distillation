#!/usr/bin/env python3
"""Builds prompts.jsonl for t1_blind_subspace.py from the same corpus this
project's other experiments already use (fineweb-edu), so the probe runs on
real text rather than requiring a new download/dependency.

One line per document, {"text": ...}, which is exactly what
t1_blind_subspace.py reads (`json.loads(l)["text"]`).
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import _load_corpus_docs

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=64,
                   help="number of documents; each yields many supervised "
                        "positions, so this need not equal --positions in "
                        "t1_blind_subspace.py")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--corpus", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--out", default="prompts.jsonl")
    a = p.parse_args()

    docs = _load_corpus_docs(a.corpus, a.seed)[: a.n]
    with open(a.out, "w") as fh:
        for d in docs:
            fh.write(json.dumps({"text": d}) + "\n")
    print(f"wrote {len(docs)} documents to {a.out}")
