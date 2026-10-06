#!/usr/bin/env python3
"""Regenerate the numeric bodies of Tables 2-4 from result files.

Tables 2 and 3 are arithmetic in d and V, so this recomputes them outright and
they can never drift from the definitions. Table 4 reads the threshold sweep, so
it needs results/e2_threshold/e2.json; if that file is absent the script says so
and emits nothing for it rather than inventing rows.

    python code/analysis/make_tables.py            # prints LaTeX bodies
"""
from __future__ import annotations
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (name, d, V). The two marked True had V taken from the published model
# configuration rather than from our sweeps; see the footnote on Table 3.
TEACHERS = [("Qwen3-1.7B", 2048, 151936, False), ("Gemma-3-4B", 2560, 262144, False),
            ("Gemma-3-12B", 3840, 262144, True), ("Llama-3.1-8B", 4096, 128256, False),
            ("Qwen3-8B", 4096, 151936, False), ("Qwen3-32B", 5120, 151936, True),
            ("Llama-3.1-70B", 8192, 128256, False)]
COMPUTE = [("Gemma-3-4B", 2560, 262144, 7.56), ("Llama-3.1-8B", 4096, 128256, 16.11),
           ("Llama-3.1-70B", 8192, 128256, 147.64)]   # body GFLOP at seqlen 8192


def storage_table() -> str:
    rows = []
    for n, d, V, footnoted in TEACHERS:
        bf16, int8, full = 2 * d, d + 4, 2 * V
        mark = "\\tnote{b}" if footnoted else ""
        rows.append(f"{n:<13} & {d:>4} & {V}{mark} & {bf16:>5} & {int8:>4} & "
                    f"{full/bf16:>5.1f} & {int8/384:>5.2f} \\\\")
    return "\n".join(rows)


def compute_table() -> str:
    rows = []
    for n, d, V, body in COMPUTE:
        head = 2 * d * V / 1e9
        rows.append(f"{n:<13} & {d:>4} & {V} & {body:>6.2f} & {head:.2f} & "
                    f"{100*body/(body+head):.1f} \\\\")
    return "\n".join(rows)


def threshold_table() -> str | None:
    path = os.path.join(ROOT, "results", "e2_threshold", "e2.json")
    if not os.path.exists(path):
        return None
    data = json.load(open(path))
    rows = []
    for t in sorted(data.get("teachers", []), key=lambda x: x["d"]):
        delta = t["collapse_k"] - (t["d"] + 1)
        cell = "$0$" if delta == 0 else f"$\\mathbf{{{delta:+d}}}$"
        rows.append(f"{t['name']:<13} & {t['d']:>4} & {t['collapse_k']:>4} & "
                    f"{t['ones_residual']:.3f} & {cell} \\\\")
    return "\n".join(rows)


if __name__ == "__main__":
    print("% ---- Table 3 (storage), recomputed from d and V ----")
    print(storage_table())
    print("\n% ---- Table 2 (compute), head = 2dV ----")
    print(compute_table())
    print("\n% ---- Table 4 (threshold) ----")
    t = threshold_table()
    if t is None:
        print("% results/e2_threshold/e2.json absent: not emitted.")
        print("% The table in main.tex was typed from that file; until it is in")
        print("% the bundle its rows cannot be regenerated or checked here.")
        sys.exit(0)
    print(t)
