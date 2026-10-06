#!/usr/bin/env python3
"""T1: blind-subspace probe. Direct test of Theorem 1. Resolves blocking item B8.

Theorem 1 makes two predictions about the SAME set of directions A-perp:
    (i)  grad L_S vanishes there exactly   -- the objective is blind
    (ii) grad m_p does not vanish there    -- but tail mass moves

The paper currently tests only the downstream drift, which several mechanisms
could produce. This probe separates them. It needs one forward/backward pass per
position and NO training.

    A       = span{w_v - w_v0 : v in S}       (student unembedding rows, centred)
    P       = projector onto A-perp
    ratio_L = ||P grad_h L_S|| / ||grad_h L_S||     predicted ~ float epsilon
    ratio_m = ||P grad_h m_p|| / ||grad_h m_p||     predicted bounded away from 0

Both gradients are computed in closed form, not by autograd, so the result is a
statement about the objective rather than about an implementation:
    grad_h L_S = c_S^p - c_S^q
    grad_h m_p = -m_p (1 - m_p) (c_S^p - c_T^p)          [Eq. 4]

The script prints what it measures and asserts nothing about the outcome. If the
prediction fails it will say so; it will not quietly pass.

Usage
-----
    python t1_blind_subspace.py \
        --teacher Qwen/Qwen3-8B --student Qwen/Qwen3-0.6B \
        --prompts prompts.jsonl --k 64 --positions 256 \
        --out results/t1_blind_subspace/t1.json

Set --k to the cache size under test. Sweep it (64, 256, 1024, and the full
vocabulary) to obtain the negative control: as k -> V the blind subspace becomes
trivial and ratio_m must fall to zero. That sweep is the cheap half of T2.
"""
from __future__ import annotations
import argparse, json, os, sys
from dataclasses import dataclass, asdict

import numpy as np

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # keep the failure legible rather than a stack trace
    sys.exit("t1 needs torch and transformers; see requirements.txt")


@dataclass
class PositionResult:
    k: int
    dim_A: int
    dim_Aperp: int
    ratio_L: float          # ||P grad L_S|| / ||grad L_S||   -> expect ~1e-16
    ratio_m: float          # ||P grad m_p|| / ||grad m_p||   -> expect >> 0
    grad_L_norm: float
    grad_m_norm: float
    m_p: float


def probe_one(W: np.ndarray, h: np.ndarray, q: np.ndarray, k: int,
              rank_tol: float = 1e-10) -> PositionResult:
    """One supervised position. W is the student unembedding (V x d').

    Two edge cases handled explicitly, both hit by the full-vocabulary
    control (k = V) and both real bugs in an earlier version of this
    function that ran to completion with silent NaNs / never finished:

    1. m_p = 0 exactly whenever S covers the whole vocabulary (k = V), so
       ~mask is empty and c_T_p = (empty sum) / 0 is a genuine 0/0, not a
       numerical-precision edge case. m_p = 0 makes grad_m identically zero
       by Eq. (4)'s own m_p*(1-m_p) prefactor regardless of c_T_p, so the
       mathematically correct value is grad_m = 0, computed without ever
       evaluating c_T_p when the tail is empty.
    2. dim(A) <= min(k-1, d') always (A is spanned by k-1 centred rows of a
       d'-dimensional space), so A already saturates the full space once
       k-1 >= d': the SVD below would return rank r = d' regardless of how
       large k grows past that point, at O(k * d'^2) cost that dominated a
       real ~1-hour SLURM budget once k approached V. A-perp = {0} is known
       in closed form once k-1 >= d', so the SVD is skipped entirely there:
       ratio_L and ratio_m are both defined as 0 (the projection onto the
       zero subspace is zero), matching what the SVD would have returned
       had it actually run to convergence.
    """
    V = W.shape[0]
    d_ = W.shape[1]
    S = np.sort(np.argsort(-q)[:k])
    mask = np.zeros(V, dtype=bool); mask[S] = True

    z = W @ h
    z = z - z.max()
    p = np.exp(z); p /= p.sum()

    q_t = q[mask] / q[mask].sum()
    p_t = p[mask] / p[mask].sum()
    m_p = float(1.0 - p[mask].sum())

    c_S_q = q_t @ W[mask]
    c_S_p = p_t @ W[mask]

    grad_L = c_S_p - c_S_q                       # gradient of L_S
    if m_p <= 0.0:
        # Empty tail: Eq. (4)'s m_p*(1-m_p) prefactor is exactly zero, so
        # grad_m is exactly zero regardless of c_T_p (which is undefined,
        # 0/0, and must not be evaluated).
        grad_m = np.zeros_like(grad_L)
    else:
        c_T_p = (p[~mask] @ W[~mask]) / m_p
        grad_m = -m_p * (1.0 - m_p) * (c_S_p - c_T_p)  # Eq. (4)

    if k - 1 >= d_:
        # A already spans R^{d'}; A-perp = {0} in closed form, no SVD
        # needed (and an SVD of a (k-1) x d' matrix at large k is the real
        # cost driver -- this is what timed out a 1-hour job at k = V).
        r = d_
        ratio_L, ratio_m = 0.0, 0.0
    else:
        # Orthonormal basis of A, then the projector onto its complement.
        # Economy SVD of the (k-1) x d' centred head rows; far cheaper than
        # a dense full_matrices SVD at realistic d'.
        Amat = W[mask][1:] - W[mask][0]
        U_, sv, Vt = np.linalg.svd(Amat, full_matrices=False)
        r = int((sv > rank_tol * max(1.0, sv[0])).sum())
        Q = Vt[:r]                                   # rows span A
        proj_perp = lambda g: g - Q.T @ (Q @ g)      # component in A-perp
        nL_full, nm_full = np.linalg.norm(grad_L), np.linalg.norm(grad_m)
        ratio_L = float(np.linalg.norm(proj_perp(grad_L)) / max(nL_full, 1e-300))
        ratio_m = float(np.linalg.norm(proj_perp(grad_m)) / max(nm_full, 1e-300))

    nL, nm = np.linalg.norm(grad_L), np.linalg.norm(grad_m)
    return PositionResult(
        k=k, dim_A=r, dim_Aperp=d_ - r,
        ratio_L=ratio_L, ratio_m=ratio_m,
        grad_L_norm=float(nL), grad_m_norm=float(nm), m_p=m_p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--prompts", required=True, help="jsonl with a 'text' field")
    ap.add_argument("--k", type=int, nargs="+", default=[64],
                    help="cache sizes to probe; include the full vocabulary for the control")
    ap.add_argument("--positions", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="float64",
                    help="float64 recommended: ratio_L is a cancellation measurement")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.teacher)
    teacher = AutoModelForCausalLM.from_pretrained(a.teacher, torch_dtype=torch.float32).eval()
    student = AutoModelForCausalLM.from_pretrained(a.student, torch_dtype=torch.float32).eval()

    W = student.get_output_embeddings().weight.detach().cpu().numpy().astype(a.dtype)
    if teacher.config.vocab_size != student.config.vocab_size:
        sys.exit("teacher and student vocabularies differ; the probe assumes a shared vocabulary")

    texts = [json.loads(l)["text"] for l in open(a.prompts)][: a.positions]
    out, n_seen = [], 0
    with torch.no_grad():
        for text in texts:
            ids = tok(text, return_tensors="pt", truncation=True, max_length=1024)
            th = teacher(**ids, output_hidden_states=True)
            sh = student(**ids, output_hidden_states=True)
            for t in range(ids["input_ids"].shape[1]):
                if n_seen >= a.positions:
                    break
                q = torch.softmax(th.logits[0, t].double(), -1).cpu().numpy()
                h = sh.hidden_states[-1][0, t].double().cpu().numpy().astype(a.dtype)
                for k in a.k:
                    out.append(asdict(probe_one(W, h, q, min(k, W.shape[0]))))
                n_seen += 1
            if n_seen >= a.positions:
                break

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"config": vars(a), "positions": out}, open(a.out, "w"), indent=1)

    print(f"{'k':>7} {'n':>5} {'dim A':>7} {'dim A-perp':>11} "
          f"{'median ratio_L':>15} {'median ratio_m':>15}")
    for k in a.k:
        rows = [r for r in out if r["k"] == min(k, W.shape[0])]
        if not rows:
            continue
        rL = np.median([r["ratio_L"] for r in rows])
        rm = np.median([r["ratio_m"] for r in rows])
        print(f"{k:>7} {len(rows):>5} {np.median([r['dim_A'] for r in rows]):>7.0f} "
              f"{np.median([r['dim_Aperp'] for r in rows]):>11.0f} {rL:>15.2e} {rm:>15.4f}")
    print("\nTheorem 1 predicts ratio_L at float precision and ratio_m bounded away")
    print("from zero, with ratio_m falling toward zero as k approaches V.")
    print("This script reports the measurement; it does not assert the prediction.")


if __name__ == "__main__":
    main()
