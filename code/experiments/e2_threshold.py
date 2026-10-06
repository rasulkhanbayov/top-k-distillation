#!/usr/bin/env python3
"""
E2 (T1). The identifiability threshold on real teachers, and its scaling with d.

Claims under test:
  (a) reconstruction error collapses at k = d+1 and not before, for every
      selection rule with full-rank D_S, independent of V and corpus;
  (b) the required k grows proportionally with d, so a fixed cache budget incurs
      a widening deficit as teachers scale.

Claim (b) is the one with the most consequence and the one we most want
falsified if it is wrong. It is also the cheapest experiment in the paper:
forward passes and linear solves, no training.

Also verifies the Section 3 standing assumptions per teacher, including how close
range(U) comes to containing the all-ones vector.

Usage:
  python e2_threshold.py --teachers qwen3-1.7b qwen3-8b llama31-70b gemma3-4b
"""
import argparse, sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch

from common import load_teacher, teacher_hidden_states, set_seed, record, TEACHERS
from hsc.reconstruct import (solve_hidden_state, solve_hidden_state_underdetermined,
                             solve_hidden_state_softcap,
                             reconstruct_logits, identifiability_report, apply_softcap)

SELECTIONS = ("topk", "random", "leverage")


def select(U, lp, k, rule, rng, leverage_order=None):
    V = U.shape[0]
    if rule == "topk":
        return np.argsort(lp)[::-1][:k].copy()
    if rule == "random":
        return rng.choice(V, k, replace=False)
    if rule == "leverage":
        # leverage_order is precomputed once per teacher by _leverage_order:
        # the ordering does not depend on k or on the sampled position, so
        # recomputing the QR per (position, k) call wasted ~64 * 9 = 576
        # redundant O(V * d^2)-ish decompositions per teacher and was the
        # dominant cost behind E2 timing out mid-sweep on qwen3-8b.
        assert leverage_order is not None, "leverage_order must be precomputed"
        return leverage_order[:k]
    raise ValueError(rule)


def _leverage_order(U, rng, n_probe):
    # approximate statistical leverage of the unembedding rows, computed once
    # per teacher and reused across every (position, k) pair in the sweep
    V = U.shape[0]
    sub = rng.choice(V, min(V, n_probe), replace=False)
    Q, _ = np.linalg.qr(U[sub])
    lev = (Q ** 2).sum(1)
    return sub[np.argsort(lev)[::-1]]


def main(a):
    set_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    # One file per invocation's teacher list, not a single shared e2.json:
    # concurrent SLURM array tasks (one per teacher) each running this script
    # would otherwise read-modify-write the same file with no locking, and a
    # task finishing while another is mid-save can silently drop that other
    # task's result. merge_e2.py (or --merge below) combines the per-slug
    # files afterward. A multi-teacher --teachers list run in one process is
    # still safe to accumulate into a single file since there is no
    # concurrency within one process.
    slug = "-".join(sorted(a.teachers)) if len(a.teachers) > 1 else a.teachers[0]
    out_path = os.path.join(a.out, f"e2_{slug}.json")
    out = _load_partial(out_path) or {"teachers": {}}

    for name in a.teachers:
        if name in out["teachers"]:
            print(f"\n=== {name} already in {out_path}, skipping ===")
            continue

        model, tok, U_t, bias_t, softcap = load_teacher(name, device=a.device)
        U = U_t.float().cpu().numpy()
        b = bias_t.float().cpu().numpy() if bias_t is not None else None
        d = U.shape[1]

        assumptions = identifiability_report(U, n_probe=a.probe, seed=a.seed)
        print(f"\n=== {name}  d={d} V={U.shape[0]} ===")
        print(f"  full column rank {assumptions['full_column_rank']}   "
              f"cond {assumptions['cond']:.1f}   "
              f"||1-proj||/||1|| {assumptions['ones_residual']:.4f}   "
              f"anisotropy {assumptions['anisotropy']:.4f}")

        # real hidden states from real text, so the entropy profile is realistic
        gs = _sample_hidden(model, tok, a.n_positions, a.seqlen)
        del model; torch.cuda.empty_cache()

        # Precompute once per teacher, not once per (position, k) call: the
        # leverage ordering of U's rows does not depend on which hidden state
        # or which k is being evaluated. See _leverage_order's docstring note
        # in select() for why this was the dominant cost in earlier runs.
        leverage_order = _leverage_order(U, rng, a.probe) if a.all_rules else None

        rows = []
        ks = sorted({max(2, int(d * f)) for f in (0.125, 0.25, 0.5, 0.9, 1.0)} |
                    {d, d + 1, d + 2, 2 * d, 4 * d})
        for g in gs:
            z = reconstruct_logits(U, g, b)
            if softcap:
                z = apply_softcap(z, softcap)
            lp = z - (z.max() + np.log(np.exp(z - z.max()).sum()))
            for k in ks:
                if k > U.shape[0]:
                    continue
                for rule in SELECTIONS if a.all_rules else ("topk",):
                    S = select(U, lp, k, rule, rng, leverage_order)
                    solve_cost = None
                    if softcap:
                        # solve_hidden_state/_underdetermined assume the
                        # linear identity log q_v - log q_v0 = (u_v-u_v0)^T g
                        # + (b_v-b_v0), which does not hold under soft-capping
                        # (see solve_hidden_state_softcap's docstring): a real
                        # run fed soft-capped Gemma log-probs into the linear
                        # solver here and produced a spurious KL spike of 8-13
                        # nats right at k=d/d+1, which looked like threshold
                        # failure but was a solver-mismatch artifact. Below
                        # d+1 there is no valid closed form under softcap
                        # either (same nonlinearity issue, underdetermined),
                        # so skip rather than report a meaningless number.
                        if k < d + 1:
                            continue
                        gh, solve_cost = solve_hidden_state_softcap(U, lp[S], S, softcap, b=b)
                        # Not filtered on solve_cost: the downstream KL below
                        # already reflects a bad solve, and dropping high-cost
                        # rows here would bias _required_k's median optimistic
                        # by silently discarding the harder cases.
                    else:
                        f = (solve_hidden_state if k >= d + 1
                             else solve_hidden_state_underdetermined)
                        try:
                            gh = f(U, lp[S], S, b=b)
                        except ValueError:
                            continue
                    zh = reconstruct_logits(U, gh, b)
                    if softcap:
                        zh = apply_softcap(zh, softcap)
                    lph = zh - (zh.max() + np.log(np.exp(zh - zh.max()).sum()))
                    p = np.exp(lp)
                    rows.append({
                        "k": int(k), "k_over_d": k / d, "rule": rule,
                        "max_logp_err": float(np.abs(lp - lph).max()),
                        "kl": float((p * (lp - lph)).sum()),
                        "entropy": float(-(p * lp).sum()),
                        "solve_cost": solve_cost,
                    })

        # required k: smallest k whose median KL falls below the tolerance
        req = _required_k(rows, a.tol)
        print(f"  required k at KL<{a.tol:g}: {req}   (d+1 = {d+1})")
        out["teachers"][name] = {"d": d, "V": int(U.shape[0]),
                                 "assumptions": assumptions,
                                 "required_k": req, "rows": rows}

        # Save after every teacher, not just at the end: a walltime kill or
        # crash partway through the --teachers list used to lose every result
        # computed so far, including teachers that had already finished. A
        # rerun with the same --out and --teachers now resumes by skipping
        # teachers already present in this invocation's output file (see the
        # skip check at the top of this loop).
        record(out_path, {"config": vars(a), **out})

    # claim (b): required k against d. Only fires when this invocation's own
    # --teachers list has 3+ entries; for the one-teacher-per-array-task case,
    # run merge_e2_threshold.py across all per-teacher files instead.
    xs = [(v["d"], v["required_k"]) for v in out["teachers"].values()
          if v["required_k"] is not None]
    if len(xs) >= 3:
        d_, k_ = np.array(xs).T
        slope, icept = np.polyfit(d_, k_, 1)
        out["scaling"] = {"slope": float(slope), "intercept": float(icept),
                          "r2": float(np.corrcoef(d_, k_)[0, 1] ** 2)}
        print(f"\nrequired_k ~ {slope:.2f} * d + {icept:.0f}   "
              f"R^2 = {out['scaling']['r2']:.3f}")
        print("Prediction (b) holds if the slope is close to 1 and R^2 is high.")

    record(out_path, {"config": vars(a), **out})


def _load_partial(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        payload = json.load(f)
    # record() wraps the payload with a content hash; unwrap back to the
    # {"config": ..., "teachers": ...} shape main() works with.
    return {"teachers": payload.get("teachers", {})}


def _required_k(rows, tol):
    import collections
    by_k = collections.defaultdict(list)
    for r in rows:
        if r["rule"] == "topk":
            by_k[r["k"]].append(r["kl"])
    for k in sorted(by_k):
        if np.median(by_k[k]) < tol:
            return int(k)
    return None


def _sample_hidden(model, tok, n, L):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                      split="train", streaming=True)
    out, it = [], iter(ds)
    while len(out) < n:
        t = tok(next(it)["text"], truncation=True, max_length=L,
                return_tensors="pt")["input_ids"].cuda()
        if t.shape[1] < 32:
            continue
        h = teacher_hidden_states(model, t)[0]
        idx = np.random.randint(0, h.shape[0], size=min(8, n - len(out)))
        out.extend(h[idx].float().cpu().numpy())
    return out[:n]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--teachers", nargs="+", default=list(TEACHERS))
    p.add_argument("--device", default="cuda",
                   help="passed to load_teacher; use 'auto' for a teacher "
                        "too large for a single GPU (e.g. llama31-70b at "
                        "~140GB bf16), which shards it across all visible "
                        "GPUs via Accelerate's device_map='auto'. Every "
                        "other teacher in this project fits on one GPU "
                        "and should keep the default.")
    p.add_argument("--n-positions", type=int, default=64)
    p.add_argument("--seqlen", type=int, default=512)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--probe", type=int, default=50000)
    p.add_argument("--all-rules", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/e2")
    main(p.parse_args())
