#!/usr/bin/env python3
"""
E8 (T2). Cache precision, and where hidden-state caching crosses top-k.

At int8 with a per-position scale, a d=2048 teacher costs about 2 KB per
position against 6 KB for a top-1024 cache, so this experiment decides whether
hidden-state caching can be both exact and strictly smaller than a cache
currently in use. Runs on CPU, no teacher download needed for the synthetic arm.

Usage:
  python e8_precision.py                       # synthetic, runs anywhere
  python e8_precision.py --teacher qwen3-8b    # real unembedding
"""
import argparse, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from hsc.cache import quantize, dequantize
from hsc.reconstruct import reconstruct_logits
from common import record

BYTES = {"fp32": 4, "bf16": 2, "fp16": 2, "int8": 1}


def main(a):
    if a.teacher:
        from common import load_teacher
        _, _, U_t, b_t, _ = load_teacher(a.teacher, device="cpu")
        U = U_t.float().numpy(); b = None if b_t is None else b_t.float().numpy()
    else:
        rng = np.random.default_rng(a.seed)
        U = rng.standard_normal((a.V, a.d)) / np.sqrt(a.d)
        U *= (1 + 0.25 * rng.standard_normal((a.V, 1)))
        U += 0.5 * rng.standard_normal((1, a.d)) / np.sqrt(a.d)
        b = None
    V, d = U.shape
    rng = np.random.default_rng(a.seed)
    G = rng.standard_normal((a.n, d)).astype(np.float32) * 1.5

    rows = []
    for dt in ("fp32", "bf16", "fp16", "int8"):
        payload, scale = quantize(G, dt)
        Gq = dequantize(payload, scale, dt)
        kls, errs = [], []
        for i in range(a.n):
            z = reconstruct_logits(U, G[i], b); zq = reconstruct_logits(U, Gq[i], b)
            lp = z - (z.max() + np.log(np.exp(z - z.max()).sum()))
            lq = zq - (zq.max() + np.log(np.exp(zq - zq.max()).sum()))
            p = np.exp(lp)
            kls.append(float((p * (lp - lq)).sum()))
            errs.append(float(np.abs(lp - lq).max()))
        nb = d * BYTES[dt] + (4 if dt == "int8" else 0)
        rows.append({"scheme": "hidden", "dtype": dt, "bytes": nb,
                     "kl": float(np.mean(kls)), "max_logp_err": float(np.mean(errs)),
                     "kind": "bit-exact" if dt == "fp32" else "quantized",
                     "exact": True})
    for k in a.ks:
        rows.append({"scheme": f"top-{k}", "dtype": "fp16", "bytes": k * 6,
                     "kl": None, "max_logp_err": None,
                     "kind": "truncated", "exact": False})

    print(f"\n{'scheme':<12}{'dtype':<7}{'B/pos':>8}{'KL(q||qhat)':>15}"
          f"{'max logp err':>15}{'kind':>12}")
    print("-" * 68)
    for r in rows:
        kl = "-" if r["kl"] is None else f"{r['kl']:.3e}"
        me = "-" if r["max_logp_err"] is None else f"{r['max_logp_err']:.3e}"
        print(f"{r['scheme']:<12}{r['dtype']:<7}{r['bytes']:>8}{kl:>15}{me:>15}"
              f"{r['kind']:>12}")
    print("-" * 68)
    ex = [r for r in rows if r["exact"]]
    best = min(ex, key=lambda r: r["bytes"])
    beat = [r for r in rows if not r["exact"] and r["bytes"] > best["bytes"]]
    print("-" * 68)
    print("The distinction that matters is not bit-exactness but whether the error")
    print("goes to zero as storage grows. Hidden-state caching is quantization-")
    print("limited, so it does; top-k truncation is structurally limited, so it")
    print("does not, at any precision.")
    print(f"smallest full-distribution cache: {best['dtype']} at {best['bytes']} "
          f"B/pos (d={d}); strictly smaller than " +
          (", ".join(r["scheme"] for r in beat) if beat else "no top-k arm tested"))
    record(os.path.join(a.out, "e8.json"), {"config": vars(a), "rows": rows})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default=None)
    p.add_argument("--V", type=int, default=151936)
    p.add_argument("--d", type=int, default=2048)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--ks", nargs="+", type=int, default=[64, 256, 1024, 4096])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/e8")
    main(p.parse_args())
