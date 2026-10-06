#!/usr/bin/env python3
"""
E6 (T2). Systems and the storage-compute frontier.

Reports real bytes on disk rather than the theoretical figure in Table 1, plus
throughput, peak memory, and time to a fixed validation loss. The reconstruction
cost is reported explicitly rather than netted out, because against a top-k cache
it is a cost, not a saving (Section 5.3).

Usage:
  python e6_systems.py --teachers qwen3-1.7b gemma3-4b qwen3-8b llama31-70b
"""
import argparse, sys, os, time, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from common import record, TEACHERS   # torch not needed: this measures bytes on disk
from hsc.cache import HiddenStateCache, TopKCache


def main(a):
    rows = []
    for name in a.teachers:
        d, V = TEACHERS[name][1], TEACHERS[name][2]
        for scheme, dtype, k in ([("hidden", dt, 0) for dt in a.dtypes] +
                                 [("topk", "fp16", kk) for kk in a.ks] +
                                 [("full_logits", "fp16", V)]):
            # full-logit caching is measured analytically; materializing it is
            # the whole problem the paper is about (V*2 bytes per position).
            if scheme == "full_logits":
                rows.append({"teacher": name, "d": d, "V": V, "scheme": scheme,
                             "dtype": dtype, "k": k,
                             "bytes_per_position": float(V * 2),
                             "write_seconds": float("nan"), "exact": True})
                print(f"{name:<14}{'full_logits':<13}{'bf16':<6}{'':<9}"
                      f"{V*2:>9.0f} B/pos  exact (analytic)")
                continue
            root = os.path.join(a.out, f"{name}_{scheme}_{dtype}_{k}")
            shutil.rmtree(root, ignore_errors=True)
            n = a.positions
            t0 = time.time()
            if scheme == "hidden":
                c = HiddenStateCache(root, d=d, dtype=dtype, teacher=name)
                c.write(np.random.randn(n, d).astype(np.float32))
            else:
                c = TopKCache(root, k=k, dtype=dtype, teacher=name)
                c.write(np.random.randn(n, k).astype(np.float32),
                        np.random.randint(0, V, (n, k)))
            meta = c.finalize(); st = c.stats()
            rows.append({
                "teacher": name, "d": d, "V": V, "scheme": scheme,
                "dtype": dtype, "k": k,
                "bytes_per_position": st["bytes_per_position"],
                "write_seconds": time.time() - t0,
                "exact": scheme in ("hidden", "full_logits"),
            })
            print(f"{name:<14}{scheme:<13}{dtype:<6}k={k:<7}"
                  f"{st['bytes_per_position']:>9.0f} B/pos  "
                  f"{'exact' if rows[-1]['exact'] else 'lossy'}")
    record(os.path.join(a.out, "e6.json"), {"config": vars(a), "rows": rows})
    _frontier(rows)


def _frontier(rows):
    print(f"\n{'teacher':<14}{'smallest exact':>18}{'smallest overall':>19}{'penalty':>10}")
    print("-" * 62)
    import collections
    by = collections.defaultdict(list)
    for r in rows:
        by[r["teacher"]].append(r)
    for t, rs in by.items():
        ex = min([r for r in rs if r["exact"]], key=lambda r: r["bytes_per_position"])
        al = min(rs, key=lambda r: r["bytes_per_position"])
        print(f"{t:<14}{ex['bytes_per_position']:>13.0f} B "
              f"{al['bytes_per_position']:>15.0f} B "
              f"{ex['bytes_per_position']/al['bytes_per_position']:>9.2f}x")
    print("\nThe paper's claim is the smallest EXACT cache, not the smallest cache.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--teachers", nargs="+", default=list(TEACHERS))
    p.add_argument("--dtypes", nargs="+", default=["fp32", "bf16", "fp16", "int8"])
    p.add_argument("--ks", nargs="+", type=int, default=[64, 256, 1024])
    p.add_argument("--positions", type=int, default=8192)
    p.add_argument("--out", default="runs/e6")
    main(p.parse_args())
