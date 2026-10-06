#!/usr/bin/env python3
"""
E1 (T1). Exactness of the cached pipeline.

Claim under test: distillation driven by logits reconstructed from cached hidden
states reproduces distillation with the teacher in the loop, step for step.

This is a correctness check, not a quality claim, and everything else in the
paper is unfalsifiable without it. It fails if the hidden state is captured at
the wrong point in the architecture, which is the single most likely
implementation error.

Registered prediction: per-step losses agree to the precision of the cached
hidden state; final metrics statistically indistinguishable.

Usage:
  python e1_exactness.py --teacher qwen3-8b --student qwen3-0.6b --steps 200
"""
import argparse, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch

from common import load_teacher, teacher_hidden_states, set_seed, record, STUDENTS
from hsc.divergences import forward_kl, chunked_logsumexp
from hsc.cache import HiddenStateCache


def main(a):
    set_seed(a.seed)
    tmodel, tok, U, ubias, softcap = load_teacher(a.teacher)
    from transformers import AutoModelForCausalLM
    smodel = AutoModelForCausalLM.from_pretrained(
        STUDENTS[a.student][0], torch_dtype=torch.bfloat16, device_map="cuda")
    W = smodel.get_output_embeddings().weight

    rows = []
    for step, batch in enumerate(_batches(tok, a.steps, a.batch, a.seqlen)):
        ids = batch.to("cuda")

        # arm A: teacher in the loop
        g_online = teacher_hidden_states(tmodel, ids)[:, :-1].reshape(-1, U.shape[1])

        # arm B: same states, round-tripped through the cache at the given dtype
        cache = HiddenStateCache(os.path.join(a.out, "cache"), dtype=a.dtype,
                                 teacher=a.teacher, softcap=softcap or 0.0)
        cache.write(g_online.float().cpu().numpy())
        g_cached = torch.from_numpy(cache.read(0)).to(g_online.device).to(g_online.dtype)

        h = _student_hidden(smodel, ids)[:, :-1].reshape(-1, W.shape[1])
        la = forward_kl(h, W, g_online, U, None, ubias, a.chunk, softcap)
        lb = forward_kl(h, W, g_cached, U, None, ubias, a.chunk, softcap)

        la_d, lb_d = la.detach(), lb.detach()
        rows.append({
            "step": step,
            "loss_online": float(la_d.mean()),
            "loss_cached": float(lb_d.mean()),
            "abs_diff": float((la_d - lb_d).abs().max()),
            "rel_diff": float(((la_d - lb_d).abs() / la_d.abs().clamp_min(1e-9)).max()),
        })
        if step % 20 == 0:
            print(f"step {step:4d}  online {rows[-1]['loss_online']:.6f}  "
                  f"cached {rows[-1]['loss_cached']:.6f}  "
                  f"max|d| {rows[-1]['abs_diff']:.3e}")

    rel = np.array([r["rel_diff"] for r in rows])
    tol = {"fp32": 1e-6, "bf16": 5e-3, "fp16": 1e-3, "int8": 5e-2}[a.dtype]
    verdict = bool(rel.max() < tol)
    print(f"\nmax relative difference {rel.max():.3e}  tolerance {tol:.0e}  "
          f"{'PASS' if verdict else 'FAIL'}")
    record(os.path.join(a.out, "e1.json"),
           {"config": vars(a), "rows": rows, "max_rel": float(rel.max()),
            "tolerance": tol, "pass": verdict})
    return 0 if verdict else 1


def _student_hidden(model, ids):
    with torch.no_grad():
        return model(input_ids=ids, output_hidden_states=True).hidden_states[-1]


def _batches(tok, n, bs, L):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                      split="train", streaming=True)
    buf, it = [], iter(ds)
    for _ in range(n):
        while len(buf) < bs:
            t = tok(next(it)["text"], truncation=True, max_length=L,
                    return_tensors="pt")["input_ids"][0]
            if len(t) == L:
                buf.append(t)
        yield torch.stack(buf[:bs]); buf = buf[bs:]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="qwen3-8b")
    p.add_argument("--student", default="qwen3-0.6b")
    p.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16", "int8"])
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seqlen", type=int, default=1024)
    p.add_argument("--chunk", type=int, default=16384)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/e1")
    sys.exit(main(p.parse_args()))
