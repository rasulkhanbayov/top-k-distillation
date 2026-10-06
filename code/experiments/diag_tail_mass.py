#!/usr/bin/env python3
"""
Standalone diagnostic, no training: how much of a teacher's next-token
probability lies outside its own top-k, on the same training stream the
Off-policy matched-storage runs used (text_batches seed 0, so the first batches
of the cache those runs read).

For each position it computes the exact full-vocabulary distribution
q = softmax(U g + b) (soft-capped where the teacher caps), then reports, over
all positions, the mean and median tail mass m_q(k) = 1 - sum of the top-k
probabilities for each requested k, and the mean entropy of q.

Usage: diag_tail_mass.py <teacher> [--batches N] [--ks 64 256 1024] [--out DIR]
"""
import argparse, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch

from common import load_teacher, teacher_hidden_states, text_batches, record


@torch.no_grad()
def tail_masses(model, tok, U, bias, softcap, ks, n_batches, bs, L, chunk_pos=512):
    per_k = {k: [] for k in ks}
    entropy = []
    U = U.cuda().float()
    b = bias.cuda().float() if bias is not None else None
    for ids in text_batches(tok, n_batches, bs=bs, L=L):
        ids = ids.to(model.device)
        g = teacher_hidden_states(model, ids)[:, :-1].reshape(-1, U.shape[1]).float().cuda()
        for i in range(0, g.shape[0], chunk_pos):
            z = g[i:i + chunk_pos] @ U.T
            if b is not None:
                z = z + b
            if softcap is not None:
                z = softcap * torch.tanh(z / softcap)  # Gemma-style cap, as in hsc.reconstruct
            logq = torch.log_softmax(z, dim=-1)
            q = logq.exp()
            entropy.append((-(q * logq).sum(-1)).cpu())
            top = torch.topk(q, max(ks), dim=-1).values.cumsum(-1)
            for k in ks:
                per_k[k].append((1.0 - top[:, k - 1]).clamp_min(0).cpu())
    out = {}
    for k in ks:
        m = torch.cat(per_k[k]).double().numpy()
        out[str(k)] = {"mean": float(m.mean()), "median": float(np.median(m)),
                       "p90": float(np.quantile(m, 0.9))}
    H = torch.cat(entropy).double().numpy()
    return out, float(H.mean()), int(H.size)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("teacher")
    p.add_argument("--batches", type=int, default=32)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seqlen", type=int, default=512)
    p.add_argument("--ks", nargs="+", type=int, default=[64, 256, 1024])
    p.add_argument("--out", default=None)
    a = p.parse_args()
    torch.manual_seed(0)
    model, tok, U, bias, softcap = load_teacher(a.teacher)
    model.eval()
    res, H, n = tail_masses(model, tok, U, bias, softcap, a.ks, a.batches, a.batch, a.seqlen)
    print(f"{a.teacher}: {n} positions, mean entropy {H:.4f} nats, softcap={softcap}")
    for k in a.ks:
        r = res[str(k)]
        print(f"  tail mass beyond top-{k}: mean {r['mean']:.5f}  median {r['median']:.5f}  p90 {r['p90']:.5f}")
    if a.out:
        record(os.path.join(a.out, f"tail_mass_{a.teacher}.json"),
               {"config": vars(a), "teacher": a.teacher, "softcap": softcap,
                "positions": n, "mean_entropy": H, "tail_mass": res})
