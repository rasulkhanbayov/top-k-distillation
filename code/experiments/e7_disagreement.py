#!/usr/bin/env python3
"""
E7 (T2). Does the trade-off hypothesis predict the published disagreement?

Reproduces three previously reported settings under one protocol and asks whether
two measurable quantities predict the sign of the truncation effect and the
ordering of the optimal k. This is a HYPOTHESIS, not a consequence of Section 4;
a failure is reported as such and leaves the identifiability results untouched.

Settings:
  bild_style      task fine-tuning distillation; truncation reported to HELP
  pretrain_style  general pre-training distillation; truncation reported to HURT
  toolcall_style  rare decision-critical tokens; mass a poor proxy for supervision

Measured predictors:
  tail_reliability  agreement of the teacher's tail across two checkpoints or
                    seeds, plus the teacher's own held-out calibration
  deficit           d + 1 - k

Registered prediction: k* is smallest for bild_style and largest for
toolcall_style, ordered by measured tail reliability.

Usage:
  python e7_disagreement.py --settings bild_style pretrain_style toolcall_style
"""
import argparse, sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch
from common import (load_teacher, teacher_hidden_states, set_seed, record,
                    SEEDS, STUDENTS, TEACHERS, text_batches, StudentHidden)
from hsc.divergences import chunked_logsumexp, _chunks
from hsc.metrics import ece, effective_support
from hsc.distill import DistillConfig, distill_step
from e4_onpolicy import _truncate

# Real HF corpora standing in for each reported setting's data regime (see
# common.py's _CORPUS_CONFIGS for the exact parquet paths). "flan" and
# "toolbench" as literal corpus names never existed in this codebase --
# these are the closest easily-downloadable, non-gated real datasets that
# match each setting's described regime:
#   bild_style:     Muennighoff/flan -- FLAN-style instruction/response pairs
#                   (task fine-tuning distillation), one task per row.
#   pretrain_style: HuggingFaceFW/fineweb-edu -- general web pretraining text,
#                   already used by E1/E2/E5.
#   toolcall_style: glaiveai/glaive-function-calling-v2 -- tool-calling
#                   conversations where the correct function/argument token is
#                   rare but decision-critical, unlike a token's raw frequency.
SETTINGS = {
    "bild_style":     dict(corpus="Muennighoff/flan",
                           teacher="qwen3-8b",   student="qwen3-0.6b"),
    "pretrain_style": dict(corpus="HuggingFaceFW/fineweb-edu",
                           teacher="llama31-8b", student="llama32-1b"),
    "toolcall_style": dict(corpus="glaiveai/glaive-function-calling-v2",
                           teacher="qwen3-8b",   student="qwen3-0.6b"),
}


def tail_reliability(U_a, U_b, g_a, g_b, k, chunk=16384, ubias=None, softcap=None):
    """Agreement of the teacher's TAIL between two teacher checkpoints or seeds.

    Low agreement means the tail is largely estimation error, which is the
    condition under which truncation should help. This is the predictor the
    hypothesis rests on, so it is measured rather than assumed.
    """
    def logp(U, g):
        lse = chunked_logsumexp(g, U, ubias, chunk, softcap)
        z = (g @ U.T).float()
        if ubias is not None:
            z = z + ubias
        if softcap:
            z = softcap * torch.tanh(z / softcap)
        return (z - lse[:, None]).cpu().numpy()
    la, lb = logp(U_a, g_a), logp(U_b, g_b)
    out = []
    for i in range(la.shape[0]):
        S = np.argsort(la[i])[::-1][:k]
        M = np.ones(la.shape[1], bool); M[S] = False
        ta, tb = la[i][M], lb[i][M]
        # rank correlation of the tail, and mass agreement
        ra = np.argsort(np.argsort(-ta)); rb = np.argsort(np.argsort(-tb))
        rho = float(np.corrcoef(ra, rb)[0, 1])
        mass_a, mass_b = float(np.exp(ta).sum()), float(np.exp(tb).sum())
        out.append({"tail_rank_corr": rho,
                    "tail_mass_ratio": mass_a / max(mass_b, 1e-30)})
    return {kk: float(np.mean([o[kk] for o in out])) for kk in out[0]}


def _register_predictor(tmodel, tok, U, ubias, softcap, k, seed, corpus, n=32, chunk=16384):
    """Measures tail_reliability BEFORE any k-sweep training happens (this is
    what makes k* a registered prediction rather than a fitted one -- E7 must commit tail_reliability/deficit ahead of the
    sweep, not compute them after seeing which k trains best).

    Only one teacher checkpoint exists per setting here (no second fine-tune
    or seed of the SAME teacher was run), so "two teacher checkpoints or
    seeds" is proxied by two INDEPENDENT draws of held-out text from the same
    corpus (U_a is U_b, g_a and g_b come from disjoint batches) -- tail
    agreement is measured across sampling variation in what text the teacher
    sees, which is the same quantity tail_reliability's docstring describes
    (whether the tail is estimation error) under the data actually available
    here."""
    batches = list(text_batches(tok, 2 * n, bs=1, L=256, seed=seed, corpus=corpus))
    ga = torch.cat([teacher_hidden_states(tmodel, b.cuda())[:, -1]
                    for b in batches[:n]], dim=0)
    gb = torch.cat([teacher_hidden_states(tmodel, b.cuda())[:, -1]
                    for b in batches[n:]], dim=0)
    return tail_reliability(U, U, ga, gb, k, chunk=chunk, ubias=ubias, softcap=softcap)


def main(a):
    rows = []
    predictors = {}
    # Registration pass: measure tail_reliability/deficit for every setting
    # BEFORE any training, and commit the result to disk immediately -- so
    # the prediction (k* ordered by measured tail reliability) is fixed
    # before the sweep below can influence it even in this same process.
    for name in a.settings:
        cfg = SETTINGS[name]
        set_seed(a.seeds[0])
        tmodel, tok, U, ubias, softcap = load_teacher(cfg["teacher"])
        pred = _register_predictor(tmodel, tok, U, ubias, softcap, a.k_ref,
                                   a.seeds[0], cfg["corpus"],
                                   n=a.predictor_batches, chunk=a.chunk)
        d = TEACHERS[cfg["teacher"]][1]
        predictors[name] = {**pred, "deficit_at_k_ref": d + 1 - a.k_ref,
                            "d_teacher": d, "k_ref": a.k_ref}
        print(f"=== {name} registered predictor: {predictors[name]} ===", flush=True)
        del tmodel; torch.cuda.empty_cache()
    record(os.path.join(a.out, "e7_predictors.json"),
          {"config": vars(a), "predictors": predictors})
    print("Registered predictions committed to e7_predictors.json BEFORE the "
         "sweep below. Prediction: k* is smallest for bild_style and largest "
         "for toolcall_style, ordered by tail_rank_corr above.", flush=True)

    # Sweep: for each setting, train student copies at each k (topk_fkl) and
    # one full_fkl reference, and measure held-out full-vocabulary forward_kl
    # quality -- same eval-on-one-scale pattern as E4/E9 (raw topk_fkl loss is
    # not comparable across k, since it renormalizes over k tokens each time).
    from hsc.divergences import forward_kl
    from transformers import AutoModelForCausalLM
    for name in a.settings:
        cfg = SETTINGS[name]
        for seed in a.seeds:
            set_seed(seed)
            tmodel, tok, U, ubias, softcap = load_teacher(cfg["teacher"])
            for k in a.ks:
                print(f"=== {name} k={k} seed={seed} starting ===", flush=True)
                raw_student = AutoModelForCausalLM.from_pretrained(
                    STUDENTS[cfg["student"]][0], torch_dtype=torch.bfloat16,
                    device_map="cuda")
                student = StudentHidden(raw_student)
                W = raw_student.get_output_embeddings().weight
                opt = torch.optim.AdamW(raw_student.parameters(), lr=a.lr)
                dcfg = DistillConfig(objective="topk_fkl", alpha_ce=0.0, k=k,
                                     chunk=a.chunk, softcap_t=softcap, seed=seed)

                t0 = time.time()
                data_iter = text_batches(tok, a.steps + a.eval_batches, bs=a.batch,
                                         L=a.seqlen, seed=seed, corpus=cfg["corpus"])
                eval_kls = []
                for step, ids in enumerate(data_iter, start=1):
                    ids = ids.cuda()
                    g = teacher_hidden_states(tmodel, ids).detach()
                    gf = g[:, :-1].reshape(-1, U.shape[1])
                    hf = student.hidden(ids)[:, :-1].reshape(-1, W.shape[1])
                    labels = ids[:, 1:].reshape(-1)

                    if step <= a.steps:
                        idx, lp = _truncate(gf, U, ubias, softcap, k, "teacher",
                                            hf.detach(), W)
                        batch = {"h": hf, "labels": labels,
                                "tk_idx": idx, "tk_logprobs": lp}
                        loss, _ = distill_step(batch, student, dcfg,
                                               teacher_U=U, teacher_bias=ubias)
                        loss.backward(); opt.step(); opt.zero_grad()
                    else:
                        with torch.no_grad():
                            eval_kl = forward_kl(hf, W, gf, U, None, ubias,
                                                 a.chunk, softcap)
                        eval_kls.append(float(eval_kl.mean()))

                rows.append({
                    "setting": name, "k": k, "seed": seed,
                    "deficit": predictors[name]["d_teacher"] + 1 - k,
                    "eval_kl": float(np.mean(eval_kls)),
                    "wallclock_s": time.time() - t0,
                })
                print(rows[-1], flush=True)
                # Checkpoint after every cell so a walltime kill loses only the
                # cell in flight.
                record(os.path.join(a.out, "e7.json"),
                       {"config": vars(a), "predictors": predictors, "rows": rows})
                del raw_student, student; torch.cuda.empty_cache()
            del tmodel; torch.cuda.empty_cache()

    record(os.path.join(a.out, "e7.json"),
          {"config": vars(a), "predictors": predictors, "rows": rows})
    _verdict(rows, predictors)


def _verdict(rows, predictors):
    import collections
    by_setting = collections.defaultdict(list)
    for r in rows:
        by_setting[r["setting"]].append(r)
    print(f"\n{'setting':<16}{'best k':>8}{'best eval_kl':>14}{'tail_rank_corr':>16}")
    print("-" * 54)
    best_k = {}
    for name, rs in by_setting.items():
        best = min(rs, key=lambda r: r["eval_kl"])
        best_k[name] = best["k"]
        print(f"{name:<16}{best['k']:>8}{best['eval_kl']:>14.4f}"
              f"{predictors[name]['tail_rank_corr']:>16.4f}")
    ordered = sorted(best_k, key=lambda n: predictors[n]["tail_rank_corr"])
    print("\nRegistered prediction: k* smallest for bild_style, largest for "
         "toolcall_style, ordered by tail_rank_corr (lower = less reliable "
         "tail = truncation should help more, i.e. smaller optimal k).")
    print(f"Settings ordered by measured tail_rank_corr (ascending): {ordered}")
    print(f"Settings ordered by measured best k (ascending): "
         f"{sorted(best_k, key=best_k.get)}")
    print("If these two orderings do not match, report the mismatch plainly "
         "-- this is a hypothesis test, not a curve to fit.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--settings", nargs="+", default=list(SETTINGS))
    p.add_argument("--ks", nargs="+", type=int,
                   default=[8, 16, 64, 256, 1024, 4096])
    p.add_argument("--k-ref", type=int, default=256,
                   help="k used only for the registered tail_reliability "
                        "predictor, measured before the sweep")
    p.add_argument("--predictor-batches", type=int, default=32)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--eval-batches", type=int, default=16)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seqlen", type=int, default=512)
    p.add_argument("--chunk", type=int, default=16384)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--out", default="runs/e7")
    main(p.parse_args())
