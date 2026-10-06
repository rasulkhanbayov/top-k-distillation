#!/usr/bin/env python3
"""
E3 (T1). Off-policy distillation against current methods, at matched storage and
separately at matched training FLOPs.

Arms (Section 6.3 of the paper):
  ce            cross-entropy only
  topk_fkl      forward KL on the cached top-k, current practice
  topp_fkl      nucleus truncation
  bild          BiLD, the strongest published claim that truncation helps
  full_rkl      reverse KL (MiniLLM-style)
  full_jsd      Jensen-Shannon / f-divergence
  full_skew     skew KL (DistiLLM)
  feature       FitNets/MiniLM hidden-state regression, the obvious alternative
                use of a cached g and the objection a reader raises first
  full_fkl_online   teacher in the loop, exact-quality upper reference
  full_fkl_cached   ours

Registered prediction: the cached arm matches the online arm within noise, lies
on the quality-versus-storage frontier at every feasible budget, sees its margin
over topk grow with corpus entropy, and beats `feature` at equal storage.

Usage:
  python e3_offpolicy.py --teacher llama31-8b --student llama32-1b \
      --arms all --match storage --budget-bytes 6144 --seeds 0 1 2
"""
import argparse, sys, os, json, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch

from common import (load_teacher, teacher_hidden_states, set_seed, record,
                    bootstrap_ci, paired_test, STUDENTS, EVAL_SUITE, SEEDS,
                    text_batches, StudentHidden)
from hsc.distill import DistillConfig, distill_step
from hsc.cache import HiddenStateCache, TopKCache
from e4_onpolicy import _truncate

ARMS = {
    # DistillConfig.alpha_ce defaults to 1.0 (see hsc/distill.py) -- meant
    # for E5's alpha_ce ablation, where including the CE term is the whole
    # point of half the arms. Every E3 arm here is a pure distillation
    # objective EXCEPT "ce" itself; only "ce" should get the auxiliary CE
    # term, so every other arm sets alpha_ce=0.0 explicitly. A first version
    # of this dict left it unset, so every non-ce arm silently trained on
    # 1.0*loss_ce + 1.0*loss_kd instead of the intended pure-KD objective --
    # confirmed as a real bug (not just theoretically wrong) by a real E3
    # run diverging (loss 2.9 -> 12 within 50 steps, reproducibly across
    # every arm and seed) in a way that did not reproduce when isolating
    # every other suspect (E5's hyperparameters, cache correctness, GPU
    # health -- see diag_e3_divergence.sh).
    "ce":              dict(objective="ce", alpha_kd=0.0),
    "topk_fkl":        dict(objective="topk_fkl", alpha_ce=0.0),
    "topp_fkl":        dict(objective="topp_fkl", alpha_ce=0.0),
    "bild":            dict(objective="bild", alpha_ce=0.0),
    "full_rkl":        dict(objective="full_rkl", alpha_ce=0.0),
    "full_jsd":        dict(objective="full_jsd", alpha_ce=0.0),
    "full_skew":       dict(objective="full_skew", alpha_ce=0.0),
    "feature":         dict(objective="feature", alpha_ce=0.0),
    "full_fkl_online": dict(objective="full_fkl", alpha_ce=0.0),
    "full_fkl_cached": dict(objective="full_fkl", alpha_ce=0.0),
}

# bytes per supervised position, used to enforce the matched-storage axis
def arm_bytes(arm, cfg, d_teacher):
    if arm in ("ce",):
        return 0
    if arm in ("topk_fkl", "topp_fkl", "bild"):
        return cfg.k * 6                      # 4B index + 2B fp16 value
    if arm in ("feature", "full_fkl_cached", "full_rkl", "full_jsd", "full_skew"):
        # all read the same hidden-state cache
        return d_teacher * {"fp32": 4, "bf16": 2, "fp16": 2, "int8": 1}[cfg_dtype]
    return 0                                   # online: teacher in the loop


cfg_dtype = "bf16"


def build_caches(a):
    """One-time pass over the teacher, writing a HiddenStateCache and a
    TopKCache from the SAME batches (same text_batches seed, default 0,
    independent of the training --seeds), so full_fkl_cached, topk_fkl and
    feature all train on identical teacher-observed data and only differ in
    what of it they're allowed to see -- the comparison the registered
    prediction is actually about. Caches are built once and reused across
    all training --seeds (those vary student-side stochasticity, not which
    teacher data exists); rebuilding per seed would make matched-storage
    comparisons noisier for no reason.

    Also writes each shard's raw input_ids next to the caches (caches only
    hold the teacher's signal, not the tokens) so the training loop can
    recompute labels/CE without needing the teacher loaded at all for the
    cached arms.
    """
    cache_root = os.path.join(a.out, "cache")
    ids_root = os.path.join(cache_root, "ids")
    if os.path.exists(os.path.join(cache_root, "hidden", "meta.json")):
        print(f"[cache] found existing caches under {cache_root}, reusing")
        return cache_root
    os.makedirs(ids_root, exist_ok=True)

    set_seed(0)
    tmodel, tok, U, ubias, softcap = load_teacher(a.teacher)
    hs_cache = HiddenStateCache(os.path.join(cache_root, "hidden"),
                                d=U.shape[1], dtype=a.cache_dtype,
                                teacher=a.teacher, softcap=softcap or 0.0)
    tk_cache = TopKCache(os.path.join(cache_root, "topk"), k=a.k,
                        dtype="fp16", teacher=a.teacher)

    print(f"[cache] building {a.cache_batches} shards "
          f"({a.cache_batches * a.batch} sequences) from {a.teacher}...")
    n_written = 0
    for ids in text_batches(tok, a.cache_batches, bs=a.batch, L=a.seqlen):
        ids = ids.cuda()
        with torch.no_grad():
            g = teacher_hidden_states(tmodel, ids)[:, :-1].reshape(-1, U.shape[1])
            idx, lp = _truncate(g, U, ubias, softcap, a.k, "teacher", None, None)
        hs_cache.write(g.float().cpu().numpy())
        tk_cache.write(lp.float().cpu().numpy(), idx.cpu().numpy())
        # Save the FULL (batch, seqlen) ids, not ids[:, :-1] or ids[:, 1:]
        # alone: the training loop needs to run the student over the full
        # sequence (student.hidden(ids), letting attention see full context,
        # same as teacher_hidden_states(tmodel, ids) above) and only slice
        # [:, :-1] on the OUTPUT hidden states / [:, 1:] on ids for labels
        # afterward -- exactly how g was computed here. An earlier version
        # saved ids[:, :-1] pre-sliced and fed that back to the student as
        # if it were a full sequence, which silently truncated context by
        # one token and, worse, used it as BOTH student input and (via a
        # reshape bug) an attempted stand-in for labels, which are actually
        # ids[:, 1:] -- shape-compatible so nothing would have crashed, just
        # trained on the wrong targets.
        np.save(os.path.join(ids_root, f"{hs_cache._shard - 1:06d}.npy"),
               ids.cpu().numpy())
        n_written += g.shape[0]
        if hs_cache._shard % 20 == 0:
            print(f"[cache]   shard {hs_cache._shard}/{a.cache_batches} "
                  f"({n_written} positions so far)", flush=True)

    hs_cache.finalize(); tk_cache.finalize()
    print(f"[cache] done: {n_written} positions, "
          f"{hs_cache.stats()['bytes_per_position']:.0f} B/pos hidden-state, "
          f"{tk_cache.stats()['bytes_per_position']:.0f} B/pos top-k")
    del tmodel; torch.cuda.empty_cache()
    return cache_root


IMPLEMENTED_ARMS = ("full_fkl_online", "full_fkl_cached", "topk_fkl", "feature",
                    "ce", "full_rkl", "full_jsd", "full_skew")
# Still not wired: topp_fkl (no read-side nucleus; distill.py routes it
# through plain top-k) and bild (_bild_loss needs a rewrite and chunking).


def main(a):
    global cfg_dtype
    cfg_dtype = a.cache_dtype
    arms = list(ARMS) if a.arms == ["all"] else a.arms
    unimplemented = [arm for arm in arms if arm not in IMPLEMENTED_ARMS]
    if unimplemented:
        raise NotImplementedError(
            f"arms {unimplemented} are not wired yet (only {IMPLEMENTED_ARMS} "
            f"are, see _train_and_eval's SCOPE NOTE) -- failing before any "
            f"cache-building or model loading rather than partway through.")
    d_t = None
    results = {}

    needs_cache = bool({"full_fkl_cached", "topk_fkl", "feature", "ce",
                        "full_rkl", "full_jsd", "full_skew"} & set(arms))
    cache_root = build_caches(a) if needs_cache else None

    # Only full_fkl_online needs the teacher live during training -- that is
    # the entire point of the comparison (caching removes the teacher from
    # the loop). Loading it unconditionally for cached/topk arms would both
    # waste GPU memory/time and silently defeat the thing being tested.
    needs_live_teacher = {"full_fkl_online"}

    per_seed_by_arm = {}   # arm -> list of raw per-seed metrics dicts, kept
                            # alongside `results`' bootstrap-aggregated means
                            # so the paired tests below can compare matched
                            # per-seed val_ppl values directly instead of
                            # re-deriving them from the aggregate.
    for arm in arms:
        per_seed = []
        for seed in a.seeds:
            set_seed(seed)
            # tok/U/ubias/softcap are needed even for cached arms (U for
            # arm_bytes' d_t, tok for the validation-perplexity split); only
            # the model itself is heavy and skippable.
            tmodel, tok, U, ubias, softcap = load_teacher(a.teacher)
            d_t = U.shape[1]
            if arm not in needs_live_teacher:
                del tmodel; torch.cuda.empty_cache()
                tmodel = None
            cfg = DistillConfig(**ARMS[arm], k=a.k, top_p=a.top_p,
                                temperature=a.temperature, chunk=a.chunk,
                                softcap_t=softcap, seed=seed,
                                matched_axis=a.match)
            nb = arm_bytes(arm, cfg, d_t)
            if a.match == "storage" and nb > a.budget_bytes:
                print(f"[skip] {arm} needs {nb} B/pos > budget {a.budget_bytes}")
                continue
            print(f"\n=== arm={arm} seed={seed} storage={nb} B/pos ===")
            metrics = _train_and_eval(a, cfg, arm, tmodel, tok, U, ubias, softcap,
                                      cache_root)
            metrics["bytes_per_position"] = nb
            per_seed.append(metrics)
            if tmodel is not None:
                del tmodel
            torch.cuda.empty_cache()
        if per_seed:
            results[arm] = _aggregate(per_seed)
            per_seed_by_arm[arm] = per_seed

    # The two comparisons the registered prediction names, on matched-seed
    # val_ppl specifically -- not _flat's old blend of every numeric field
    # (train_loss, val_ppl, AND bytes_per_position mixed together, which
    # produced nonsense mean_diff values around 2000-2700 when val_ppl
    # itself was ~11-12; bytes_per_position dominated because it's the
    # largest number in the mix, not because it means anything paired
    # against another arm's storage size).
    def _val_ppls(arm):
        return [m["val_ppl"] for m in per_seed_by_arm.get(arm, [])]

    if "full_fkl_cached" in results and "full_fkl_online" in results:
        results["_test_cached_vs_online"] = paired_test(
            _val_ppls("full_fkl_cached"), _val_ppls("full_fkl_online"))
    if "full_fkl_cached" in results and "topk_fkl" in results:
        results["_test_cached_vs_topk"] = paired_test(
            _val_ppls("full_fkl_cached"), _val_ppls("topk_fkl"))
    if "full_fkl_cached" in results and "feature" in results:
        results["_test_cached_vs_feature"] = paired_test(
            _val_ppls("full_fkl_cached"), _val_ppls("feature"))

    # per_seed_by_arm is saved alongside the aggregate so a paired test can be
    # re-run independently later (e.g. code/analysis/stats.py, which
    # needs the raw seed-matched arrays, not just the aggregate CI already
    # computed above) without needing to redo any training.
    record(os.path.join(a.out, "e3.json"), {"config": vars(a), "results": results,
                                             "per_seed_by_arm": per_seed_by_arm})
    _print_table(results)


def _train_and_eval(a, cfg, arm, tmodel, tok, U, ubias, softcap, cache_root):
    """Runs the training loop and a validation-perplexity check.

    Teacher signal source depends on the arm: full_fkl_cached reads
    HiddenStateCache; topk_fkl reads TopKCache; full_fkl_online runs the
    teacher every step (tmodel is non-None only for this arm, see main()).

    SCOPE NOTE: wires the three arms the registered prediction directly names
    (full_fkl_cached vs full_fkl_online vs topk_fkl) plus feature the only arm separating "exact reconstruction helps" from
    "hidden-state availability alone helps"), all with a lightweight
    validation-perplexity eval, not the full 10-arm / lm-eval-suite (MMLU,
    GSM8K, HumanEval, ...) version the docstring above originally described.
    ce/topp_fkl/bild/full_rkl/full_jsd/full_skew and the full EVAL_SUITE are
    still unimplemented -- deliberately, to validate the cache-building
    infrastructure (genuinely new, unlike E5's live-teacher-only path) on
    the load-bearing comparisons first, same approach as E5's smoke test.
    feature's own plumbing (the proj linear layer, its optimizer param group,
    the "feature" -> "hidden" cache_dir mapping, distill_step's proj argument)
    was already present when this arm was still gated out; only the two
    NotImplementedError checks needed widening.
    """
    if arm not in IMPLEMENTED_ARMS:
        raise NotImplementedError(
            f"arm={arm!r} is not wired yet. Only {IMPLEMENTED_ARMS} are "
            f"implemented so far. topp_fkl/bild and the full lm-eval "
            f"suite (EVAL_SUITE) are "
            f"deliberately left for later, same as E5's staged "
            f"smoke-test-first approach. Pass --arms explicitly rather than "
            f"--arms all.")

    from transformers import AutoModelForCausalLM
    raw_student = AutoModelForCausalLM.from_pretrained(
        STUDENTS[a.student][0], torch_dtype=torch.bfloat16, device_map="cuda")
    student = StudentHidden(raw_student)
    W = raw_student.get_output_embeddings().weight
    opt = torch.optim.AdamW(raw_student.parameters(), lr=a.lr,
                            weight_decay=a.weight_decay)
    proj = None
    if cfg.objective == "feature":
        proj = torch.nn.Linear(U.shape[1], raw_student.config.hidden_size,
                               bias=False).cuda().to(torch.bfloat16)
        opt.add_param_group({"params": proj.parameters()})

    if arm == "full_fkl_online":
        data_iter = text_batches(tok, a.steps, bs=a.batch, L=a.seqlen)
        cache = None
    else:
        # ce only needs the ids shards stored next to the caches; it reads
        # the hidden cache too but distill_step ignores g for objective="ce".
        cache_dir = {"full_fkl_cached": "hidden", "topk_fkl": "topk",
                     "feature": "hidden", "ce": "hidden", "full_rkl": "hidden",
                     "full_jsd": "hidden", "full_skew": "hidden"}[arm]
        cache_path = os.path.join(cache_root, cache_dir)
        with open(os.path.join(cache_path, "meta.json")) as fh:
            meta_json = json.load(fh)
        n_shards = meta_json["n_shards"]
        # Read-side dtype MUST come from the actual meta.json, not from
        # HiddenStateCache/TopKCache's constructor defaults (bf16/fp16) --
        # those only happen to match what build_caches wrote if --cache-dtype
        # was left at its own default; passing e.g. --cache-dtype fp32 would
        # otherwise dequantize with the wrong dtype and silently corrupt
        # every value read back.
        cache = (HiddenStateCache(cache_path, dtype=meta_json["dtype"])
                if cache_dir == "hidden" else
                TopKCache(cache_path, k=a.k, dtype=meta_json["dtype"]))
        data_iter = None

    losses = []
    for step in range(a.steps):
        if data_iter is not None:
            ids = next(data_iter).cuda()
            with torch.no_grad():
                g_full = teacher_hidden_states(tmodel, ids)[:, :-1].reshape(-1, U.shape[1])
        else:
            shard = step % n_shards
            ids = torch.from_numpy(
                np.load(os.path.join(cache_root, "ids", f"{shard:06d}.npy"))
            ).cuda()
            if arm == "topk_fkl":
                lp, idx = cache.read(shard)
                lp = torch.from_numpy(lp).cuda()
                idx = torch.from_numpy(idx.astype(np.int64)).cuda()
            else:
                g_full = torch.from_numpy(cache.read(shard)).cuda().to(torch.bfloat16)

        # Run the student over the FULL sequence (attention sees full
        # context, matching how g was computed for the teacher), then slice
        # [:, :-1] on the output hidden states -- not on ids beforehand --
        # to align with labels = ids[:, 1:]. Same pattern as E5's probe()
        # and training loop (hsc/distill.py's docstring explains why
        # distill_step itself can't do this reshaping: it needs the
        # (batch, seqlen)-shaped forward pass to have already happened).
        hf = student.hidden(ids)[:, :-1].reshape(-1, W.shape[1])
        labels = ids[:, 1:].reshape(-1)

        batch = {"h": hf, "labels": labels}
        if arm == "topk_fkl":
            batch["tk_idx"] = idx
            batch["tk_logprobs"] = lp
        else:
            batch["g"] = g_full

        loss, parts = distill_step(batch, student, cfg, teacher_U=U,
                                   teacher_bias=ubias, proj=proj)
        loss.backward()
        opt.step(); opt.zero_grad()
        losses.append(float(loss))
        if step % 50 == 0:
            print(f"  step {step}/{a.steps} loss={float(loss):.4f}", flush=True)

    val_ppl = _val_perplexity(raw_student, tok, a)
    del raw_student, student
    torch.cuda.empty_cache()
    return {"train_loss": float(np.mean(losses[-20:])), "val_ppl": val_ppl}


def _val_perplexity(student, tok, a, n_batches=8):
    """Held-out (different text_batches seed) next-token perplexity. Not the
    full lm-eval suite (MMLU/GSM8K/...) the paper's declared protocol wants
    eventually -- see _train_and_eval's SCOPE NOTE."""
    total_nll, total_tokens = 0.0, 0
    with torch.no_grad():
        for ids in text_batches(tok, n_batches, bs=a.batch, L=a.seqlen, seed=1):
            ids = ids.cuda()
            # HF's ForCausalLM shifts `labels` internally (labels[..., 1:]
            # against logits from the FULL input_ids) -- it expects the same
            # whole sequence passed as both, not pre-shifted tensors. Passing
            # ids[:, :-1]/ids[:, 1:] separately, as an earlier version of
            # this did, would have HF shift the already-shifted labels
            # again, comparing mismatched positions.
            out = student(input_ids=ids, labels=ids)
            n = ids[:, 1:].numel()
            total_nll += float(out.loss) * n
            total_tokens += n
    return float(np.exp(total_nll / total_tokens))


def _aggregate(per_seed):
    keys = [k for k in per_seed[0] if isinstance(per_seed[0][k], (int, float))]
    return {k: dict(zip(("mean", "lo", "hi"),
                        bootstrap_ci([m[k] for m in per_seed]))) for k in keys}


def _print_table(results):
    print(f"\n{'arm':<18}{'B/pos':>8}{'val ppl':>12}{'MMLU':>9}{'GSM8K':>9}")
    print("-" * 60)
    for arm, r in results.items():
        if arm.startswith("_"):
            continue
        g = lambda k: r.get(k, {}).get("mean", float("nan"))
        print(f"{arm:<18}{g('bytes_per_position'):>8.0f}{g('val_ppl'):>12.4f}"
              f"{g('mmlu'):>9.3f}{g('gsm8k'):>9.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="llama31-8b")
    p.add_argument("--student", default="llama32-1b")
    p.add_argument("--arms", nargs="+", default=["all"])
    p.add_argument("--match", choices=["storage", "flops", "wallclock"], default="storage")
    p.add_argument("--budget-bytes", type=int, default=6144)
    p.add_argument("--k", type=int, default=256)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--cache-dtype", default="bf16")
    p.add_argument("--chunk", type=int, default=16384)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.1,
                   help="a real E3 full run diverged (loss 2.9->12 within "
                        "50 steps) at the original hardcoded 0.1; E5's "
                        "training (same distill_step path) is stable at "
                        "AdamW's own default 0.01 -- pass --weight-decay "
                        "0.01 to test that hypothesis before trusting this "
                        "default further")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seqlen", type=int, default=512)
    p.add_argument("--cache-batches", type=int, default=500,
                   help="number of batches (shards) to cache, each --batch "
                        "sequences of --seqlen tokens; training cycles "
                        "through these shards if --steps exceeds this")
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--out", default="runs/e3")
    main(p.parse_args())
