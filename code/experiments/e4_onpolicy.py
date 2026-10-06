#!/usr/bin/env python3
"""
E4 (T1). On-policy distillation.

This is the experiment that decides how far the paper's claims reach. On-policy
distillation is now the dominant post-training paradigm, and hidden-state caching
gives NO storage benefit there because the teacher is in the loop by
construction. What should carry over is the theory: Proposition 1 constrains any
supervision transmitting fewer than d+1 teacher log-probabilities, however the
supervised sequences arose, and on-policy implementations truncate for the same
memory reasons offline ones do.

Two comparisons:
  (A) GKD and a DistiLLM-2 style contrastive recipe, truncated top-k teacher
      supervision on student rollouts versus exact full-vocabulary supervision,
      matched on wall-clock.
  (B) replay-based variants where generations are refreshed every R steps: the
      saving from caching teacher hidden states once per refresh and reusing
      them within the epoch.

Registered prediction: the truncated-versus-exact gap persists on-policy and is
larger on high-entropy rollouts. If it does not, the theory's reach is confined
to off-policy distillation and we say so in the paper.

Caveat that must be honored in the implementation: several on-policy recipes
select the observation set from the STUDENT's top-k, which makes S a function of
h. Theorems 2 and 3 are not stated for that case (Section 3). Both selection
rules are run so the difference is measured rather than assumed.

Usage:
  python e4_onpolicy.py --recipe gkd --teacher qwen3-8b --student qwen3-0.6b \
      --supervision topk exact --select teacher student --refresh 0
"""
import argparse, sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch

from common import (load_teacher, teacher_hidden_states, set_seed, record,
                    STUDENTS, SEEDS, prompt_batches)
from hsc.divergences import forward_kl, reverse_kl, jsd, topk_forward_kl

RECIPES = ("gkd", "distillm2", "seqkd_replay")


def rollout(student, tok, prompts, max_new, temperature=1.0):
    """Student-generated trajectories. On-policy data, so nothing can be cached
    ahead of time; this is the cost the paper does not claim to remove.

    Returns (seqs, attention_mask): the prompt is left-padded (see main()),
    so the generated sequence's attention_mask must extend the PROMPT's mask
    with 1s over every newly generated token (never padding, since
    generation only appends real tokens after the prompt) -- needed by every
    downstream forward pass on `seqs` (teacher_hidden_states, the student's
    own training forward pass) so padding positions aren't attended to as if
    they were real content."""
    with torch.no_grad():
        out = student.generate(**prompts, do_sample=True,
                               temperature=temperature, max_new_tokens=max_new,
                               return_dict_in_generate=True)
    seqs = out.sequences
    n_generated = seqs.shape[1] - prompts["attention_mask"].shape[1]
    gen_mask = torch.ones(seqs.shape[0], n_generated, dtype=torch.long,
                          device=seqs.device)
    attention_mask = torch.cat([prompts["attention_mask"], gen_mask], dim=1)
    return seqs, attention_mask


def main(a):
    rows = []
    for seed in a.seeds:
        for sup in a.supervision:            # "topk" | "exact"
            for sel in (a.select if sup == "topk" else ["teacher"]):
                set_seed(seed)
                tmodel, teacher_tok, U, ubias, softcap = load_teacher(a.teacher)
                from transformers import AutoModelForCausalLM, AutoTokenizer
                student = AutoModelForCausalLM.from_pretrained(
                    STUDENTS[a.student][0], torch_dtype=torch.bfloat16,
                    device_map="cuda")
                W = student.get_output_embeddings().weight
                opt = torch.optim.AdamW(student.parameters(), lr=a.lr)

                # tok must be the STUDENT's own tokenizer, not the teacher's
                # (teacher_hidden_states/topk_forward_kl etc. below index
                # into U/W by token id -- using the teacher's tokenizer to
                # build the student's generation input would silently feed
                # it token ids from the wrong vocabulary whenever the two
                # families differ; the default --teacher qwen3-8b --student
                # qwen3-0.6b pairing happens to share a tokenizer, which
                # would have hidden this). Left-padding is required for
                # batched generation with a decoder-only model -- default
                # padding_side is "right" for most tokenizers, which
                # generate() cannot use correctly for a padded batch (it
                # would try to continue generating after right-side pad
                # tokens); AutoTokenizer.from_pretrained does not turn this
                # on automatically.
                tok = AutoTokenizer.from_pretrained(STUDENTS[a.student][0])
                tok.padding_side = "left"
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                # The teacher's forward pass below (teacher_hidden_states)
                # runs on `seqs`, which is tokenized with the STUDENT's
                # tokenizer -- correct only when both share a vocabulary.
                # True for every current --teacher/--student default pairing
                # (verified: Qwen3-8B and Qwen3-0.6B produce identical token
                # ids for the same text) but not guaranteed for an arbitrary
                # pairing, and there is no re-tokenization step here to
                # handle a real mismatch -- fail clearly rather than run the
                # teacher on token ids from a vocabulary it doesn't share.
                assert teacher_tok.vocab_size == tok.vocab_size, (
                    f"teacher ({a.teacher}) and student ({a.student}) "
                    f"tokenizers have different vocab sizes "
                    f"({teacher_tok.vocab_size} vs {tok.vocab_size}) -- "
                    f"on-policy distillation across tokenizers needs a "
                    f"re-tokenization step this script does not implement.")

                t0 = time.time()
                ent_bucket = {"low": [], "high": []}
                cached_g, refresh_at = None, -1
                prompt_iter = prompt_batches(tok, a.steps, bs=a.batch,
                                             L=a.prompt_len, seed=seed)

                for step in range(a.steps):
                    # (B) replay: reuse the SAME rollout (seqs/amask) for R
                    # steps, refreshing both the generation and the cached
                    # teacher hidden states together every R steps -- an
                    # earlier version generated a NEW rollout every step
                    # regardless of --refresh, but only cached the teacher's
                    # hidden states from the refresh step, so on non-refresh
                    # steps `g` (stale rollout) and `h` (current step's fresh
                    # rollout) came from different token sequences entirely,
                    # not just a shape mismatch. Harmless at the default
                    # --refresh 0 (fully on-policy, this branch never taken)
                    # but broken for any real replay run.
                    if a.refresh <= 0 or step >= refresh_at:
                        prompts = next(prompt_iter).to("cuda")
                        seqs, amask = rollout(student, tok, prompts, a.max_new, a.temp)
                        if a.refresh > 0:
                            cached_g = teacher_hidden_states(tmodel, seqs, amask).detach()
                            cached_seqs, cached_amask = seqs, amask
                            refresh_at = step + a.refresh
                    if a.refresh > 0:
                        seqs, amask, g = cached_seqs, cached_amask, cached_g
                    else:
                        g = teacher_hidden_states(tmodel, seqs, amask)

                    h = student(input_ids=seqs, attention_mask=amask,
                               output_hidden_states=True).hidden_states[-1]
                    hf = h[:, :-1].reshape(-1, W.shape[1])
                    gf = g[:, :-1].reshape(-1, U.shape[1])

                    if sup == "exact":
                        loss = (forward_kl(hf, W, gf, U, None, ubias, a.chunk, softcap)
                                if a.recipe == "gkd" else
                                jsd(hf, W, gf, U, None, ubias, a.chunk, softcap))
                    else:
                        idx, lp = _truncate(gf, U, ubias, softcap, a.k, sel, hf, W)
                        loss = topk_forward_kl(hf, W, lp, idx)

                    # topk_forward_kl computes KL between BOTH sides
                    # renormalized over the k-token cached set (see its
                    # docstring) -- a structurally smaller-scale quantity
                    # than forward_kl's exact full-vocabulary KL, not
                    # comparable to it on the same axis. A real smoke test
                    # showed topk with LOWER raw loss than exact and initially
                    # read as topk "winning", which would have been a
                    # methodologically meaningless comparison of two
                    # different metrics, not evidence about the paper's
                    # actual claim. eval_kl is the SAME exact full-vocabulary
                    # forward_kl computed as a held-out measurement (no_grad,
                    # not used for .backward()) for every arm regardless of
                    # what it trained on, so topk-vs-exact is compared on one
                    # consistent scale: does training on truncated
                    # supervision produce a student that matches the
                    # teacher's FULL distribution worse than training on
                    # exact supervision does.
                    with torch.no_grad():
                        eval_kl = forward_kl(hf, W, gf, U, None, ubias,
                                             a.chunk, softcap)

                    ent = _teacher_entropy(gf, U, ubias, softcap, a.chunk)
                    med = float(np.median(ent))
                    ent_bucket["low"].append(float(eval_kl[ent <= med].mean()))
                    ent_bucket["high"].append(float(eval_kl[ent > med].mean()))

                    loss.mean().backward(); opt.step(); opt.zero_grad()

                rows.append({
                    "seed": seed, "supervision": sup, "selection": sel,
                    "recipe": a.recipe, "refresh": a.refresh,
                    "wallclock_s": time.time() - t0,
                    "loss_low_entropy": float(np.mean(ent_bucket["low"][-50:])),
                    "loss_high_entropy": float(np.mean(ent_bucket["high"][-50:])),
                })
                print(rows[-1])
                del tmodel, student; torch.cuda.empty_cache()

    record(os.path.join(a.out, "e4.json"), {"config": vars(a), "rows": rows})
    _verdict(rows)


def _topk_indices(g, U, ubias, softcap, k, h, W, scores):
    """One top-k selection pass (teacher-scored if scores is None, else
    student-scored). Factored out of _truncate so "union" can call it twice per row and combine the results -- this is the
    same chunked top-k accumulation _truncate always did, just returning
    only the indices rather than also computing teacher logprobs, since a
    union of two k-sized sets needs deduping before logprobs are computed
    for the combined (variable-size) set."""
    from hsc.divergences import _chunks
    V = U.shape[0]
    best_v = torch.full((g.shape[0], k), -1, dtype=torch.long, device=g.device)
    best_s = torch.full((g.shape[0], k), -float("inf"), device=g.device)
    for aa, bb in _chunks(V, 16384):
        z = (g @ U[aa:bb].T).float() if scores is None else (h @ W[aa:bb].T).float()
        if scores is None and ubias is not None:
            z = z + ubias[aa:bb]
        if scores is None and softcap:
            z = softcap * torch.tanh(z / softcap)
        cat_s = torch.cat([best_s, z], 1)
        cat_v = torch.cat([best_v, torch.arange(aa, bb, device=g.device)
                           .expand(g.shape[0], -1)], 1)
        best_s, order = cat_s.topk(k, dim=1)
        best_v = torch.gather(cat_v, 1, order)
    return best_v


def _truncate(g, U, ubias, softcap, k, rule, h, W):
    """Teacher-selected top-k (Theorems 2 and 3 apply), student-selected, or
    union the union of the teacher's and the student's
    own top-k, the student-dependent case of \\citet{sfd2026} the theorems
    do NOT cover -- measured for comparison, not because they apply).

    Under "union" each row's true observation set size varies (between k and
    2k, depending on teacher/student overlap), but topk_forward_kl needs a
    fixed-shape (B, k') tensor. Padded to 2k with the row's own first real
    index repeated and its teacher logprob set to -inf: exp(-inf)=0 in the
    renormalization, so padding slots contribute nothing to the loss (not a
    real, separate vocabulary position being double-counted) and every row
    still gets a valid index at that column (not an out-of-bounds -1)."""
    from hsc.divergences import chunked_logsumexp
    if rule == "union":
        v_teacher = _topk_indices(g, U, ubias, softcap, k, h, W, scores=None)
        v_student = _topk_indices(g, U, ubias, softcap, k, h, W, scores="student")
        B = g.shape[0]
        best_v = torch.full((B, 2 * k), -1, dtype=torch.long, device=g.device)
        valid = torch.zeros((B, 2 * k), dtype=torch.bool, device=g.device)
        for b in range(B):
            u = torch.unique(torch.cat([v_teacher[b], v_student[b]]))
            n = u.shape[0]
            best_v[b, :n] = u
            valid[b, :n] = True
            if n < 2 * k:
                best_v[b, n:] = u[0]     # pad with a real, already-included index
    else:
        scores = None if rule == "teacher" else "student"
        best_v = _topk_indices(g, U, ubias, softcap, k, h, W, scores)
        valid = None
    lse_q = chunked_logsumexp(g, U, ubias, 16384, softcap)
    # NOT torch.einsum("bkd,bd->bk", ...): its backward crashes with a real
    # "Triton Error [CUDA]: an illegal memory access" at this shape class
    # (B~2000, k~1024, D~2048) on this cluster's PyTorch 2.13.0+cu130 -- see
    # hsc/divergences.py's topk_forward_kl for the full diagnosis (confirmed
    # via synthetic reproduction) and its identical fix. Every caller here
    # passes an already-detached g, so this specific call site never actually
    # backpropagated through the einsum in practice, but the identical
    # broadcast-multiply-and-sum form is used for consistency and to remove
    # the latent risk for any future caller that doesn't detach first.
    #
    # NOT a single (B, k, d) materialization either: at large k (e.g. T2's
    # k=4096 sweep against llama31-8b's d=4096), U[best_v.reshape(-1)]
    # .reshape(*best_v.shape, -1) is a dense (B, k, d) float32 tensor --
    # 2044*4096*4096*4 bytes = ~137GB, which crashed a real SLURM run
    # (t2_drift_vs_k, a run) with a CUDA OOM trying to allocate exactly
    # that many bytes. k up to 2k under "union" makes this worse, not
    # better. Chunking over k (like chunked_logsumexp does over V above)
    # does not help here -- at k=4096 a single k-chunk of 16384 already
    # covers the whole thing, since k < that chunk size; the (B, k, d)
    # volume is the same however it is chunked. Chunked over the BATCH
    # dimension instead, in groups sized to keep each chunk under ~10GB
    # regardless of k or d, since B (batch*seqlen, ~2000 here) is the axis
    # with headroom to shrink without hurting accuracy.
    zq_chunks = []
    B_total = best_v.shape[0]
    d_ = U.shape[1]
    chunk_B = max(1, int(1e10 / (best_v.shape[1] * d_ * 4)))
    for bb_a in range(0, B_total, chunk_B):
        bb_b = min(bb_a + chunk_B, B_total)
        v_chunk = best_v[bb_a:bb_b]
        z = (U[v_chunk.reshape(-1)].reshape(*v_chunk.shape, -1).float()
            * g[bb_a:bb_b].float().unsqueeze(1)).sum(-1)
        if ubias is not None:
            z = z + ubias[v_chunk]
        if softcap:
            z = softcap * torch.tanh(z / softcap)
        zq_chunks.append(z)
    zq = torch.cat(zq_chunks, dim=0)
    lp = zq - lse_q[:, None]
    if valid is not None:
        # Padding slots (see "union" above) get a large-but-FINITE negative
        # value here, AFTER lse_q's subtraction (lse_q itself must stay the
        # real teacher log-normalizer over the full vocabulary, unaffected
        # by padding) -- NOT -inf. topk_forward_kl computes
        # qt * (log_qt - log_pt) where qt = exp(log_qt - logsumexp(log_qt)):
        # with a genuine -inf entry, qt correctly underflows to exactly 0.0,
        # but 0.0 * (-inf - log_pt) is 0 * inf, which is NaN in IEEE float
        # arithmetic, not 0 -- confirmed by direct reproduction before
        # picking -1e30 instead. -1e30 still underflows qt to exactly 0.0
        # after the softmax (its logsumexp contribution is negligible next
        # to real, near-zero teacher logprobs) while keeping the surrounding
        # arithmetic finite, so padding slots contribute exactly 0 to the
        # loss instead of poisoning the whole row with NaN.
        lp = lp.masked_fill(~valid, -1e30)
    return best_v, lp


def _teacher_entropy(g, U, ubias, softcap, chunk):
    from hsc.divergences import chunked_logsumexp, _chunks
    lse = chunked_logsumexp(g, U, ubias, chunk, softcap)
    acc = torch.zeros_like(lse)
    with torch.no_grad():
        for aa, bb in _chunks(U.shape[0], chunk):
            z = (g @ U[aa:bb].T).float()
            if ubias is not None:
                z = z + ubias[aa:bb]
            if softcap:
                z = softcap * torch.tanh(z / softcap)
            q = torch.exp(z - lse[:, None])
            acc += (q * z).sum(1)
    return (lse - acc).cpu().numpy()


def _verdict(rows):
    import collections
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["supervision"], r["selection"])].append(r)
    # loss_low/high_entropy are the SAME held-out exact full-vocabulary
    # forward_kl for every row regardless of what that row trained on (see
    # the eval_kl comment in main()) -- comparable across supervision types.
    # Do not confuse this with the training loss itself, which for "topk"
    # rows is topk_forward_kl (both sides renormalized over k tokens, a
    # different-scale quantity) and would make topk look better than exact
    # by measuring less divergence, not better distillation.
    print(f"\n{'supervision':<14}{'selection':<12}{'low-H loss':>12}{'high-H loss':>13}{'wallclock':>11}")
    print("-" * 64)
    for kk, v in by.items():
        print(f"{kk[0]:<14}{kk[1]:<12}"
              f"{np.mean([x['loss_low_entropy'] for x in v]):>12.4f}"
              f"{np.mean([x['loss_high_entropy'] for x in v]):>13.4f}"
              f"{np.mean([x['wallclock_s'] for x in v]):>11.0f}")
    print("\nPrediction holds if the truncated-versus-exact gap is positive and")
    print("larger in the high-entropy bucket. If it is absent, report that the")
    print("theory's reach is confined to off-policy distillation.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--recipe", default="gkd", choices=RECIPES)
    p.add_argument("--teacher", default="qwen3-8b")
    p.add_argument("--student", default="qwen3-0.6b")
    p.add_argument("--supervision", nargs="+", default=["topk", "exact"])
    p.add_argument("--select", nargs="+", default=["teacher", "student"])
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--refresh", type=int, default=0, help="0 = fully on-policy")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--prompt-len", type=int, default=256)
    p.add_argument("--max-new", type=int, default=256)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--chunk", type=int, default=16384)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--out", default="runs/e4")
    main(p.parse_args())
