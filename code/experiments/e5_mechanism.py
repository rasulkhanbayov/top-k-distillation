#!/usr/bin/env python3
"""
E5 (T1). Mechanism: tail mass, calibration, and the cross-entropy ablation.

This targets the coordinate Theorem 3 identifies as unconstrained and is far more
sensitive than aggregate benchmark accuracy. It is also where Theorem 2's lower
bound is measured on real models, replacing the synthetic check in Appendix C.

The decisive variant sets alpha_ce = 0, leaving the distillation term as the only
signal. Registered prediction: with cross-entropy removed, top-k students show
tail-mass error that tracks their initialization rather than the teacher and does
not decrease with training, while full-vocabulary students converge to the
teacher's tail mass. With cross-entropy present the effect is attenuated, which
would establish that tail-mass control in current pipelines comes from the
auxiliary term rather than from distillation.

Usage:
  python e5_mechanism.py --arms topk_fkl full_fkl --alpha-ce 0.0 1.0 --seeds 0 1 2
"""
import argparse, sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch

from common import (load_teacher, teacher_hidden_states, set_seed, record,
                    STUDENTS, SEEDS, text_batches, StudentHidden)
from hsc.distill import DistillConfig, distill_step
from hsc.divergences import chunked_logsumexp
from hsc.metrics import (ece, separation_constants, residual_identity,
                         effective_support, head_tail_centroids)
from e4_onpolicy import _truncate


def probe(student, tmodel, tok, U, ubias, softcap, W, k, n=256, chunk=16384,
         select="teacher", track_absorption=False):
    """Everything Theorems 2 and 3 make predictions about, measured on real text.
    A measurement pass, not a training step -- no_grad throughout.

    select: "teacher" (default, the fixed-S case the theorems are stated
    for) or "union" S is the union of the teacher's and
    the student's own top-k, recomputed every position -- the
    student-dependent case of \\citet{sfd2026} that Theorems 2/3 do NOT
    cover; this probe measures it for comparison, not because the theorems
    apply to it. |S| varies row-to-row under "union" (union size is between
    k and 2k depending on overlap), unlike "teacher" where |S|=k exactly.

    track_absorption: Accumulates, for every row, the
    student's tail probability mass (V \\ S) broken down per token id, using
    the same S/lp already computed below for the separation/residual
    quantities -- no extra forward pass. Off by default since the returned
    dict is then a length-V array (accepted; this probe already runs the
    O(V) chunked_logsumexp above, so this adds no new asymptotic cost, only
    a per-row scatter-add)."""
    out = {"tail_mass_student": [], "tail_mass_teacher": [], "ratio": [],
           "gamma": [], "Gamma": [], "lower_bound": [], "residual": [],
           "grad_trunc_norm": [],
           "tightness": [], "identity_mismatch": [], "entropy": [],
           "eff_support": [], "conf": [], "correct": [], "skipped_no_tail": 0,
           "skipped_teacher_zero": 0, "skipped_student_zero": 0}
    absorbed = np.zeros(U.shape[0] if track_absorption else 0, dtype=np.float64)
    # Cheap, permanent progress markers (flush=True): a prior run on this
    # shared account stalled silently for 2 hours with zero output between
    # model loading and job kill, and the cause (contention from other jobs
    # on the same account vs. a real bug) could not be determined after the
    # fact from an empty log. These make the next stall diagnosable live.
    t_probe = time.time()
    print(f"  [probe] n={n} batches, starting", flush=True)
    for bi, ids in enumerate(text_batches(tok, n)):
        print(f"  [probe] batch {bi+1}/{n} ready at t+{time.time()-t_probe:.1f}s",
              flush=True)
        ids = ids.cuda()
        with torch.no_grad():
            g = teacher_hidden_states(tmodel, ids)[:, :-1].reshape(-1, U.shape[1])
            h = student(input_ids=ids, output_hidden_states=True
                        ).hidden_states[-1][:, :-1].reshape(-1, W.shape[1])
            labels = ids[:, 1:].reshape(-1)
            lse_q = chunked_logsumexp(g, U, ubias, chunk, softcap)
            lse_p = chunked_logsumexp(h, W, None, chunk)
            # W doesn't change within a batch; converting it to CPU/NumPy
            # inside the chunk loop redid the same (V, D) copy up to 8x/batch
            # (510 rows / 64-row chunks) for nothing -- hoisted out.
            Wn = W.float().cpu().numpy()

            # exact top-k of the teacher, then the Theorem 2 and 3 quantities
            for i in range(0, h.shape[0], 64):
                hs, gs = h[i:i+64], g[i:i+64]
                zq = (gs @ U.T).float()
                if ubias is not None:
                    zq = zq + ubias
                if softcap:
                    zq = softcap * torch.tanh(zq / softcap)
                lq = (zq - lse_q[i:i+64, None]).cpu().numpy()
                zp = (hs @ W.T).float()
                lp = (zp - lse_p[i:i+64, None]).cpu().numpy()
                for j in range(lq.shape[0]):
                    S = np.argsort(lq[j])[::-1][:k]
                    if select == "union":
                        S_student = np.argsort(lp[j])[::-1][:k]
                        S = np.union1d(S, S_student)
                    if track_absorption:
                        # Tail = V \ S under the student's own distribution
                        # (exp(lp[j])), the quantity the objective is blind
                        # to per Theorem 3 -- accumulated per-token-id across
                        # every row in this probe rather than per-row, since
                        # a single row's tail is too sparse to be informative
                        # on its own.
                        tail_mask = np.ones(lp[j].shape[0], dtype=bool)
                        tail_mask[S] = False
                        absorbed[tail_mask] += np.exp(lp[j][tail_mask])
                    # separation_constants and residual_identity both need
                    # head_centroid/tail_centroid for the SAME (logp, Wn, S)
                    # triple -- computed once here and shared, instead of
                    # each function recomputing both from scratch (see
                    # head_tail_centroids's docstring for the real cost this
                    # avoided: it was the actual bottleneck behind a real
                    # ~50-minute-per-batch stall).
                    tc = head_tail_centroids(lq[j], Wn, S)
                    sc = head_tail_centroids(lp[j], Wn, S)
                    sep = separation_constants(lp[j], lq[j], Wn, S,
                                               teacher_centroids=tc, student_centroids=sc)
                    res = residual_identity(lp[j], lq[j], Wn, S,
                                            teacher_centroids=tc, student_centroids=sc)
                    if not sep["valid"]:
                        # Includes rows where the teacher (or student) put
                        # ~all mass on the top-k observation set, leaving no
                        # tail to form a centroid from (see
                        # head_tail_centroids/separation_constants) -- a real
                        # ~0.06% of rows in a real E5 run, not a bug. Skipped
                        # rather than let a single NaN poison this whole
                        # batch's np.mean over 8000+ rows.
                        #
                        # A real full grid run showed this rate climbing from
                        # ~0.18% to ~38% over 500 steps, specifically in the
                        # full_fkl arm -- tracked separately by which side
                        # (teacher vs student) is degenerate, to tell apart
                        # "the student is becoming overconfident under
                        # full-vocabulary distillation" (real finding) from
                        # something else.
                        out["skipped_no_tail"] += 1
                        if tc[1][1] <= 0:
                            out["skipped_teacher_zero"] += 1
                        if sc[1][1] <= 0:
                            out["skipped_student_zero"] += 1
                        continue
                    r = float(np.linalg.norm(res["direct"]))
                    # (residual plateau): needs
                    # ||grad L_S|| tracked alongside the existing residual
                    # (||grad L_full|| = res["direct"]) and tail term
                    # (res["identity"]) to show grad L_S decaying to zero
                    # while the other two plateau. grad L_S = c_S^p - c_S^q
                    # (Section 4's own derivation) -- tc[0]/sc[0] are the
                    # teacher's/student's head centroids over the SAME idx,
                    # already computed above for sep/res, so this is free.
                    out["grad_trunc_norm"].append(
                        float(np.linalg.norm(sc[0] - tc[0])))
                    out["tail_mass_student"].append(sep["m_p"])
                    out["tail_mass_teacher"].append(sep["m_q"])
                    out["ratio"].append(sep["m_p"] / sep["m_q"])
                    out["gamma"].append(sep["gamma"]); out["Gamma"].append(sep["Gamma"])
                    out["lower_bound"].append(sep["lower_bound"])
                    out["residual"].append(r)
                    out["tightness"].append(sep["lower_bound"] / max(r, 1e-30))
                    out["identity_mismatch"].append(res["rel_mismatch"])
                    out["entropy"].append(float(-(np.exp(lq[j]) * lq[j]).sum()))
                    out["eff_support"].append(effective_support(lq[j]))
                    out["conf"].append(float(np.exp(lp[j]).max()))
                    out["correct"].append(float(np.argmax(lp[j]) == int(labels[i+j])))
    out["ece"] = ece(out["conf"], out["correct"])
    n_total = len(out["tail_mass_student"]) + out["skipped_no_tail"]
    print(f"  [probe] done after {time.time()-t_probe:.1f}s "
          f"({out['skipped_no_tail']}/{n_total} rows skipped: "
          f"teacher_zero={out['skipped_teacher_zero']} "
          f"student_zero={out['skipped_student_zero']})", flush=True)
    if track_absorption:
        top = np.argsort(absorbed)[::-1][:200]
        out["absorption_top_ids"] = [int(t) for t in top]
        out["absorption_top_mass"] = [float(absorbed[t]) for t in top]
        out["absorption_total_tail_mass"] = float(absorbed.sum())
    return {kk: (float(np.mean(v)) if isinstance(v, list) and kk not in
                ("ece", "absorption_top_ids", "absorption_top_mass") else v)
            for kk, v in out.items()}


def main(a):
    rows = []
    if a.measure_every:
        # (residual plateau): needs denser tracking than
        # the default 4-point {0, 10%, 50%, 100%} schedule to actually show
        # a plateau rather than just two distant endpoints, especially at
        # T4's own 5000-step spec (10x the other Mechanism runs' 500).
        checkpoints = sorted(set(range(0, a.steps + 1, a.measure_every))
                             | {a.steps})
    else:
        checkpoints = sorted({0, a.steps // 10, a.steps // 2, a.steps})
    for arm in a.arms:
        for ace in a.alpha_ce:
            for smoothing in a.ce_smoothing:
                for seed in a.seeds:
                    print(f"=== arm={arm} alpha_ce={ace} "
                          f"ce_smoothing={smoothing} seed={seed} starting ===",
                          flush=True)
                    set_seed(seed)
                    tmodel, tok, U, ubias, softcap = load_teacher(a.teacher)
                    from transformers import AutoModelForCausalLM
                    raw_student = AutoModelForCausalLM.from_pretrained(
                        STUDENTS[a.student][0], torch_dtype=torch.bfloat16,
                        device_map="cuda")
                    print("  both models loaded", flush=True)
                    student = StudentHidden(raw_student)
                    W = raw_student.get_output_embeddings().weight
                    opt = torch.optim.AdamW(raw_student.parameters(), lr=a.lr)
                    cfg = DistillConfig(objective=arm, alpha_ce=ace, k=a.k,
                                        chunk=a.chunk, softcap_t=softcap,
                                        seed=seed, ce_label_smoothing=smoothing)

                    trajectory = []

                    def do_probe(step, absorb=False):
                        p = probe(raw_student, tmodel, tok, U, ubias, softcap, W,
                                 a.k, n=a.probe_batches, select=a.select,
                                 track_absorption=absorb)
                        p["step"] = step
                        trajectory.append(p)
                        print(f"[{arm} ace={ace} sm={smoothing} seed={seed} "
                              f"step={step}] tail mass student "
                              f"{p['tail_mass_student']:.4f} vs teacher "
                              f"{p['tail_mass_teacher']:.4f}")

                    do_probe(0)
                    print("  training loop starting", flush=True)
                    t_train = time.time()
                    data_iter = text_batches(tok, a.steps, bs=a.batch, L=a.seqlen)
                    for step, ids in enumerate(data_iter, start=1):
                        if step <= 3 or step % 50 == 0:
                            print(f"  step {step}/{a.steps} batch ready at "
                                  f"t+{time.time()-t_train:.1f}s", flush=True)
                        ids = ids.cuda()
                        g = teacher_hidden_states(tmodel, ids).detach()
                        gf = g[:, :-1].reshape(-1, U.shape[1])
                        # One forward pass, gradients attached; flatten to (B, D)
                        # per distill_step's contract (see hsc/distill.py). The
                        # selection step below (_truncate, teacher-side top-k)
                        # only needs the VALUES, not gradients through h, so it
                        # gets a detached view -- the actual training loss still
                        # backprops through the same hf tensor via batch["h"].
                        hf = student.hidden(ids)[:, :-1].reshape(-1, W.shape[1])
                        batch = {"h": hf, "labels": ids[:, 1:].reshape(-1)}
                        if arm in ("topk_fkl", "topp_fkl"):
                            idx, lp = _truncate(gf, U, ubias, softcap, a.k, a.select,
                                                hf.detach(), W)
                            batch["tk_idx"] = idx
                            batch["tk_logprobs"] = lp
                        else:
                            batch["g"] = gf

                        loss, parts = distill_step(batch, student, cfg,
                                                   teacher_U=U, teacher_bias=ubias)
                        loss.backward()
                        opt.step(); opt.zero_grad()
                        if step <= 3:
                            print(f"  step {step}/{a.steps} optimizer step done, "
                                  f"loss={float(loss):.4f} at "
                                  f"t+{time.time()-t_train:.1f}s", flush=True)

                        if step in checkpoints:
                            # absorption tracking only on the
                            # final checkpoint of the run this flag is asked
                            # for -- "one pass over a checkpoint" per the note,
                            # not every checkpoint of every arm/seed/smoothing.
                            absorb = (a.absorb_report and step == a.steps and
                                     arm == a.absorb_report and ace == 0.0 and
                                     smoothing == 0.0)
                            do_probe(step, absorb=absorb)

                    rows.append({
                        "arm": arm, "alpha_ce": ace, "ce_smoothing": smoothing,
                        "seed": seed, "trajectory": trajectory,
                    })
                    del tmodel, raw_student, student; torch.cuda.empty_cache()

                    # Checkpoint after every cell, not just at the end: job
                    # 403903 (T5b smoothing sweep) ran 38/48 cells over a
                    # full 3-day walltime, hit the limit, and lost all of
                    # them because record() was only called once at the end
                    # of main(). Re-writing the whole file each cell is
                    # cheap (record() overwrites, not appends) and bounds
                    # the loss from any future timeout/preemption to the
                    # one cell in flight.
                    record(os.path.join(a.out, "e5.json"), {"config": vars(a), "rows": rows})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="llama31-8b")
    p.add_argument("--student", default="llama32-1b")
    p.add_argument("--arms", nargs="+", default=["topk_fkl", "full_fkl"])
    p.add_argument("--alpha-ce", nargs="+", type=float, default=[0.0, 1.0])
    p.add_argument("--ce-smoothing", nargs="+", type=float, default=[0.0],
                   help="label smoothing epsilon on the "
                        "auxiliary CE term (hsc.distill.DistillConfig's "
                        "ce_label_smoothing), swept independently of "
                        "--alpha-ce. Standard CE (alpha_ce's weight) "
                        "bundles two effects -- full-vocabulary reach via "
                        "the softmax normalizer, and hard-label sharpening "
                        "toward the single correct token. Raising this at "
                        "FIXED alpha_ce keeps the same full-vocabulary "
                        "reach but softens the target, isolating the "
                        "sharpening axis from the weight axis -- lets T5's "
                        "confound (does ECE track alpha_ce's weight or its "
                        "hardness?) be tested directly. Default [0.0] "
                        "reproduces every prior run's exact behavior "
                        "(standard CE, unchanged).")
    p.add_argument("--k", type=int, default=256)
    p.add_argument("--select", default="teacher", choices=["teacher", "union"],
                   help="observation-set selection rule for topk_fkl/topp_fkl: "
                        "'teacher' is the fixed-S case Theorems 2/3 are stated "
                        "for (default); 'union' unions the "
                        "teacher's and student's own top-k, which the theorems "
                        "do not cover -- measured for comparison only")
    p.add_argument("--probe-batches", type=int, default=32)
    p.add_argument("--measure-every", type=int, default=0,
                   help="if set, probe every N steps instead of the default "
                        "sparse {0, 10%%, 50%%, 100%%} schedule -- needed by "
                        "(residual plateau), which must show "
                        "denser trajectory than four points to demonstrate "
                        "a plateau rather than just report two endpoints")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seqlen", type=int, default=512)
    p.add_argument("--chunk", type=int, default=16384)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--absorb-report", default=None,
                   help="name an arm (e.g. topk_fkl) to "
                        "record, on that arm's alpha_ce=0.0 final checkpoint "
                        "only, which tail token ids absorb the student's "
                        "tail probability mass -- written to "
                        "e5.json's trajectory row for that checkpoint as "
                        "absorption_top_ids/absorption_top_mass. Off by "
                        "default (no cost added to a normal Mechanism run).")
    p.add_argument("--out", default="runs/e5")
    main(p.parse_args())
