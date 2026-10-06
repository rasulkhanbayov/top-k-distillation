#!/usr/bin/env python3
"""
E9 (T2). Student capacity and the teacher-student gap.

Separates the identifiability deficit (a property of the teacher's d and the
cache's k) from the capacity gap (a property of the student). Vary the student at
fixed teacher, and the teacher at fixed student, and relate the result to
distillation scaling laws.

The confound this addresses: a reader can attribute any truncation effect to the
student being too small rather than to the supervision being underdetermined.
Only the two-way sweep separates them.

Usage:
  python e9_capacity.py --teachers qwen3-1.7b qwen3-8b qwen3-32b \
                        --students qwen3-0.6b qwen3-1.7b --ks 64 256 1024
"""
import argparse, sys, os, time, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch

from common import (load_teacher, teacher_hidden_states, set_seed, record,
                    TEACHERS, STUDENTS, SEEDS, text_batches, StudentHidden)
from hsc.distill import DistillConfig, distill_step
from hsc.divergences import forward_kl
from e4_onpolicy import _truncate

OBJECTIVES = ("topk_fkl", "full_fkl")


def _eval_kl(student, tmodel, tok, U, ubias, softcap, W, a, seed):
    """Held-out, full-vocabulary exact forward KL(teacher || student), computed
    the same way regardless of what the arm actually trained on. Raw training
    loss for topk_fkl is NOT comparable to full_fkl's raw loss (topk_fkl
    renormalizes over only k tokens, so its scale is different) -- see E4's
    eval_kl fix. Held out by drawing from a different text_batches seed
    offset than training, not by re-consuming the training iterator (a
    training data_iter of length a.steps is already exhausted after the
    training loop anyway)."""
    total, n = 0.0, 0
    with torch.no_grad():
        for ids in text_batches(tok, a.eval_batches, bs=a.batch, L=a.seqlen,
                                seed=seed + 10_000):
            ids = ids.cuda()
            g = teacher_hidden_states(tmodel, ids)[:, :-1].reshape(-1, U.shape[1])
            h = student.hidden(ids)[:, :-1].reshape(-1, W.shape[1])
            loss = forward_kl(h, W, g, U, None, ubias, a.chunk, softcap)
            total += float(loss.sum())
            n += loss.shape[0]
    return total / max(n, 1)


def main(a):
    grid = list(itertools.product(a.teachers, a.students, a.ks, a.seeds))
    print(f"{len(grid)} cells x {len(OBJECTIVES)} objectives: "
          f"teacher x student x k x seed x objective")
    print(f"{'teacher':<13}{'d_t':>6}{'student':<13}{'d_s':>6}{'k':>7}"
          f"{'k/(d_t+1)':>11}{'k/(d_s+1)':>11}")
    print("-" * 70)
    for t, s, k, seed in grid:
        if seed != a.seeds[0]:
            continue
        dt, ds = TEACHERS[t][1], STUDENTS[s][1]
        print(f"{t:<13}{dt:>6}{s:<13}{ds:>6}{k:>7}"
              f"{k/(dt+1):>11.3f}{k/(ds+1):>11.3f}")
    print("-" * 70)
    print("Both ratios matter: k/(d_t+1) governs teacher identifiability")
    print("(Proposition 1), k/(d_s+1) governs whether the objective determines")
    print("the student (Theorem 3). The grid is designed so the two vary")
    print("independently, which a single teacher-student pair cannot do.")

    rows = []
    for obj in a.objectives:
        for t, s, k, seed in grid:
            print(f"=== obj={obj} teacher={t} student={s} k={k} seed={seed} "
                  f"starting ===", flush=True)
            set_seed(seed)
            tmodel, tok, U, ubias, softcap = load_teacher(t)
            from transformers import AutoModelForCausalLM
            raw_student = AutoModelForCausalLM.from_pretrained(
                STUDENTS[s][0], torch_dtype=torch.bfloat16, device_map="cuda")
            print("  both models loaded", flush=True)
            student = StudentHidden(raw_student)
            W = raw_student.get_output_embeddings().weight
            opt = torch.optim.AdamW(raw_student.parameters(), lr=a.lr)
            # alpha_ce=0.0 is mandatory: leaving it unset defaults to 1.0
            # (DistillConfig's own default) and silently trains on CE+KD
            # rather than pure KD -- confirmed as a real divergence bug in
            # E3's history (see e3_offpolicy.py's ARMS comment). Every cell
            # here is a pure distillation objective.
            cfg = DistillConfig(objective=obj, alpha_ce=0.0, k=k,
                                chunk=a.chunk, softcap_t=softcap, seed=seed)

            t_train = time.time()
            losses = []
            data_iter = text_batches(tok, a.steps, bs=a.batch, L=a.seqlen,
                                     seed=seed)
            for step, ids in enumerate(data_iter, start=1):
                if step <= 3 or step % 50 == 0:
                    print(f"  step {step}/{a.steps} batch ready at "
                          f"t+{time.time()-t_train:.1f}s", flush=True)
                ids = ids.cuda()
                g = teacher_hidden_states(tmodel, ids).detach()
                gf = g[:, :-1].reshape(-1, U.shape[1])
                hf = student.hidden(ids)[:, :-1].reshape(-1, W.shape[1])
                batch = {"h": hf, "labels": ids[:, 1:].reshape(-1)}
                if obj in ("topk_fkl", "topp_fkl"):
                    idx, lp = _truncate(gf, U, ubias, softcap, k, "teacher",
                                        hf.detach(), W)
                    batch["tk_idx"] = idx
                    batch["tk_logprobs"] = lp
                else:
                    batch["g"] = gf

                loss, parts = distill_step(batch, student, cfg,
                                           teacher_U=U, teacher_bias=ubias)
                loss.backward()
                opt.step(); opt.zero_grad()
                losses.append(float(loss))
                if step <= 3:
                    print(f"  step {step}/{a.steps} optimizer step done, "
                          f"loss={float(loss):.4f} at "
                          f"t+{time.time()-t_train:.1f}s", flush=True)

            eval_kl = _eval_kl(student, tmodel, tok, U, ubias, softcap, W, a,
                               seed)
            wallclock = time.time() - t_train
            dt, ds = TEACHERS[t][1], STUDENTS[s][1]
            rows.append({
                "objective": obj, "teacher": t, "student": s, "k": k,
                "seed": seed, "d_t": dt, "d_s": ds,
                "k_over_dt1": k / (dt + 1), "k_over_ds1": k / (ds + 1),
                "train_loss_final": float(np.mean(losses[-20:])) if losses else None,
                "eval_kl": eval_kl, "wallclock_s": wallclock,
            })
            print(f"  [{obj} {t}/{s} k={k} seed={seed}] "
                  f"train_loss={rows[-1]['train_loss_final']:.4f} "
                  f"eval_kl={eval_kl:.4f} ({wallclock:.1f}s)", flush=True)

            del tmodel, raw_student, student; torch.cuda.empty_cache()

    _print_partial_effects(rows)
    record(os.path.join(a.out, "e9.json"), {"config": vars(a), "rows": rows})


def _print_partial_effects(rows):
    """Simple grouped-mean summary so a human can see the two effects (teacher
    identifiability vs. student capacity) are separable without loading the
    JSON into a notebook -- mirrors e6_systems.py/e8_precision.py's own
    end-of-main summary tables. Not a full regression: grouped means at fixed
    student (varying k/(d_t+1)) and at fixed teacher (varying k/(d_s+1)),
    which is exactly the partial-effect view the docstring/stub asked for
    ("partial effects, not a single regression on k alone")."""
    if not rows:
        print("no rows collected, nothing to summarize")
        return
    for obj in sorted({r["objective"] for r in rows}):
        orows = [r for r in rows if r["objective"] == obj]
        print(f"\n=== {obj}: eval KL vs k/(d_t+1), grouped by student "
              f"(teacher-identifiability effect at fixed student) ===")
        for s in sorted({r["student"] for r in orows}):
            srows = [r for r in orows if r["student"] == s]
            print(f"  student={s}")
            by_ratio = {}
            for r in srows:
                by_ratio.setdefault(round(r["k_over_dt1"], 4), []).append(r["eval_kl"])
            for ratio in sorted(by_ratio):
                vals = np.array(by_ratio[ratio])
                print(f"    k/(d_t+1)={ratio:>8.3f}  eval_kl mean={vals.mean():.4f} "
                      f"std={vals.std():.4f} n={len(vals)}")

        print(f"\n=== {obj}: eval KL vs k/(d_s+1), grouped by teacher "
              f"(student-capacity effect at fixed teacher) ===")
        for t in sorted({r["teacher"] for r in orows}):
            trows = [r for r in orows if r["teacher"] == t]
            print(f"  teacher={t}")
            by_ratio = {}
            for r in trows:
                by_ratio.setdefault(round(r["k_over_ds1"], 4), []).append(r["eval_kl"])
            for ratio in sorted(by_ratio):
                vals = np.array(by_ratio[ratio])
                print(f"    k/(d_s+1)={ratio:>8.3f}  eval_kl mean={vals.mean():.4f} "
                      f"std={vals.std():.4f} n={len(vals)}")
    print()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--teachers", nargs="+", default=["qwen3-1.7b", "qwen3-8b", "qwen3-32b"])
    p.add_argument("--students", nargs="+", default=["qwen3-0.6b", "qwen3-1.7b"])
    p.add_argument("--ks", nargs="+", type=int, default=[64, 256, 1024])
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--objectives", nargs="+", default=list(OBJECTIVES))
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seqlen", type=int, default=512)
    p.add_argument("--chunk", type=int, default=16384)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--out", default="runs/e9")
    main(p.parse_args())
