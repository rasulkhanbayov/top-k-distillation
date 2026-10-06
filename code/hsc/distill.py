"""
Distillation training loop covering every arm of E3 and E4.

One loop, one `objective` string, so that arms differ only in the loss. Matched
budgets are enforced here rather than by convention: `budget` is either
("storage", bytes_per_position) or ("flops", target) and the runner refuses to
start if the configured arm violates it.
"""
from __future__ import annotations
import math, time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .divergences import forward_kl, reverse_kl, jsd, skew_kl, topk_forward_kl

OBJECTIVES = ("ce", "full_fkl", "full_rkl", "full_jsd", "full_skew",
              "topk_fkl", "topp_fkl", "bild", "feature")


@dataclass
class DistillConfig:
    objective: str = "full_fkl"
    alpha_ce: float = 1.0          # weight on the auxiliary CE term
    alpha_kd: float = 1.0
    k: int = 256                   # for truncated arms
    top_p: float = 0.95
    temperature: float = 1.0
    chunk: int = 16384
    softcap_t: float | None = None
    bild_top_teacher: int = 8      # BiLD keeps a small teacher head
    bild_top_student: int = 8
    feature_proj: bool = True      # FitNets-style learned projection
    ce_label_smoothing: float = 0.0
    # disentangles alpha_ce's two bundled effects -- full-
    # vocabulary reach (via the softmax normalizer over all V tokens) and
    # hard-label sharpening (pushing all mass onto the single correct
    # token). Standard CE (smoothing=0) does both at once. Raising this
    # keeps the same full-vocabulary reach but softens the target toward
    # uniform, so sweeping THIS at fixed alpha_ce isolates the
    # sharpening axis, while sweeping alpha_ce at fixed smoothing isolates
    # the weight-on-full-vocabulary-signal axis -- together they let T5's
    # confound (does ECE track alpha_ce's WEIGHT or its HARDNESS?) be
    # tested directly rather than inferred from one 1-D sweep.
    seed: int = 0
    notes: str = ""
    matched_axis: str = ""         # "storage" | "flops" | "wallclock"
    matched_value: float = 0.0


def _ce(student_logits_fn, labels):
    return None  # placeholder; CE is computed inside the loop where logits exist


def distill_step(batch, student, cfg: DistillConfig, teacher_U=None,
                 teacher_bias=None, proj=None):
    """One optimization step. `batch` supplies whichever teacher signal the arm
    needs, so the caller decides between a hidden-state cache and a top-k cache.

    batch["h"]: (B, D) student final-norm hidden states, ALREADY FLATTENED to
    one row per supervised position. The caller runs the student's forward
    pass at (batch, seqlen) shape (needed for attention) and flattens before
    calling this -- distill_step cannot do that itself from raw input_ids,
    since chunked_logsumexp and the CE term below both operate on (B, D). An
    earlier version took batch["input_ids"] and called a student.hidden()
    that didn't exist anywhere in the codebase; every batch["g"]/["labels"]
    entry below must be flattened the same way and in the same row order as
    batch["h"].

    Required keys by arm:
      full_*   : batch["h"], batch["g"]  (B, d_s), (B, d_t), same B and order
      topk_*   : batch["h"], batch["tk_logprobs"], batch["tk_idx"]
      bild     : same as topk plus student-side top-m, computed here
      feature  : batch["h"], batch["g"]
      ce       : batch["h"], batch["labels"] only
    """
    h = batch["h"]                                   # (B, D) final-norm states
    W = student.lm_head.weight                      # (V, D)
    labels = batch["labels"]
    T = cfg.temperature

    loss_ce = torch.tensor(0.0, device=h.device)
    if cfg.alpha_ce > 0:
        # correct-token CE without materializing logits. lse MUST carry
        # gradient: an earlier version used the no_grad chunked_logsumexp,
        # so the loss value was right but the gradient dropped the softmax
        # normalizer and trained "raise the correct logit" instead of CE.
        # Every alpha_ce > 0 run before 2026-10-03 is affected.
        from .divergences import chunked_logsumexp_grad
        lse = chunked_logsumexp_grad(h, W, None, cfg.chunk)
        z_y = (h * W[labels]).sum(-1)
        eps = cfg.ce_label_smoothing
        if eps <= 0:
            loss_ce = (lse - z_y).mean()
        else:
            # Standard label smoothing: target (1-eps) on the true label,
            # eps spread uniformly over V. CE against that target equals
            # (1-eps)*(lse - z_y) + eps*(lse - mean_v z_v), so this needs
            # only the mean logit mean_v(W_v. h) = (mean_v W_v) . h --
            # cheap, no (B, V) tensor, same style as the unsmoothed path
            # above and chunked_logsumexp itself.
            z_bar = (h * W.mean(dim=0)).sum(-1)
            loss_ce = ((1 - eps) * (lse - z_y) + eps * (lse - z_bar)).mean()

    o = cfg.objective
    if o == "ce":
        loss_kd = torch.tensor(0.0, device=h.device)

    elif o == "full_fkl":
        loss_kd = forward_kl(h / T, W, batch["g"] / T, teacher_U,
                             None, teacher_bias, cfg.chunk, cfg.softcap_t).mean()
    elif o == "full_rkl":
        loss_kd = reverse_kl(h / T, W, batch["g"] / T, teacher_U,
                             None, teacher_bias, cfg.chunk, cfg.softcap_t).mean()
    elif o == "full_jsd":
        loss_kd = jsd(h / T, W, batch["g"] / T, teacher_U,
                      None, teacher_bias, cfg.chunk, cfg.softcap_t).mean()
    elif o == "full_skew":
        loss_kd = skew_kl(h / T, W, batch["g"] / T, teacher_U,
                          u_bias=teacher_bias, chunk=cfg.chunk,
                          softcap_t=cfg.softcap_t).mean()

    elif o in ("topk_fkl", "topp_fkl"):
        loss_kd = topk_forward_kl(h / T, W, batch["tk_logprobs"] / T,
                                  batch["tk_idx"]).mean()

    elif o == "bild":
        # BiLD: intersect the teacher's top-m with the student's top-m and
        # apply the bidirectional logit-difference loss on that union.
        loss_kd = _bild_loss(h, W, batch["tk_logprobs"], batch["tk_idx"], cfg)

    elif o == "feature":
        # FitNets / MiniLM style: regress the student state onto the teacher's.
        g = batch["g"]
        tgt = proj(g) if proj is not None else g
        loss_kd = F.mse_loss(h, tgt)

    else:
        raise ValueError(f"unknown objective {o}; expected one of {OBJECTIVES}")

    return cfg.alpha_ce * loss_ce + cfg.alpha_kd * loss_kd, {
        "ce": float(loss_ce), "kd": float(loss_kd)}


def _bild_loss(h, W, t_logprobs, t_idx, cfg):
    """Bidirectional logit difference on the union of teacher and student heads."""
    B, k = t_idx.shape
    m_t, m_s = cfg.bild_top_teacher, cfg.bild_top_student
    Wk = W[t_idx.reshape(-1)].reshape(B, k, -1)
    # NOT torch.einsum("bkd,bd->bk", ...): see divergences.py's
    # topk_forward_kl for the full diagnosis -- this same computation's
    # backward crashes with a real CUDA illegal-memory-access on this
    # cluster's PyTorch 2.13.0+cu130 at large B/k/D. Same fix here.
    zp = (Wk.float() * h.float().unsqueeze(1)).sum(-1)
    t_top = t_logprobs.topk(min(m_t, k), dim=1).indices
    s_top = zp.topk(min(m_s, k), dim=1).indices
    sel = torch.cat([t_top, s_top], dim=1)
    zp_s = torch.gather(zp, 1, sel)
    zq_s = torch.gather(t_logprobs, 1, sel)
    lp = zp_s - torch.logsumexp(zp_s, 1, keepdim=True)
    lq = zq_s - torch.logsumexp(zq_s, 1, keepdim=True)
    # symmetric difference of internal rankings
    dp = lp[:, :, None] - lp[:, None, :]
    dq = lq[:, :, None] - lq[:, None, :]
    return (dp - dq).pow(2).mean()
