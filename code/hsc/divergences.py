"""
Full-vocabulary divergences that never materialize a (B, V) logit tensor.

The teacher side is reconstructed on the fly from a cached hidden state, so the
teacher's body never runs during training. Only its unembedding does.

Design note. We write explicit chunked backward passes rather than relying on
autograd, for the same reason CCE and the Liger fused kernels do: autograd would
retain every chunk's logits. For forward KL the gradient has the CCE form,

    dL/dh   = sum_v (p_v - q_v) w_v        dL/dW_v = (p_v - q_v) h,

which is chunk-separable given the two log-normalizers. Reverse KL and JSD are
also chunk-separable once the scalar loss is known; see the derivations inline.

This is a reference implementation in PyTorch, not a Triton kernel. It is exact
and memory-bounded; for wall-clock parity with the published fused kernels, wire
in one of those instead.

Peak extra memory is O(B * chunk), independent of V.
"""

from __future__ import annotations

import torch

__all__ = [
    "forward_kl",
    "reverse_kl",
    "jsd",
    "skew_kl",
    "topk_forward_kl",
    "chunked_logsumexp",
]


def _chunks(V, chunk):
    for s in range(0, V, chunk):
        yield s, min(s + chunk, V)


@torch.no_grad()
def chunked_logsumexp(h, W, bias=None, chunk=16384, softcap=None):
    """Online log-sum-exp over the vocabulary. h: (B, D), W: (V, D)."""
    V = W.shape[0]
    m = torch.full((h.shape[0],), -float("inf"), device=h.device, dtype=torch.float32)
    s = torch.zeros_like(m)
    for a, b in _chunks(V, chunk):
        z = (h @ W[a:b].T).float()
        if bias is not None:
            z = z + bias[a:b]
        if softcap is not None:
            z = softcap * torch.tanh(z / softcap)
        cm = z.max(dim=1).values
        nm = torch.maximum(m, cm)
        s = s * torch.exp(m - nm) + torch.exp(z - nm[:, None]).sum(1)
        m = nm
    return m + torch.log(s)


def chunked_logsumexp_grad(h, W, bias=None, chunk=16384):
    """Differentiable counterpart of chunked_logsumexp (which is no_grad).

    Each vocabulary chunk's logsumexp is checkpointed, so memory stays
    O(B * chunk) while the gradient keeps the softmax-normalizer term. Any
    loss built from the student's lse that needs a gradient must use this:
    using the no_grad version keeps the loss VALUE right but silently drops
    the normalizer from the gradient (this broke the CE term in distill_step
    and the old jsd/skew_kl).
    """
    from torch.utils.checkpoint import checkpoint

    def _piece(a, b, h_, W_):
        z = (h_ @ W_.T).float()
        if bias is not None:
            z = z + bias[a:b]
        return torch.logsumexp(z, dim=1)

    return torch.logsumexp(torch.stack(
        [checkpoint(_piece, a, b, h, W[a:b], use_reentrant=False)
         for a, b in _chunks(W.shape[0], chunk)], dim=1), dim=1)


def _teacher_chunk(g, U, a, b, bias=None, softcap=None):
    z = (g @ U[a:b].T).float()
    if bias is not None:
        z = z + bias[a:b]
    if softcap is not None:
        z = softcap * torch.tanh(z / softcap)
    return z


class _ForwardKL(torch.autograd.Function):
    """KL(q || p) with q frozen. Two chunked passes forward, one backward."""

    @staticmethod
    def forward(ctx, h, W, g, U, w_bias, u_bias, chunk, softcap_t):
        V = W.shape[0]
        with torch.no_grad():
            lse_p = chunked_logsumexp(h, W, w_bias, chunk)
            lse_q = chunked_logsumexp(g, U, u_bias, chunk, softcap_t)
            # A = sum_v q_v z^q_v ; B = sum_v q_v z^p_v
            A = torch.zeros_like(lse_p)
            B = torch.zeros_like(lse_p)
            for a, b in _chunks(V, chunk):
                zq = _teacher_chunk(g, U, a, b, u_bias, softcap_t)
                q = torch.exp(zq - lse_q[:, None])
                zp = (h @ W[a:b].T).float()
                if w_bias is not None:
                    zp = zp + w_bias[a:b]
                A += (q * zq).sum(1)
                B += (q * zp).sum(1)
            # KL = (A - lse_q) - (B - lse_p)
            loss = (A - lse_q) - (B - lse_p)
        ctx.save_for_backward(h, W, g, U, lse_p, lse_q)
        ctx.extras = (w_bias, u_bias, chunk, softcap_t)
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        h, W, g, U, lse_p, lse_q = ctx.saved_tensors
        w_bias, u_bias, chunk, softcap_t = ctx.extras
        V = W.shape[0]
        gh = torch.zeros_like(h, dtype=torch.float32)
        gW = torch.zeros_like(W, dtype=torch.float32) if W.requires_grad else None
        go = grad_out[:, None]
        for a, b in _chunks(V, chunk):
            zp = (h @ W[a:b].T).float()
            if w_bias is not None:
                zp = zp + w_bias[a:b]
            p = torch.exp(zp - lse_p[:, None])
            zq = _teacher_chunk(g, U, a, b, u_bias, softcap_t)
            q = torch.exp(zq - lse_q[:, None])
            r = (p - q) * go                       # (B, chunk)
            gh += r @ W[a:b].float()
            if gW is not None:
                gW[a:b] += r.T @ h.float()
        return (gh.to(h.dtype), None if gW is None else gW.to(W.dtype),
                None, None, None, None, None, None)


def forward_kl(h, W, g, U, w_bias=None, u_bias=None, chunk=16384, softcap_t=None):
    """Exact KL(teacher || student), memory O(B * chunk)."""
    return _ForwardKL.apply(h, W, g, U, w_bias, u_bias, chunk, softcap_t)


def reverse_kl(h, W, g, U, w_bias=None, u_bias=None, chunk=16384, softcap_t=None):
    """KL(p || q), the MiniLLM-style mode-seeking objective.

    dL/dz^p_j = p_j (s_j - L) with s_j = log p_j - log q_j, since the +1 terms
    cancel against sum_v p_v = 1. Chunk-separable once L is known, so we do an
    extra forward pass to obtain L before accumulating gradients.
    """
    return _ReverseKL.apply(h, W, g, U, w_bias, u_bias, chunk, softcap_t)


class _ReverseKL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, W, g, U, w_bias, u_bias, chunk, softcap_t):
        V = W.shape[0]
        with torch.no_grad():
            lse_p = chunked_logsumexp(h, W, w_bias, chunk)
            lse_q = chunked_logsumexp(g, U, u_bias, chunk, softcap_t)
            loss = torch.zeros_like(lse_p)
            for a, b in _chunks(V, chunk):
                zp = (h @ W[a:b].T).float()
                if w_bias is not None:
                    zp = zp + w_bias[a:b]
                zq = _teacher_chunk(g, U, a, b, u_bias, softcap_t)
                p = torch.exp(zp - lse_p[:, None])
                s = (zp - lse_p[:, None]) - (zq - lse_q[:, None])
                loss += (p * s).sum(1)
        ctx.save_for_backward(h, W, g, U, lse_p, lse_q, loss)
        ctx.extras = (w_bias, u_bias, chunk, softcap_t)
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        h, W, g, U, lse_p, lse_q, loss = ctx.saved_tensors
        w_bias, u_bias, chunk, softcap_t = ctx.extras
        V = W.shape[0]
        gh = torch.zeros_like(h, dtype=torch.float32)
        gW = torch.zeros_like(W, dtype=torch.float32) if W.requires_grad else None
        go = grad_out[:, None]
        for a, b in _chunks(V, chunk):
            zp = (h @ W[a:b].T).float()
            if w_bias is not None:
                zp = zp + w_bias[a:b]
            zq = _teacher_chunk(g, U, a, b, u_bias, softcap_t)
            p = torch.exp(zp - lse_p[:, None])
            s = (zp - lse_p[:, None]) - (zq - lse_q[:, None])
            r = p * (s - loss[:, None]) * go
            gh += r @ W[a:b].float()
            if gW is not None:
                gW[a:b] += r.T @ h.float()
        return (gh.to(h.dtype), None if gW is None else gW.to(W.dtype),
                None, None, None, None, None, None)


def _mixture_div(h, W, g, U, w_bias, u_bias, chunk, softcap_t, beta, wp, wq):
    """wp*KL(p||m) + wq*KL(q||m) with m = beta*p + (1-beta)*q, chunked over V.

    Chunked autograd with per-chunk checkpointing (memory O(B * chunk)). The
    student's log-normalizer lse_p must carry gradient: an earlier version
    computed it from h.detach(), which kept the loss value right but dropped
    the normalizer's term from the gradient (off by ~4x the true gradient's
    scale on a CPU check against a dense reference).
    """
    from torch.utils.checkpoint import checkpoint

    V = W.shape[0]
    lse_p = chunked_logsumexp_grad(h, W, w_bias, chunk)
    lse_q = chunked_logsumexp(g, U, u_bias, chunk, softcap_t)

    def _piece(a, b, h_, W_, lse_p_):
        zp = (h_ @ W_.T).float()
        if w_bias is not None:
            zp = zp + w_bias[a:b]
        zq = _teacher_chunk(g, U, a, b, u_bias, softcap_t)
        lp = zp - lse_p_[:, None]
        lq = zq - lse_q[:, None]
        p, q = lp.exp(), lq.exp()
        lm = torch.log((beta * p + (1 - beta) * q).clamp_min(1e-30))
        out = 0.0
        if wp:
            out = out + wp * (p * (lp - lm)).sum(1)
        if wq:
            out = out + wq * (q * (lq - lm)).sum(1)
        return out

    total = 0.0
    for a, b in _chunks(V, chunk):
        total = total + checkpoint(_piece, a, b, h, W[a:b], lse_p,
                                   use_reentrant=False)
    return total


def jsd(h, W, g, U, w_bias=None, u_bias=None, chunk=16384, softcap_t=None,
        beta=0.5):
    """Generalized Jensen-Shannon: beta*KL(p||m) + (1-beta)*KL(q||m),
    m = beta*p + (1-beta)*q."""
    return _mixture_div(h, W, g, U, w_bias, u_bias, chunk, softcap_t,
                        beta, beta, 1 - beta)


def skew_kl(h, W, g, U, alpha=0.1, w_bias=None, u_bias=None, chunk=16384,
            softcap_t=None):
    """DistiLLM skew KL: KL(q || alpha*q + (1-alpha)*p), teacher term only.

    An earlier version returned jsd(beta=1-alpha), which has the right mixture
    but adds a 0.9*KL(p||m) student term that DistiLLM's SKL does not have.
    """
    return _mixture_div(h, W, g, U, w_bias, u_bias, chunk, softcap_t,
                        1 - alpha, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Truncated baselines. These are what the paper argues against; they are here so
# that E3 compares against a faithful implementation rather than a straw man.
# --------------------------------------------------------------------------- #
def topk_forward_kl(h, W, teacher_logprobs, teacher_idx, w_bias=None):
    """KL(q~ || p~) with both sides renormalized over the cached top-k set.

    teacher_logprobs : (B, k) cached values, already renormalized or not
    teacher_idx      : (B, k) int64 vocabulary indices
    """
    B, k = teacher_idx.shape
    D = W.shape[1]
    # NOT a single Wk = W[teacher_idx.reshape(-1)].reshape(B, k, -1) then
    # torch.einsum("bkd,bd->bk", ...) or torch.bmm: the einsum/bmm backward
    # crashes with a real "Triton Error [CUDA]: an illegal memory access"
    # (confirmed via a synthetic reproduction at B~2000/k=1024/D=2048) on
    # this cluster's PyTorch 2.13.0+cu130 -- silently produces a NaN
    # gradient on W on a first call and an outright crash on a repeat call
    # with the same inputs (E7's first pretrain_style/k=1024 full-sweep run
    # produced eval_kl=nan for all 3 seeds because of this). The elementwise
    # broadcast-multiply-and-sum below avoids that kernel. But at large B*k*D
    # (e.g. T2's k=4096 sweep, B~2044, D=2048) even THAT materializes a dense
    # (B, k, D) float32 tensor too large for one GPU -- 2044*4096*2048*4
    # bytes = ~68.6GB, which crashed a real SLURM run
    # (t2_k4096_smoke, a run) with a CUDA OOM matching that exact byte
    # count. Chunked over the batch dimension instead (same fix, same
    # ~10GB/chunk target, as experiments/e4_onpolicy.py's _truncate needed
    # for the identical shape of bug in the observation-set gather) --
    # gradients still flow through W via the indexing op in each chunk,
    # unlike _truncate's no-grad selection pass, so this function's autograd
    # contract is unchanged, only how the forward computation is tiled.
    chunk_B = max(1, int(1e10 / (k * D * 4)))
    zp_chunks = []
    for bb_a in range(0, B, chunk_B):
        bb_b = min(bb_a + chunk_B, B)
        idx_chunk = teacher_idx[bb_a:bb_b]
        Wk_chunk = W[idx_chunk.reshape(-1)].reshape(idx_chunk.shape[0], k, -1)
        zp_chunks.append((Wk_chunk.float() * h[bb_a:bb_b].float().unsqueeze(1)).sum(-1))
    zp = torch.cat(zp_chunks, dim=0)
    if w_bias is not None:
        zp = zp + w_bias[teacher_idx]
    log_pt = zp - torch.logsumexp(zp, dim=1, keepdim=True)
    log_qt = teacher_logprobs - torch.logsumexp(teacher_logprobs, dim=1, keepdim=True)
    qt = log_qt.exp()
    return (qt * (log_qt - log_pt)).sum(1)


def topp_indices(logprobs, p=0.95, max_k=1024):
    """Nucleus truncation, for the E3 top-p baseline and the E4 discriminator."""
    srt, idx = torch.sort(logprobs, dim=-1, descending=True)
    cum = srt.exp().cumsum(-1)
    keep = (cum - srt.exp()) < p
    keep[..., 0] = True
    keep[..., max_k:] = False
    return idx, keep
