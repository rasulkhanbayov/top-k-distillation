"""
Metrics for E5 (mechanism) and E3 (quality).

The tail-mass and calibration metrics are the ones Theorem 3 makes predictions
about, so they are the load-bearing measurements in the paper, more sensitive
than aggregate benchmark accuracy.
"""
from __future__ import annotations
import numpy as np

__all__ = ["tail_mass", "tail_mass_error", "entropy", "ece", "tail_centroid",
           "head_centroid", "head_tail_centroids", "separation_constants",
           "residual_identity", "effective_support"]


def entropy(logp):
    p = np.exp(logp)
    return float(-(p * logp).sum(-1))


def tail_mass(logp, idx):
    """1 - sum of mass on the observation set."""
    p = np.exp(logp)
    return float(1.0 - p[..., idx].sum(-1))


def tail_mass_error(logp_student, logp_teacher, idx):
    """Signed error; Theorem 3 predicts it does not shrink with training."""
    return tail_mass(logp_student, idx) - tail_mass(logp_teacher, idx)


def effective_support(logp, delta=1e-3):
    """Smallest number of entries carrying 1 - delta of the mass."""
    p = np.sort(np.exp(logp))[::-1]
    return int(np.searchsorted(np.cumsum(p), 1 - delta) + 1)


def ece(confidences, correct, n_bins=15):
    """Standard equal-width expected calibration error."""
    conf = np.asarray(confidences); corr = np.asarray(correct, float)
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        total += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return float(total)


# --------------------------------------------------------------------------- #
# Quantities appearing in Theorems 2 and 3. Used by E5 to test the bound.
# --------------------------------------------------------------------------- #
def head_centroid(logp, W, idx):
    p = np.exp(logp[idx]); p = p / p.sum()
    return p @ W[idx]


def tail_centroid(logp, W, idx):
    """Kept for API compatibility; prefer head_tail_centroids when you also
    need the head centroid or call this for both student and teacher, since
    it recomputes the full-vocabulary sum this function's docstring warns
    about avoiding. See head_tail_centroids for why and the real-world cost
    that motivated it."""
    _, (cT, m) = head_tail_centroids(logp, W, idx)
    return cT, m


def head_tail_centroids(logp, W, idx):
    """(head_centroid, (tail_centroid, tail_mass)) computed together in one
    pass over the full vocabulary, instead of tail_centroid's own full
    O(V*d) mask/exp/matmul from scratch.

    tail_centroid used to build a (V,) boolean mask and redo the exp and a
    (V-k, d) matmul on every call. Since logp is a normalized log-probability
    distribution (sum_v p_v = 1 exactly, up to float error) and idx's
    complement is "everything else", the tail sum is just the full-vocabulary
    sum minus the idx-restricted sum -- both O(V*d), but ONE dense contiguous
    matmul (p_full @ W) instead of a masked/fancy-indexed one, and the
    idx-restricted part is exactly what head_centroid already computes, so
    nothing is duplicated between the two centroids.

    This was the actual bottleneck behind a real ~50-minute-per-batch stall
    in E5's probe(): separation_constants and residual_identity are called
    back to back on the SAME (logp_teacher, W, idx) triple in probe()'s
    per-row loop, so the old code computed head_centroid and both
    tail_centroids twice each, ~510 rows/batch, at full V=128256 -- roughly
    2000+ full-vocabulary mask/exp/matmul operations per batch on CPU. Traced
    with a chain of diagnostics that ruled out GPU contention, node health,
    and the model forward pass before landing here (see diag_real_probe.sh
    and diag_probe_workload2.sh in jobs/ for the elimination process).

    Returns
    -------
    head : (d,) head_centroid(logp, W, idx)
    (tail, tail_mass) : (d,) tail centroid and its probability mass
    """
    p_full = np.exp(logp)
    p_idx = p_full[idx]
    idx_sum_p = p_idx.sum()
    idx_sum_pw = p_idx @ W[idx]
    head = idx_sum_pw / idx_sum_p

    full_sum_p = p_full.sum()          # ~1.0 for a normalized logp
    full_sum_pw = p_full @ W
    tail_mass_ = full_sum_p - idx_sum_p
    # Real teacher distributions can be confident enough that idx (the top-k
    # observation set) captures the observed distribution's mass to within
    # float32 precision, leaving tail_mass_ at exactly 0.0 or a value so
    # small the division below is not meaningful (found in a real E5 run:
    # 5 of 8176 rows, m_q == 0.0 exactly). There genuinely is no tail
    # centroid in that case -- not a bug to paper over with a fallback
    # value, since Theorem 2's centroids are only defined for m in (0,1).
    # Return NaN explicitly rather than let float division silently produce
    # it, and let tail_mass itself (0.0, still a real, useful number) tell
    # the caller why: check `tail_mass < eps`, not `isnan(tail)`, to detect
    # this case, since deciding "how small is degenerate" belongs with the
    # caller (separation_constants) that knows what the value feeds into.
    if tail_mass_ <= 0:
        tail = np.full_like(head, np.nan)
    else:
        tail = (full_sum_pw - idx_sum_pw) / tail_mass_
    return head, (tail, float(tail_mass_))


def separation_constants(logp_student, logp_teacher, W, idx,
                         teacher_centroids=None, student_centroids=None):
    """gamma and Gamma of Theorem 2, plus the projections they bound.

    teacher_centroids/student_centroids: optional precomputed
    head_tail_centroids(...) results, so a caller that also needs
    residual_identity on the same (logp, W, idx) triple (probe() does, every
    row) does not pay for head_centroid/tail_centroid twice. Computed here if
    not supplied, so existing callers are unaffected."""
    if teacher_centroids is None:
        teacher_centroids = head_tail_centroids(logp_teacher, W, idx)
    if student_centroids is None:
        student_centroids = head_tail_centroids(logp_student, W, idx)
    cS, (cTq, mq) = teacher_centroids
    _, (cTp, mp) = student_centroids
    # mq/mp == 0 (or so close float32 rounds it to 0) means idx already
    # captured ~all of that distribution's mass -- no tail centroid exists,
    # not a value we can average in. See head_tail_centroids's comment.
    if mq <= 0 or mp <= 0:
        return dict(valid=False)
    n = np.linalg.norm(cS)
    if n == 0:
        return dict(valid=False)
    cs = cS / n
    proj_q = float((cS - cTq) @ cs)
    proj_p = float((cS - cTp) @ cs)
    return dict(valid=True, gamma=min(proj_q, proj_p), Gamma=max(proj_q, proj_p),
                proj_q=proj_q, proj_p=proj_p, m_q=mq, m_p=mp,
                lower_bound=mp * min(proj_q, proj_p) - mq * max(proj_q, proj_p))


def residual_identity(logp_student, logp_teacher, W, idx,
                      teacher_centroids=None, student_centroids=None):
    """Both sides of Equation (5). Their agreement is a correctness check.

    See separation_constants for teacher_centroids/student_centroids."""
    if teacher_centroids is None:
        teacher_centroids = head_tail_centroids(logp_teacher, W, idx)
    if student_centroids is None:
        student_centroids = head_tail_centroids(logp_student, W, idx)
    cS, (cTq, mq) = teacher_centroids
    _, (cTp, mp) = student_centroids
    p = np.exp(logp_student); q = np.exp(logp_teacher)
    direct = (p - q) @ W
    identity = mq * (cS - cTq) - mp * (cS - cTp)
    return dict(direct=direct, identity=identity,
                rel_mismatch=float(np.linalg.norm(direct - identity) /
                                   max(np.linalg.norm(direct), 1e-30)))
