"""
Reconstruction of a teacher's full next-token distribution from a cached hidden
state, and the identifiability diagnostics the paper's Proposition 1 depends on.

Everything here is NumPy and runs without a GPU. The heavy training code lives in
hsc/divergences.py and hsc/distill.py.

Paper reference: Section 3 (setup), Proposition 1, Section 5 (method).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "reconstruct_logits",
    "solve_hidden_state",
    "solve_hidden_state_softcap",
    "solve_hidden_state_underdetermined",
    "identifiability_report",
    "one_in_range_margin",
    "apply_softcap",
    "invert_softcap",
]


# --------------------------------------------------------------------------- #
# Forward direction: hidden state -> logits
# --------------------------------------------------------------------------- #
def reconstruct_logits(U, g, b=None, softcap=None):
    """z = U g + b, optionally soft-capped.

    Parameters
    ----------
    U : (V, d) unembedding
    g : (d,) or (n, d) hidden state(s), taken AFTER the final norm
    b : (V,) optional output bias
    softcap : float or None. Gemma-style c * tanh(z / c).

    Returns
    -------
    (V,) or (n, V) logits.
    """
    z = g @ U.T
    if b is not None:
        z = z + b
    if softcap is not None:
        z = apply_softcap(z, softcap)
    return z


def apply_softcap(z, c):
    return c * np.tanh(z / c)


def invert_softcap(z_capped, c):
    """Soft-capping is monotone and elementwise, so it inverts exactly.

    Needed when a logging pipeline emits post-cap logits but the reconstruction
    must operate on pre-cap ones (paper Section 5.4).
    """
    r = np.clip(z_capped / c, -1 + 1e-7, 1 - 1e-7)
    return c * np.arctanh(r)


# --------------------------------------------------------------------------- #
# Inverse direction: observed log-probabilities -> hidden state
# --------------------------------------------------------------------------- #
def solve_hidden_state(U, logprobs, indices, b=None, rcond=None, return_diag=False):
    """Recover g from k observed log-probabilities (Proposition 1(ii)).

    Renormalization cancels log Z, so for any v, v0 in the observation set

        log q_v - log q_v0 = (u_v - u_v0)^T g + (b_v - b_v0),

    giving k-1 independent linear constraints on d unknowns. Requires
    rank(D_S) = d, hence k >= d + 1.

    Parameters
    ----------
    U : (V, d)
    logprobs : (k,) log-probabilities at `indices`. May be renormalized over the
        observation set or not; the difference cancels.
    indices : (k,) int
    b : (V,) optional bias
    rcond : cutoff passed to lstsq

    Returns
    -------
    g_hat : (d,), and optionally a diagnostics dict.
    """
    idx = np.asarray(indices)
    k = len(idx)
    d = U.shape[1]
    if k < d + 1:
        raise ValueError(
            f"k={k} observations cannot identify a d={d} hidden state; "
            f"Proposition 1 requires k >= d+1. Reconstruct anyway only if you "
            f"intend to measure the deficit."
        )
    v0, rest = idx[0], idx[1:]
    D = U[rest] - U[v0]                       # (k-1, d)
    delta = logprobs[1:] - logprobs[0]
    if b is not None:
        delta = delta - (b[rest] - b[v0])
    g_hat, residuals, rank, sv = np.linalg.lstsq(D, delta, rcond=rcond)
    if not return_diag:
        return g_hat
    return g_hat, {
        "rank": int(rank),
        "d": int(d),
        "full_rank": bool(rank == d),
        "sigma_min": float(sv[-1]) if len(sv) else 0.0,
        "sigma_max": float(sv[0]) if len(sv) else 0.0,
        "cond": float(sv[0] / sv[-1]) if len(sv) and sv[-1] > 0 else np.inf,
        "residual": float(residuals[0]) if len(residuals) else 0.0,
    }


def solve_hidden_state_softcap(U, logprobs, indices, c, b=None):
    """Recover g from k observed log-probabilities when the teacher applies
    Gemma-style soft-capping z_capped = c * tanh(z / c) before the softmax.

    solve_hidden_state's linear identity log q_v - log q_v0 = (u_v-u_v0)^T g +
    (b_v-b_v0) assumes q = softmax(Ug+b) with NO nonlinearity in between; under
    soft-capping it is log q_v - log q_v0 = z_capped,v - z_capped,v0, which is
    linear in z_capped but not in g, since tanh is elementwise-nonlinear and
    the differencing happens on its output. Recovering z_capped absolutely
    (not just its differences) additionally requires knowing s = z_capped[v0],
    a single extra scalar (equivalent to knowing the log-partition function) --
    the observed log-probabilities alone only fix z_capped up to that one
    additive unknown.

    Naively treating this as a joint (1+d)-dimensional nonlinear least-squares
    problem over (s, g) is numerically unreliable: scipy.optimize.least_squares
    converges to bad local minima unpredictably, including well above k=d+1
    where the problem should be easy (verified empirically: ~40-66% of KL mass
    wrong on some seeds even at k=2d, regardless of multi-restart or
    warm-starting from the linear solution). The fix is to profile out g
    analytically: for any FIXED s, z_capped is fully determined, z = c *
    arctanh(z_capped / c) is determined, and g is then the ordinary LINEAR
    least-squares solution of U_obs @ g = z. So only the single scalar s needs
    a nonlinear search; g is recovered in closed form for each candidate s.
    This reduces an unreliable (1+d)-dimensional search to a robust bounded 1-D
    one and gives machine-precision recovery in testing (from toy scale up to
    d=2048, V=20000, matching Gemma-3-4B's real dimensions).

    Requires k >= d + 1 for the same rank reason as solve_hidden_state.

    Parameters
    ----------
    U : (V, d)
    logprobs : (k,) log-probabilities at `indices`
    indices : (k,) int
    c : softcap constant
    b : (V,) optional bias

    Returns
    -------
    g_hat : (d,), cost : float (final sum-of-squares residual; near zero means
        a clean recovery, large means either k is too close to d+1 for this
        realization or the search bounds were degenerate -- treat a cost above
        ~1e-6 as suspect and worth checking rather than trusting blindly).
    """
    from scipy.optimize import minimize_scalar

    idx = np.asarray(indices)
    k = len(idx)
    d = U.shape[1]
    if k < d + 1:
        raise ValueError(
            f"k={k} observations cannot identify a d={d} hidden state under "
            f"soft-capping; requires k >= d+1 for the same rank reason as "
            f"solve_hidden_state."
        )
    U_obs = U[idx]
    b_obs = b[idx] if b is not None else 0.0
    lp_rel = logprobs - logprobs[0]           # zc_i - zc_0, exact and known

    # s = zc_0 must keep every zc = s + lp_rel strictly inside (-c, c).
    lo = max(-c - lp_rel.min(), -c) + 1e-6
    hi = min(c - lp_rel.max(), c) - 1e-6

    # U_obs is the SAME for every candidate s the search tries -- only the
    # target z(s) changes. Factor it ONCE via QR and reuse for every trial
    # instead of calling the SVD-based np.linalg.lstsq(U_obs, ...) inside
    # cost(): at Gemma-3-12B's scale (d=3840, k=3841) a single lstsq call
    # alone takes ~9-10s, and the bounded scalar search evaluates cost()
    # roughly 10-20 times, so the un-cached version made the whole E2 sweep
    # (hundreds of these calls per teacher) infeasible within any reasonable
    # walltime -- confirmed: it did not return within 100s even for a single
    # call. QR + cached triangular solve turns each trial into a fast
    # solve_triangular against a new right-hand side, roughly 5x faster
    # end-to-end and correctness-equivalent (same normal-equations solution,
    # verified against the lstsq answer to ~1e-13 relative error).
    Q, R = np.linalg.qr(U_obs)
    QT = Q.T

    def solve_g(z):
        return np.linalg.solve(R, QT @ z)

    def cost(s):
        zc = np.clip(s + lp_rel, -c + 1e-9, c - 1e-9)
        z = c * np.arctanh(zc / c)
        g_hat = solve_g(z - b_obs)
        resid = U_obs @ g_hat + b_obs - z
        return float(np.sum(resid ** 2))

    res = minimize_scalar(cost, bounds=(lo, hi), method="bounded",
                           options={"xatol": 1e-13, "maxiter": 500})
    zc = np.clip(res.x + lp_rel, -c + 1e-9, c - 1e-9)
    z = c * np.arctanh(zc / c)
    g_hat = solve_g(z - b_obs)
    return g_hat, res.fun


def solve_hidden_state_underdetermined(U, logprobs, indices, b=None):
    """Minimum-norm solution when k < d + 1.

    Not a recovery: by Proposition 1(i) the fiber is an affine subspace of
    dimension at least d - k + 1 and this returns one point on it. Provided so
    that E2 can measure the deficit rather than refusing to run.
    """
    idx = np.asarray(indices)
    v0, rest = idx[0], idx[1:]
    D = U[rest] - U[v0]
    delta = logprobs[1:] - logprobs[0]
    if b is not None:
        delta = delta - (b[rest] - b[v0])
    g_hat, *_ = np.linalg.lstsq(D, delta, rcond=None)
    return g_hat


# --------------------------------------------------------------------------- #
# Standing assumptions of Section 3
# --------------------------------------------------------------------------- #
def one_in_range_margin(U, n_probe=None, seed=0):
    """How close is span(1) to range(U)?

    Section 3 assumes 1 is NOT in range(U); otherwise softmax shift invariance
    makes g unrecoverable (though q is still determined). Representation
    degeneration pushes toward violation, so this is worth reporting per teacher.

    Returns the relative residual of projecting the all-ones vector onto
    range(U): 1.0 means orthogonal to range(U), 0.0 means contained in it.
    """
    V, d = U.shape
    rng = np.random.default_rng(seed)
    if n_probe is not None and n_probe < V:
        rows = rng.choice(V, n_probe, replace=False)
        A = U[rows]
    else:
        A = U
    ones = np.ones(A.shape[0])
    # least-squares projection of `ones` onto range(A)
    coef, *_ = np.linalg.lstsq(A, ones, rcond=None)
    resid = ones - A @ coef
    return float(np.linalg.norm(resid) / np.linalg.norm(ones))


def identifiability_report(U, b=None, n_probe=20000, seed=0):
    """Everything E2 needs to check the standing assumptions on a real teacher."""
    V, d = U.shape
    rng = np.random.default_rng(seed)
    rows = rng.choice(V, min(n_probe, V), replace=False)
    A = U[rows]
    sv = np.linalg.svd(A, compute_uv=False)
    return {
        "V": int(V),
        "d": int(d),
        "rank_estimate": int((sv > sv[0] * 1e-10).sum()),
        "full_column_rank": bool((sv > sv[0] * 1e-10).sum() == d),
        "sigma_max": float(sv[0]),
        "sigma_min": float(sv[-1]),
        "cond": float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf,
        "ones_residual": one_in_range_margin(U, n_probe=n_probe, seed=seed),
        "row_norm_mean": float(np.linalg.norm(U, axis=1).mean()),
        "anisotropy": float(
            np.linalg.norm(U.mean(0)) / np.linalg.norm(U, axis=1).mean()
        ),
    }
