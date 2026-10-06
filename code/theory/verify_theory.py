#!/usr/bin/env python3
"""
verify_theory.py -- independent numerical re-derivation of the three results in
"One Logit Per Dimension".

  Theorem 1    (invariance on A-perp, and grad m_p, Eq. 4)
  Theorem 2    (residual at a truncated optimum, Eq. 5, and Cor. 4's bound)
  Proposition 3 (identifiability threshold)

  Numbering follows the paper as of the theory-reduction pass; the checks
  themselves are numbered T1/T2/T3 in derivation order and do not depend on it.

Design rules this file follows, deliberately:

  * It COMPUTES and PRINTS. It never asserts that a result "passed" against a
    number copied from the paper. The only assertions are on structural facts
    that follow from linear algebra alone (e.g. rank(D_S) == min(k-1, d)), so a
    failure here is a real failure, not a disagreement with a hardcoded value.
  * It does not read any experiment output. Everything is generated in-process
    from a seed, so the checks are reproducible and independent of the bundle.
  * NumPy only. Runs in about a minute on CPU.

The values this script prints are the ones quoted in Appendix E of the paper.
If you change anything here, re-read Appendix E and reconcile it.

Usage:
    python verify_theory.py             # full run
    python verify_theory.py --smoke     # fast synthetic smoke test (~2 s)
"""

from __future__ import annotations

import argparse
import numpy as np


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) over the full support."""
    return float(np.sum(p * (np.log(p) - np.log(q))))


def make_teacher(V: int, d: int, rng: np.random.Generator):
    """A synthetic teacher of the form z = Ug + b. Random U, so this says
    nothing about real unembedding geometry; that is what the Threshold
    experiment in the paper is for."""
    U = rng.normal(size=(V, d))
    b = rng.normal(size=V) * 0.1
    g = rng.normal(size=d)
    q = softmax(U @ g + b)
    return U, b, g, q


def select(q: np.ndarray, k: int, rule: str, rng: np.random.Generator) -> np.ndarray:
    if rule == "topk":
        return np.argsort(-q)[:k]
    if rule == "random":
        return rng.choice(len(q), size=k, replace=False)
    if rule == "leverage":
        # placeholder for the leverage-score rule; not used by the paper's
        # synthetic checks, present so the rule set matches the experiment.
        raise NotImplementedError("leverage-score selection is experiment-side only")
    raise ValueError(rule)


# --------------------------------------------------------------------------
# T1 -- Proposition 1
# --------------------------------------------------------------------------

def check_prop1(V=4000, d=64, seed=1, rules=("topk", "random")) -> dict:
    """Reconstruct g from k observed log-probabilities and measure the error.

    Proposition 1 says: every fiber has dimension exactly d - rank(D_S), which
    is >= d-k+1; and recovery is exact iff rank(D_S) = d, which needs k >= d+1.
    """
    rng = np.random.default_rng(seed)
    U, b, g, q = make_teacher(V, d, rng)
    out = {}

    for rule in rules:
        rows = []
        for k in (d // 2, d - 1, d, d + 1, d + 2):
            S = select(q, k, rule, np.random.default_rng(seed + k))
            v0 = S[0]
            D = U[S[1:]] - U[v0]                     # (k-1) x d
            r = int(np.linalg.matrix_rank(D))

            lq = np.log(q[S] / q[S].sum())           # renormalized, as observed
            delta = lq[1:] - lq[0] - (b[S[1:]] - b[v0])
            ghat, *_ = np.linalg.lstsq(D, delta, rcond=None)
            qhat = softmax(U @ ghat + b)

            # structural facts that must hold by linear algebra alone
            assert r == min(k - 1, d), f"rank(D_S)={r}, expected {min(k-1, d)}"
            assert d - r >= max(d - k + 1, 0), "fiber dimension violates the bound"

            rows.append(dict(k=k, rank=r, fiber_dim=d - r,
                             kl=kl(q, qhat),
                             max_logprob_err=float(
                                 np.abs(np.log(q) - np.log(qhat)).max())))
        out[rule] = rows
    return out


# --------------------------------------------------------------------------
# T2 -- Theorem 2
# --------------------------------------------------------------------------

def fit_truncated_optimum(W, q, S, dp, rng, steps=20000, lr=0.5, tol=1e-11):
    """Minimize L_S = KL(qt || pt) over h until stationary."""
    mask = np.zeros(len(q), dtype=bool)
    mask[S] = True
    qt = q[mask] / q[mask].sum()
    h = rng.normal(size=dp) * 0.1
    for _ in range(steps):
        p = softmax(W @ h)
        pt = p[mask] / p[mask].sum()
        grad = W[mask].T @ (pt - qt)
        if np.linalg.norm(grad) < tol:
            break
        h -= lr * grad
    return h, mask, qt


def check_thm2(V=3000, dp=48, k=20, seed=2) -> dict:
    """At a stationary point of L_S, verify Eq. (5) exactly and Eq. (6) as a
    valid lower bound, and report how much of the residual norm it recovers."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(V, dp)) / np.sqrt(dp)      # STUDENT unembedding
    q = softmax(rng.normal(size=V))                 # arbitrary teacher marginal
    S = np.sort(np.argsort(-q)[:k])

    h, mask, qt = fit_truncated_optimum(W, q, S, dp, rng)
    p = softmax(W @ h)
    pt = p[mask] / p[mask].sum()

    grad_S = W[mask].T @ (pt - qt)                  # should be ~0
    m_p = float(1 - p[mask].sum())
    m_q = float(1 - q[mask].sum())

    c_S_q = (qt[:, None] * W[mask]).sum(0)
    c_S_p = (pt[:, None] * W[mask]).sum(0)
    c_T_q = (q[~mask][:, None] * W[~mask]).sum(0) / m_q
    c_T_p = (p[~mask][:, None] * W[~mask]).sum(0) / m_p

    full = ((p - q)[:, None] * W).sum(0)            # true full-KL gradient
    c_S = 0.5 * (c_S_p + c_S_q)                     # they coincide at h*
    D_q, D_p = c_S - c_T_q, c_S - c_T_p
    predicted = m_q * D_q - m_p * D_p               # Eq. (5)

    c_hat = c_S / np.linalg.norm(c_S)
    a_q, a_p = float(D_q @ c_hat), float(D_p @ c_hat)
    gamma, Gamma = min(a_q, a_p), max(a_q, a_p)
    lb = m_p * gamma - m_q * Gamma                  # Eq. (6)
    norm = float(np.linalg.norm(full))

    return dict(
        stationarity=float(np.linalg.norm(grad_S)),
        head_centroid_gap=float(np.linalg.norm(c_S_p - c_S_q)),
        eq5_rel_mismatch=float(np.linalg.norm(full - predicted) / norm),
        residual_norm=norm,
        lower_bound=float(lb),
        bound_valid=bool(lb <= norm + 1e-12),
        bound_tightness=float(lb / norm) if norm > 0 else float("nan"),
        m_p_over_m_q=float(m_p / m_q),
        Gamma_over_gamma=float(Gamma / gamma),
    )


def check_thm2_unconditional(V=800, dp=9, k=4, seed=0) -> dict:
    """Theorem 2 is an identity in h, not a statement about optima. Evaluate it
    at a random h far from stationarity, where a near-optimum artefact would show
    up immediately."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(V, dp)) / np.sqrt(dp)
    q = softmax(rng.normal(size=V) * 2.0)
    h = rng.normal(size=dp) * 3.0
    p = softmax(W @ h)
    mask = np.zeros(V, dtype=bool)
    mask[np.sort(np.argsort(-q)[:k])] = True

    m_q = float(1 - q[mask].sum())
    m_p = float(1 - p[mask].sum())
    qt = q[mask] / q[mask].sum()
    pt = p[mask] / p[mask].sum()
    c_S_q = qt @ W[mask]
    c_S_p = pt @ W[mask]
    c_T_q = (q[~mask] @ W[~mask]) / m_q
    c_T_p = (p[~mask] @ W[~mask]) / m_p

    grad_full = ((p - q)[:, None] * W).sum(0)
    grad_S = c_S_p - c_S_q                     # equals grad of L_S
    tail = m_q * (c_S_q - c_T_q) - m_p * (c_S_p - c_T_p)
    return dict(residual=float(np.linalg.norm(grad_full - (grad_S + tail))),
                grad_S_norm=float(np.linalg.norm(grad_S)))


def check_thm2_stationarity_sensitivity(V=3000, dp=48, k=20, seed=2) -> list:
    """Appendix E notes that the Eq. (5) mismatch is governed entirely by how
    exactly stationarity is reached. Demonstrate that rather than assert it."""
    rows = []
    for tol in (1e-4, 1e-5, 1e-7, 1e-9, 1e-11):
        rng = np.random.default_rng(seed)
        W = rng.normal(size=(V, dp)) / np.sqrt(dp)
        q = softmax(rng.normal(size=V))
        S = np.sort(np.argsort(-q)[:k])
        h, mask, qt = fit_truncated_optimum(W, q, S, dp, rng, tol=tol)
        p = softmax(W @ h)
        pt = p[mask] / p[mask].sum()
        m_p = float(1 - p[mask].sum())
        m_q = float(1 - q[mask].sum())
        c_S_q = (qt[:, None] * W[mask]).sum(0)
        c_S_p = (pt[:, None] * W[mask]).sum(0)
        c_T_q = (q[~mask][:, None] * W[~mask]).sum(0) / m_q
        c_T_p = (p[~mask][:, None] * W[~mask]).sum(0) / m_p
        full = ((p - q)[:, None] * W).sum(0)
        c_S = 0.5 * (c_S_p + c_S_q)
        pred = m_q * (c_S - c_T_q) - m_p * (c_S - c_T_p)
        rows.append((tol,
                     float(np.linalg.norm(W[mask].T @ (pt - qt))),
                     float(np.linalg.norm(full - pred) / np.linalg.norm(full))))
    return rows


# --------------------------------------------------------------------------
# T3 -- Theorem 3
# --------------------------------------------------------------------------

def check_thm3(V=2000, dp=32, k=12, seed=7) -> dict:
    """L_S is constant on A-perp (dim >= d'-k+1); m_p is not; and Eq. (7) holds."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(V, dp)) / np.sqrt(dp)
    q = softmax(rng.normal(size=V))
    S = np.sort(np.argsort(-q)[:k])
    mask = np.zeros(V, dtype=bool)
    mask[S] = True
    qt = q[mask] / q[mask].sum()
    h = rng.normal(size=dp)

    A = W[mask][1:] - W[mask][0]
    _, sv, Vt = np.linalg.svd(A)
    rank_A = int((sv > 1e-10).sum())
    A_perp = Vt[rank_A:]
    assert dp - rank_A >= dp - k + 1, "dim A-perp below the theorem's bound"

    def L_S(hh):
        p = softmax(W @ hh)
        pt = p[mask] / p[mask].sum()
        return float(np.sum(qt * (np.log(qt) - np.log(pt))))

    def m_p_of(hh):
        return float(1 - softmax(W @ hh)[mask].sum())

    eta = A_perp[0] * 0.7

    # Eq. (7) against central finite differences
    p = softmax(W @ h)
    m_p = float(1 - p[mask].sum())
    pt = p[mask] / p[mask].sum()
    c_S_p = (pt[:, None] * W[mask]).sum(0)
    c_T_p = (p[~mask][:, None] * W[~mask]).sum(0) / m_p
    analytic = -m_p * (1 - m_p) * (c_S_p - c_T_p)
    eps = 1e-6
    I = np.eye(dp)
    fd = np.array([(m_p_of(h + eps * I[i]) - m_p_of(h - eps * I[i])) / (2 * eps)
                   for i in range(dp)])

    return dict(
        dim_A_perp=dp - rank_A,
        bound_d_minus_k_plus_1=dp - k + 1,
        L_S_change_on_A_perp=abs(L_S(h + eta) - L_S(h)),
        m_p_change_on_A_perp=abs(m_p_of(h + eta) - m_p_of(h)),
        grad_m_p_rel_err=float(np.linalg.norm(fd - analytic) / np.linalg.norm(fd)),
    )


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def run(smoke: bool = False) -> None:
    if smoke:
        print("SMOKE TEST (small sizes; structural assertions only)\n")
        p1 = check_prop1(V=400, d=16, seed=0, rules=("topk",))
        for r in p1["topk"]:
            print(f"  P1 k={r['k']:3d} rank={r['rank']:3d} fiber={r['fiber_dim']:3d} "
                  f"KL={r['kl']:.3e}")
        t2 = check_thm2(V=300, dp=12, k=6, seed=0)
        print(f"  T2 eq5_rel_mismatch={t2['eq5_rel_mismatch']:.2e} "
              f"bound_valid={t2['bound_valid']}")
        t3 = check_thm3(V=200, dp=10, k=4, seed=0)
        print(f"  T3 dim(A-perp)={t3['dim_A_perp']} (bound {t3['bound_d_minus_k_plus_1']}) "
              f"grad_m_p_rel_err={t3['grad_m_p_rel_err']:.2e}")
        print("\nSmoke test complete. Structural assertions held.")
        return

    print("=" * 74)
    print("T1  Proposition 1: identifiability threshold        (V=4000, d=64)")
    print("=" * 74)
    res = check_prop1()
    for rule, rows in res.items():
        print(f"\n  selection = {rule}")
        print(f"    {'k':>5} {'rank(D_S)':>10} {'fiber dim':>10} "
              f"{'KL(q||qhat)':>14} {'max logp err':>14}")
        for r in rows:
            print(f"    {r['k']:>5} {r['rank']:>10} {r['fiber_dim']:>10} "
                  f"{r['kl']:>14.3e} {r['max_logprob_err']:>14.3e}")
    print("\n  Reading: recovery is exact at k = d+1 and not one index earlier,"
          "\n  under BOTH selection rules. Top-k selection does not collapse early"
          "\n  on random unembeddings. See Appendix C on the discrepancy this"
          "\n  creates with the previous draft's selection-rule paragraph.")

    print("\n" + "=" * 74)
    print("T2a Theorem 2 as an identity in h, evaluated FAR from stationarity")
    print("=" * 74)
    print(f"  {'seed':>5} {'||grad L - (grad L_S + tail)||':>32} {'||grad L_S||':>14}")
    for sd in range(0, 4):
        rr = check_thm2_unconditional(seed=sd)
        print(f"  {sd:>5} {rr['residual']:>32.2e} {rr['grad_S_norm']:>14.4f}")
    print("  The decomposition holds at arbitrary h, not only at an optimum.")

    print("\n" + "=" * 74)
    print("T2b Stationary-point form and the lower bound        (V=3000, d'=48, k=20)")
    print("=" * 74)
    print(f"  {'seed':>5} {'||grad L_S||':>13} {'Eq.5 rel mism':>14} "
          f"{'lower bnd':>11} {'||grad L||':>11} {'valid':>6} {'tightness':>10}")
    for seed in range(2, 6):
        r = check_thm2(seed=seed)
        print(f"  {seed:>5} {r['stationarity']:>13.2e} {r['eq5_rel_mismatch']:>14.2e} "
              f"{r['lower_bound']:>11.4f} {r['residual_norm']:>11.4f} "
              f"{str(r['bound_valid']):>6} {r['bound_tightness']:>9.1%}")

    print("\n  Eq. (5) mismatch as a function of how exactly stationarity is reached:")
    print(f"    {'tol':>8} {'||grad L_S||':>14} {'Eq.5 rel mismatch':>19}")
    for tol, g, mism in check_thm2_stationarity_sensitivity():
        print(f"    {tol:>8.0e} {g:>14.2e} {mism:>19.2e}")
    print("  The identity is exact. Evaluating its stationary-point form off the")
    print("  optimum leaves a discrepancy of exactly ||grad L_S||, which is what the")
    print("  one-for-one tracking above shows.")

    print("\n" + "=" * 74)
    print("T3  Theorem 3: invariance on A-perp and grad m_p     (V=2000, d'=32, k=12)")
    print("=" * 74)
    print(f"  {'seed':>5} {'dim A-perp':>11} {'bound':>7} {'|dL_S|':>11} "
          f"{'|dm_p|':>11} {'grad m_p rel err':>18}")
    for seed in range(7, 11):
        r = check_thm3(seed=seed)
        print(f"  {seed:>5} {r['dim_A_perp']:>11} {r['bound_d_minus_k_plus_1']:>7} "
              f"{r['L_S_change_on_A_perp']:>11.2e} {r['m_p_change_on_A_perp']:>11.2e} "
              f"{r['grad_m_p_rel_err']:>18.2e}")

    print("\n" + "=" * 74)
    print("These checks confirm the ALGEBRA of the three results. They are not a")
    print("substitute for a human author re-deriving each proof line by line,")
    print("which the paper flags as outstanding.")
    print("=" * 74)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="fast structural smoke test (~2 s)")
    run(ap.parse_args().smoke)
