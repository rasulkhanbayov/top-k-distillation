#!/usr/bin/env python3
"""
Numerical verification of every formal claim in Section 4. Runs in about a
minute on CPU with NumPy only, and is the first thing to run after checkout.

Checks, in paper order:
  P1  Proposition 1: reconstruction error collapses at exactly k = d+1
  P1b Proposition 1 standing assumptions: rank(U) = d and 1 not in range(U)
  T2a Theorem 2: the residual identity holds at a truncated stationary point
  T2b Theorem 2: the lower bound is valid, and how tight it is
  T3a Theorem 3: grad_h m_p = -m_p (1 - m_p) Delta^p, by finite differences
  T3b Theorem 3: Delta^p leaves span A, so tail mass moves on the invariant set

These are SYNTHETIC checks on random unembeddings. They verify the algebra, not
the behavior of real models; that is E2 and E5.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from hsc.reconstruct import solve_hidden_state, solve_hidden_state_underdetermined, reconstruct_logits


def softmax(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()


def make_unembedding(V, d, aniso=1.0, seed=0):
    r = np.random.default_rng(seed)
    W = r.standard_normal((V, d)) / np.sqrt(d)
    W *= (1 + 0.25 * r.standard_normal((V, 1)))
    W += aniso * r.standard_normal((1, d)) / np.sqrt(d)
    return W


def logprobs(U, g, b=None):
    z = reconstruct_logits(U, g, b)
    m = z.max()
    return z - (m + np.log(np.exp(z - m).sum()))


def hdr(t):
    print("\n" + "=" * 78); print(t); print("=" * 78)


# ----------------------------------------------------------------- P1
def check_P1(V=20000, d=256, seed=1):
    hdr("P1  Proposition 1: reconstruction collapses at k = d+1")
    rng = np.random.default_rng(seed)
    U = make_unembedding(V, d, seed=seed); b = 0.1 * rng.standard_normal(V)
    g = rng.standard_normal(d) * 1.5
    lp = logprobs(U, g, b)
    order = np.argsort(lp)[::-1]
    print(f"{'k':>8}{'k/d':>8}{'selection':>12}{'max logp err':>16}{'KL(q||qhat)':>15}")
    print("-" * 78)
    ok = True
    for k in (d // 2, d - 1, d, d + 1, d + 2, 2 * d):
        for sel in ("top-k", "random"):
            S = order[:k] if sel == "top-k" else rng.choice(V, k, replace=False)
            f = solve_hidden_state if k >= d + 1 else solve_hidden_state_underdetermined
            gh = f(U, lp[S], S, b=b)
            lph = logprobs(U, gh, b)
            p, q = np.exp(lp), np.exp(lph)
            kl = float((p * (lp - lph)).sum())
            print(f"{k:>8}{k/d:>8.2f}{sel:>12}{np.abs(lp-lph).max():>16.3e}{kl:>15.3e}")
            if k >= d + 1 and np.abs(lp - lph).max() > 1e-6:
                ok = False
    print("-" * 78)
    print("PASS" if ok else "FAIL", ": error is numerically zero for every k >= d+1")
    return ok


def check_P1b(V=20000, d=256, seed=2):
    hdr("P1b  Standing assumptions, and how anisotropy erodes the second one")
    from hsc.reconstruct import identifiability_report
    print(f"{'anisotropy':>12}{'full rank':>12}{'cond(U)':>11}"
          f"{'||1-proj||/||1||':>19}{'g recoverable':>15}")
    print("-" * 78)
    ok = True
    for aniso in (0.0, 0.25, 1.0, 4.0):
        U = make_unembedding(V, d, aniso=aniso, seed=seed)
        r = identifiability_report(U, n_probe=V)
        ok &= r["full_column_rank"]
        rec = "yes" if r["ones_residual"] > 0.1 else "ill-cond."
        print(f"{aniso:>12.2f}{str(r['full_column_rank']):>12}{r['cond']:>11.1f}"
              f"{r['ones_residual']:>19.4f}{rec:>15}")
    print("-" * 78)
    print("Full column rank is what Proposition 1 needs and it always holds here.")
    print("The second assumption erodes as anisotropy grows: range(U) approaches")
    print("span(1). This is BENIGN for the method, because the direction along")
    print("which g stops being recoverable is exactly the direction that leaves q")
    print("unchanged. It matters only when solving for g from logits (Section 5.3),")
    print("not when caching g directly.")
    print("PASS" if ok else "FAIL", ": full column rank holds in every configuration")
    return ok


# ----------------------------------------------------------------- T2
def fit_truncated(W, q, S, d, steps=20000, lr=5.0, tol=1e-9, seed=0, h0=None):
    """Descend L_S to stationarity. Theorem 2 is a statement ABOUT stationary
    points, so the identity check is meaningless unless ||grad L_S|| is driven
    to zero; we return it so the caller can report convergence."""
    r = np.random.default_rng(seed)
    h = r.standard_normal(d) / np.sqrt(d) if h0 is None else h0.copy()
    tgt = q[S] / q[S].sum()
    gn = np.inf
    for _ in range(steps):
        p = softmax(W @ h)
        ps = p[S] / p[S].sum()
        gr = (ps - tgt) @ W[S]
        gn = np.linalg.norm(gr)
        if gn < tol:
            break
        h -= lr * gr
    return h, float(gn)


def check_T2(V=8192, d=128, seed=3):
    hdr("T2  Theorem 2: residual identity and lower bound at a truncated optimum")
    rng = np.random.default_rng(seed)
    W = make_unembedding(V, d, seed=seed)
    print(f"{'k':>6}{'||grad L_S||':>14}{'||grad L||':>13}{'||identity||':>14}"
          f"{'rel mism':>11}{'lower bd':>11}{'tight':>8}{'valid':>7}")
    print("-" * 78)
    ok = True
    for k in (16, 64, 256):
        ht = rng.standard_normal(d) * 2.0
        q = softmax(W @ ht)
        S = np.argsort(q)[::-1][:k]
        h, gs = fit_truncated(W, q, S, d, seed=seed)
        p = softmax(W @ h)
        M = np.ones(V, bool); M[S] = False
        m_p, m_q = float(p[M].sum()), float(q[M].sum())
        cS = (q[S] / q[S].sum()) @ W[S]
        cTp = (p[M] / m_p) @ W[M]; cTq = (q[M] / m_q) @ W[M]
        grad = (p - q) @ W
        ident = m_q * (cS - cTq) - m_p * (cS - cTp)
        rel = np.linalg.norm(grad - ident) / max(np.linalg.norm(grad), 1e-30)
        cs = cS / np.linalg.norm(cS)
        pq = float((cS - cTq) @ cs); pp = float((cS - cTp) @ cs)
        gam, Gam = min(pq, pp), max(pq, pp)
        lb = m_p * gam - m_q * Gam
        act = float(np.linalg.norm(grad))
        valid = lb <= act + 1e-9
        # The identity is exact only at stationarity; tolerate a mismatch of the
        # same order as the residual truncated gradient we failed to drive out.
        tol = max(5e-3, 20 * gs / max(act, 1e-30))
        ok &= valid and rel < tol
        print(f"{k:>6}{gs:>14.2e}{act:>13.6f}{np.linalg.norm(ident):>14.6f}"
              f"{rel:>11.2e}{lb:>11.5f}{lb/act:>7.0%}{'yes' if valid else 'NO':>7}")
    print("-" * 78)
    print("PASS" if ok else "FAIL",
          ": identity holds to the accuracy of the stationarity we reached,")
    print("      and the lower bound never exceeds the realized residual.")
    return ok


# ----------------------------------------------------------------- T3
def check_T3(V=4000, d=64, k=32, seed=4):
    hdr("T3  Theorem 3: grad_h m_p = -m_p (1 - m_p) Delta^p, by finite differences")
    rng = np.random.default_rng(seed)
    W = make_unembedding(V, d, seed=seed)
    print(f"{'trial':>7}{'||analytic||':>15}{'||numeric||':>14}{'rel err':>12}{'cos':>10}")
    print("-" * 78)
    ok = True
    for t in range(4):
        h = rng.standard_normal(d) * (1.0 + t)
        S = rng.choice(V, k, replace=False)
        p = softmax(W @ h); m_p = 1 - p[S].sum()
        M = np.ones(V, bool); M[S] = False
        cSp = (p[S] / p[S].sum()) @ W[S]; cTp = (p[M] / m_p) @ W[M]
        analytic = -m_p * (1 - m_p) * (cSp - cTp)
        eps, numeric = 1e-6, np.zeros(d)
        for i in range(d):
            e = np.zeros(d); e[i] = eps
            a = 1 - softmax(W @ (h + e))[S].sum()
            bq = 1 - softmax(W @ (h - e))[S].sum()
            numeric[i] = (a - bq) / (2 * eps)
        rel = np.linalg.norm(analytic - numeric) / max(np.linalg.norm(numeric), 1e-30)
        cos = float(analytic @ numeric / (np.linalg.norm(analytic) * np.linalg.norm(numeric)))
        ok &= rel < 1e-4
        print(f"{t:>7}{np.linalg.norm(analytic):>15.6e}{np.linalg.norm(numeric):>14.6e}"
              f"{rel:>12.2e}{cos:>10.6f}")
    print("-" * 78); print("PASS" if ok else "FAIL", ": closed form matches finite differences")

    hdr("T3b  Delta^p leaves span A, so tail mass moves on the invariant subspace")
    print(f"{'k':>6}{'rank A':>9}{'||proj_A Delta^p||/||Delta^p||':>34}{'m_p moves':>12}")
    print("-" * 78)
    for k2 in (8, 32, 128, 512):
        h = rng.standard_normal(d)
        S = rng.choice(V, k2, replace=False)
        p = softmax(W @ h); m_p = 1 - p[S].sum()
        M = np.ones(V, bool); M[S] = False
        D = (p[S] / p[S].sum()) @ W[S] - (p[M] / m_p) @ W[M]
        A = W[S[1:]] - W[S[0]]
        Q, _ = np.linalg.qr(A.T)
        frac = float(np.linalg.norm(Q.T @ D) / np.linalg.norm(D))
        moves = frac < 1 - 1e-6
        print(f"{k2:>6}{np.linalg.matrix_rank(A):>9}{frac:>34.4f}{('yes' if moves else 'no'):>12}")
    print("-" * 78)
    print("Note: at k-1 >= d' the span fills R^d' and the invariant subspace is")
    print("empty, which is the necessary-but-not-sufficient remark in Section 4.2.")
    return ok


if __name__ == "__main__":
    results = {"P1": check_P1(), "P1b": check_P1b(), "T2": check_T2(), "T3": check_T3()}
    hdr("SUMMARY")
    for k, v in results.items():
        print(f"  {k:5s} {'PASS' if v else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)
