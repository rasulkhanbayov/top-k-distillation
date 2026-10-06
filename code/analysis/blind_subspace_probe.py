"""Prototype of the missing direct test of Theorem 1.

Theorem 1 says two things about the SAME set of directions A-perp:
  (i)  grad L_S is exactly zero there   (the objective is blind)
  (ii) grad m_p is NOT zero there       (but tail mass moves)
Both are one forward/backward pass on any checkpoint. Neither is currently
measured; the paper tests only the downstream drift, which many mechanisms
could produce. This script establishes that the probe is well-posed and
reports what it costs.
"""
import numpy as np, time

def softmax(z):
    z = np.asarray(z, float); z = z - z.max(); e = np.exp(z); return e/e.sum()

def probe(V, dp, k, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(V, dp))/np.sqrt(dp)
    q = softmax(rng.normal(size=V)); h = rng.normal(size=dp)
    S = np.sort(np.argsort(-q)[:k]); M = np.zeros(V, bool); M[S] = True
    qt = q[M]/q[M].sum()
    p = softmax(W@h); pt = p[M]/p[M].sum()

    # orthonormal basis of A = span{w_v - w_v0 : v in S};  A-perp is its complement
    Amat = W[M][1:] - W[M][0]
    _, sv, Vt = np.linalg.svd(Amat, full_matrices=True)
    r = int((sv > 1e-10*max(1.0, sv[0])).sum())
    Aperp = Vt[r:]                                   # rows span A-perp

    g_LS = W[M].T @ (pt - qt)                        # grad of L_S
    m_p  = 1 - p[M].sum()
    cSp  = pt @ W[M]; cTp = (p[~M] @ W[~M])/m_p
    g_mp = -m_p*(1-m_p)*(cSp - cTp)                  # grad of m_p, Eq. (4)

    proj = lambda g: np.linalg.norm(Aperp @ g)
    return dict(dimAperp=Vt.shape[0]-r,
                ratio_LS = proj(g_LS)/max(np.linalg.norm(g_LS), 1e-300),
                ratio_mp = proj(g_mp)/max(np.linalg.norm(g_mp), 1e-300))

print(f"{'V':>6} {'dp':>5} {'k':>5} {'dim A-perp':>11} "
      f"{'||P grad L_S||/||grad L_S||':>28} {'||P grad m_p||/||grad m_p||':>28}")
for V,dp,k in [(2000,64,8),(2000,64,64),(20000,512,64),(20000,512,256),(20000,512,512)]:
    R=[probe(V,dp,k,s) for s in range(3)]
    m=lambda key: float(np.mean([x[key] for x in R]))
    print(f"{V:>6} {dp:>5} {k:>5} {m('dimAperp'):>11.0f} "
          f"{m('ratio_LS'):>28.2e} {m('ratio_mp'):>28.4f}")

t0=time.time(); probe(50000, 2048, 256); dt=time.time()-t0
print(f"\ncost at V=50k, d'=2048, k=256 (single position, CPU, dense SVD): {dt:.2f}s")
print("On GPU with the student's own W this is one backward pass plus a")
print("(k-1) x d' SVD, i.e. negligible next to a training step. No training needed.")
print("\nPREDICTION the probe would test:")
print("  ratio_LS  -> 0 at float precision   (the objective is blind)")
print("  ratio_mp  -> bounded away from 0    (tail mass is not)")
print("  and ratio_mp -> 0 as k -> V         (blindness disappears; negative control)")
