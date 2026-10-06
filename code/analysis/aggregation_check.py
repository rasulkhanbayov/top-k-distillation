"""ADVERSARIAL CHECK.

Theorem 1 gives a blind subspace A_i^perp at EACH position i. Training optimises
shared parameters, so what matters is the intersection over positions,
  cap_i A_i^perp  =  ( sum_i A_i )^perp.
If that intersection is trivial, the flat-direction story does not survive
aggregation and the paper's training-time reading is wrong.

Theorem 2 is a per-position identity that sums, so it should survive. Check both.
"""
import numpy as np
def softmax(z):
    z=np.asarray(z,float); z=z-z.max(); e=np.exp(z); return e/e.sum()

def agg_blind_dim(V, dp, k, N, seed=0):
    """dim of the intersection of N per-position blind subspaces."""
    rng=np.random.default_rng(seed)
    W=rng.normal(size=(V,dp))/np.sqrt(dp)
    rows=[]
    for i in range(N):
        q=softmax(rng.normal(size=V))
        S=np.sort(np.argsort(-q)[:k]); v0=S[0]
        rows.append(W[S[1:]]-W[v0])                 # generators of A_i
    M=np.vstack(rows)
    return dp-int(np.linalg.matrix_rank(M))

print("Intersection of per-position blind subspaces, V=20000, d'=512, k=64")
print(f"  {'positions N':>12} {'dim cap A_i^perp':>18} {'N(k-1)':>9}")
for N in [1,2,4,8,9,16,64,1000]:
    print(f"  {N:>12} {agg_blind_dim(20000,512,64,N):>18} {N*63:>9}")
print("  -> the intersection collapses to {0} once N(k-1) >= d', i.e. after")
print("     about d'/(k-1) ~ 8 positions. At k=64, d'=2048 that is ~33 positions.")
print("     Millions of positions are trained on. THE FLAT SUBSPACE DOES NOT")
print("     SURVIVE AGGREGATION OVER POSITIONS.")

print("\nDoes Theorem 2's decomposition survive aggregation? (it is a sum of")
print("per-position identities, so it should)")
rng=np.random.default_rng(3)
V,dp,k,N=3000,16,5,40
W=rng.normal(size=(V,dp))/np.sqrt(dp)
tot_full=np.zeros(dp); tot_tr=np.zeros(dp); tot_tail=np.zeros(dp)
for i in range(N):
    q=softmax(rng.normal(size=V)*1.5); h=rng.normal(size=dp)*2
    p=softmax(W@h); S=np.sort(np.argsort(-q)[:k]); M=np.zeros(V,bool); M[S]=True
    mq=1-q[M].sum(); mp=1-p[M].sum()
    qt=q[M]/q[M].sum(); pt=p[M]/p[M].sum()
    cSq=qt@W[M]; cSp=pt@W[M]
    cTq=(q[~M]@W[~M])/mq; cTp=(p[~M]@W[~M])/mp
    tot_full += ((p-q)[:,None]*W).sum(0)
    tot_tr   += cSp-cSq
    tot_tail += mq*(cSq-cTq)-mp*(cSp-cTp)
print(f"  || sum_i gradL_i  -  (sum_i gradL_S,i + sum_i tail_i) || = "
      f"{np.linalg.norm(tot_full-(tot_tr+tot_tail)):.3e}")
print("  -> Theorem 2 aggregates exactly. Theorem 1's flat subspace does not.")
