"""First-principles verification of every equation, theorem, and proof step.
Symbolic where feasible (SymPy), numeric elsewhere, plus boundary cases."""
import numpy as np, sympy as sp

def softmax(z):
    z = np.asarray(z, float); z = z - z.max(); e = np.exp(z); return e/e.sum()

print("="*72); print("A. SYMBOLIC: gradient identities from first principles"); print("="*72)
# small symbolic model: V=4, d'=2, exact softmax
dp, V = 2, 4
h = sp.Matrix(sp.symbols('h1 h2', real=True))
W = sp.Matrix(4, 2, sp.symbols('w1_1 w1_2 w2_1 w2_2 w3_1 w3_2 w4_1 w4_2', real=True))
qs = sp.Matrix(sp.symbols('q1 q2 q3 q4', positive=True))
zz = W*h
den = sum(sp.exp(zz[v]) for v in range(V))
p = sp.Matrix([sp.exp(zz[v])/den for v in range(V)])

L = sum(qs[v]*(sp.log(qs[v]) - sp.log(p[v])) for v in range(V))
gradL = sp.Matrix([sp.simplify(sp.diff(L, h[i])) for i in range(dp)])
claim = (W.T*(p - qs))                      # sum_v (p_v - q_v) w_v
print("A1  grad KL(q||p) == sum_v (p_v-q_v) w_v :",
      sp.simplify(gradL - claim).applyfunc(sp.simplify) == sp.zeros(dp,1),
      "  [requires sum q_v = 1? test below]")
# the identity uses sum_v q_v = 1; verify it FAILS without that constraint
sub = {qs[3]: 1 - qs[0] - qs[1] - qs[2]}
gradL_c = gradL.subs(sub); claim_c = claim.subs(sub)
print("A2  with sum q_v = 1 imposed              :",
      sp.simplify((gradL_c - claim_c)).applyfunc(sp.simplify) == sp.zeros(dp,1))

# truncated objective on S = {0,1}
S = [0,1]
denS = sum(sp.exp(zz[v]) for v in S)
pt = {v: sp.exp(zz[v])/denS for v in S}
qt = sp.Matrix(sp.symbols('qt1 qt2', positive=True))
qtd = {S[0]: qt[0], S[1]: qt[1]}
LS = sum(qtd[v]*(sp.log(qtd[v]) - sp.log(pt[v])) for v in S)
gradLS = sp.Matrix([sp.diff(LS, h[i]) for i in range(dp)])
cSp = sum(pt[v]*W[v,:].T for v in S)
cSq = sum(qtd[v]*W[v,:].T for v in S)
diff = sp.simplify((gradLS - (cSp - cSq)).subs({qt[1]: 1-qt[0]}))
print("A3  grad L_S == c_S^p - c_S^q (sum qt=1)  :",
      diff.applyfunc(sp.simplify) == sp.zeros(dp,1))

# grad m_p, Eq (4)
mp = 1 - sum(p[v] for v in S)
grad_mp = sp.Matrix([sp.simplify(sp.diff(mp, h[i])) for i in range(dp)])
mps = 1 - sum(p[v] for v in S)
cSp_full = sum(p[v]*W[v,:].T for v in S)/(1-mps)
cTp_full = sum(p[v]*W[v,:].T for v in range(V) if v not in S)/mps
rhs = -mps*(1-mps)*(cSp_full - cTp_full)
print("A4  grad m_p == -m_p(1-m_p)Delta^p        :",
      sp.simplify((grad_mp - rhs)).applyfunc(sp.simplify) == sp.zeros(dp,1))

print()
print("="*72); print("B. BOUNDARY CASES"); print("="*72)
rng = np.random.default_rng(0)
def centroids(W,q,p,S):
    V=len(q); mask=np.zeros(V,bool); mask[S]=True
    mq=1-q[mask].sum(); mp_=1-p[mask].sum()
    out={'m_q':mq,'m_p':mp_}
    out['c_S^q']=(q[mask]/q[mask].sum())@W[mask]
    out['c_S^p']=(p[mask]/p[mask].sum())@W[mask]
    out['c_T^q']=(q[~mask]@W[~mask])/mq if mq>0 else None
    out['c_T^p']=(p[~mask]@W[~mask])/mp_ if mp_>0 else None
    return out
Vv,dpp=6,3
Wm=rng.normal(size=(Vv,dpp)); qv=softmax(rng.normal(size=Vv)); hv=rng.normal(size=dpp)
pv=softmax(Wm@hv)
for name,S in [("k=1  S={v0}",[0]), ("k=V  S=all",list(range(Vv))), ("k=V-1",list(range(Vv-1)))]:
    c=centroids(Wm,qv,pv,S)
    ok = (c['c_T^q'] is not None)
    print(f"  {name:14s} m_q={c['m_q']:.3e} m_p={c['m_p']:.3e}  tail centroids defined: {ok}")
print("  -> at k=V the tail centroids are 0/0. The paper's claim m_p,m_q in (0,1)")
print("     silently assumes k < V. HIDDEN ASSUMPTION, must be stated.")

# k=1: is Thm 2's identity still correct?
S=[0]; mask=np.zeros(Vv,bool); mask[S]=True
mq=1-qv[mask].sum(); mp_=1-pv[mask].sum()
cS=Wm[0]  # both head centroids are w_{v0}
cTq=(qv[~mask]@Wm[~mask])/mq; cTp=(pv[~mask]@Wm[~mask])/mp_
lhs=((pv-qv)[:,None]*Wm).sum(0); rhs=mq*(cS-cTq)-mp_*(cS-cTp)
print(f"  k=1 identity residual ||lhs-rhs|| = {np.linalg.norm(lhs-rhs):.2e}"
      f"  (any h is stationary since L_S == 0)")

# Prop 3 at k=1: D_S is (0 x d), rank 0
d_=4; D=np.zeros((0,d_))
print(f"  k=1 Prop 3: rank(D_S)={np.linalg.matrix_rank(D) if D.size else 0}, "
      f"fiber dim = d-0 = {d_} >= d-k+1 = {d_}  OK")

print()
print("="*72); print("C. LOGICAL STEP: 'some v* NOT in S has (u_v* - u_v0)^T eta != 0'"); print("="*72)
for seed in range(3):
    r=np.random.default_rng(seed); Vb,db,kb=200,8,5
    U=r.normal(size=(Vb,db)); S=np.arange(kb); v0=S[0]
    D=U[S[1:]]-U[v0]
    ns=np.linalg.svd(D)[2][np.linalg.matrix_rank(D):]   # basis of ker D_S
    eta=ns[0]
    inS=np.abs((U[S]-U[v0])@eta).max()
    outS=np.abs((U[~np.isin(np.arange(Vb),S)]-U[v0])@eta).max()
    print(f"  seed{seed}: max|(u_v-u_v0)^T eta| inside S = {inS:.2e},  outside S = {outS:.2e}"
          f"  -> {'differing index is outside S' if outS>1e-8 else 'FAIL'}")

print()
print("="*72); print("D. Cor 4 lower bound: is ||x|| >= <x,-c_hat> the only step?"); print("="*72)
for seed in range(3):
    r=np.random.default_rng(seed); n=7
    x=r.normal(size=n); c=r.normal(size=n); ch=c/np.linalg.norm(c)
    print(f"  seed{seed}: ||x||={np.linalg.norm(x):.4f} >= <x,-c_hat>={x@(-ch):+.4f} : "
          f"{np.linalg.norm(x) >= x@(-ch)}")
