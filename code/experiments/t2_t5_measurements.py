#!/usr/bin/env python3
"""T2-T5: measurement code for the four remaining unrun experiments.

These four need training runs, and this repository does not ship a training
harness: the runs behind the paper were driven by the scripts in jobs/.
Writing a second, untested trainer here would be worse than useless, so this
module supplies the parts that are actually specific to these experiments --
the per-step measurements and the configurations -- and names the two hooks the
existing harness must call. Everything else is already in the harness.

    from t2_t5_measurements import tail_mass, residual_terms
    ...
    if step % cfg.measure_every == 0:
        record(step, tail_mass(student_logits, S),
                     residual_terms(student_logits, teacher_logits, S, W))

T2  drift versus k, with the full-vocabulary negative control
T3  student-selected observation sets, closing the scope gap to Liu et al.
T4  residual plateau past convergence
T5  alpha_ce sweep, separating attenuation from the calibration regime change

Every function computes; none asserts an outcome.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import numpy as np


# ----------------------------------------------------------------- measurements
def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def tail_mass(logits: np.ndarray, S: np.ndarray) -> float:
    """m_p = 1 - sum_{v in S} p_v. The quantity Theorem 1 says is unconstrained."""
    p = _softmax(np.asarray(logits, float))
    return float(1.0 - p[..., S].sum(-1).mean())


def residual_terms(student_logits, teacher_logits, S, W) -> dict:
    """The two halves of Eq. (5), measured separately.

    Theorem 2 says grad L = grad L_S + (m_q D^q - m_p D^p). Training drives the
    first to zero and cannot touch the second, so tracking them apart is what
    T4 needs: the first should decay, the second should plateau.
    """
    W = np.asarray(W, float)
    p = _softmax(np.asarray(student_logits, float))
    q = _softmax(np.asarray(teacher_logits, float))
    V = p.shape[-1]
    mask = np.zeros(V, bool); mask[S] = True

    m_p = float(1 - p[mask].sum()); m_q = float(1 - q[mask].sum())
    p_t = p[mask] / p[mask].sum(); q_t = q[mask] / q[mask].sum()
    c_S_p = p_t @ W[mask]; c_S_q = q_t @ W[mask]
    c_T_p = (p[~mask] @ W[~mask]) / m_p
    c_T_q = (q[~mask] @ W[~mask]) / m_q

    grad_trunc = c_S_p - c_S_q
    tail_term = m_q * (c_S_q - c_T_q) - m_p * (c_S_p - c_T_p)
    grad_full = ((p - q)[:, None] * W).sum(0)
    return dict(
        grad_trunc_norm=float(np.linalg.norm(grad_trunc)),
        tail_term_norm=float(np.linalg.norm(tail_term)),
        grad_full_norm=float(np.linalg.norm(grad_full)),
        identity_residual=float(np.linalg.norm(grad_full - (grad_trunc + tail_term))),
        m_p=m_p, m_q=m_q)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    """ECE. T5 needs it alongside the tail-mass gap: the paper's own Mechanism
    run shows calibration shifting two orders of magnitude between alpha_ce
    settings, so an attenuation reported without ECE is confounded."""
    probs = np.asarray(probs, float); labels = np.asarray(labels, int)
    conf = probs.max(-1); pred = probs.argmax(-1); acc = (pred == labels).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def select_S(teacher_logits, student_logits, k: int, rule: str) -> np.ndarray:
    """Observation set. 'teacher' is the fixed-S case the theorems assume;
    'union' is the student-dependent case of Liu et al., which T3 probes and
    which the theorems do NOT cover."""
    if rule == "teacher":
        return np.sort(np.argsort(-np.asarray(teacher_logits))[:k])
    if rule == "union":
        t = set(np.argsort(-np.asarray(teacher_logits))[:k].tolist())
        s = set(np.argsort(-np.asarray(student_logits))[:k].tolist())
        return np.array(sorted(t | s))
    if rule == "full":
        return np.arange(len(teacher_logits))
    raise ValueError(f"unknown selection rule {rule!r}")


# --------------------------------------------------------------- configurations
@dataclass
class RunConfig:
    experiment: str
    teacher: str = "meta-llama/Llama-3.1-8B"
    student: str = "meta-llama/Llama-3.2-1B"
    k: int = 64
    selection: str = "teacher"       # teacher | union | full
    alpha_ce: float = 0.0
    steps: int = 500
    seeds: tuple = (0, 1, 2, 3, 4, 5)   # six, not five: see note below
    measure_every: int = 25
    lr: float = 1e-5
    warmup: int = 50
    batch_tokens: int = 65536
    notes: str = ""


# Six seeds, not five. An exact two-sided paired permutation test over n
# seed-matched runs cannot return a p below 2/2^n. At n=5 that floor is 0.0625,
# which never reaches 0.05; n=6 gives 0.031. See code/analysis/stats.py.
MIN_SEEDS_FOR_SIGNIFICANCE = 6

T2 = [RunConfig("T2_drift_vs_k", k=k, alpha_ce=0.0, selection=sel,
                notes="drift versus k; the full-vocabulary arm is the negative control")
      for k, sel in [(16, "teacher"), (64, "teacher"), (256, "teacher"),
                     (1024, "teacher"), (4096, "teacher"), (0, "full")]]

T3 = [RunConfig("T3_student_selected", k=10, selection=rule, alpha_ce=0.0,
                notes="closes the scope gap to Liu et al.; theorems do not cover 'union'")
      for rule in ("teacher", "union")]

T4 = [RunConfig("T4_residual_plateau", k=64, alpha_ce=0.0, steps=5000,
                measure_every=100,
                notes="train past convergence; grad_trunc_norm should decay, "
                      "tail_term_norm should plateau")]

T5 = [RunConfig("T5_alpha_sweep", k=64, alpha_ce=a,
                notes="report the tail-mass gap AND ECE at each alpha")
      for a in (0.0, 0.1, 0.3, 1.0)]

# The two arms B5 needs, without which no quality claim can be made.
B5 = [RunConfig("B5_matched_storage", k=1024, alpha_ce=1.0,
                notes="top-1024 at 6144 B/position, matched against the 8192 B "
                      "hidden-state cache; the paper's own falsification criterion"),
      RunConfig("B5_feature_matching", k=0, selection="full", alpha_ce=1.0,
                notes="feature matching; the only arm separating 'exact reconstruction "
                      "helps' from 'hidden-state availability helps'")]

ALL = {"T2": T2, "T3": T3, "T4": T4, "T5": T5, "B5": B5}


if __name__ == "__main__":
    import json, sys
    if len(sys.argv) > 1 and sys.argv[1] == "--emit-configs":
        for name, runs in ALL.items():
            for i, c in enumerate(runs):
                print(json.dumps({"config_id": f"{name}_{i}", **asdict(c)}))
        sys.exit(0)
    print(__doc__)
    print(f"Configurations: " + ", ".join(f"{n} ({len(v)} runs x {len(v[0].seeds)} seeds)"
                                          for n, v in ALL.items()))
    total = sum(len(v) * len(v[0].seeds) for v in ALL.values())
    print(f"Total runs to complete the design: {total}")
    print(f"Minimum seeds for an exact permutation test to reach p<0.05: "
          f"{MIN_SEEDS_FOR_SIGNIFICANCE}")
    print("\nSelf-check of the measurement functions on synthetic data:")
    rng = np.random.default_rng(0); V, dp, k = 2000, 32, 8
    W = rng.normal(size=(V, dp)) / np.sqrt(dp)
    tl = rng.normal(size=V); sl = rng.normal(size=V)
    S = select_S(tl, sl, k, "teacher")
    r = residual_terms(sl, tl, S, W)
    print(f"  Eq. (5) identity residual on random inputs: {r['identity_residual']:.2e}")
    print(f"  tail mass m_p = {r['m_p']:.4f}, m_q = {r['m_q']:.4f}")
    print(f"  |S| teacher={len(S)}, union={len(select_S(tl, sl, k, 'union'))}")
