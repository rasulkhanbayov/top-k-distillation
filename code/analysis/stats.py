#!/usr/bin/env python3
"""Uncertainty and significance for every comparison in the paper. Resolves B4.

The paper currently reports point estimates from three seeds with no intervals
and one significance test. This module supplies what the protocol promised:
bootstrap confidence intervals, paired permutation tests, and effect sizes.

Design choices, and why:

* Permutation over t-tests. Held-out KL and perplexity are bounded, skewed, and
  measured on a fixed set of evaluation items, so normality is not available.
  An exact paired permutation test over seed-matched arms makes no distributional
  assumption. With n seeds there are 2^n sign assignments; at n=5 that is 32, so
  the test is enumerated exactly rather than sampled.
* n=3 cannot support an equivalence claim. The smallest attainable two-sided
  p-value from a paired permutation test at n=3 is 2/8 = 0.25, so no comparison
  at three seeds can reach any conventional threshold. reported_min_p() makes
  that explicit rather than leaving a reader to infer it from a p-value of 0.75.
* Bootstrap over seeds, not over evaluation items. The quantity of interest is
  run-to-run variation, so seeds are the resampling unit.

numpy only. Everything here computes; nothing asserts an outcome.
"""
from __future__ import annotations
import itertools
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class Comparison:
    n_seeds: int
    mean_a: float
    mean_b: float
    diff: float                 # mean_a - mean_b
    ci_lo: float                # bootstrap CI on the paired difference
    ci_hi: float
    p_value: float              # exact paired permutation, two-sided
    min_attainable_p: float     # floor imposed by n_seeds
    cohens_dz: float            # paired effect size
    conclusive: bool            # p can reach 0.05 at this n at all


def min_attainable_p(n: int) -> float:
    """Smallest two-sided p an exact paired permutation test can return."""
    return 2.0 / (2 ** n)


def paired_permutation(a: np.ndarray, b: np.ndarray) -> float:
    """Exact two-sided paired permutation test on seed-matched arms."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("arms must be seed-matched")
    d = a - b
    n = len(d)
    if n > 22:
        raise ValueError("enumeration is exponential; sample instead above n=22")
    obs = abs(d.mean())
    count = sum(1 for signs in itertools.product([1, -1], repeat=n)
                if abs((d * np.array(signs)).mean()) >= obs - 1e-15)
    return count / (2 ** n)


def bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 20000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI on the paired difference, resampling seeds."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def compare(a, b, n_boot: int = 20000, seed: int = 0) -> Comparison:
    """Full paired comparison of two seed-matched arms."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    lo, hi = bootstrap_ci(a, b, n_boot, seed=seed)
    p = paired_permutation(a, b)
    floor = min_attainable_p(len(d))
    sd = d.std(ddof=1)
    return Comparison(
        n_seeds=len(d), mean_a=float(a.mean()), mean_b=float(b.mean()),
        diff=float(d.mean()), ci_lo=lo, ci_hi=hi, p_value=p,
        min_attainable_p=floor,
        cohens_dz=float(d.mean() / sd) if sd > 0 else float("nan"),
        conclusive=floor <= 0.05)


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, for a family of comparisons."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def _selftest() -> None:
    rng = np.random.default_rng(0)
    print("n  min attainable two-sided p")
    for n in (3, 5, 8, 10):
        print(f"{n:>2}  {min_attainable_p(n):.4f}"
              + ("   <- cannot reach 0.05" if min_attainable_p(n) > 0.05 else ""))
    print("\nnull case (arms identical in distribution), 5 seeds:")
    a = rng.normal(0, 1, 5); b = a + rng.normal(0, 0.02, 5)
    print("  ", asdict(compare(a, b)))
    print("\nclear effect, 5 seeds:")
    a = rng.normal(1.0, 0.02, 5); b = rng.normal(0.8, 0.02, 5)
    c = compare(a, b)
    print("  ", asdict(c))
    print("\nthe paper's off-policy comparison as run (n=3):")
    print(f"   min attainable p = {min_attainable_p(3):.3f}; the reported p=0.754")
    print("   therefore carries no evidence either way. This is B4.")
    print("\nHolm over a family of four:", [round(x, 4) for x in holm([0.01, 0.04, 0.03, 0.2])])


if __name__ == "__main__":
    _selftest()
