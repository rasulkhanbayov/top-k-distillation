#!/usr/bin/env python3
"""Re-derive the paper's reported numbers from the released result files.

Each check reads one JSON under results/, recomputes a quantity the paper
reports, and compares it with the value printed in the paper at the paper's
own rounding. Prints PASS/FAIL per quantity. No model is run; numpy only.

    python3 verify_reported_numbers.py            # exit code 0 iff all pass
    python3 verify_reported_numbers.py --verbose  # also print computed values

Checks for results that are not present (runs still in progress) are
reported as SKIP, not PASS.
"""
from __future__ import annotations

import itertools, json, math, sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent / "results"
VERBOSE = "--verbose" in sys.argv
counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}


def load(rel):
    return json.loads((R / rel).read_text())


def check(label, computed, paper, tol):
    ok = abs(computed - paper) <= tol
    tag = "PASS" if ok else "FAIL"
    counts[tag] += 1
    if VERBOSE or not ok:
        print(f"  [{tag}] {label:62s} computed={computed:<12.6g} paper={paper:g}")
    else:
        print(f"  [{tag}] {label}")


def skip(label, why):
    counts["SKIP"] += 1
    print(f"  [SKIP] {label} ({why})")


def section(name, needs):
    print(f"\n{name}")
    missing = [n for n in needs if not (R / n).exists()]
    if missing:
        skip(name, "missing " + ", ".join(missing))
        return False
    return True


def perm_p(a, b):
    """Exact two-sided paired sign-flip permutation p-value."""
    d = np.asarray(a, float) - np.asarray(b, float)
    obs = abs(d.mean())
    hits = sum(abs((d * np.array(s)).mean()) >= obs - 1e-15
               for s in itertools.product([1, -1], repeat=len(d)))
    return hits / 2 ** len(d)


def dz(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    return d.mean() / d.std(ddof=1)


def final(traj):
    return max(traj, key=lambda t: t["step"])


# --------------------------------------------------------------------------
if section("Exactness (E1)", ["e1_exactness/e1.json"]):
    e1 = load("e1_exactness/e1.json")
    check("number of steps compared", len(e1["rows"]), 200, 0)
    check("max |loss_online - loss_cached| over all steps",
          max(r["abs_diff"] for r in e1["rows"]), 0.0, 0.0)

if section("Threshold (E2)", ["e2_threshold/e2.json"]):
    e2 = load("e2_threshold/e2.json")
    T = e2["teachers"]
    check("teachers measured", len(T), 7, 0)
    exact = sum(v["required_k"] == v["d"] + 1 for v in T.values())
    check("teachers collapsing exactly at k=d+1", exact, 5, 0)
    for name in ("llama31-8b", "llama31-70b"):
        check(f"{name}: crossing index k-(d+1)", T[name]["required_k"] - (T[name]["d"] + 1), -1, 0)
    check("fit slope of collapse k on d", e2["scaling"]["slope"], 1.00, 0.005)
    check("fit R^2", e2["scaling"]["r2"], 1.000, 0.0005)
    res = [v["assumptions"]["ones_residual"] for v in T.values()]
    check("min relative distance of 1 to range(U)", min(res), 0.062, 0.0005)
    check("max relative distance of 1 to range(U)", max(res), 0.130, 0.0005)

    def median_kl(name, k):
        return float(np.median([r["kl"] for r in T[name]["rows"]
                                if r["k"] == k and r["rule"] == "topk"]))
    for name, k, paper, tol in [
            ("llama31-8b", 3686, 9.98e-5, 0.005e-5), ("llama31-8b", 4096, 1.39e-7, 0.005e-7),
            ("llama31-8b", 4097, 2.72e-8, 0.005e-8), ("llama31-8b", 4098, -1.48e-8, 0.005e-8),
            ("llama31-8b", 8192, -2.59e-10, 0.005e-10),
            ("llama31-70b", 7372, 2.06e-5, 0.005e-5), ("llama31-70b", 8192, 3.48e-8, 0.005e-8),
            ("llama31-70b", 8193, 2.45e-8, 0.005e-8), ("llama31-70b", 8194, 5.75e-9, 0.005e-9),
            ("llama31-70b", 16384, -2.90e-9, 0.005e-9)]:
        check(f"{name}: median KL(q||qhat) at k={k}", median_kl(name, k), paper, tol)
    for name in ("gemma3-4b", "gemma3-12b"):
        d_ = T[name]["d"]
        check(f"{name}: smallest k measured is d+1 (soft-capped, 1=yes)",
              float(min(r["k"] for r in T[name]["rows"]) == d_ + 1), 1.0, 0)
    for name in T:
        check(f"{name}: full column rank (1=yes)", float(bool(T[name]["assumptions"]["full_column_rank"])), 1.0, 0)

if section("Off-policy, three seeds (E3, appendix)", ["e3_offpolicy_3seed/e3.json"]):
    r = load("e3_offpolicy_3seed/e3.json")["results"]
    check("online train loss", r["full_fkl_online"]["train_loss"]["mean"], 0.34808, 5e-6)
    check("cached train loss", r["full_fkl_cached"]["train_loss"]["mean"], 0.34791, 5e-6)
    check("top-256 train loss", r["topk_fkl"]["train_loss"]["mean"], 0.3277, 5e-5)
    check("online perplexity", r["full_fkl_online"]["val_ppl"]["mean"], 11.655, 5e-4)
    check("cached perplexity", r["full_fkl_cached"]["val_ppl"]["mean"], 11.656, 5e-4)
    check("top-256 perplexity", r["topk_fkl"]["val_ppl"]["mean"], 12.114, 5e-4)
    # This file stores only Monte Carlo p-values (20,000 sign flips); at n=3 the
    # exact test can only return 0.25, 0.5, 0.75 or 1, which the paper reports.
    check("cached vs online p (MC estimate of exact 0.75)", r["_test_cached_vs_online"]["p_value"], 0.75, 0.01)
    check("cached vs top-256 p (MC estimate of exact 0.25)", r["_test_cached_vs_topk"]["p_value"], 0.25, 0.01)

if section("Off-policy, six seeds (E3)", ["e3_offpolicy_6seed/e3.json"]):
    ps = load("e3_offpolicy_6seed/e3.json")["per_seed_by_arm"]
    ppl = {a: [s["val_ppl"] for s in ps[a]] for a in ps}
    check("cached perplexity", np.mean(ppl["full_fkl_cached"]), 11.656, 5e-4)
    check("online perplexity", np.mean(ppl["full_fkl_online"]), 11.653, 5e-4)
    check("cached vs online exact p", perm_p(ppl["full_fkl_cached"], ppl["full_fkl_online"]), 0.0625, 1e-9)
    check("cached vs online Cohen's d_z", dz(ppl["full_fkl_cached"], ppl["full_fkl_online"]), 1.28, 0.005)
    check("top-256 perplexity", np.mean(ppl["topk_fkl"]), 12.150, 5e-4)
    check("cached vs top-256 exact p", perm_p(ppl["full_fkl_cached"], ppl["topk_fkl"]), 0.03125, 1e-9)
    check("cached vs top-256 Cohen's d_z", dz(ppl["full_fkl_cached"], ppl["topk_fkl"]), -3.35, 0.005)
    # Equivalence (TOST) of cached and online against a +/-0.01 perplexity margin:
    # the larger of two exact one-sided sign-flip p-values.
    dd = np.asarray(ppl["full_fkl_cached"]) - np.asarray(ppl["full_fkl_online"])
    def one_sided_below(x):  # H0: mean >= 0, H1: mean < 0
        obs = x.mean()
        return sum((x * np.array(sg)).mean() <= obs + 1e-15
                   for sg in itertools.product([1, -1], repeat=len(x))) / 2 ** len(x)
    tost = max(one_sided_below(dd - 0.01), one_sided_below(-(dd + 0.01)))
    check("cached vs online TOST p at +/-0.01 perplexity", tost, 0.016, 5e-4)
    se = dd.std(ddof=1) / np.sqrt(len(dd))
    tcrit = 2.015048372669157  # t_{0.95, 5}
    check("90% interval of the gap, lower", dd.mean() - tcrit * se, 0.0012, 5e-5)
    check("90% interval of the gap, upper", dd.mean() + tcrit * se, 0.0055, 5e-5)
    check("margin as share of the top-256 gap, percent",
          100 * 0.01 / (np.mean(ppl["topk_fkl"]) - np.mean(ppl["full_fkl_cached"])), 2.0, 0.05)

for name, cached, topk, d_z, feat, ftol, cbytes in [
        ("qwen3-1.7b", 18.615, 18.336, 40, 83248, 0.5, 4096),
        ("gemma3-4b", 14.275, 15.417, -23.6, 244352, 0.5, 5120),
        ("qwen3-4b", 13.885, 13.747, 25.2, 117018, 0.5, 5120),
        ("llama3.2-3b", 11.680, 11.718, -6.8, 55517, 0.5, 6144),
        ("smollm2-1.7b", 12.558, 12.569, -2.4, 684.7, 0.05, 4096)]:
    f = f"e3_matched_{name}/e3.json"
    if section(f"Matched storage vs top-1024, {name} (E3)", [f]):
        ps = load(f)["per_seed_by_arm"]
        ppl = {a: [s["val_ppl"] for s in ps[a]] for a in ps}
        check("cached perplexity", np.mean(ppl["full_fkl_cached"]), cached, 5e-4)
        check("top-1024 perplexity", np.mean(ppl["topk_fkl"]), topk, 5e-4)
        check("cached vs top-1024 exact p", perm_p(ppl["full_fkl_cached"], ppl["topk_fkl"]), 0.03125, 1e-9)
        dec = 0 if abs(d_z) >= 30 else 1
        check("cached vs top-1024 Cohen's d_z",
              dz(ppl["full_fkl_cached"], ppl["topk_fkl"]), d_z, 0.5 * 10 ** -dec)
        check("feature-matching perplexity", np.mean(ppl["feature"]), feat, ftol)
        check("cached bytes per position", ps["full_fkl_cached"][0]["bytes_per_position"], cbytes, 0)
        check("top-1024 bytes per position", ps["topk_fkl"][0]["bytes_per_position"], 6144, 0)
        check("cached vs feature exact p", perm_p(ppl["full_fkl_cached"], ppl["feature"]), 0.03125, 1e-9)

for on, m1, o_mean, p_oc, dz_oc, dz_ot in [
        ("e3_online_qwen3-1.7b", "e3_matched_qwen3-1.7b", 18.618, 0.34375, 0.40, 31.6),
        ("e3_online_qwen3-4b", "e3_matched_qwen3-4b", 13.881, 0.3125, -0.45, 20.7)]:
    if section(f"Online arm vs cache and top-1024, {on} (appendix)", [f"{on}/e3.json", f"{m1}/e3.json"]):
        o = [s_["val_ppl"] for s_ in load(f"{on}/e3.json")["per_seed_by_arm"]["full_fkl_online"]]
        mm = load(f"{m1}/e3.json")["per_seed_by_arm"]
        c = [s_["val_ppl"] for s_ in mm["full_fkl_cached"]]
        t = [s_["val_ppl"] for s_ in mm["topk_fkl"]]
        check("seeds", len(o), 6, 0)
        check("online perplexity", np.mean(o), o_mean, 5e-4)
        check("online vs cached exact p", perm_p(o, c), p_oc, 1e-9)
        check("online vs cached Cohen's d_z", dz(o, c), dz_oc, 0.005)
        check("online vs top-1024 exact p", perm_p(o, t), 0.03125, 1e-9)
        check("online vs top-1024 Cohen's d_z", dz(o, t), dz_ot, 0.05)

TAIL = {"gemma3-4b": (0.0121, 0.0201, 0.151, 3.59), "llama32-3b": (0.0024, 0.0097, 0.064, 2.28),
        "smollm2-1.7b": (0.0020, 0.0087, 0.061, 2.20), "qwen3-1.7b": (0.0006, 0.0070, 0.036, 1.68),
        "qwen3-4b": (0.0004, 0.0055, 0.028, 1.53)}
if section("Tail mass beyond top-k per teacher (appendix)", [f"tail_mass/tail_mass_{t}.json" for t in TAIL]):
    tm = {t: load(f"tail_mass/tail_mass_{t}.json") for t in TAIL}
    for t, (med, mean1024, mean64, H) in TAIL.items():
        check(f"{t} positions", tm[t]["positions"], 65408, 0)
        check(f"{t} median tail beyond top-1024", tm[t]["tail_mass"]["1024"]["median"], med, 5e-5)
        check(f"{t} mean tail beyond top-1024", tm[t]["tail_mass"]["1024"]["mean"], mean1024, 5e-5)
        check(f"{t} mean tail beyond top-64", tm[t]["tail_mass"]["64"]["mean"], mean64, 5e-4)
        check(f"{t} mean entropy", tm[t]["mean_entropy"], H, 5e-3)
    gain = {"gemma3-4b": 1.142, "llama32-3b": 0.038, "smollm2-1.7b": 0.011, "qwen3-1.7b": -0.279, "qwen3-4b": -0.138}
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
        for pos, i in enumerate(order): r[i] = pos
        return np.array(r, float)
    ts = list(TAIL)
    for key in ("median", "mean"):
        a = ranks([tm[t]["tail_mass"]["1024"][key] for t in ts]); b = ranks([gain[t] for t in ts])
        check(f"Spearman, {key} tail beyond top-1024 vs cache gain", np.corrcoef(a, b)[0, 1], 0.9, 1e-9)
    a = ranks([tm[t]["mean_entropy"] for t in ts]); b = ranks([gain[t] for t in ts])
    check("Spearman, entropy vs cache gain", np.corrcoef(a, b)[0, 1], 0.9, 1e-9)

if section("Unembedding geometry of the matched-storage teachers", ["e2_threshold/e2.json"]):
    T = load("e2_threshold/e2.json")["teachers"]
    geo = {n: T[n]["assumptions"] for n in ("qwen3-1.7b", "gemma3-4b")}
    for n in ("qwen3-4b", "llama32-3b", "smollm2-1.7b"):
        gq = R / f"geometry/geometry_{n}.json"
        if gq.exists():
            geo[n] = json.loads(gq.read_text())["report"]
    for n, c, a in [("qwen3-1.7b", 33.7, 0.226), ("gemma3-4b", 30.0, 0.166), ("qwen3-4b", 59.5, 0.285),
                    ("llama32-3b", 60.8, 0.371), ("smollm2-1.7b", 115.2, 0.270)]:
        if n not in geo:
            skip(f"{n} geometry", "geometry file missing"); continue
        check(f"{n} condition number", geo[n]["cond"], c, 0.05)
        check(f"{n} anisotropy", geo[n]["anisotropy"], a, 0.0005)

if section("On-policy, six seeds (E4)", ["e4_onpolicy_6seed/e4.json"]):
    rows = [r for r in load("e4_onpolicy_6seed/e4.json")["rows"] if r["selection"] == "teacher"]
    def col(sup, key):
        rs = sorted((r for r in rows if r["supervision"] == sup), key=lambda r: r["seed"])
        return [r[key] for r in rs]
    tl, el = col("topk", "loss_low_entropy"), col("exact", "loss_low_entropy")
    th, eh = col("topk", "loss_high_entropy"), col("exact", "loss_high_entropy")
    check("seeds per arm", len(tl), 6, 0)
    check("low entropy: truncated KL", np.mean(tl), 0.4513, 5e-5)
    check("low entropy: exact KL", np.mean(el), 0.4182, 5e-5)
    check("high entropy: truncated KL", np.mean(th), 4.0296, 5e-5)
    check("high entropy: exact KL", np.mean(eh), 0.5945, 5e-5)
    check("low entropy exact p", perm_p(tl, el), 0.03125, 1e-9)
    check("high entropy exact p", perm_p(th, eh), 0.03125, 1e-9)
    check("low entropy Cohen's d_z", dz(tl, el), 12.7, 0.05)
    check("high entropy Cohen's d_z", dz(th, eh), 4.0, 0.05)
    check("high/low gap ratio", (np.mean(th) - np.mean(eh)) / (np.mean(tl) - np.mean(el)), 104, 0.5)
    check("low entropy relative gap, percent", 100 * (np.mean(tl) - np.mean(el)) / np.mean(el), 8, 0.5)
    check("high entropy relative gap, percent", 100 * (np.mean(th) - np.mean(eh)) / np.mean(eh), 578, 0.5)

if section("Mechanism, alpha_ce=0, six seeds (E5)", ["e5_mechanism_6seed/e5.json"]):
    rows = [r for r in load("e5_mechanism_6seed/e5.json")["rows"] if r["alpha_ce"] == 0.0]
    def arm(a, key="tail_mass_student", step=None):
        rs = sorted((r for r in rows if r["arm"] == a), key=lambda r: r["seed"])
        out = []
        for r in rs:
            t = final(r["trajectory"]) if step is None else next(x for x in r["trajectory"] if x["step"] == step)
            out.append(t[key])
        return out
    tk, fv = arm("topk_fkl"), arm("full_fkl")
    check("final student tail mass, top-k", np.mean(tk), 0.0784, 5e-5)
    check("final student tail mass, full vocabulary", np.mean(fv), 0.0471, 5e-5)
    check("teacher tail mass", np.mean(arm("topk_fkl", "tail_mass_teacher")), 0.0216, 5e-5)
    check("top-k vs full exact p", perm_p(tk, fv), 0.03125, 1e-9)
    check("top-k vs full Cohen's d_z", dz(tk, fv), 2031, 0.5)
    for step, a_tk, a_fv in [(0, 0.0496, 0.0496), (50, 0.0664, 0.0476), (250, 0.0760, 0.0474), (500, 0.0784, 0.0471)]:
        check(f"Fig. 1b tail mass at step {step}, top-k", np.mean(arm("topk_fkl", step=step)), a_tk, 5e-5)
        check(f"Fig. 1b tail mass at step {step}, full", np.mean(arm("full_fkl", step=step)), a_fv, 5e-5)

if section("Mechanism, alpha_ce=1, six seeds (E5)", ["e5_mechanism_ace1_topk/e5.json", "e5_mechanism_ace1_full/e5.json"]):
    def runs(a):
        return sorted(load(f"e5_mechanism_ace1_{a}/e5.json")["rows"], key=lambda r: r["seed"])
    tk, fv = runs("topk"), runs("full")
    check("seeds per arm", min(len(tk), len(fv)), 6, 0)
    def traj(rs, key="tail_mass_student"):
        return np.array([[t[key] for t in r["trajectory"]] for r in rs])
    T, F = traj(tk), traj(fv)
    for i, (a, b) in enumerate([(0.0496, 0.0496), (0.0525, 0.0468), (0.0544, 0.0471), (0.0542, 0.0468)]):
        check(f"top-k tail mass, checkpoint {i}", T[:, i].mean(), a, 5e-5)
        check(f"full-vocab tail mass, checkpoint {i}", F[:, i].mean(), b, 5e-5)
    check("seed sd at most 2e-5 (1=yes)", float(max(T.std(0, ddof=1).max(), F.std(0, ddof=1).max()) <= 2.5e-5), 1.0, 0)
    check("final gap exact p", perm_p(T[:, -1], F[:, -1]), 0.03125, 1e-9)
    check("final gap Cohen's d_z", dz(T[:, -1], F[:, -1]), 497, 0.5)
    mq = tk[0]["trajectory"][0]["tail_mass_teacher"]
    check("teacher tail mass", mq, 0.0216, 5e-5)
    check("top-k |m_p-m_q| final", np.mean(np.abs(T[:, -1] - mq)), 0.0326, 5e-5)
    check("full |m_p-m_q| final", np.mean(np.abs(F[:, -1] - mq)), 0.0252, 5e-5)
    check("final gap", T[:, -1].mean() - F[:, -1].mean(), 0.0074, 5e-5)
    check("alpha_ce=0 gap / alpha_ce=1 gap", 0.0312818347 / (T[:, -1].mean() - F[:, -1].mean()), 4.25, 0.005)
    check("top-k final ECE", traj(tk, "ece")[:, -1].mean(), 0.019, 5e-4)
    check("full final ECE", traj(fv, "ece")[:, -1].mean(), 0.013, 5e-4)

if section("Mechanism rerun stability (E5, appendix)",
           ["e5_mechanism_3seed/e5.json", "e5_mechanism_6seed/e5.json"]):
    a = load("e5_mechanism_3seed/e5.json")["rows"]; b = load("e5_mechanism_6seed/e5.json")["rows"]
    diffs = []
    for r in a:
        m = next(x for x in b if x["arm"] == r["arm"] and x["alpha_ce"] == r["alpha_ce"] and x["seed"] == r["seed"])
        for t in r["trajectory"]:
            u = next(x for x in m["trajectory"] if x["step"] == t["step"])
            diffs.append(abs(t["tail_mass_student"] - u["tail_mass_student"]))
    check("shared arm-seed-checkpoint cells", len(diffs), 24, 0)
    check("max |tail-mass difference| between the two runs", max(diffs), 7e-5, 0.5e-5)
    check("mean |tail-mass difference|", float(np.mean(diffs)), 2e-5, 0.5e-5)

if section("Drift versus k (T2)", ["t2_drift_vs_k"]):
    def drift(sub, arm_name):
        d = load(f"t2_drift_vs_k/{sub}/e5.json")
        vals = [abs(final(r["trajectory"])["tail_mass_student"] - final(r["trajectory"])["tail_mass_teacher"])
                for r in d["rows"] if r["arm"] == arm_name and r["alpha_ce"] == 0.0]
        return float(np.mean(vals))
    for k, paper in [(16, 0.279), (64, 0.143), (256, 0.057), (1024, 0.018), (4096, 0.0046)]:
        check(f"final drift at k={k}", drift(f"k{k}", "topk_fkl"), paper, 0.5 * 10 ** -(3 if paper >= 0.01 else 4))
    check("final drift, full vocabulary", drift("full_vocab", "full_fkl"), 0.0255, 5e-5)

if section("Student-selected observation sets (T3)", ["t3_union_select"]):
    def fd(sub):
        rs = sorted(load(f"t3_union_select/{sub}/e5.json")["rows"], key=lambda r: r["seed"])
        return [abs(final(r["trajectory"])["tail_mass_student"] - final(r["trajectory"])["tail_mass_teacher"]) for r in rs]
    u, t = fd("union"), fd("teacher")
    check("final drift, union selection", np.mean(u), 0.200, 5e-4)
    check("final drift, teacher selection", np.mean(t), 0.327, 5e-4)
    check("union vs teacher exact p", perm_p(u, t), 0.03125, 1e-9)
    check("union vs teacher |Cohen's d_z|", abs(dz(u, t)), 884, 0.5)

if section("Residual plateau (T4)", ["t4_residual_plateau"]):
    rows = load("t4_residual_plateau/seeds012/e5.json")["rows"] + load("t4_residual_plateau/seeds345/e5.json")["rows"]
    def mean_at(key, step):
        return float(np.mean([next(t for t in r["trajectory"] if t["step"] == step)[key] for r in rows]))
    steps = sorted(t["step"] for t in rows[0]["trajectory"])
    check("seeds", len(rows), 6, 0)
    check("||grad L_S|| at step 0", mean_at("grad_trunc_norm", 0), 0.162, 5e-4)
    check("||grad L_S|| at step 250", mean_at("grad_trunc_norm", 250), 0.134, 1e-3)
    check("||grad L|| at step 250", mean_at("residual", 250), 0.152, 5e-4)
    check("||grad L|| at final step", mean_at("residual", steps[-1]), 0.173, 5e-4)
    check("lower bound at step 250", mean_at("lower_bound", 250), 0.057, 5e-4)
    check("lower bound at final step", mean_at("lower_bound", steps[-1]), 0.079, 5e-4)
    check("lower-bound rise from step 250 (%)", 100 * (mean_at("lower_bound", steps[-1]) / mean_at("lower_bound", 250) - 1), 38, 0.5)
    check("residual rise from step 250 (%)", 100 * (mean_at("residual", steps[-1]) / mean_at("residual", 250) - 1), 14, 0.5)
    check("max tightness (bound / residual)", max(mean_at("tightness", s) for s in steps), 0.42, 0.005)
    rb = [mean_at("residual", s) for s in steps]; lb = [mean_at("lower_bound", s) for s in steps]
    i = steps.index(250)
    check("corr(residual, lower bound), all checkpoints", float(np.corrcoef(rb, lb)[0, 1]), 0.79, 0.005)
    check("corr(residual, lower bound), from step 250", float(np.corrcoef(rb[i:], lb[i:])[0, 1]), 0.9995, 0.00005)

if section("Blind-subspace probe (T1)", ["t1_blind_subspace/t1.json"]):
    P = load("t1_blind_subspace/t1.json")["positions"]
    ks = sorted({p["k"] for p in P})
    for k in ks[:-1]:
        med = float(np.median([p["ratio_L"] for p in P if p["k"] == k]))
        check(f"median ratio for grad L_S at k={k} in [2e-15, 3e-15] (1=yes)", float(2e-15 <= med <= 3e-15), 1.0, 0)
    check("median ratio for grad L_S at k=V", float(np.median([p["ratio_L"] for p in P if p["k"] == ks[-1]])), 0.0, 1e-15)
    for k, paper in zip(ks, [0.32, 0.17, 0.0027, 0.0]):
        med = float(np.median([p["ratio_m"] for p in P if p["k"] == k]))
        check(f"median ratio for grad m_p at k={k}", med, paper, 0.005 if paper >= 0.01 else 5e-5)

if section("Disagreement (E7)", ["e7_disagreement/e7.json", "e7_disagreement/e7_predictors.json"]):
    pr = load("e7_disagreement/e7_predictors.json")["predictors"]
    for s, v in [("bild_style", 0.455), ("pretrain_style", 0.395), ("toolcall_style", 0.181)]:
        check(f"{s} tail rank correlation", pr[s]["tail_rank_corr"], v, 5e-4)
    rows = load("e7_disagreement/e7.json")["rows"]
    def kl(s, k, rr=rows): return float(np.mean([r["eval_kl"] for r in rr if r["setting"] == s and r["k"] == k]))
    for s in ("bild_style", "pretrain_style", "toolcall_style"):
        kstar = min((8, 64, 256, 1024), key=lambda k: kl(s, k))
        check(f"{s}: best k in the original sweep", kstar, 1024, 0)
    for s, k, v in [("bild_style", 64, 0.967), ("bild_style", 256, 0.885), ("bild_style", 1024, 0.858),
                    ("toolcall_style", 64, 0.508), ("toolcall_style", 256, 0.496), ("toolcall_style", 1024, 0.491)]:
        check(f"{s} KL at k={k}", kl(s, k), v, 5e-4)
    if (R / "e7_disagreement_wide/e7.json").exists():
        wide = load("e7_disagreement_wide/e7.json")["rows"]
        allr = rows + wide
        for s in ("bild_style", "pretrain_style", "toolcall_style"):
            seq = [kl(s, k, allr) for k in (8, 64, 256, 1024, 2048, 4096)]
            check(f"{s}: KL monotone decreasing to k=4096 (1=yes)", float(all(a > b for a, b in zip(seq, seq[1:]))), 1.0, 0)

if section("Disagreement, widened sweep, six seeds (E7)", ["e7_disagreement_wide_6seed/e7.json"]):
    rows6 = load("e7_disagreement_wide_6seed/e7.json")["rows"]
    def v6(s, k): return [r["eval_kl"] for r in sorted((r for r in rows6 if r["setting"] == s and r["k"] == k), key=lambda r: r["seed"])]
    for s, d1, d2 in [("bild_style", 0.573, 0.280), ("pretrain_style", 0.339, 0.121), ("toolcall_style", 0.226, 0.136)]:
        m = {k: float(np.mean(v6(s, k))) for k in (1024, 2048, 4096)}
        check(f"{s} seeds per k", len(v6(s, 4096)), 6, 0)
        check(f"{s} percent drop 1024->2048", 100 * (m[1024] - m[2048]) / m[1024], d1, 5e-4)
        check(f"{s} percent drop 2048->4096", 100 * (m[2048] - m[4096]) / m[2048], d2, 5e-4)
        check(f"{s}: best k in the widened sweep", min(m, key=m.get), 4096, 0)
        for a, b in [(1024, 2048), (2048, 4096), (1024, 4096)]:
            check(f"{s} k={a} vs k={b} exact p", perm_p(v6(s, b), v6(s, a)), 0.03125, 1e-9)
    check("largest 1024->4096 relative gain, percent (bild_style)",
          100 * (np.mean(v6("bild_style", 1024)) - np.mean(v6("bild_style", 4096))) / np.mean(v6("bild_style", 1024)), 0.85, 0.005)

if section("Precision (E8)", ["e8_precision_real/e8.json", "e8_precision_synthetic/e8.json"]):
    def row(f, dtype):
        return next(r for r in load(f)["rows"] if r["scheme"] == "hidden" and r["dtype"] == dtype)
    real, syn = row("e8_precision_real/e8.json", "int8"), row("e8_precision_synthetic/e8.json", "int8")
    check("real-teacher int8 KL", real["kl"], 2.02e-4, 0.005e-4)
    check("synthetic int8 KL", syn["kl"], 2.60e-11, 0.005e-11)
    check("real / synthetic KL ratio", real["kl"] / syn["kl"], 7.8e6, 0.05e6)
    check("real-teacher int8 max log-prob error", real["max_logp_err"], 0.0943, 5e-5)
    check("synthetic int8 max log-prob error", syn["max_logp_err"], 0.0887, 5e-5)

if section("Capacity (E9)", ["e9_capacity/e9.json"]):
    rows = [r for r in load("e9_capacity/e9.json")["rows"] if r["objective"] == "topk_fkl"]
    def m(t, s, k): return float(np.mean([r["eval_kl"] for r in rows if r["teacher"] == t and r["student"] == s and r["k"] == k]))
    for k, v in [(64, 0.825), (256, 0.753), (1024, 0.731)]:
        check(f"Qwen3-8B->Qwen3-0.6B KL at k={k}", m("qwen3-8b", "qwen3-0.6b", k), v, 5e-4)
    for (t, s), v in [(("qwen3-1.7b", "qwen3-1.7b"), 0.0013), (("qwen3-1.7b", "qwen3-0.6b"), 0.488),
                      (("qwen3-8b", "qwen3-1.7b"), 0.430)]:
        check(f"{t}->{s} KL at k=64", m(t, s, 64), v, 5e-4 if v > 0.01 else 5e-5)
    check("pair factor at k=64 (three distinct pairs)",
          m("qwen3-8b", "qwen3-0.6b", 64) / m("qwen3-8b", "qwen3-1.7b", 64), 1.92, 0.005)
    check("k factor, Qwen3-8B->Qwen3-0.6B",
          m("qwen3-8b", "qwen3-0.6b", 64) / m("qwen3-8b", "qwen3-0.6b", 1024), 1.13, 0.005)

if section("Storage arithmetic (Table: cache size per position)", []):
    teachers = {"Qwen3-1.7B": (2048, 151936), "Gemma-3-4B": (2560, 262144), "Gemma-3-12B": (3840, 262144),
                "Llama-3.1-8B": (4096, 128256), "Qwen3-8B": (4096, 151936), "Qwen3-32B": (5120, 151936),
                "Llama-3.1-70B": (8192, 128256)}
    paper = {"Qwen3-1.7B": (74.2, 5.34), "Gemma-3-4B": (102.4, 6.68), "Gemma-3-12B": (68.3, 10.01),
             "Llama-3.1-8B": (31.3, 10.68), "Qwen3-8B": (37.1, 10.68), "Qwen3-32B": (29.7, 13.34),
             "Llama-3.1-70B": (15.7, 21.34)}
    for t, (d, V) in teachers.items():
        check(f"{t}: full logits / bf16 cache", 2 * V / (2 * d), paper[t][0], 0.05)
        check(f"{t}: int8 cache / top-64", (d + 4) / 384, paper[t][1], 0.005)
        check(f"{t}: head GFLOPs 2dV", 2 * d * V / 1e9,
              {"Qwen3-1.7B": 0.62, "Gemma-3-4B": 1.34, "Gemma-3-12B": 2.01, "Llama-3.1-8B": 1.05,
               "Qwen3-8B": 1.24, "Qwen3-32B": 1.56, "Llama-3.1-70B": 2.10}[t], 0.005)

print(f"\n{counts['PASS']} passed, {counts['FAIL']} failed, {counts['SKIP']} skipped")
sys.exit(1 if counts["FAIL"] else 0)
