# The Gradient That Top-k Distillation Never Computes — code and results

Anonymous release accompanying the AISTATS 2027 submission. It contains the
full distillation, caching and reconstruction pipeline, the job script behind
every reported run, the per-run JSON results each reported number is drawn
from, and a script that re-derives those numbers from the JSON files.

## Check the reported numbers (no GPU, a few seconds)

```bash
pip install numpy
python3 verify_reported_numbers.py            # exit code 0 iff every check passes
python3 verify_reported_numbers.py --verbose  # also print computed vs. reported values
```

Each check loads one file under `results/`, recomputes a quantity the paper
reports (means, exact paired permutation p-values, Cohen's d_z, crossing
indices, medians, ratios, storage and FLOP arithmetic), and compares it with
the value printed in the paper at the paper's own rounding.

## Layout

```
code/hsc/           reconstruction, chunked divergences, caches, distillation step, metrics
code/experiments/   one script per experiment, plus helpers
code/analysis/      statistics (exact paired permutation tests, bootstrap CIs) and post-hoc analyses
code/theory/        numerical checks of the theorems and the identifiability threshold
jobs/               the SLURM scripts that produced the results
results/            per-run JSON results (raw teacher caches are not included)
```

## Setup

```bash
conda create -n hsc python=3.11 && conda activate hsc
pip install -r code/requirements.txt
echo "HF_TOKEN=<your token>" > .env     # Llama and Gemma checkpoints are gated on Hugging Face
```

The job scripts were written for a SLURM cluster with NVIDIA H200 GPUs
(one per job; two for the Llama-3.1-70B Threshold run). Partition, account
and QoS are left as `<gpu-partition>`, `<account>`, `<qos>`; fill them in or
drop those lines. Each script resolves the repository root from its own
location (or from `$REPO`) and writes to its `results/` directory. Scripts
can also be run directly with `bash jobs/<script>.sh`.

## Where each result comes from

| Paper | Job script(s) | Results |
|---|---|---|
| Exactness | `02_e1_exactness.sh` | `e1_exactness` |
| Threshold (7 teachers, `tab:e2`, error-vs-k curves) | `01_e2_threshold.sh`, `22_e2_llama70b.sh`, then `code/experiments/merge_e2_threshold.py` | `e2_threshold` |
| Off-policy, three seeds (appendix) | `04_e3_offpolicy.sh` | `e3_offpolicy_3seed` |
| Off-policy, six seeds | `10_e3_seeds6.sh` | `e3_offpolicy_6seed` |
| Matched storage vs. top-1024 | `11_e3_m1_arms.sh`, `19_e3_m1_gemma.sh`, `20_e3_m1_qwen4b.sh`, `24_e3_m1_llama3b.sh`, `23_e3_m1_smollm2.sh` | `e3_matched_*` |
| Unembedding geometry | `python3 code/experiments/diag_teacher_geometry.py <teacher> --device cpu --out results/geometry` | `geometry`, plus the `assumptions` block of each `e2_threshold` file |
| Online arm on the Qwen3 matched-storage pairs | `27_qwen_online.sh` | `e3_online_qwen3-1.7b`, `e3_online_qwen3-4b` |
| Tail mass beyond top-k per teacher | `28_tail_mass.sh` (`code/experiments/diag_tail_mass.py`) | `tail_mass` |
| On-policy | `05_e4_onpolicy.sh` (three seeds), `12_e4_seeds6.sh` (six seeds) | `e4_onpolicy_3seed`, `e4_onpolicy_6seed` |
| Mechanism | `03_e5_mechanism.sh` (three seeds), `13_e5_seeds6.sh` (six seeds), `26b_e5_ace1_fixed_topk.sh`, `26c_e5_ace1_fixed_full.sh` (alpha_ce = 1) | `e5_mechanism_*` |
| Token absorption | `18_r10_absorption.sh`; decode with `code/analysis/decode_absorption.py` | `e5_token_absorption` |
| Systems (storage, write cost) | `06_e6_systems.sh` | `e6_systems` |
| Disagreement | `09_e7_disagreement.sh`, `20_e7_wide_sweep.sh`, `25_e7_wide_sweep_6seed.sh` | `e7_disagreement*` |
| Precision | `07_e8_precision.sh` (synthetic), `07b_e8_precision_real.sh` | `e8_precision_*` |
| Capacity | `08_e9_capacity.sh` | `e9_capacity` |
| Blind-subspace probe | `t1_blind_subspace.sh` | `t1_blind_subspace` |
| Drift versus k | `14_t2_drift_vs_k.sh`, `14b_t2_k4096.sh` | `t2_drift_vs_k` |
| Student-selected sets | `15_t3_union_select.sh` | `t3_union_select` |
| Residual plateau | `16_t4_residual_plateau_seeds012.sh`, `..._seeds345.sh` | `t4_residual_plateau` |
| Theorem checks | `python3 code/theory/verify_theory.py` (CPU) | printed |

## Notes on the result files

- **Mechanism, alpha_ce > 0.** The original three- and six-seed Mechanism
  runs also trained an alpha_ce = 1 arm. Those runs used a cross-entropy term
  whose gradient was wrong (see `BUGFIXES.md`), so their rows have been
  removed from `e5_mechanism_3seed` and `e5_mechanism_6seed`. The alpha_ce = 1
  arm was rerun with the corrected code, six seeds (`e5_mechanism_ace1_*`); that
  rerun is what the paper reports. No alpha_ce = 0 result is affected.
- **Three-seed Off-policy p-values.** `e3_offpolicy_3seed` predates exact
  p-values and stores a 20,000-draw Monte Carlo estimate (e.g. 0.247). At
  three seeds the exact sign-flip test can only return 0.25, 0.5, 0.75 or 1;
  the paper reports the exact values. All other p-values are exact.
- **Caches.** Hidden-state and top-k caches are tens of GB per experiment and
  are not included; each job rebuilds its own.
