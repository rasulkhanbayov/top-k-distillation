# Implementation bugs found and fixed during the project

Bugs caught by checking results against sanity checks or independently
reasoned expected values. Each is fixed in the released code. The two marked
**affected results** changed what had been measured; the reruns or removals
are described there. The others were caught before any reported number
depended on them.

1. **Gemma-3 padded output head.** Gemma-3's LM head has 262,208 rows against
   a 262,144-token vocabulary. `load_teacher` slices `U` and the bias to the
   declared vocabulary.
2. **Threshold walltime.** Leverage-score selection recomputed a QR
   decomposition for every (position, k) pair; the order is now precomputed.
3. **`distill_step` interface.** It expected token ids and a student method
   that did not exist; it now takes flattened final-norm hidden states.
4. **Slow tail/head centroids.** Mask-based centroid code repeated
   full-vocabulary work about four times; replaced by an algebraic identity
   (about 100x faster).
5. **0/0 in separation constants.** When a teacher's tail mass rounds to
   zero in float32, the position is now marked invalid and counted, instead
   of producing NaN.
6. **Distillation arms silently trained with cross-entropy.** The off-policy
   arm table left `alpha_ce` at its default of 1.0. Every pure-distillation
   arm now sets `alpha_ce=0.0` explicitly.
7. **On-policy data pipeline.** Repeated prompts, the wrong tokenizer, missing
   left padding, a missing attention mask, and misaligned replay content.
8. **Dataset loading.** Multi-split Hugging Face datasets and parquet paths
   that existed only on a conversion branch; corpora are now read from their
   actual files.
9. **Unbounded resampling.** An exact-length filter could never match
   documents shorter than the sequence length. `text_batches` now raises
   after 200,000 rejected draws instead of looping.
10. **PyTorch kernel crash.** On PyTorch 2.13 + CUDA 13.0, the backward pass
    of `einsum("bkd,bd->bk")` and of `bmm` hits an illegal memory access at
    large batch x k x d. Replaced by an elementwise multiply-and-sum, chunked
    over the batch to bound memory.
11. **Results lost on timeout.** Sweep scripts saved results only at the end.
    `e5_mechanism.py` and `e7_disagreement.py` now save after every cell.
12. **Monte Carlo p-values.** `paired_test` used 20,000 random sign flips;
    it now enumerates all sign patterns exactly for up to 16 seeds.
    **Affected results:** two three-seed p-values in the appendix had been
    quoted as 0.247 and 0.754; the exact values, 0.25 and 0.75, are reported.
13. **Cross-tokenizer pairs.** Full-vocabulary KL compares teacher and
    student token by token, so the two must share a tokenizer. A pairing of
    gpt-oss-20b with a Gemma student was caught by a shape error and never
    run; every reported pair shares a tokenizer.
14. **Cross-entropy, JSD and skew-KL gradients.** These losses used a
    log-sum-exp computed without gradient, so the loss value was correct but
    the gradient dropped the softmax normalizer. For cross-entropy this
    trained "raise the correct logit" rather than cross-entropy. Skew KL also
    added a student-side term that DistiLLM's skew KL does not have. All three
    now use a differentiable chunked log-sum-exp. Every loss in
    `hsc/divergences.py` and `hsc/distill.py` used by a reported result
    (forward KL, truncated forward KL, cross-entropy, feature matching) has
    been checked against a dense reference on both value and gradient.
    **Affected results:** every run with alpha_ce > 0. Those rows were
    removed from the result files, and the alpha_ce = 1 Mechanism arm was
    rerun with the corrected code. JSD and skew KL were never used in a
    reported result.
