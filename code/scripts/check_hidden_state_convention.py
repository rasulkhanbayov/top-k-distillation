#!/usr/bin/env python3
""". Verify that common.py::teacher_hidden_states captures the
exact tensor lm_head consumes for a given architecture, by checking that it
reproduces the model's own logits exactly.

Run this once per new model family before trusting any cache built from it.
A mismatch here is the failure mode that looks like a small quality regression
instead of a bug.

PRECISION NOTE, found the hard way (diag_capture_point.sh/2.sh): this script
used to reconstruct logits as `h.float() @ U.float().T`, upcasting both
operands to fp32 BEFORE the matmul. On qwen3-8b that gave a 2.17e-3 relative
mismatch and looked exactly like a capture-point bug -- it sent a real
capture-point "fix" (model.model(...).last_hidden_state instead of
hidden_states[-1]) into common.py and four experiment files, none of which
changed the result at all, because hidden_states[-1] was never wrong: it is
bit-identical to last_hidden_state, and model.lm_head(hidden_states[-1])
reproduces the model's own logits with 0.0 error. The actual cause is that
bf16 @ bf16 matmul (what F.linear does internally, and what hsc/divergences.py
already does correctly: `z = (h @ W[a:b].T).float()`, upcasting only the
RESULT) is not bit-equivalent to upcast-then-multiply in fp32 -- different
rounding/accumulation order. Fixed here by calling model.lm_head(...) itself
rather than hand-reconstructing with U/bias, since that is both simpler and
exactly matches what the real training code path does."""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from experiments.common import load_teacher, teacher_hidden_states, TEACHERS


def main(a):
    model, tok, U, bias, softcap = load_teacher(a.teacher)
    ids = tok("The quick brown fox jumps over the lazy dog.",
              return_tensors="pt")["input_ids"].cuda()
    h = teacher_hidden_states(model, ids)
    with torch.no_grad():
        out = model(input_ids=ids)
        z_recon = model.lm_head(h)[0].float()
        if softcap:
            z_recon = softcap * torch.tanh(z_recon / softcap)
    z_model = out.logits[0].float()
    err = (z_model - z_recon).abs().max().item()
    rel = err / z_model.abs().max().item()
    print(f"{a.teacher}: max |logit diff| = {err:.3e}   relative {rel:.3e}")
    ok = rel < 1e-3
    print("PASS: teacher_hidden_states is the correct capture point" if ok else
          "FAIL: teacher_hidden_states is NOT the right capture point for this "
          "model.\n      Find the tensor the lm_head actually consumes and "
          "cache that.")
    return 0 if ok else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="qwen3-8b", choices=list(TEACHERS))
    sys.exit(main(p.parse_args()))
