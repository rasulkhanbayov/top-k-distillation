"""Shared plumbing for E1-E9: teacher loading, hidden-state extraction, seeds,
and the statistical protocol declared in Section 6.2 of the paper."""
from __future__ import annotations
import json, os, random, hashlib
import numpy as np

SEEDS = (0, 1, 2)          # Section 6.2: every training comparison at 3 seeds

TEACHERS = {
    # name: (hf_id, d, V, softcap)
    "qwen3-1.7b":  ("Qwen/Qwen3-1.7B",            2048, 151936, None),
    "qwen3-4b":    ("Qwen/Qwen3-4B",              2560, 151936, None),
    "qwen3-8b":    ("Qwen/Qwen3-8B",              4096, 151936, None),
    "qwen3-32b":   ("Qwen/Qwen3-32B",             5120, 151936, None),
    "llama31-8b":  ("meta-llama/Llama-3.1-8B",    4096, 128256, None),
    "llama31-70b": ("meta-llama/Llama-3.1-70B",   8192, 128256, None),
    "gemma3-4b":   ("google/gemma-3-4b-pt",       2560, 262144, 30.0),
    "gemma3-12b":  ("google/gemma-3-12b-pt",      3840, 262144, 30.0),
    "gpt-oss-20b": ("openai/gpt-oss-20b",         2880, 201088, None),
    "smollm2-1.7b": ("HuggingFaceTB/SmolLM2-1.7B", 2048, 49152, None),
    "llama32-3b":  ("meta-llama/Llama-3.2-3B",    3072, 128256, None),
}

STUDENTS = {
    "qwen3-0.6b": ("Qwen/Qwen3-0.6B", 1024),
    "qwen3-1.7b": ("Qwen/Qwen3-1.7B", 2048),
    "llama32-1b": ("meta-llama/Llama-3.2-1B", 2048),
    "gemma3-1b":  ("google/gemma-3-1b-pt", 1152),
    "smollm2-360m": ("HuggingFaceTB/SmolLM2-360M", 960),
}

CORPORA = ("fineweb-edu", "starcoder", "openmath", "flan", "multilingual", "toolbench")

EVAL_SUITE = ("mmlu", "arc_challenge", "hellaswag", "gsm8k", "humaneval",
              "ifeval", "multilingual_ppl")


def set_seed(s):
    """NOT full bit-exact reproducibility across separate runs, even with the
    same seed: warn_only=True lets non-deterministic CUDA kernels run anyway
    (PyTorch itself warns "cuDNN Attention defaults to a non-deterministic
    algorithm" during training here), and small per-step float differences
    compound over hundreds of optimizer steps. Confirmed in practice: a real
    E5 full-grid run (full_fkl/alpha_ce=0.0/seed=0) showed its
    zero-tail-mass skip rate grow from ~0.18% to ~38% over 500 steps; two
    separate reruns of the identical config both stayed flat near 0.18%
    through step 300, matching each other but not the original run. Setting
    warn_only=False would enforce determinism (erroring instead of silently
    falling back on ops without a deterministic implementation) at some
    speed cost; not done here since the paper's reproducibility claims are
    about the statistical protocol (seeds, bootstrap CIs, permutation tests
    -- see README.md) rather than bit-exact single-run reproduction."""
    random.seed(s); np.random.seed(s)
    try:
        import torch
        torch.manual_seed(s); torch.cuda.manual_seed_all(s)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def load_teacher(name, device="cuda", dtype="bfloat16"):
    """Returns (model, tokenizer, U, bias, softcap). U is the unembedding as a
    torch tensor on `device`. Handles tied embeddings.

    device="auto" passes device_map="auto" straight through to
    from_pretrained, letting Accelerate shard the model across all visible
    GPUs -- needed for Llama-3.1-70B (bf16 weights alone are ~140GB, leaving
    no real headroom on a single H200's 141GB). Callers that immediately
    move U/bias to CPU/numpy (as every current caller does) are unaffected
    by sharding, since only the forward pass touches GPU placement; nothing
    downstream of load_teacher in this codebase assumes single-device
    model internals.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_id, d, V, softcap = TEACHERS[name]
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=getattr(torch, dtype), device_map=device)
    model.eval()
    head = model.get_output_embeddings()
    U = head.weight.detach()
    bias = head.bias.detach() if getattr(head, "bias", None) is not None else None
    if U.shape[0] != V:
        # Some checkpoints (e.g. Gemma-3's multimodal LM head) pad the output
        # embedding table to a hardware-friendly row count beyond the real
        # tokenizer vocabulary; the extra rows are untrained and are not real
        # logits. z = Ug + b must only ever see the true vocabulary, so slice
        # rather than accept the padded shape.
        assert U.shape[0] > V, (
            f"{hf_id}: unembedding has fewer rows ({U.shape[0]}) than the "
            f"declared vocab size ({V}); this is not padding, investigate.")
        U = U[:V]
        if bias is not None:
            bias = bias[:V]
    assert U.shape == (V, d), f"expected ({V},{d}), got {tuple(U.shape)}"
    return model, tok, U, bias, softcap


@torch.no_grad() if False else (lambda f: f)
def teacher_hidden_states(model, input_ids, attention_mask=None):
    """Final hidden states AFTER the final norm, which is what Equation (1)
    requires. output_hidden_states[-1] is post-norm for Llama, Qwen and Gemma;
    verify per architecture with scripts/check_hidden_state_convention.py before
    trusting a new family.

    CONFIRMED (diag_capture_point.sh, transformers 5.15.1, qwen3-8b):
    out.hidden_states[-1] and model.model(...).last_hidden_state are bit-
    identical tensors, and model.lm_head(hidden_states[-1]) reproduces the
    model's own logits with 0.0 error. An earlier version of this function was
    changed to call model.model(...) instead, based on a static reading of
    modeling_qwen3.py that turned out to be a red herring -- the real bug was
    in check_hidden_state_convention.py's verification method (see that file),
    not in hidden state capture. Reverted; see diag_capture_point2.sh's output
    for the full trace of how this was narrowed down."""
    import torch
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True)
    return out.hidden_states[-1]


class StudentHidden:
    """Adapts a raw AutoModelForCausalLM to the (student.hidden(ids), student.
    lm_head.weight) interface hsc.distill.distill_step expects. Everything
    else delegates straight through to the wrapped model (parameters(),
    .eval(), etc.), so this is a thin shim, not a reimplementation.

    student.hidden() must NOT use torch.no_grad() -- unlike
    teacher_hidden_states, which is always called on a frozen teacher, this
    is the student and needs gradients to flow through for training. Uses
    output_hidden_states=True / hidden_states[-1]; see
    teacher_hidden_states's docstring for why that is the correct
    post-final-norm capture point (not model.model(...).last_hidden_state).

    Shared across E3/E5 (and any future distill_step caller) instead of each
    experiment script defining its own copy -- originally written inside
    e5_mechanism.py, moved here when E3 needed the identical adapter."""

    def __init__(self, model):
        self._model = model

    def hidden(self, input_ids, attention_mask=None):
        return self._model(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True).hidden_states[-1]

    def __getattr__(self, name):
        return getattr(self._model, name)


def bootstrap_ci(values, n=10000, alpha=0.05, seed=0):
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    boots = rng.choice(v, size=(n, len(v)), replace=True).mean(1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(v.mean()), float(lo), float(hi)


def paired_test(a, b):
    """Paired difference across matched seeds and tasks, with a permutation test
    rather than a t-test, since the per-task metric distributions are not normal."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape
    d = a - b
    obs = d.mean()
    if len(d) <= 16:
        # Exact: enumerate all 2^n sign patterns. The earlier 20,000-draw
        # Monte Carlo version printed p=0.247 for an exact 0.25 at n=3, which
        # the paper then quoted as if it were below the n=3 floor.
        signs = 1 - 2 * ((np.arange(2 ** len(d))[:, None] >> np.arange(len(d))) & 1)
    else:
        rng = np.random.default_rng(0)
        signs = rng.choice([-1, 1], size=(20000, len(d)))
    null = (signs * d).mean(1)
    p = float((np.abs(null) >= abs(obs)).mean())
    return {"mean_diff": float(obs), "p_value": p, "n": int(len(d))}


def text_batches(tok, n, bs=4, L=512, seed=0, corpus="HuggingFaceFW/fineweb-edu"):
    """Yields `n` batches of `bs` sequences, each exactly `L` tokens after
    truncation, from a LOCALLY CACHED slice of `corpus` -- not the streaming
    iterator every E1/E2/E4/E5/E7 script previously rolled by hand
    (`load_dataset(..., streaming=True)` + `next(it)` per document).

    That per-document streaming pattern stalled for ~46 minutes on 2 of 4
    requested batches during a real E5 smoke test run on our cluster (batch 1:
    12s, batches 2-3: ~2791s and ~2797s each), with fewer than 200 documents
    scanned in each stall -- a network/rate-limit issue on individual
    streaming fetches, not a code bug, but one this function avoids entirely
    by downloading a single non-streamed parquet shard once (HF's normal
    resumable, cached download path) and sampling from it in memory
    afterward.

    MUST target an explicit file via `data_files=` (see _CORPUS_CONFIGS),
    NOT `load_dataset(corpus, name, split="train[:N]")`: a first attempt at
    that split-slicing syntax downloaded the entire sample-10BT config (all
    14 parquet shards, 52GB) before applying the row slice in memory --
    `datasets` does not push the slice down to a partial/lazy read for this
    dataset builder, it materializes the full split first. Naming one
    shard's file path directly downloads only that ~2GB file. First call for
    a given corpus is slower (one real download); every call after -- same
    process or a future job -- reuses the on-disk HF cache."""
    import time
    docs = _load_corpus_docs(corpus, seed)
    rng = random.Random(seed)
    buf = []
    for batch_i in range(n):
        t0 = time.time()
        draws = 0
        while len(buf) < bs:
            draws += 1
            # Hard failure guard: `if len(t) == L` below can only ever match
            # a document that tokenizes to >= L tokens (truncation clips a
            # longer doc down to exactly L; a shorter doc can never reach L
            # no matter how many times it's redrawn). If the corpus's real
            # length distribution tops out below L, this loop runs forever
            # with 0 accepted -- a real ~23-hour silent hang hit exactly this
            # (Muennighoff/flan's ag_news_subset shard maxes out at 273
            # tokens, requested against --seqlen 512, in E7's first full
            # sweep submission). Fail loudly instead of spinning.
            if draws > 200_000:
                raise RuntimeError(
                    f"text_batches: {draws} draws and still 0/{bs} accepted "
                    f"for corpus={corpus!r} at L={L}. This corpus's documents "
                    f"likely never reach {L} tokens after tokenization -- "
                    f"check its real length distribution (see common.py's "
                    f"_load_corpus_docs docstring) and either lower L or pick "
                    f"a longer-form corpus/shard. This would otherwise hang "
                    f"forever, not just run slowly.")
            doc = docs[rng.randrange(len(docs))]
            t = tok(doc, truncation=True, max_length=L,
                    return_tensors="pt")["input_ids"][0]
            if len(t) == L:
                buf.append(t)
            if draws % 500 == 0:
                # Heartbeat inside the retry loop itself, not just between
                # batches: a real E5 run showed ~50-minute gaps between
                # yielded batches that survived ruling out GPU contention
                # (idle + healthy on both candidate nodes) and the actual
                # model/chunked_logsumexp workload (fast in isolation, <4s
                # total). If this loop is where the time actually goes, this
                # print pinpoints it with an exact draw count; if it never
                # fires during a stall, the bottleneck is elsewhere (e.g.
                # something in the calling code between text_batches calls).
                print(f"    [text_batches] batch {batch_i+1}/{n}: "
                      f"{draws} draws, {len(buf)}/{bs} accepted, "
                      f"{time.time()-t0:.1f}s elapsed", flush=True)
        import torch
        yield torch.stack(buf[:bs]); buf = buf[bs:]


# One explicit file per corpus, not a config name + split slice: see
# text_batches's docstring for why the slice form downloads everything.
_CORPUS_CONFIGS = {
    "HuggingFaceFW/fineweb-edu": dict(
        data_files="sample/10BT/000_00000.parquet", field="text"),
    "HuggingFaceH4/ultrachat_200k": dict(
        data_files="data/train_sft-00000-of-00003-a3ecf92756993583.parquet",
        field="prompt"),
    # E7 bild_style: FLAN-style instruction/response pairs, one task per row.
    # `field` is a list here (see _load_corpus_docs) -- inputs+targets are
    # concatenated into one plain-text doc, since text_batches/prompt_batches
    # only ever consume a single string per document. builder="json": this
    # repo's `main` branch ships raw per-task .jsonl files (train/<task>_
    # train.jsonl), NOT parquet -- a first attempt guessed a parquet path
    # under "default/train/0000.parquet" by analogy with fineweb-edu/
    # ultrachat_200k, which raised FileNotFoundError: that path only exists
    # on the auto-converted refs/convert/parquet ref, not on main, and hf://
    # with no revision resolves against main. Verified via the repo's real
    # file listing (HF API) before picking this path.
    #
    # cnn_dailymail, not ag_news_subset: a first choice of the ag_news task
    # shard caused a real ~23-hour silent hang in E7's first full-sweep
    # submission (a run) -- ag_news's classification-style
    # inputs+targets top out at 273 tokens, so text_batches' exact-length
    # filter (`if len(t) == L`) could never match at --seqlen 512 and looped
    # forever with 0/n accepted (see text_batches' new draw-count guard,
    # added after this was found, which now fails loudly instead of hanging
    # for any corpus/L combination this narrow). cnn_dailymail is also a
    # better match for bild_style's actual regime (task-specific
    # summarization fine-tuning, not short classification) and has documents
    # long enough for L=512 (67% of a 500-doc sample reached >=512 tokens
    # after inputs+targets concatenation, vs 0% for ag_news).
    "Muennighoff/flan": dict(
        builder="json",
        data_files="train/cnn_dailymail_10templates_train.jsonl",
        field=["inputs", "targets"]),
    # E7 toolcall_style: tool-calling conversations already serialized as one
    # string (role markers like USER:/ASSISTANT:/FUNCTION CALL: embedded in
    # the text itself, not a list of turn dicts). builder="json": this repo's
    # `main` branch ships one single JSON file, not parquet -- same class of
    # mistake as Muennighoff/flan above (guessed a parquet path that only
    # exists on refs/convert/parquet), fixed the same way.
    "glaiveai/glaive-function-calling-v2": dict(
        builder="json",
        data_files="glaive-function-calling-v2.json", field="chat"),
}
_corpus_cache = {}


def _load_corpus_docs(corpus, seed):
    """Downloads (once, cached by `datasets` on disk) exactly one shard of
    `corpus` and returns a plain list of strings (field name from
    _CORPUS_CONFIGS -- "text" for fineweb-edu, "prompt" for ultrachat_200k),
    shuffled once with `seed`. Cached in-process too (`_corpus_cache`) so
    repeated calls within one script run -- e.g. probe() called many times
    over training -- don't re-read the parquet file from disk each time.

    Uses a generic builder ("parquet" or "json", per-corpus in
    _CORPUS_CONFIGS's `builder` key, default "parquet") against an hf:// URI,
    NOT load_dataset(corpus, data_files=...): the latter still resolves
    `corpus` to its own registered dataset config/loading script, which for a
    multi-split dataset (ultrachat_200k declares train_sft/train_gen/
    test_sft/test_gen) makes `datasets` verify that ALL declared splits are
    present and raise ExpectedMoreSplitsError, even though only one
    data_files entry was requested and no other split was ever asked for.
    fineweb-edu's simpler config didn't hit this, which is why it wasn't
    caught until e4_onpolicy.py's ultrachat_200k usage was actually run.
    The generic builder reads the named file directly and has no such split
    bookkeeping to satisfy.

    `data_files` must be a real path on the repo's `main` branch: bare hf://
    URIs (no revision pin) resolve against main, not the auto-converted
    refs/convert/parquet ref some HF datasets tools list by default -- a real
    mistake made wiring E7 (Muennighoff/flan, glaiveai/glaive-function-
    calling-v2): both were first pointed at a plausible-looking
    "default/train/0000.parquet" path that exists only on refs/convert/
    parquet, raising FileNotFoundError against main. Fixed by checking each
    repo's actual main-branch file listing (HF's /api/datasets/<repo> siblings
    list) and using its real raw file (.jsonl or .json, hence builder="json"
    for both) instead of assuming every corpus ships parquet on main."""
    key = (corpus, seed)
    if key in _corpus_cache:
        return _corpus_cache[key]
    from datasets import load_dataset
    cfg = _CORPUS_CONFIGS[corpus]
    ds = load_dataset(cfg.get("builder", "parquet"),
                      data_files=f"hf://datasets/{corpus}/{cfg['data_files']}",
                      split="train")
    field = cfg["field"]
    if isinstance(field, list):
        # e.g. flan's inputs+targets: concatenate per-row into one doc rather
        # than treating them as separate documents, so downstream token-length
        # truncation (text_batches) sees the whole instruction+response pair.
        cols = [ds[f] for f in field]
        docs = ["\n".join(parts) for parts in zip(*cols)]
    else:
        docs = list(ds[field])
    random.Random(seed).shuffle(docs)
    _corpus_cache[key] = docs
    return docs


def prompt_batches(tok, n, bs=4, L=256, seed=0,
                   corpus="HuggingFaceH4/ultrachat_200k"):
    """Yields `n` batches of `bs` PADDED (not exact-length) tokenized
    prompts, from the same locally-cached-shard approach as text_batches --
    see that function's docstring for why streaming + next(iter(ds)) is
    unsafe (network stalls) AND wrong here specifically: e4_onpolicy.py's
    original _prompt_batch called `next(iter(ds))` fresh on every single
    call, which creates a NEW iterator every time and always returns the
    dataset's first bs prompts -- every "on-policy" step was training on the
    literal same few prompts repeatedly, not real corpus diversity, on top
    of the same per-call streaming-stall risk. Padding (not truncation to
    exactly L) is appropriate for prompts, unlike text_batches' fixed-length
    training sequences -- callers pass the tokenized, padded batch directly
    to student.generate()."""
    docs = _load_corpus_docs(corpus, seed)
    rng = random.Random(seed)
    for _ in range(n):
        batch_docs = [docs[rng.randrange(len(docs))] for _ in range(bs)]
        yield tok(batch_docs, return_tensors="pt", padding=True,
                  truncation=True, max_length=L)


def record(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload["_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return payload["_hash"]
