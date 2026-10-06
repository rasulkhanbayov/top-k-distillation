"""
Cache formats.

Two writers, one per arm of the central comparison in E3:

  HiddenStateCache : d values per supervised position, exact, no index array.
  TopKCache        : k values plus k indices, lossy, the current standard.

Both are sharded flat files with a JSON sidecar so that E6 can report real bytes
on disk rather than a theoretical figure. Sizes are the numbers in Table 1 of the
paper; do not recompute them by hand, read them off `stats()`.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import numpy as np

__all__ = ["HiddenStateCache", "TopKCache", "quantize", "dequantize", "DTYPES"]

DTYPES = {
    "fp32": np.float32,
    "fp16": np.float16,
    "bf16": "bf16",   # stored as uint16, handled below
    "int8": "int8",   # per-position scale
}


def quantize(x, kind):
    """Returns (payload, scale_or_None). x is (n, d) float32."""
    if kind == "fp32":
        return x.astype(np.float32), None
    if kind == "fp16":
        return x.astype(np.float16), None
    if kind == "bf16":
        u = x.astype(np.float32).view(np.uint32)
        # round-to-nearest-even truncation to the top 16 bits
        rounded = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
        return rounded, None
    if kind == "int8":
        scale = np.abs(x).max(axis=1, keepdims=True) / 127.0
        scale = np.maximum(scale, 1e-12)
        return np.round(x / scale).astype(np.int8), scale.astype(np.float32)
    raise ValueError(kind)


def dequantize(payload, scale, kind):
    if kind == "fp32":
        return payload.astype(np.float32)
    if kind == "fp16":
        return payload.astype(np.float32)
    if kind == "bf16":
        u = (payload.astype(np.uint32) << 16)
        return u.view(np.float32)
    if kind == "int8":
        return payload.astype(np.float32) * scale
    raise ValueError(kind)


@dataclass
class CacheMeta:
    kind: str
    dtype: str
    d: int = 0
    k: int = 0
    n_positions: int = 0
    n_shards: int = 0
    bytes_on_disk: int = 0
    teacher: str = ""
    softcap: float = 0.0


class _Base:
    def __init__(self, root, meta=None):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.meta = meta
        self._shard = 0

    def _path(self, name):
        return os.path.join(self.root, name)

    def finalize(self):
        total = 0
        for f in os.listdir(self.root):
            if f.endswith(".npy") or f.endswith(".bin"):
                total += os.path.getsize(self._path(f))
        self.meta.bytes_on_disk = total
        self.meta.n_shards = self._shard
        with open(self._path("meta.json"), "w") as fh:
            json.dump(asdict(self.meta), fh, indent=2)
        return self.meta

    def stats(self):
        with open(self._path("meta.json")) as fh:
            m = json.load(fh)
        m["bytes_per_position"] = m["bytes_on_disk"] / max(m["n_positions"], 1)
        return m


class HiddenStateCache(_Base):
    """d values per position. Exact under reconstruct.reconstruct_logits."""

    def __init__(self, root, d=None, dtype="bf16", teacher="", softcap=0.0):
        super().__init__(root, CacheMeta("hidden_state", dtype, d=d or 0,
                                         teacher=teacher, softcap=softcap))

    def write(self, g):
        """g : (n, d) float32 teacher hidden states, post final norm."""
        g = np.asarray(g, dtype=np.float32)
        if self.meta.d == 0:
            self.meta.d = g.shape[1]
        payload, scale = quantize(g, self.meta.dtype)
        np.save(self._path(f"g_{self._shard:06d}.npy"), payload)
        if scale is not None:
            np.save(self._path(f"s_{self._shard:06d}.npy"), scale)
        self.meta.n_positions += g.shape[0]
        self._shard += 1

    def read(self, shard):
        payload = np.load(self._path(f"g_{shard:06d}.npy"))
        sp = self._path(f"s_{shard:06d}.npy")
        scale = np.load(sp) if os.path.exists(sp) else None
        return dequantize(payload, scale, self.meta.dtype)


class TopKCache(_Base):
    """k log-probabilities plus k int32 indices. The baseline being argued against."""

    def __init__(self, root, k, dtype="fp16", teacher=""):
        super().__init__(root, CacheMeta("topk", dtype, k=k, teacher=teacher))

    def write(self, logprobs, indices):
        lp, _ = quantize(np.asarray(logprobs, np.float32), self.meta.dtype)
        np.save(self._path(f"lp_{self._shard:06d}.npy"), lp)
        np.save(self._path(f"ix_{self._shard:06d}.npy"),
                np.asarray(indices, np.int32))
        self.meta.n_positions += lp.shape[0]
        self._shard += 1

    def read(self, shard):
        lp = np.load(self._path(f"lp_{shard:06d}.npy"))
        ix = np.load(self._path(f"ix_{shard:06d}.npy"))
        return dequantize(lp, None, self.meta.dtype), ix
