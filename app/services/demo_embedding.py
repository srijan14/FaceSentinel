"""
Deterministic *demo* face embeddings — a drop-in stand-in for the ArcFace model
so FaceSentinel runs end-to-end without the ~200 MB ONNX weights.

Why this exists
---------------
The real product embeds faces with InsightFace ArcFace (512-d). Downloading and
running that model is heavy and can fail on locked-down / free-tier hosts. For a
portable demo (and a cheap deployment link) we derive a **stable, unit-norm
512-d vector from the image bytes themselves**:

    embedding = normalize( RNG(seed = sha256(image_bytes)).standard_normal(dim) )

Properties that make it a faithful demo of the *pipeline*:

* Same image  -> identical vector -> cosine 1.0  -> similarity 1.0  (a match).
* Different images -> ~orthogonal vectors -> cosine ~0 -> similarity ~0.5
  (an "unrelated face", well below the review threshold).

So re-uploading an enrolled customer's photo under a *different* identity
reproduces the hero fraud case, and a brand-new photo reads as CLEAR — exactly
the behaviour the fraud engine expects, with zero model download.

This is clearly labelled everywhere as ``embedding_mode = "demo"``. It does NOT
recognise that two *different* photos of the same person are the same face — set
``EMBEDDING_MODE=insightface`` (with models present) for true face matching.
"""
from __future__ import annotations

import hashlib
from typing import List

import numpy as np


def demo_embedding_from_bytes(image_data: bytes, dim: int = 512) -> List[float]:
    """Return a deterministic, L2-normalised ``dim``-d embedding for these bytes."""
    digest = hashlib.sha256(image_data).digest()
    seed = int.from_bytes(digest[:8], "big", signed=False)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return vec.tolist()
