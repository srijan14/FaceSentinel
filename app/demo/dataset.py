"""
Loader for the bundled, curated FaceSentinel demo dataset.

The dataset (``app/demo/dataset/``) is built offline from the public PAN-card
OCR dataset by ``scripts/build_demo_dataset.py``. It contains:

    manifest.json      gallery + planted probes with real OCR'd identities
    faces/<cid>.jpg    enrolled gallery face crops (real people)
    probes/<pid>.jpg   planted probe face crops (a *different* photo)
    embeddings.npz     precomputed 512-d ArcFace vectors for every face

Two consumption paths share this data:
  * local model mode  -> faces are re-embedded live by ArcFace at seed/probe time
  * hosted read-only  -> precomputed vectors are used directly (no model needed)

Faces & identity fields come from a public CC BY 4.0 dataset; fraud-applicant
personas are fictional. Illustrative / demo use only. See ``manifest.json``
``attribution``.
"""
from __future__ import annotations

import functools
import json
import os
from typing import Dict, List, Optional

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(_DIR, "dataset")
MANIFEST_PATH = os.path.join(DATASET_DIR, "manifest.json")
EMB_PATH = os.path.join(DATASET_DIR, "embeddings.npz")


def available() -> bool:
    """True when the curated dataset (manifest + embeddings) is bundled."""
    return os.path.isfile(MANIFEST_PATH) and os.path.isfile(EMB_PATH)


@functools.lru_cache(maxsize=1)
def _manifest() -> Dict:
    with open(MANIFEST_PATH) as fh:
        return json.load(fh)


@functools.lru_cache(maxsize=1)
def _embeddings() -> Dict[str, np.ndarray]:
    with np.load(EMB_PATH) as z:
        return {k: z[k].astype(np.float32) for k in z.files}


def attribution() -> str:
    return _manifest().get("attribution", "")


def embedding_model() -> str:
    return _manifest().get("embedding_model", "insightface ArcFace (512-d)")


def _face_bytes(rel_path: str) -> bytes:
    with open(os.path.join(DATASET_DIR, rel_path), "rb") as fh:
        return fh.read()


def gallery() -> List[Dict]:
    """Enrolled customers. Each dict: identity fields + ``face_bytes`` + ``vector`` (list)."""
    embs = _embeddings()
    out = []
    for g in _manifest()["gallery"]:
        cid = g["customer_id"]
        out.append({
            **{k: g.get(k) for k in ("customer_id", "full_name", "id_type", "id_number", "phone", "dob", "city")},
            "face_bytes": _face_bytes(g["face"]),
            "vector": embs[cid].tolist(),
        })
    return out


def probes() -> List[Dict]:
    """Planted onboarding probes (each carries a *different* photo of a person)."""
    embs = _embeddings()
    out = []
    for p in _manifest()["probes"]:
        pid = p["probe_id"]
        out.append({
            "probe_id": pid,
            "label": p["label"],
            "expected_verdict": p["expected_verdict"],
            "identity": p["identity"],
            "true_person": p.get("true_person", ""),
            "matches_customer_id": p.get("matches_customer_id"),
            "face_bytes": _face_bytes(p["face"]),
            "vector": embs[f"probe::{pid}"].tolist(),
        })
    return out


def probe_by_id(probe_id: str) -> Optional[Dict]:
    for p in probes():
        if p["probe_id"] == probe_id:
            return p
    return None
