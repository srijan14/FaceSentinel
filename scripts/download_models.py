#!/usr/bin/env python3
"""
Download the InsightFace ONNX weights the app needs into ``models/``.

The vendored ``FaceAnalysis`` (app/core/src/face_analysis.py) simply globs
``models/*.onnx`` and routes each file to a detector / recognizer by its tensor
shape, so we only need two files:

    * a SCRFD face **detector**      -> models/det_10g.onnx   (buffalo_l)
    * an ArcFace face **recognizer** -> models/w600k_r50.onnx (buffalo_l, 512-d)

We fetch them from the community Hugging Face mirror ``immich-app/buffalo_l``
(the official ``storage.insightface.ai`` host is not reachable from many
networks, and the GitHub release asset returns 403). ``requests`` automatically
honours ``HTTPS_PROXY``/``HTTP_PROXY`` so this works behind an outbound proxy.

Usage:
    python scripts/download_models.py                 # buffalo_l (accurate, default)
    python scripts/download_models.py --pack buffalo_s  # smaller/faster (CPU)
    python scripts/download_models.py --force         # re-download even if present

If your network blocks Hugging Face, download the two ONNX files by hand and
drop them into ``models/`` — any correctly shaped SCRFD + ArcFace pair works.
"""
from __future__ import annotations

import argparse
import os
import sys

import requests

# repo_root/models  (this file lives in repo_root/scripts/)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(REPO_ROOT, "models")

HF = "https://huggingface.co/immich-app/{pack}/resolve/main/{kind}/model.onnx"

# expected_size is used as an integrity check; None = only sanity-check (> 1 MB).
PACKS = {
    "buffalo_l": {
        "detection": {"filename": "det_10g.onnx", "expected_size": 16_923_827},
        "recognition": {"filename": "w600k_r50.onnx", "expected_size": 174_383_860},
    },
    "buffalo_s": {
        "detection": {"filename": "det_500m.onnx", "expected_size": None},
        "recognition": {"filename": "w600k_mbf.onnx", "expected_size": None},
    },
}


def _ok_size(path: str, expected: int | None) -> bool:
    if not os.path.exists(path):
        return False
    size = os.path.getsize(path)
    if expected is not None:
        return size == expected
    return size > 1_000_000  # > 1 MB sanity check when exact size is unknown


def download_one(pack: str, kind: str, spec: dict, force: bool) -> None:
    dest = os.path.join(MODELS_DIR, spec["filename"])
    expected = spec["expected_size"]

    if not force and _ok_size(dest, expected):
        print(f"  [skip] {spec['filename']} already present "
              f"({os.path.getsize(dest):,} bytes)")
        return

    url = HF.format(pack=pack, kind=kind)
    print(f"  [get ] {spec['filename']}  <-  {url}")
    tmp = dest + ".part"
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done / total
                    print(f"\r         {done:,}/{total:,} bytes ({pct:5.1f}%)",
                          end="", flush=True)
        print()

    if expected is not None and os.path.getsize(tmp) != expected:
        os.remove(tmp)
        raise SystemExit(
            f"  [FAIL] {spec['filename']} size mismatch: "
            f"got {os.path.getsize(tmp) if os.path.exists(tmp) else 0}, "
            f"expected {expected}. Deleted partial file."
        )
    os.replace(tmp, dest)
    print(f"  [done] {spec['filename']} ({os.path.getsize(dest):,} bytes)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Download InsightFace ONNX weights into models/")
    ap.add_argument("--pack", choices=list(PACKS), default="buffalo_l",
                    help="Model pack (buffalo_l = accurate default, buffalo_s = faster on CPU)")
    ap.add_argument("--force", action="store_true", help="Re-download even if files exist")
    args = ap.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"Provisioning '{args.pack}' models into {MODELS_DIR}")
    for kind in ("detection", "recognition"):
        download_one(args.pack, kind, PACKS[args.pack][kind], args.force)

    present = sorted(f for f in os.listdir(MODELS_DIR) if f.endswith(".onnx"))
    print(f"\nONNX files in models/: {present}")
    if len(present) < 2:
        print("WARNING: expected at least a detector + recognizer .onnx", file=sys.stderr)
        return 1
    print("Models ready. The API will load these on startup.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
