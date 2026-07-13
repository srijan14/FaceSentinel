#!/usr/bin/env python3
"""
Seed the sample-KYC gallery into the active vector database (Pinecone by default).

This talks to the *running FaceSentinel API*, so it enrols faces into whichever
backend the API is configured with — set ``VECTOR_BACKEND=pinecone`` (or just
``PINECONE_API_KEY``) on the API and this script indexes straight into Pinecone.

What it does
------------
1. Health-checks the API and prints the active backend + embedding mode.
2. Calls ``POST /v1/dedup/face/demo/seed`` to enrol ~16 fictional IDBI customers
   (with procedurally-drawn avatar faces).
3. Saves the returned onboarding *probes* (fraud / duplicate / clear) to
   ``data/fraud_probes/`` and writes ``data/fraud_manifest.csv`` so the Streamlit
   console's "Batch Probe Test" and planted-probe replay work out of the box.

Usage:
    # start the API first, e.g.:
    #   PINECONE_API_KEY=... VECTOR_BACKEND=pinecone uvicorn app.main:app --port 8000
    python scripts/seed_pinecone.py                 # seed + write manifest
    python scripts/seed_pinecone.py --reset         # purge & re-enrol first
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import sys

import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBE_DIR = os.path.join(REPO_ROOT, "data", "fraud_probes")
MANIFEST = os.path.join(REPO_ROOT, "data", "fraud_manifest.csv")


def main() -> int:
    ap = argparse.ArgumentParser(description="Seed the sample-KYC gallery into Pinecone via the API")
    ap.add_argument("--api-url", default=os.environ.get("API_URL", "http://localhost:8000"))
    ap.add_argument("--api-key", default=os.environ.get("API_KEY", "dev-local-key-change-me"))
    ap.add_argument("--reset", action="store_true", help="Purge & re-enrol existing sample customers")
    args = ap.parse_args()

    # 1) Health check
    try:
        h = requests.get(f"{args.api_url}/health", timeout=15).json()
        svc = h.get("services", {})
        print(f"API status     : {h.get('status')}")
        print(f"Vector backend : {svc.get('vector_backend')}")
        print(f"Embedding mode : {svc.get('embedding_mode')}")
        if not svc.get("overall", False):
            print("WARNING: API health is not fully green — attempting to seed anyway.")
    except Exception as e:
        print(f"ERROR: cannot reach API at {args.api_url} ({e}). Start the server first.")
        return 2

    # 2) Seed the gallery
    print("\nSeeding sample-KYC gallery ...")
    try:
        r = requests.post(
            f"{args.api_url}/v1/dedup/face/demo/seed",
            headers={"Authorization": args.api_key},
            data={"reset": "true" if args.reset else "false"},
            timeout=180,
        )
    except Exception as e:
        print(f"ERROR: seed request failed: {e}")
        return 2
    if r.status_code != 200:
        print(f"ERROR: seed failed ({r.status_code}): {r.text[:300]}")
        return 2

    res = r.json()
    print(f"  enrolled       : {res.get('seeded')}")
    print(f"  skipped (exist): {res.get('skipped_existing')}")
    print(f"  failed         : {res.get('failed')}")
    if res.get("errors"):
        for e in res["errors"]:
            print(f"    ! {e}")
    print(f"  gallery size   : {res.get('gallery_size')}  (backend={res.get('vector_backend')})")

    # 3) Write probe images + manifest for the console
    os.makedirs(PROBE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    rows = []
    for i, p in enumerate(res.get("probes", [])):
        img = base64.b64decode(p["image_b64"])
        dst = os.path.join(PROBE_DIR, f"probe_{i:02d}_{p['expected_verdict'].lower()}.jpg")
        with open(dst, "wb") as fh:
            fh.write(img)
        rows.append({
            "probe_image": dst,
            "true_person": p.get("true_person", ""),
            "probe_identity": json.dumps(p["identity"]),
            "expected_verdict": p["expected_verdict"],
        })
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["probe_image", "true_person", "probe_identity", "expected_verdict"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {len(rows)} probes -> {PROBE_DIR}")
    print(f"Manifest        -> {MANIFEST}")
    print("\nDone. Open the console:  streamlit run ui/app.py")
    print("Then use the '🎯 Onboarding Check' tab and pick a planted probe, "
          "or '🚨 Batch Probe Test' to run them all.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
