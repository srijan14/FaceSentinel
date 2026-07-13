#!/usr/bin/env python3
"""Run every planted probe through POST /check and compare to the expected verdict.

Usage:  python scripts/run_probes.py [--api-url ...] [--api-key ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO_ROOT, "data", "fraud_manifest.csv")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-url", default=os.environ.get("API_URL", "http://localhost:8000"))
    ap.add_argument("--api-key", default=os.environ.get("API_KEY", "dev-local-key-change-me"))
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(MANIFEST)))
    ok = 0
    print(f"{'RES':4}{'probe':16}{'expected':34}{'actual':34}{'risk':>5} {'bestsim':>8}")
    print("-" * 105)
    for i, p in enumerate(rows):
        ident = json.loads(p["probe_identity"])
        with open(p["probe_image"], "rb") as f:
            files = {"image": (os.path.basename(p["probe_image"]), f, "image/jpeg")}
            data = {"transaction_id": f"VERIFY-{i}", "metadata": json.dumps(ident), "limit": str(args.limit)}
            r = requests.post(f"{args.api_url}/v1/dedup/face/check",
                              headers={"Authorization": args.api_key}, files=files, data=data, timeout=60)
        j = r.json() if r.status_code == 200 else {"verdict": f"HTTP{r.status_code}"}
        got, exp = j.get("verdict"), p["expected_verdict"]
        good = got == exp
        ok += good
        bm = j.get("best_match") or {}
        name = os.path.basename(p["probe_image"])
        print(f"{'OK ' if good else 'XX '} {name:16}{exp:34}{str(got):34}{str(j.get('risk_score','')):>5} "
              f"{str(bm.get('similarity_score','')):>8}   reasons={','.join(j.get('reason_codes', []))}")
    print("-" * 105)
    print(f"{ok}/{len(rows)} probes correct")
    return 0 if ok == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
