#!/usr/bin/env python3
"""
Seed the face-dedup gallery from the LFW dataset and synthesize fraud scenarios.

For each person we synthesize a plausible bank identity (name + a fake PAN-style
id_number + phone + customer_id) and enrol one face into the gallery via the
running API's ``/store`` endpoint.

Fraud synthesis (``--fraud-pairs N``): for N people that have >= 2 photos we
enrol photo #1 under identity A, then save photo #2 (the *same face*) as an
onboarding *probe* tagged with a **different** identity B. Running ``/check`` on
that probe should return ``FRAUD_ALERT_DIFFERENT_IDENTITY`` — the hero demo.

We also emit a few genuine "new applicant" probes (people NOT in the gallery,
expected ``CLEAR``) and legitimate re-KYC probes (same person, same identity,
expected ``DUPLICATE_SAME_IDENTITY``). Everything is recorded in
``data/fraud_manifest.csv``.

Usage:
    # start the API first (uvicorn app.main:app ...), then:
    python scripts/ingest_lfw.py --limit 200 --fraud-pairs 3
    python scripts/ingest_lfw.py --images-dir ./my_faces   # offline fallback
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from datetime import datetime, timezone

import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBE_DIR = os.path.join(REPO_ROOT, "data", "fraud_probes")
MANIFEST = os.path.join(REPO_ROOT, "data", "fraud_manifest.csv")

FRAUD = "FRAUD_ALERT_DIFFERENT_IDENTITY"
DUPLICATE = "DUPLICATE_SAME_IDENTITY"
CLEAR = "CLEAR"


def synth_identity(name: str, salt: str = "") -> dict:
    """Deterministically synthesize a bank identity from a person's name."""
    h = hashlib.sha256((name + "|" + salt).encode()).hexdigest()
    letters = "".join(chr(65 + (int(h[i:i + 2], 16) % 26)) for i in range(0, 10, 2))
    digits = f"{int(h[10:18], 16) % 10000:04d}"
    last = chr(65 + (int(h[18:20], 16) % 26))
    pan = f"{letters}{digits}{last}"                 # e.g. ABCDE1234F
    phone = "9" + f"{int(h[20:30], 16) % 1000000000:09d}"
    cust = "CUST-" + h[:8].upper()
    return {
        "full_name": name.replace("_", " "),
        "id_type": "PAN",
        "id_number": pan,
        "phone": phone,
        "customer_id": cust,
    }


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_people(min_faces: int, images_dir: str | None):
    """Return {person_name: [image_path, ...]} either from LFW or a local dir."""
    if images_dir:
        base = images_dir
        people = {}
        for name in sorted(os.listdir(base)):
            d = os.path.join(base, name)
            if os.path.isdir(d):
                imgs = sorted(os.path.join(d, f) for f in os.listdir(d)
                              if f.lower().endswith((".jpg", ".jpeg", ".png")))
                if imgs:
                    people[name] = imgs
        return people
    # LFW path
    from sklearn.datasets import fetch_lfw_people, get_data_home
    print(f"Fetching LFW (min_faces_per_person={min_faces}) — first run downloads ~200MB...")
    fetch_lfw_people(min_faces_per_person=min_faces, color=True, resize=0.4,
                     download_if_missing=True)
    home = os.path.join(get_data_home(), "lfw_home", "lfw_funneled")
    people = {}
    for name in sorted(os.listdir(home)):
        d = os.path.join(home, name)
        if not os.path.isdir(d):
            continue
        imgs = sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(".jpg"))
        if imgs:
            people[name] = imgs
    return people


def enroll(api: str, key: str, txn: str, img_path: str, identity: dict, timeout=60):
    """Enrol a face via POST /store. Returns (ok, message)."""
    meta = dict(identity)
    meta["image_path"] = img_path
    meta["created_on"] = now_iso()
    with open(img_path, "rb") as fh:
        files = {"image": (os.path.basename(img_path), fh, "image/jpeg")}
        data = {"transaction_id": txn, "metadata": json.dumps(meta)}
        r = requests.post(f"{api}/v1/dedup/face/store", headers={"Authorization": key},
                          files=files, data=data, timeout=timeout)
    if r.status_code == 200:
        return True, "ok"
    try:
        return False, r.json().get("error", {}).get("message", r.text)[:120]
    except Exception:
        return False, f"HTTP {r.status_code}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Seed the dedup gallery from LFW + synthesize fraud")
    ap.add_argument("--api-url", default=os.environ.get("API_URL", "http://localhost:8000"))
    ap.add_argument("--api-key", default=os.environ.get("API_KEY", "dev-local-key-change-me"))
    ap.add_argument("--limit", type=int, default=200, help="Max people to enrol into the gallery")
    ap.add_argument("--min-faces", type=int, default=2, help="LFW min_faces_per_person filter")
    ap.add_argument("--fraud-pairs", type=int, default=3, help="Same-face/different-identity probes to plant")
    ap.add_argument("--clear-probes", type=int, default=2, help="Genuine new-applicant probes (expected CLEAR)")
    ap.add_argument("--dup-probes", type=int, default=1, help="Legit re-KYC probes (expected DUPLICATE)")
    ap.add_argument("--images-dir", default=None, help="Offline: ingest local <dir>/<person>/*.jpg instead of LFW")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(PROBE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)

    # Health check
    try:
        h = requests.get(f"{args.api_url}/health", timeout=10).json()
        if not h.get("services", {}).get("overall", False):
            print(f"WARNING: API health is not fully green: {h}")
    except Exception as e:
        print(f"ERROR: cannot reach API at {args.api_url} ({e}). Start the server first.")
        return 2

    people = load_people(args.min_faces, args.images_dir)
    names = list(people.keys())
    rng.shuffle(names)
    if not names:
        print("No people found. Check the dataset / --images-dir.")
        return 2
    print(f"Found {len(names)} people with photos.")

    multi = [n for n in names if len(people[n]) >= 2]
    # Reserve people for fraud pairs, dup probes, and clear probes.
    fraud_people = multi[:args.fraud_pairs]
    dup_people = multi[args.fraud_pairs:args.fraud_pairs + args.dup_probes]
    reserved_clear = set(names[-args.clear_probes:]) if args.clear_probes else set()

    manifest_rows = []
    enrolled = skipped = 0

    # 1) Enrol the gallery (one face per person), excluding the reserved-clear people.
    gallery_names = [n for n in names if n not in reserved_clear][:args.limit]
    print(f"Enrolling up to {len(gallery_names)} gallery faces via {args.api_url} ...")
    for i, name in enumerate(gallery_names):
        identity = synth_identity(name)
        txn = f"GAL-{i:05d}"
        ok, msg = enroll(args.api_url, args.api_key, txn, people[name][0], identity)
        if ok:
            enrolled += 1
        else:
            skipped += 1
            if skipped <= 5:
                print(f"  skip {name}: {msg}")
        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{len(gallery_names)} processed (enrolled={enrolled}, skipped={skipped})")

    # 2) Fraud probes: same face as an enrolled gallery person, but a DIFFERENT identity.
    for j, name in enumerate(fraud_people):
        if name in reserved_clear or name not in gallery_names:
            continue
        probe_img = people[name][1]           # a different photo of the same person
        fake_name = f"Fraudster {j+1}"
        identity_b = synth_identity(fake_name, salt="FRAUD")   # different id_number + name
        dst = os.path.join(PROBE_DIR, f"fraud_{j:02d}.jpg")
        with open(probe_img, "rb") as s, open(dst, "wb") as d:
            d.write(s.read())
        manifest_rows.append({
            "probe_image": dst, "true_person": name.replace("_", " "),
            "probe_identity": json.dumps(identity_b),
            "expected_verdict": FRAUD,
        })

    # 3) Duplicate probes: same person, SAME identity (legit re-KYC).
    for k, name in enumerate(dup_people):
        if name not in gallery_names:
            continue
        probe_img = people[name][1]
        identity_same = synth_identity(name)   # identical identity to the gallery record
        dst = os.path.join(PROBE_DIR, f"dup_{k:02d}.jpg")
        with open(probe_img, "rb") as s, open(dst, "wb") as d:
            d.write(s.read())
        manifest_rows.append({
            "probe_image": dst, "true_person": name.replace("_", " "),
            "probe_identity": json.dumps(identity_same),
            "expected_verdict": DUPLICATE,
        })

    # 4) Clear probes: brand-new people not present in the gallery.
    for m, name in enumerate(sorted(reserved_clear)):
        probe_img = people[name][0]
        identity_new = synth_identity(name)
        dst = os.path.join(PROBE_DIR, f"clear_{m:02d}.jpg")
        with open(probe_img, "rb") as s, open(dst, "wb") as d:
            d.write(s.read())
        manifest_rows.append({
            "probe_image": dst, "true_person": name.replace("_", " "),
            "probe_identity": json.dumps(identity_new),
            "expected_verdict": CLEAR,
        })

    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["probe_image", "true_person", "probe_identity", "expected_verdict"])
        w.writeheader()
        w.writerows(manifest_rows)

    print("\n=== Ingestion summary ===")
    print(f"  Gallery enrolled : {enrolled}")
    print(f"  Skipped (no face): {skipped}")
    print(f"  Fraud probes     : {sum(1 for r in manifest_rows if r['expected_verdict'] == FRAUD)}")
    print(f"  Duplicate probes : {sum(1 for r in manifest_rows if r['expected_verdict'] == DUPLICATE)}")
    print(f"  Clear probes     : {sum(1 for r in manifest_rows if r['expected_verdict'] == CLEAR)}")
    print(f"  Manifest         : {MANIFEST}")
    print(f"  Probe images     : {PROBE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
