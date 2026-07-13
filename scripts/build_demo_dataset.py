#!/usr/bin/env python3
"""
Build the curated FaceSentinel demo dataset from a public PAN-card OCR dataset.

The repo already ships a pre-built dataset in ``app/demo/dataset/`` — this script
regenerates it (or builds one from your own copy of the source data). It:

  1. detects + embeds the primary face on every card (ArcFace R50, ONNX)
  2. clusters embeddings to find the SAME person across DIFFERENT cards
     (these give genuinely-different photos of one person)
  3. OCRs the annotated name / PAN / DOB boxes for real identities
  4. selects a gallery of distinct people + planted probes:
       - FRAUD     : a different photo of an enrolled person under a *fictional* PAN
       - DUPLICATE : a different photo of an enrolled person, same PAN
       - CLEAR     : a distinct person not enrolled
  5. writes detectable face crops + a manifest + precomputed embeddings.

Source: Roboflow Universe `panocr/ocr-qaxqg` (CC BY 4.0), TensorFlow export
(a folder of *.jpg + `_annotations.csv` with name/father/dob/pan boxes).

Usage:
    python scripts/build_demo_dataset.py --src /path/to/ocr.v6i.tensorflow/train \
        [--gallery 20] [--frauds 3] [--out app/demo/dataset]

Requires the full model runtime (requirements.txt) + the `tesseract` binary.
Fraud-applicant personas are fictional; enrolled identities are real fields from
the public dataset. Illustrative / hackathon use only.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from app.core.src.face_analysis import FaceAnalysis  # noqa: E402

ATTRIBUTION = ("Faces & identity fields derive from the public 'ocr' PAN-card dataset "
               "(Roboflow Universe, panocr/ocr-qaxqg, CC BY 4.0). Fraud-applicant personas "
               "are fictional. Demo/illustrative use only.")
EMBEDDING_MODEL = "insightface ArcFace w600k_r50 (512-d)"
CITIES = ["Mumbai", "Delhi", "Kolkata", "Chennai", "Pune", "Hyderabad", "Ahmedabad", "Jaipur",
          "Lucknow", "Nagpur", "Bhubaneswar", "Kochi", "Patna", "Indore", "Ranchi", "Guwahati"]
FRAUD_PERSONAS = [
    {"full_name": "RUKSANA PARVEEN", "city": "Kolkata"},
    {"full_name": "SUNITA DEVI", "city": "Patna"},
    {"full_name": "IMRAN QURESHI", "city": "Mumbai"},
    {"full_name": "FIROZ ANSARI", "city": "Ranchi"},
]
_PAN_RE = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")


# --------------------------- image / ocr helpers ---------------------------
def load_bgr(path, max_side=2200):
    img = cv2.imread(path)
    if img is None:
        from PIL import Image
        img = np.ascontiguousarray(np.array(Image.open(path).convert("RGB"))[:, :, ::-1])
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def ocr(crop, psm=7):
    if crop is None or crop.size == 0:
        return ""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as t:
        cv2.imwrite(t.name, g)
        p = t.name
    try:
        return subprocess.run(["tesseract", p, "-", "--psm", str(psm)],
                              capture_output=True, text=True, timeout=25).stdout.strip()
    except Exception:
        return ""
    finally:
        os.unlink(p)


def clean_name(s):
    return re.sub(r"\s+", " ", re.sub(r"[^A-Za-z .]", " ", s).strip()).upper()


def clean_dob(s):
    m = re.search(r"(\d{2})[/\-.](\d{2})[/\-.](\d{4})", s)
    return f"{m.group(1)}/{m.group(2)}/{m.group(3)}" if m else ""


def clean_pan(s):
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def coerce_pan(s, seed):
    s = clean_pan(s)
    if _PAN_RE.match(s):
        return s
    h = hashlib.sha256(seed.encode()).hexdigest().upper()
    letters = re.sub(r"[^A-Z]", "", h)
    digits = re.sub(r"[^0-9]", "", h)
    return letters[:5] + digits[:4] + letters[5:6]


def phone_for(seed):
    return "9" + str(int(hashlib.sha256(seed.encode()).hexdigest(), 16) % 900000000 + 100000000)


def largest(faces):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def square_crop(img, bbox, margin, out=320):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * (0.5 + margin)
    H, W = img.shape[:2]
    crop = img[int(max(0, cy - half)):int(min(H, cy + half)),
               int(max(0, cx - half)):int(min(W, cx + half))]
    return None if crop.size == 0 else cv2.resize(crop, (out, out), interpolation=cv2.INTER_AREA)


# ------------------------------- pipeline ----------------------------------
def base_name(fn):
    return re.sub(r"\.rf\..*$", "", fn)


def scan(fa, src_dir):
    """Detect + embed the primary face on each (augmentation-deduped) card."""
    files = sorted(glob.glob(os.path.join(src_dir, "*.jpg")))
    seen = {}
    for f in files:
        seen.setdefault(base_name(os.path.basename(f)), f)
    uniq = sorted(seen.values())
    recs, embs = [], []
    for i, f in enumerate(uniq):
        img = load_bgr(f)
        try:
            faces = fa.get(img)
        except Exception:
            continue
        if not faces:
            continue
        face = largest(faces)
        bw, bh = face.bbox[2] - face.bbox[0], face.bbox[3] - face.bbox[1]
        if min(bw, bh) < 60 or face.det_score < 0.62:
            continue
        recs.append({"idx": len(recs), "src": os.path.basename(f), "base": base_name(os.path.basename(f)),
                     "facepx": [int(bw), int(bh)], "det": round(float(face.det_score), 3)})
        embs.append(np.asarray(face.normed_embedding, dtype=np.float32))
        if (i + 1) % 200 == 0:
            print(f"  scanned {i+1}/{len(uniq)} kept={len(recs)}", flush=True)
    return recs, (np.vstack(embs) if embs else np.zeros((0, 512), np.float32))


def clusters(recs, E, lo=0.60, hi=0.93):
    """Same-person groups across different cards, with genuinely different photos
    (pairwise cosine in [lo, hi] avoids exact re-scans)."""
    n = len(recs)
    S = E @ E.T
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a in range(n):
        for b in range(a + 1, n):
            if S[a, b] >= lo and recs[a]["base"] != recs[b]["base"]:
                parent[find(a)] = find(b)
    groups = defaultdict(list)
    for a in range(n):
        groups[find(a)].append(a)
    out = []
    for members in groups.values():
        if len(members) < 2 or len(members) > 8:
            continue
        if len({recs[m]["base"] for m in members}) < 2:
            continue
        ms = np.array(members)
        off = S[np.ix_(ms, ms)][np.triu_indices(len(ms), 1)]
        if off.min() < 0.45:
            continue
        out.append({"members": members, "smin": float(off.min()), "smax": float(off.max())})
    # prefer clusters with a genuinely different photo (moderate max sim), good size
    out.sort(key=lambda c: (c["smax"] <= hi, len(c["members"]), -c["smax"]), reverse=True)
    return out


def ident_of(fa_ann, src_dir, src):
    boxes = fa_ann.get(src, {})
    res = {"name": "", "dob": "", "pan": ""}
    if not boxes:
        return res
    img = load_bgr(os.path.join(src_dir, src))
    H, W = img.shape[:2]
    for cls, fn in (("name", clean_name), ("dob", clean_dob), ("pan", clean_pan)):
        if cls in boxes:
            x1, y1, x2, y2 = boxes[cls]
            res[cls] = fn(ocr(img[max(0, y1 - 3):min(H, y2 + 3), max(0, x1 - 3):min(W, x2 + 3)]))
    return res


def redetectable_crop(fa, src_dir, src):
    card = load_bgr(os.path.join(src_dir, src))
    faces = fa.get(card)
    if not faces:
        return None, None
    bbox = largest(faces).bbox
    for margin in (0.55, 0.9, 1.4, 2.2):
        crop = square_crop(card, bbox, margin)
        if crop is None:
            continue
        f2 = fa.get(crop)
        if f2:
            return crop, np.asarray(largest(f2).normed_embedding, dtype=np.float32)
    h, w = card.shape[:2]
    s = 640 / max(h, w)
    return cv2.resize(card, (int(w * s), int(h * s))), np.asarray(largest(faces).normed_embedding, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dataset train/ dir (jpgs + _annotations.csv)")
    ap.add_argument("--out", default=os.path.join(ROOT, "app/demo/dataset"))
    ap.add_argument("--gallery", type=int, default=20, help="# distinct singleton customers")
    ap.add_argument("--frauds", type=int, default=3, help="# FRAUD probes")
    args = ap.parse_args()

    ann = {}
    with open(os.path.join(args.src, "_annotations.csv")) as f:
        for r in csv.DictReader(f):
            ann.setdefault(r["filename"], {})[r["class"]] = (
                int(r["xmin"]), int(r["ymin"]), int(r["xmax"]), int(r["ymax"]))

    fa = FaceAnalysis(allowed_modules=["detection", "recognition"])
    fa.prepare(ctx_id=0)

    print("scanning faces…", flush=True)
    recs, E = scan(fa, args.src)
    print(f"kept {len(recs)} faces; clustering…", flush=True)
    cls = clusters(recs, E)
    in_cluster = {m for c in cls for m in c["members"]}

    need_clusters = args.frauds + 1  # + 1 DUPLICATE
    chosen = cls[:need_clusters]
    if len(chosen) < need_clusters:
        raise SystemExit(f"only found {len(chosen)} usable same-person clusters (need {need_clusters})")

    # distinct, quality, legibly-identified singletons for the gallery
    pool = [r for r in recs if r["idx"] not in in_cluster
            and min(r["facepx"]) >= 85 and r["det"] >= 0.72]
    pool.sort(key=lambda r: -(min(r["facepx"]) * r["det"]))
    singles, sing_emb = [], []
    for r in pool:
        e = E[r["idx"]]
        if sing_emb and max(float(e @ pe) for pe in sing_emb) > 0.45:
            continue
        idn = ident_of(ann, args.src, r["src"])
        if not _PAN_RE.match(coerce_pan(idn["pan"], idn["name"])) or len(idn["name"].split()) < 2:
            continue
        singles.append({**r, **idn})
        sing_emb.append(e)
        if len(singles) >= args.gallery + 1:  # +1 reserved for CLEAR
            break
    if len(singles) < args.gallery + 1:
        raise SystemExit(f"only found {len(singles)} legible singletons (need {args.gallery + 1})")

    os.makedirs(args.out, exist_ok=True)
    faces_dir = os.path.join(args.out, "faces")
    probes_dir = os.path.join(args.out, "probes")
    for d in (faces_dir, probes_dir):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)

    gallery, probes, vectors = [], [], {}
    gi = [0]

    def add_gallery(src, name, pan, dob):
        gi[0] += 1
        cid = f"IDBI-1000{gi[0]:02d}"
        crop, emb = redetectable_crop(fa, args.src, src)
        cv2.imwrite(os.path.join(faces_dir, f"{cid}.jpg"), crop)
        vectors[cid] = emb
        gallery.append({"customer_id": cid, "full_name": name, "id_type": "PAN", "id_number": pan,
                        "dob": dob or "", "phone": phone_for(cid), "city": CITIES[gi[0] % len(CITIES)],
                        "face": f"faces/{cid}.jpg"})
        return cid

    for s in singles[:args.gallery]:
        add_gallery(s["src"], s["name"], coerce_pan(s["pan"], s["name"]), s["dob"])

    cluster_enroll = {}
    for ci, c in enumerate(chosen):
        m = c["members"][0]
        idn = ident_of(ann, args.src, recs[m]["src"])
        nm = idn["name"] or f"CUSTOMER {ci}"
        cid = add_gallery(recs[m]["src"], nm, coerce_pan(idn["pan"], nm), idn["dob"])
        cluster_enroll[ci] = (cid, nm)

    pi = [0]

    def add_probe(src, label, verdict, identity, true_person, matches):
        pi[0] += 1
        pid = f"APP-9000{pi[0]:02d}"
        crop, emb = redetectable_crop(fa, args.src, src)
        cv2.imwrite(os.path.join(probes_dir, f"{pid}.jpg"), crop)
        vectors[f"probe::{pid}"] = emb
        probes.append({"probe_id": pid, "label": label, "expected_verdict": verdict,
                       "identity": {"customer_id": pid, "id_type": "PAN", **identity},
                       "true_person": true_person, "matches_customer_id": matches,
                       "face": f"probes/{pid}.jpg"})

    # FRAUD probes: different photo (member[1]) of an enrolled person, fictional identity
    for ci in range(args.frauds):
        m = chosen[ci]["members"][1]
        ecid, ename = cluster_enroll[ci]
        persona = FRAUD_PERSONAS[ci % len(FRAUD_PERSONAS)]
        add_probe(recs[m]["src"],
                  f"Same face as {ename.title()} — new PAN ({persona['full_name'].title()})",
                  "FRAUD_ALERT_DIFFERENT_IDENTITY",
                  {"full_name": persona["full_name"], "id_number": coerce_pan("", persona["full_name"] + "fraud"),
                   "phone": phone_for(persona["full_name"]), "dob": "", "city": persona["city"]},
                  ename.title(), ecid)

    # DUPLICATE probe: different photo, SAME identity
    dci = args.frauds
    m = chosen[dci]["members"][1]
    ecid, ename = cluster_enroll[dci]
    genr = next(g for g in gallery if g["customer_id"] == ecid)
    add_probe(recs[m]["src"], f"Re-KYC of {ename.title()} (same identity)", "DUPLICATE_SAME_IDENTITY",
              {"full_name": genr["full_name"], "id_number": genr["id_number"], "phone": genr["phone"],
               "dob": genr["dob"], "city": genr["city"]}, ename.title(), ecid)

    # CLEAR probe: the reserved distinct singleton, not enrolled
    cs = singles[args.gallery]
    add_probe(cs["src"], f"New applicant {cs['name'].title()} (not enrolled)", "CLEAR",
              {"full_name": cs["name"], "id_number": coerce_pan(cs["pan"], cs["name"]),
               "phone": phone_for(cs["name"]), "dob": cs["dob"], "city": "Surat"}, cs["name"].title(), None)

    json.dump({"attribution": ATTRIBUTION, "embedding_model": EMBEDDING_MODEL,
               "gallery": gallery, "probes": probes},
              open(os.path.join(args.out, "manifest.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(args.out, "embeddings.npz"),
                        **{k: v.astype(np.float32) for k, v in vectors.items()})
    print(f"\nDONE: gallery={len(gallery)} probes={len(probes)} -> {args.out}")
    for p in probes:
        print(f"  {p['probe_id']} {p['expected_verdict']:32} true={p['true_person']}")


if __name__ == "__main__":
    main()
