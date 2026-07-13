"""
FaceSentinel — Fraud De-Duplication Console
A bank-analyst UI over the Face De-Duplication API. Seed a sample KYC gallery,
screen applicants at onboarding, and visibly catch the same face applying under a
different identity (duplicate / synthetic-identity fraud).

Run:  streamlit run ui/app.py
"""
from __future__ import annotations

import os

# pyarrow (pulled in by st.dataframe) bundles a mimalloc allocator that segfaults
# in mi_thread_init on macOS when Streamlit serialises a dataframe on its
# ScriptRunner thread. Select Arrow's system allocator BEFORE pyarrow initialises.
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

# Load .env (non-overriding) so the console defaults to the SAME API_KEY / API_URL
# as the API when both are launched from the project directory — avoids a 401 from
# a key mismatch. Real env vars (e.g. on Render) always take precedence.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import base64
import csv
import json
import time

import requests
import streamlit as st

from api_client import DedupClient

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO_ROOT, "data", "fraud_manifest.csv")

st.set_page_config(page_title="FaceSentinel — Fraud De-Duplication Console",
                   page_icon="🛡️", layout="wide")

VERDICT_STYLE = {
    "CLEAR": {"label": "CLEAR — No duplicate found", "color": "#0B8043", "bg": "#E6F4EA", "emoji": "✅"},
    "REVIEW": {"label": "REVIEW — Manual review needed", "color": "#B26A00", "bg": "#FEF7E0", "emoji": "⚠️"},
    "DUPLICATE_SAME_IDENTITY": {"label": "DUPLICATE — Same person, same identity (re-KYC)",
                                "color": "#1A73E8", "bg": "#E8F0FE", "emoji": "🔁"},
    "FRAUD_ALERT_DIFFERENT_IDENTITY": {"label": "FRAUD ALERT — Same face, DIFFERENT identity",
                                       "color": "#C5221F", "bg": "#FCE8E6", "emoji": "🚨"},
}
ID_TYPES = ["PAN", "AADHAAR", "PASSPORT", "VOTER_ID", "DL"]


# ------------------------------- HTML helpers ------------------------------
def hero_html() -> str:
    return (
        '<div style="background:linear-gradient(100deg,#0B6E3B 0%,#12894a 45%,#E8590C 100%);'
        'padding:22px 26px;border-radius:14px;margin-bottom:6px;color:#fff;">'
        '<div style="font-size:30px;font-weight:800;letter-spacing:.3px;">🛡️ FaceSentinel</div>'
        '<div style="font-size:16px;font-weight:600;opacity:.95;margin-top:2px;">'
        'One face. One identity. — real-time facial de-duplication for KYC fraud prevention.</div>'
        '<div style="font-size:13.5px;opacity:.92;margin-top:8px;max-width:900px;line-height:1.5;">'
        'Traditional KYC checks each <i>document</i> in isolation, so it never sees the <b>same face</b> '
        'enrolling under a <b>different</b> PAN / Aadhaar — the root of duplicate-account, mule and '
        'synthetic-identity fraud. FaceSentinel matches every applicant face against the whole customer '
        'base and raises a verdict <b>before the account goes live.</b></div>'
        '</div>'
    )


def pipeline_html() -> str:
    steps = [
        ("📸", "Applicant face"),
        ("🧠", "Irreversible 512-d template"),
        ("🔎", "1:N vector search"),
        ("🪪", "Identity compare"),
        ("⚖️", "Verdict + risk score"),
    ]
    cells = []
    for i, (emo, label) in enumerate(steps):
        cells.append(
            f'<div style="flex:1;text-align:center;padding:8px 6px;">'
            f'<div style="font-size:20px;">{emo}</div>'
            f'<div style="font-size:12px;color:#3c4043;font-weight:600;margin-top:2px;">{label}</div></div>')
        if i < len(steps) - 1:
            cells.append('<div style="align-self:center;color:#9aa0a6;font-size:18px;">→</div>')
    return ('<div style="display:flex;align-items:stretch;background:#f8f9fa;border:1px solid #e8eaed;'
            'border-radius:10px;padding:4px 6px;margin:10px 0 4px 0;">' + "".join(cells) + '</div>')


def banner_html(verdict: str) -> str:
    s = VERDICT_STYLE.get(verdict, {"label": verdict, "color": "#3c4043", "bg": "#f1f3f4", "emoji": "•"})
    return (f'<div style="background:{s["bg"]};border-left:9px solid {s["color"]};'
            f'padding:18px 22px;border-radius:10px;margin:6px 0 4px 0;">'
            f'<div style="font-size:26px;font-weight:800;color:{s["color"]};line-height:1.2;">'
            f'{s["emoji"]} {s["label"]}</div></div>')


def risk_meter_html(score: int) -> str:
    color = "#0B8043" if score < 30 else "#B26A00" if score < 60 else "#C5221F"
    return (f'<div style="margin:10px 0 4px 0;">'
            f'<div style="display:flex;justify-content:space-between;font-size:13px;color:#5f6368;">'
            f'<span>Fraud risk score</span><span style="font-weight:800;color:{color};">{score}/100</span></div>'
            f'<div style="background:#e8eaed;border-radius:9px;height:16px;overflow:hidden;margin-top:3px;">'
            f'<div style="width:{max(2, score)}%;background:{color};height:16px;"></div></div></div>')


def chips_html(codes) -> str:
    if not codes:
        return ""
    chips = "".join(
        f'<span style="display:inline-block;background:#f1f3f4;color:#3c4043;border-radius:14px;'
        f'padding:4px 11px;margin:3px 4px 3px 0;font-size:12px;font-weight:600;">{c}</span>'
        for c in codes)
    return f'<div style="margin:6px 0 2px 0;">{chips}</div>'


def show_thumb(col, image_bytes, image_path, caption):
    if image_bytes:
        col.image(image_bytes, caption=caption, width=150)
    elif image_path and os.path.exists(image_path):
        col.image(image_path, caption=caption, width=150)
    else:
        col.markdown(f'<div style="width:150px;height:150px;background:#f1f3f4;border-radius:8px;'
                     f'display:flex;align-items:center;justify-content:center;color:#9aa0a6;">no image</div>'
                     f'<div style="font-size:12px;color:#5f6368;">{caption}</div>', unsafe_allow_html=True)


def render_result(res: dict, query_image_bytes=None):
    verdict = res.get("verdict", "?")
    st.markdown(banner_html(verdict), unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(risk_meter_html(int(res.get("risk_score", 0))), unsafe_allow_html=True)
    with c2:
        st.markdown(f"**Enrolled to gallery:** {'yes' if res.get('enrolled') else 'no — held'}  \n"
                    f"**Candidate matches:** {res.get('total_matches', 0)}")
    st.markdown("**Why:** " + " ".join(res.get("reason_codes", [])), help="Explainability reason codes")
    st.markdown(chips_html(res.get("reason_codes", [])), unsafe_allow_html=True)

    st.markdown("#### Applicant vs. gallery matches")
    cols = st.columns([1, 3])
    with cols[0]:
        if query_image_bytes:
            st.image(query_image_bytes, caption="Applicant", width=150)
        qi = res.get("query_identity", {})
        st.markdown(f"**{qi.get('full_name') or '—'}**  \n{qi.get('id_type') or ''} {qi.get('id_number') or ''}")
    with cols[1]:
        matches = res.get("matches", [])
        if not matches:
            st.info("No near-duplicate faces in the gallery.")
        for m in matches:
            sim = float(m.get("similarity_score", 0)) * 100
            same = m.get("identity_match")
            badge = ("<span style='color:#0B8043;font-weight:700;'>● SAME IDENTITY</span>" if same
                     else "<span style='color:#C5221F;font-weight:700;'>▲ DIFFERENT IDENTITY</span>")
            mc = st.columns([1, 3])
            show_thumb(mc[0], None, m.get("image_path"), f"{sim:.1f}% match")
            ident = m.get("identity", {})
            diffs = ", ".join(m.get("field_diffs", [])) or "—"
            mc[1].markdown(
                f"**{ident.get('full_name') or '—'}** · {ident.get('id_type') or ''} "
                f"{ident.get('id_number') or ''}  \n"
                f"txn `{m.get('transaction_id')}` · similarity **{sim:.1f}%** · {badge}  \n"
                f"<span style='font-size:12px;color:#5f6368;'>differing fields: {diffs}</span>",
                unsafe_allow_html=True)


# ------------------------------- Probe helpers -----------------------------
def probes_from_manifest():
    """Local fallback: read data/fraud_manifest.csv written by the seed scripts."""
    if not os.path.exists(MANIFEST):
        return []
    out = []
    with open(MANIFEST) as f:
        for row in csv.DictReader(f):
            img = None
            if row.get("probe_image") and os.path.exists(row["probe_image"]):
                with open(row["probe_image"], "rb") as fh:
                    img = fh.read()
            out.append({
                "label": f"{row.get('true_person', '')} · expect {row['expected_verdict']}",
                "expected_verdict": row["expected_verdict"],
                "identity": json.loads(row["probe_identity"]),
                "image_bytes": img,
                "true_person": row.get("true_person", ""),
            })
    return out


def get_probes(client):
    """Planted probes: prefer ones cached from the API, else the local manifest."""
    if st.session_state.get("probes"):
        return st.session_state["probes"]
    try:
        data = client.demo_probes()
        probes = []
        for p in data.get("probes", []):
            probes.append({
                "label": p["label"],
                "expected_verdict": p["expected_verdict"],
                "identity": p["identity"],
                "image_bytes": base64.b64decode(p["image_b64"]) if p.get("image_b64") else None,
                "true_person": p.get("true_person", ""),
            })
        if probes:
            st.session_state["probes"] = probes
            return probes
    except Exception:
        pass
    return probes_from_manifest()


# ----------------------------- Sidebar -------------------------------------
st.sidebar.markdown("## 🛡️ FaceSentinel")
st.sidebar.caption("Identity De-Duplication Security Fabric")
api_url = st.sidebar.text_input("API base URL", value=os.environ.get("API_URL", "http://localhost:8000"))
api_key = st.sidebar.text_input("API key", value=os.environ.get("API_KEY", "dev-local-key-change-me"), type="password")
limit = st.sidebar.slider("Max candidate matches", 1, 20, 8)
client = DedupClient(api_url, api_key)

health, stats = {}, {}
api_ok = False
try:
    health = client.health()
    stats = client.stats()
    api_ok = health.get("services", {}).get("overall", False)
except Exception as e:
    st.sidebar.error(f"API unreachable: {e}")

svc = health.get("services", {})
backend = (stats.get("vector_backend") or svc.get("vector_backend") or "—")
emode = (stats.get("embedding_mode") or svc.get("embedding_mode") or "—")
gallery_size = stats.get("gallery_size", "?")

st.sidebar.markdown(f"{'🟢' if api_ok else '🔴'} **API {'healthy' if api_ok else 'unhealthy'}**")
emode_label = "demo" if emode == "demo" else ("production" if emode == "insightface" else str(emode))
st.sidebar.markdown(
    f"**Vector DB:** `{backend}`  \n"
    f"**Embedding:** `{emode_label}`"
    + ("  ·  _demo (no model)_" if emode == "demo" else "  ·  _full engine_" if emode == "insightface" else "")
)
st.sidebar.markdown(f"**Gallery:** {gallery_size} faces")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🌱 Demo data")
if st.sidebar.button("Seed sample KYC gallery", use_container_width=True, type="primary"):
    with st.spinner("Enrolling fictional IDBI customers into the vector DB…"):
        try:
            r = client.seed_demo(reset=False)
            if r.status_code == 200:
                data = r.json()
                st.session_state["probes"] = [{
                    "label": p["label"], "expected_verdict": p["expected_verdict"],
                    "identity": p["identity"],
                    "image_bytes": base64.b64decode(p["image_b64"]) if p.get("image_b64") else None,
                    "true_person": p.get("true_person", ""),
                } for p in data.get("probes", [])]
                st.sidebar.success(f"Seeded {data.get('seeded')} (skipped {data.get('skipped_existing')} existing). "
                                   f"Gallery: {data.get('gallery_size')}.")
                st.rerun()
            else:
                st.sidebar.error(f"{r.status_code}: {r.text[:200]}")
        except Exception as ex:
            st.sidebar.error(f"Seed failed: {ex}")
if st.sidebar.button("Reset & reseed", use_container_width=True):
    with st.spinner("Purging and re-enrolling sample customers…"):
        try:
            r = client.seed_demo(reset=True)
            if r.status_code == 200:
                st.session_state.pop("probes", None)
                st.sidebar.success("Gallery reset & reseeded.")
                st.rerun()
            else:
                st.sidebar.error(f"{r.status_code}: {r.text[:200]}")
        except Exception as ex:
            st.sidebar.error(f"Reset failed: {ex}")

# ------------------------------- Header ------------------------------------
st.markdown(hero_html(), unsafe_allow_html=True)
st.markdown(pipeline_html(), unsafe_allow_html=True)

# KPI row
k1, k2, k3, k4 = st.columns(4)
k1.metric("Gallery faces", gallery_size)
k2.metric("Vector DB", str(backend).title())
k3.metric("Embedding", "Demo" if emode == "demo" else ("Production" if emode == "insightface" else str(emode)))
k4.metric("API", "Healthy" if api_ok else "Down")

empty_gallery = (isinstance(gallery_size, int) and gallery_size == 0)
if empty_gallery:
    st.warning("👈 The gallery is empty. Click **Seed sample KYC gallery** in the sidebar to load "
               "16 fictional customers and planted fraud probes, then try the **Onboarding Check** below.")

tab_check, tab_enroll, tab_batch, tab_bench, tab_about = st.tabs(
    ["🎯 Onboarding Check", "➕ Enroll Customer", "🚨 Batch Probe Test", "📊 Benchmarks", "ℹ️ About"])

# ----------------------------- Onboarding Check ----------------------------
with tab_check:
    st.markdown("Screen a new applicant against every enrolled face. If the same face already "
                "exists under a **different** government identity, it is flagged as fraud.")
    probes = get_probes(client)
    probe_map = {p["label"]: p for p in probes}

    src = st.radio("Applicant image source", ["Planted demo probe", "Upload"], horizontal=True)
    img_bytes, img_name, preset = None, "applicant.jpg", {}
    if src == "Planted demo probe" and probe_map:
        sel = st.selectbox("Choose a probe", list(probe_map.keys()))
        p = probe_map[sel]
        img_bytes = p.get("image_bytes")
        preset = p.get("identity", {})
        st.caption(f"Expected verdict for this probe: **{p['expected_verdict']}**")
    elif src == "Planted demo probe":
        st.info("No probes yet. Click **Seed sample KYC gallery** in the sidebar first.")
    else:
        up = st.file_uploader("Applicant face", type=["jpg", "jpeg", "png"])
        if up:
            img_bytes, img_name = up.getvalue(), up.name

    with st.form("check_form"):
        st.markdown("**Applicant identity (as submitted at onboarding)**")
        a, b = st.columns(2)
        full_name = a.text_input("Full name", value=preset.get("full_name", "") or "")
        phone = b.text_input("Phone", value=preset.get("phone", "") or "")
        c, d, e = st.columns(3)
        id_type = c.selectbox("ID type", ID_TYPES,
                              index=ID_TYPES.index(preset.get("id_type", "PAN")) if preset.get("id_type") in ID_TYPES else 0)
        id_number = d.text_input("ID number", value=preset.get("id_number", "") or "")
        customer_id = e.text_input("Customer/App ID", value=preset.get("customer_id", "") or "")
        submitted = st.form_submit_button("🔍 Run onboarding check", type="primary", use_container_width=True)

    if submitted:
        if not img_bytes:
            st.error("Please choose or upload an applicant image.")
        else:
            identity = {"full_name": full_name, "id_type": id_type, "id_number": id_number,
                        "phone": phone, "customer_id": customer_id}
            txn = f"CHK-{int(time.time()*1000)}"
            try:
                r = client.check(img_bytes, img_name, txn, identity, limit=limit)
                if r.status_code == 200:
                    render_result(r.json(), query_image_bytes=img_bytes)
                else:
                    st.error(f"{r.status_code}: {r.text[:300]}")
            except Exception as ex:
                st.error(f"Request failed: {ex}")

# ----------------------------- Enroll --------------------------------------
with tab_enroll:
    st.markdown("Add a known customer's face to the gallery.")
    up = st.file_uploader("Customer face", type=["jpg", "jpeg", "png"], key="enroll_up")
    with st.form("enroll_form"):
        a, b = st.columns(2)
        e_name = a.text_input("Full name")
        e_phone = b.text_input("Phone")
        c, d, e = st.columns(3)
        e_idtype = c.selectbox("ID type", ID_TYPES, key="e_idtype")
        e_idnum = d.text_input("ID number")
        e_cust = e.text_input("Customer/App ID")
        e_sub = st.form_submit_button("➕ Enroll", type="primary")
    if e_sub:
        if not up:
            st.error("Upload a face image first.")
        else:
            identity = {"full_name": e_name, "id_type": e_idtype, "id_number": e_idnum,
                        "phone": e_phone, "customer_id": e_cust}
            txn = f"ENR-{int(time.time()*1000)}"
            try:
                r = client.store(up.getvalue(), up.name, txn, identity)
                if r.status_code == 200:
                    st.success(f"Enrolled as {txn}. Gallery now: {client.stats().get('gallery_size')} faces.")
                else:
                    st.error(f"{r.status_code}: {r.text[:300]}")
            except Exception as ex:
                st.error(f"Request failed: {ex}")

# ----------------------------- Batch Probe Test ----------------------------
with tab_batch:
    st.markdown("Run every planted probe and check the verdict against the expected outcome — "
                "a live accuracy demonstration of the fraud engine.")
    st.caption("Each run first restores a clean sample gallery, so the accuracy is reproducible "
               "however many times you run it.")
    if not get_probes(client):
        st.info("No probes yet. Click **Seed sample KYC gallery** in the sidebar first.")
    elif st.button("▶ Run all probes", type="primary"):
        # Restore a known-clean gallery before measuring. An onboarding CLEAR is
        # auto-enrolled by design, so a naive re-run would later flag those same
        # applicants as duplicates and skew the score — reseeding keeps the
        # measurement idempotent.
        with st.spinner("Restoring a clean sample gallery…"):
            try:
                sr = client.seed_demo(reset=True)
                if sr.status_code == 200:
                    st.session_state.pop("probes", None)
                else:
                    st.warning(f"Could not reseed (HTTP {sr.status_code}); using the current gallery.")
            except Exception as ex:
                st.warning(f"Could not reseed ({ex}); using the current gallery.")
        # Managed vector indexes (Pinecone) are eventually consistent — give the
        # freshly re-seeded gallery a moment to become queryable so no planted
        # fraud probe is mis-scored as CLEAR against a not-yet-visible match.
        if str(backend).lower() == "pinecone":
            with st.spinner("Syncing the vector index…"):
                time.sleep(6)
        probes = get_probes(client)
        rows, correct = [], 0
        prog = st.progress(0.0)
        for i, p in enumerate(probes):
            if not p.get("image_bytes"):
                continue
            ident = p["identity"]
            try:
                r = client.check(p["image_bytes"], f"probe_{i}.jpg",
                                 f"BATCH-{i}-{int(time.time()*1000)}", ident, limit=limit)
                got = r.json().get("verdict", "ERR") if r.status_code == 200 else f"HTTP{r.status_code}"
                risk = r.json().get("risk_score", "") if r.status_code == 200 else ""
            except Exception as ex:
                got, risk = f"ERR:{ex}", ""
            ok = (got == p["expected_verdict"])
            correct += ok
            rows.append({"true face": p.get("true_person", f"probe_{i}"),
                         "submitted as": ident.get("full_name", "—"),
                         "expected verdict": p["expected_verdict"], "actual verdict": got,
                         "risk": risk, "✓": "✅" if ok else "❌"})
            prog.progress((i + 1) / len(probes))
        prog.empty()
        st.dataframe(rows, use_container_width=True, hide_index=True)
        pct = int(round(100 * correct / len(rows))) if rows else 0
        a, b = st.columns([1, 3])
        a.metric("Probe accuracy", f"{correct}/{len(rows)}")
        if correct == len(rows):
            b.success(f"✅ All {len(rows)} planted probes classified correctly ({pct}%) — "
                      f"every same-face-different-identity fraud caught, genuine re-KYC and new "
                      f"applicants passed cleanly.")
        else:
            b.warning(f"{correct}/{len(rows)} correct ({pct}%).")

# ----------------------------- Benchmarks ----------------------------------
with tab_bench:
    out = os.path.join(REPO_ROOT, "benchmarks", "out")
    st.markdown("Model accuracy (LFW) and 1:N search latency at scale.")
    if not os.path.isdir(out):
        st.info("No benchmarks yet. Run: python benchmarks/accuracy_lfw.py and python benchmarks/latency_scale.py")
    else:
        pngs = sorted(f for f in os.listdir(out) if f.endswith(".png"))
        csvs = sorted(f for f in os.listdir(out) if f.endswith(".csv"))
        for png in pngs:
            st.image(os.path.join(out, png), caption=png, use_container_width=True)
        for c in csvs:
            st.markdown(f"**{c}**")
            with open(os.path.join(out, c)) as f:
                st.dataframe(list(csv.DictReader(f)), use_container_width=True)

# ----------------------------- About ---------------------------------------
with tab_about:
    st.markdown("""
### FaceSentinel — Identity De-Duplication Security Fabric
**Problem.** Traditional KYC validates each *document* in isolation, so it cannot see that the **same face**
has already been enrolled under a **different identity** — the root of duplicate accounts, mule accounts and
synthetic-identity fraud (India: ₹36,014 cr reported bank fraud in FY25; ~1.33M mule accounts frozen in 2025).

**How it works.** Every enrolled face becomes an irreversible 512-d biometric template via a proprietary
deep-metric embedding engine. At onboarding we run a
**1:N similarity search** across the whole customer base (**Pinecone** or Redis vector DB); when a high face
match collides with a **different** government ID, we raise a **FRAUD ALERT** with a risk score, reason codes
and a review queue — before the account goes live.

**Verdicts.**
- ✅ `CLEAR` — no duplicate face; safe to enroll.
- ⚠️ `REVIEW` — borderline / unverifiable → human review.
- 🔁 `DUPLICATE_SAME_IDENTITY` — same person, same ID → legitimate re-KYC.
- 🚨 `FRAUD_ALERT_DIFFERENT_IDENTITY` — same face, **different** government ID → the fraud case.

**Demo mode.** This deployment can run with deterministic *demo* embeddings (no 200 MB model), so the whole
Pinecone pipeline is demonstrable anywhere. In demo mode, matching is exact-image based; enable the
production embedding engine (see the repository README) for true cross-photo face matching.

**Privacy by design (DPDP-ready).** Only irreversible embeddings are stored; the `/purge` endpoint is the
right-to-erasure; runs fully on-prem or in your own managed Pinecone — no data leaves the bank.
    """)
