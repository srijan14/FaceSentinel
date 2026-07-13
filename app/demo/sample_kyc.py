"""
Sample KYC dataset + procedural avatars for the FaceSentinel demo.

This is a small, self-contained gallery of **fictional** bank customers used to
seed the vector database (Pinecone or Redis) so the fraud engine can be shown
off without any real customer data or the 200 MB face model.

Every customer gets a deterministic, procedurally-drawn "face" avatar (a coloured
gradient + initials) so:

* thumbnails render in the review console, and
* in demo embedding mode the avatar's *bytes* are the face embedding's input,
  so re-submitting a customer's avatar under a **different** identity reproduces
  the hero fraud case (same face, different PAN), while a brand-new avatar reads
  as CLEAR.

Nothing here is a real person; PAN/phone numbers are synthetic and illustrative.
"""
from __future__ import annotations

import hashlib
import io
import os
from typing import Dict, List, Optional

# Bundled synthetic portraits (one per customer_id) so the console shows real
# faces instead of drawn placeholders. Keyed by customer_id so a probe that
# reuses a gallery customer's id gets byte-identical input (the demo-mode match).
FACES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faces")

IDENTITY_KEYS = ["customer_id", "full_name", "id_type", "id_number", "phone", "dob"]

# ---------------------------------------------------------------------------
# Fictional gallery: 16 customers already enrolled with IDBI Bank.
# ---------------------------------------------------------------------------
SAMPLE_CUSTOMERS: List[Dict] = [
    {"customer_id": "IDBI-100001", "full_name": "Aarav Sharma",    "id_type": "PAN", "id_number": "ABCPS1234A", "phone": "9810012345", "dob": "1990-04-12", "city": "Mumbai"},
    {"customer_id": "IDBI-100002", "full_name": "Priya Nair",      "id_type": "PAN", "id_number": "BNZPN5678B", "phone": "9820023456", "dob": "1988-09-30", "city": "Kochi"},
    {"customer_id": "IDBI-100003", "full_name": "Rohan Verma",     "id_type": "PAN", "id_number": "CDEPV9012C", "phone": "9830034567", "dob": "1995-01-22", "city": "Delhi"},
    {"customer_id": "IDBI-100004", "full_name": "Ananya Iyer",     "id_type": "PAN", "id_number": "DKLPI3456D", "phone": "9840045678", "dob": "1992-07-08", "city": "Chennai"},
    {"customer_id": "IDBI-100005", "full_name": "Vikram Singh",    "id_type": "PAN", "id_number": "EMNPS7890E", "phone": "9850056789", "dob": "1985-11-15", "city": "Jaipur"},
    {"customer_id": "IDBI-100006", "full_name": "Sneha Reddy",     "id_type": "PAN", "id_number": "FOPQR1234F", "phone": "9860067890", "dob": "1993-03-03", "city": "Hyderabad"},
    {"customer_id": "IDBI-100007", "full_name": "Arjun Menon",     "id_type": "PAN", "id_number": "GQRPM5678G", "phone": "9870078901", "dob": "1991-06-19", "city": "Bengaluru"},
    {"customer_id": "IDBI-100008", "full_name": "Kavya Joshi",     "id_type": "PAN", "id_number": "HRSPJ9012H", "phone": "9880089012", "dob": "1997-12-25", "city": "Pune"},
    {"customer_id": "IDBI-100009", "full_name": "Rahul Gupta",     "id_type": "PAN", "id_number": "ITUPG3456I", "phone": "9890090123", "dob": "1989-02-14", "city": "Lucknow"},
    {"customer_id": "IDBI-100010", "full_name": "Meera Desai",     "id_type": "PAN", "id_number": "JVWPD7890J", "phone": "9900101234", "dob": "1994-08-27", "city": "Ahmedabad"},
    {"customer_id": "IDBI-100011", "full_name": "Karan Malhotra",  "id_type": "PAN", "id_number": "KWXPM1234K", "phone": "9910112345", "dob": "1986-05-05", "city": "Chandigarh"},
    {"customer_id": "IDBI-100012", "full_name": "Divya Pillai",    "id_type": "PAN", "id_number": "LXYPP5678L", "phone": "9920123456", "dob": "1996-10-11", "city": "Thiruvananthapuram"},
    {"customer_id": "IDBI-100013", "full_name": "Aditya Rao",      "id_type": "PAN", "id_number": "MYZPR9012M", "phone": "9930134567", "dob": "1990-01-01", "city": "Nagpur"},
    {"customer_id": "IDBI-100014", "full_name": "Ishita Banerjee", "id_type": "PAN", "id_number": "NZAPB3456N", "phone": "9940145678", "dob": "1992-04-18", "city": "Kolkata"},
    {"customer_id": "IDBI-100015", "full_name": "Nikhil Kulkarni", "id_type": "PAN", "id_number": "OABPK7890O", "phone": "9950156789", "dob": "1987-07-23", "city": "Nashik"},
    {"customer_id": "IDBI-100016", "full_name": "Tara Krishnan",   "id_type": "PAN", "id_number": "PBCPK1234P", "phone": "9960167890", "dob": "1998-09-09", "city": "Coimbatore"},
]

# Fraudster personas: the SAME face as an enrolled customer resubmitted under a
# fresh identity (new name + new PAN) — the duplicate/synthetic-identity signature.
FRAUD_APPLICANTS: List[Dict] = [
    {"customer_id": "APP-900001", "full_name": "Sameer Khan",   "id_type": "PAN", "id_number": "ZZZPX9999Z", "phone": "9700009999", "dob": "1990-04-12"},
    {"customer_id": "APP-900002", "full_name": "Neha Kapoor",   "id_type": "PAN", "id_number": "YYYPQ8888Y", "phone": "9700008888", "dob": "1988-09-30"},
]

# A genuine brand-new applicant (not in the gallery) — expected CLEAR.
NEW_APPLICANT: Dict = {
    "customer_id": "APP-900100", "full_name": "Manish Agarwal", "id_type": "PAN",
    "id_number": "QCDPA5678Q", "phone": "9700007777", "dob": "1993-11-02", "city": "Indore",
}

# Verdict labels (kept in sync with app.services.fraud_decision).
FRAUD = "FRAUD_ALERT_DIFFERENT_IDENTITY"
DUPLICATE = "DUPLICATE_SAME_IDENTITY"
CLEAR = "CLEAR"


def identity_of(customer: Dict) -> Dict[str, Optional[str]]:
    """Project a customer record down to the identity fields the API stores."""
    return {k: customer.get(k) for k in IDENTITY_KEYS}


# ---------------------------------------------------------------------------
# Procedural avatar generation (deterministic from the customer id)
# ---------------------------------------------------------------------------
def _initials(name: str) -> str:
    parts = [p for p in name.split() if p]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def _hsv_to_rgb(h: float, s: float, v: float):
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def avatar_bytes(customer: Dict, size: int = 320) -> bytes:
    """Return a face image for a customer.

    Prefers a bundled real portrait (``faces/<customer_id>.jpg``); falls back to
    a deterministic drawn avatar (gradient + initials) if none is bundled.
    Keyed by customer_id so a probe reusing a gallery id gets identical bytes.
    Requires Pillow (in requirements.txt / requirements-demo.txt).
    """
    face_path = os.path.join(FACES_DIR, f"{customer.get('customer_id', '')}.jpg")
    if os.path.isfile(face_path):
        with open(face_path, "rb") as fh:
            return fh.read()

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Pillow is required to generate demo avatars "
                           "(pip install Pillow)") from e

    seed = int(hashlib.sha256(customer["customer_id"].encode()).hexdigest()[:8], 16)
    hue = (seed % 360) / 360.0
    top = _hsv_to_rgb(hue, 0.55, 0.95)
    bottom = _hsv_to_rgb((hue + 0.08) % 1.0, 0.70, 0.55)

    img = Image.new("RGB", (size, size))
    px = img.load()
    for y in range(size):
        t = y / max(1, size - 1)
        row = (int(top[0] + (bottom[0] - top[0]) * t),
               int(top[1] + (bottom[1] - top[1]) * t),
               int(top[2] + (bottom[2] - top[2]) * t))
        for x in range(size):
            px[x, y] = row

    draw = ImageDraw.Draw(img)
    # Head circle
    pad = size // 5
    draw.ellipse([pad, pad, size - pad, size - pad], fill=(255, 255, 255, 255))
    # Initials
    text = _initials(customer.get("full_name", "?"))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size // 4)
    except Exception:
        font = ImageFont.load_default()
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx, ty = (size - tw) / 2 - bbox[0], (size - th) / 2 - bbox[1]
    except Exception:
        tw, th = draw.textsize(text, font=font)
        tx, ty = (size - tw) / 2, (size - th) / 2
    draw.text((tx, ty), text, fill=bottom, font=font)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def build_demo_plan(n_customers: int = 16) -> Dict:
    """Return the full seed plan: gallery records + planted onboarding probes.

    ``probes`` is a list of dicts with keys:
        label, expected_verdict, identity, source_customer_id
    where ``source_customer_id`` names the gallery customer whose avatar bytes
    the probe reuses (``None`` -> the probe brings its own fresh avatar via the
    NEW_APPLICANT record). The caller renders/attaches the actual image bytes.
    """
    gallery = SAMPLE_CUSTOMERS[:max(4, min(n_customers, len(SAMPLE_CUSTOMERS)))]

    probes: List[Dict] = []
    # Two fraud probes: same face as gallery[2] / gallery[5], different identity.
    fraud_targets = [gallery[2], gallery[5]] if len(gallery) > 5 else [gallery[0]]
    for persona, target in zip(FRAUD_APPLICANTS, fraud_targets):
        probes.append({
            "label": f"Same face as {target['full_name']} — new PAN ({persona['full_name']})",
            "expected_verdict": FRAUD,
            "identity": identity_of(persona),
            "source_customer_id": target["customer_id"],
            "true_person": target["full_name"],
        })
    # One duplicate probe: same face + same identity (legit re-KYC).
    dup_target = gallery[8] if len(gallery) > 8 else gallery[1]
    probes.append({
        "label": f"Re-KYC of {dup_target['full_name']} (same identity)",
        "expected_verdict": DUPLICATE,
        "identity": identity_of(dup_target),
        "source_customer_id": dup_target["customer_id"],
        "true_person": dup_target["full_name"],
    })
    # One clear probe: a genuinely new applicant, not in the gallery.
    probes.append({
        "label": f"New applicant {NEW_APPLICANT['full_name']} (not enrolled)",
        "expected_verdict": CLEAR,
        "identity": identity_of(NEW_APPLICANT),
        "source_customer_id": None,
        "true_person": NEW_APPLICANT["full_name"],
    })

    return {"gallery": gallery, "probes": probes, "new_applicant": NEW_APPLICANT}
