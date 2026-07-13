# FaceSentinel — Facial De-Duplication for Fraud Prevention

Real-time **1:N facial de-duplication** that catches the same face being enrolled under a
**different identity** — the signature of duplicate-account, mule-account and
synthetic-identity fraud at KYC onboarding.

Built for **IDBI Innovate 2026 — Open Track**. Turns a face into an irreversible 512-d
biometric embedding, searches the whole customer gallery with a vector database, and returns a
**fraud verdict + risk score + reason codes** the onboarding workflow can act on.

> One face. One identity.

---

## Why this exists

Traditional KYC validates each *document* in isolation, so it cannot see that the **same
face** already exists under a **different** PAN / Aadhaar / name. That blind spot is exactly
what fraud rings exploit (India: ₹36,014 cr reported bank fraud in FY25; ~1.33M mule accounts
frozen in 2025; synthetic-identity fraud up ~450% since 2022). FaceSentinel closes it by
matching faces across the entire base at onboarding, **before the account goes live**.

## How it works

```
Applicant face ─▶ face detect ─▶ 512-d embedding ─▶ 1:N vector search (Pinecone / Redis)
                                                                     │
                        identity compare (id_number / name)  ◀───────┘
                                                                     │
                                                                     ▼
        CLEAR · REVIEW · DUPLICATE_SAME_IDENTITY · FRAUD_ALERT_DIFFERENT_IDENTITY
                        (+ risk score 0-100 + reason codes)
```

- **Detection + embedding:** production-grade face detection + deep-metric 512-d embedding, served CPU-only via the ONNX standard. A model-free **demo embedding mode** lets the whole pipeline run without the 200 MB weights (see below).
- **Vector store / search (pluggable):** **Pinecone** (managed, serverless — recommended for a hosted deployment) **or** **Redis Stack / RediSearch** (self-hosted). Cosine similarity, backend-independent thresholds. Select with `VECTOR_BACKEND` (`auto` picks Pinecone when `PINECONE_API_KEY` is set).
- **Fraud decision:** a strong face match combined with a **different government id_number** ⇒ `FRAUD_ALERT_DIFFERENT_IDENTITY`. Same id ⇒ legitimate re-KYC. A configurable multi-threshold policy engine (`app/services/fraud_decision.py`) turns the score into `CLEAR` / `REVIEW` / `DUPLICATE` / `FRAUD_ALERT`.

## Architecture

| Layer | Component |
|---|---|
| API | FastAPI (`app/main.py`, `app/routers/dedup.py`) |
| Orchestration | `app/services/dedup_service.py` (`onboarding_check`) |
| Face model | `app/services/embedding.py` → vendored face engine in `app/core/` (+ `demo_embedding.py` model-free mode) |
| Vector DB | `app/services/vector_store.py` factory → `pinecone_service.py` (Pinecone) or `redis_service.py` (RediSearch), 512-d, COSINE |
| Fraud engine | `app/services/fraud_decision.py` (pure, unit-tested) |
| Sample KYC / seeding | `app/demo/sample_kyc.py` + `scripts/seed_pinecone.py` + `/demo/seed` endpoint |
| Review console | `ui/app.py` (Streamlit) → talks to the API over HTTP |

---

## Quick start (Docker — one command)

Requires Docker with Compose. Brings up Redis Stack, the API (auto-downloads the ONNX models
on first boot), and the Streamlit console. If the auto-download fails, fetch the weights
manually from [Google Drive](https://drive.google.com/drive/folders/1O6YbF41QSwPOnuPUVspTmPOX9kkvP5ar?usp=sharing).

```bash
cp .env.example .env            # optional: set your API_KEY
docker compose up --build
# API:      http://localhost:8000  (Swagger at /docs)
# Console:  http://localhost:8501
```

## Quick start (Pinecone + demo mode — fastest, no models, no Redis) ⭐

The lightest way to get a running deployment/demo. Uses **managed Pinecone** for the vector
DB and **demo embeddings** (deterministic, no 200 MB model download).

```bash
pip install -r requirements-demo.txt           # light deps (no face-model runtime)

export API_KEY=dev-local-key-change-me
export PINECONE_API_KEY=pc-xxxxxxxx             # free key from https://app.pinecone.io

uvicorn app.main:app --host 0.0.0.0 --port 8000   # boots, auto-creates the Pinecone index
streamlit run ui/app.py                            # in another shell → open http://localhost:8501
```

Then click **🌱 Seed sample KYC gallery** in the console sidebar (or run
`python scripts/seed_pinecone.py`) and try the **🎯 Onboarding Check** tab. The planted
"same face / different PAN" probe returns **🚨 FRAUD ALERT**, re-KYC returns **DUPLICATE**,
and a new applicant returns **CLEAR**.

## API

All endpoints are under `/v1/dedup/face` and require an `Authorization: <API_KEY>` header
(raw key, no `Bearer` prefix).

### `POST /check` — screen an applicant (the fraud gate)

```bash
curl -X POST http://localhost:8000/v1/dedup/face/check \
  -H "Authorization: dev-local-key-change-me" \
  -F "image=@applicant.jpg" \
  -F "transaction_id=APP-1001" \
  -F 'metadata={"full_name":"Ravi Kumar","id_type":"PAN","id_number":"ABCDE1234F","phone":"9990001111"}'
```

```jsonc
{
  "status": "success",
  "transaction_id": "APP-1001",
  "verdict": "FRAUD_ALERT_DIFFERENT_IDENTITY",
  "risk_score": 87,
  "reason_codes": ["FACE_MATCH_HIGH", "ID_NUMBER_MISMATCH", "NAME_MISMATCH"],
  "enrolled": false,
  "best_match": {
    "transaction_id": "CUST-88231",
    "similarity_score": 0.8535,
    "identity_match": false,
    "field_diffs": ["id_number", "full_name", "phone"],
    "identity": {"full_name": "Priya Nair", "id_type": "PAN", "id_number": "ZZZZZ9999Z"}
  },
  "matches": [ /* ranked candidates */ ],
  "total_matches": 3
}
```

**Verdicts:** `CLEAR` (safe, face enrolled) · `REVIEW` (borderline / unverifiable → human) ·
`DUPLICATE_SAME_IDENTITY` (same person, legit re-KYC) · `FRAUD_ALERT_DIFFERENT_IDENTITY`
(same face, different government identity).

### Other endpoints
- `POST /store` — enrol a known customer face (`image`, `transaction_id`, `metadata` JSON).
- `POST /search` — raw 1:N similarity search (`image`, `threshold`, `limit`).
- `POST /purge` — delete a record (`{"transaction_id": "..."}`) — the **right-to-erasure** hook.
- `POST /demo/seed` — one-click: enrol the fictional sample-KYC gallery + return planted probes.
- `GET  /demo/probes` — the planted onboarding probes (base64 avatars) for the console.
- `GET  /stats` — gallery size + active vector backend + embedding mode.
- `GET  /health` — vector-store + model readiness (reports backend & embedding mode).

Identity metadata fields (all optional): `customer_id`, `full_name`, `id_type`, `id_number`,
`phone`, `dob`. `id_number` is the authoritative identity key for the fraud decision.
