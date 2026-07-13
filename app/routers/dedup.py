from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from typing import Optional
import base64
import json
import os
from datetime import datetime
from app.models.requests import StoreMetadata, SearchMetadata, SearchRequest, PurgeRequest, CheckMetadata
from app.models.responses import StoreResponse, SearchResponse, PurgeResponse, CheckResponse, MatchDetail
from app.services.dedup_service import dedup_service
from app.utils.auth import verify_token
from app.utils.exceptions import DedupException, create_http_exception
from app.config import settings

router = APIRouter(prefix="/v1/dedup/face", tags=["deduplication"])

# Identity fields persisted alongside every enrolled vector.
_ID_FIELDS = ("customer_id", "full_name", "id_type", "id_number", "phone", "dob")


def _require_live_inference():
    """Reject custom-image / live-inference requests unless the real model is running.

    Uploads need the ArcFace model to embed the image; on a read-only box (or one
    without the ONNX models) that path is unavailable and would otherwise mix
    embedding spaces with the pre-indexed gallery.
    """
    if not settings.inference_enabled:
        raise HTTPException(
            status_code=403,
            detail={"error": {
                "code": "INFERENCE_DISABLED",
                "message": "Custom image uploads and enrollment are disabled on this hosted demo.",
                "details": "This server is too resource-constrained to run the ~200 MB face-embedding "
                           "model, so it runs read-only and serves only the pre-indexed dataset. To "
                           "process your own images, set up FaceSentinel locally with the ArcFace "
                           "model (see the README)."}},
        )
    if dedup_service.embedding_service.get_health_status().get("embedding_mode") != "insightface":
        raise HTTPException(
            status_code=503,
            detail={"error": {
                "code": "MODEL_UNAVAILABLE",
                "message": "The ArcFace face model is not loaded on this instance.",
                "details": "Custom uploads/enrollment need the ONNX models present in models/. "
                           "Add them and restart, or use the planted demo probes."}},
        )


def _use_live_inference() -> bool:
    """True only when we should run the real model (enabled + models loaded)."""
    if not settings.inference_enabled:
        return False
    return dedup_service.embedding_service.get_health_status().get("embedding_mode") == "insightface"


def _engine_label(embedding_mode: str) -> str:
    """Human, technical label for the face-embedding engine (for dashboards)."""
    if not settings.inference_enabled:
        return "ArcFace R50 · pre-indexed"
    if embedding_mode == "insightface":
        return "ArcFace R50 (512-d)"
    if embedding_mode == "demo":
        return "deterministic (no model)"
    return str(embedding_mode)


@router.post("/store", response_model=StoreResponse)
async def store_record(
        image: UploadFile = File(..., description="Customer image file"),
        transaction_id: str = Form(..., description="Unique transaction identifier"),
        metadata: str = Form(..., description="JSON string containing customer metadata"),
        token: str = Depends(verify_token)
):
    """
    Store a new customer record with image and metadata.
    Generates embeddings and stores them in the vector database.
    """
    _require_live_inference()
    try:
        # Validate image content type
        if image.content_type not in settings.allowed_image_types:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "code": "INVALID_REQUEST",
                        "message": f"Unsupported image format: {image.content_type}",
                        "details": f"Allowed formats: {', '.join(settings.allowed_image_types)}"
                    }
                }
            )

        # Parse and validate metadata
        try:
            metadata_dict = json.loads(metadata)
            if "created_on" not in metadata_dict:
                from datetime import datetime
                metadata_dict["created_on"] = datetime.utcnow().isoformat() + "Z"
            
            customer_metadata = StoreMetadata(**metadata_dict)
            
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "code": "INVALID_REQUEST",
                        "message": "Invalid metadata JSON format",
                        "details": str(e)
                    }
                }
            )
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "code": "INVALID_REQUEST",
                        "message": "Invalid metadata values",
                        "details": str(e)
                    }
                }
            )

        # Read image data
        image_data = await image.read()

        # Store customer record
        await dedup_service.store_customer(
            transaction_id,
            image_data,
            customer_metadata
        )

        return StoreResponse(
            status="success",
            transaction_id=transaction_id,
            message="Record inserted successfully"
        )

    except DedupException as e:
        raise create_http_exception(e)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "Internal server error",
                    "details": str(e)
                }
            }
        )


@router.post("/search", response_model=SearchResponse)
async def search_similar(
        image: UploadFile = File(..., description="Query image for similarity search"),
        transaction_id: str = Form(..., description="Unique transaction identifier"),
        metadata: str = Form(..., description="JSON string containing query metadata"),
        threshold: Optional[float] = Form(default=settings.default_similarity_threshold,
                                          description="Similarity threshold (0.0-1.0)"),
        limit: Optional[int] = Form(default=settings.max_search_results,
                                    description="Maximum results (1-1000)"),
        token: str = Depends(verify_token)
):
    """
    Search for matching customer records based on image similarity.
    Uses vector embeddings and configurable similarity thresholds.
    """
    _require_live_inference()
    try:
        # Validate image content type
        if image.content_type not in settings.allowed_image_types:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "code": "INVALID_REQUEST",
                        "message": f"Unsupported image format: {image.content_type}",
                        "details": f"Allowed formats: {', '.join(settings.allowed_image_types)}"
                    }
                }
            )

        # Parse and validate metadata
        try:
            metadata_dict = json.loads(metadata)
            search_metadata = SearchMetadata(**metadata_dict)
            
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "code": "INVALID_REQUEST",
                        "message": "Invalid metadata JSON format",
                        "details": str(e)
                    }
                }
            )
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "code": "INVALID_REQUEST",
                        "message": "Invalid metadata values",
                        "details": str(e)
                    }
                }
            )

        # Validate search parameters
        if threshold is not None and (threshold < 0.0 or threshold > 1.0):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "code": "INVALID_REQUEST",
                        "message": "Threshold must be between 0.0 and 1.0",
                        "details": None
                    }
                }
            )

        if limit is not None and (limit < 1 or limit > 1000):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "code": "INVALID_REQUEST",
                        "message": "Limit must be between 1 and 1000",
                        "details": None
                    }
                }
            )

        # Read image data
        image_data = await image.read()

        # Search for similar customers by face-embedding similarity
        results = await dedup_service.search_similar_customers(
            transaction_id,
            image_data,
            search_metadata.model_dump(),  # Convert to dict for dedup service
            threshold or settings.default_similarity_threshold,
            limit or settings.max_search_results
        )

        # Return response matching contract format
        return SearchResponse(
            status="success",
            transaction_id=transaction_id,  # Return transaction_id as-is
            total_matches=len(results),
            metadata=search_metadata,  # Return SearchMetadata object
            results=results
        )

    except DedupException as e:
        raise create_http_exception(e)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "Internal server error",
                    "details": str(e)
                }
            }
        )


@router.post("/check", response_model=CheckResponse)
async def onboarding_check(
        image: UploadFile = File(..., description="Applicant face image"),
        transaction_id: str = Form(..., description="Unique transaction identifier"),
        metadata: str = Form(..., description="JSON string with applicant identity fields"),
        threshold: Optional[float] = Form(default=None,
                                          description="Candidate retrieval threshold (0.0-1.0); default t_candidate"),
        limit: Optional[int] = Form(default=10, description="Maximum candidate matches (1-100)"),
        token: str = Depends(verify_token)
):
    """
    Screen an applicant face for duplicate / synthetic-identity fraud at onboarding.

    Generates the face embedding, runs a 1:N similarity search across the whole
    gallery, compares the applicant identity against every near-duplicate, and
    returns a verdict (CLEAR / REVIEW / DUPLICATE_SAME_IDENTITY /
    FRAUD_ALERT_DIFFERENT_IDENTITY) with a 0-100 risk score and reason codes.
    A CLEAR face is enrolled into the gallery; duplicates and fraud alerts are
    never auto-enrolled.
    """
    _require_live_inference()
    try:
        # Validate image content type
        if image.content_type not in settings.allowed_image_types:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "code": "INVALID_REQUEST",
                        "message": f"Unsupported image format: {image.content_type}",
                        "details": f"Allowed formats: {', '.join(settings.allowed_image_types)}"
                    }
                }
            )

        # Parse and validate applicant identity metadata
        try:
            metadata_dict = json.loads(metadata)
            applicant = CheckMetadata(**metadata_dict)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400,
                detail={"error": {"code": "INVALID_REQUEST",
                                  "message": "Invalid metadata JSON format", "details": str(e)}}
            )
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail={"error": {"code": "INVALID_REQUEST",
                                  "message": "Invalid metadata values", "details": str(e)}}
            )

        # Validate parameters
        if threshold is not None and (threshold < 0.0 or threshold > 1.0):
            raise HTTPException(
                status_code=400,
                detail={"error": {"code": "INVALID_REQUEST",
                                  "message": "Threshold must be between 0.0 and 1.0", "details": None}}
            )
        if limit is not None and (limit < 1 or limit > 100):
            raise HTTPException(
                status_code=400,
                detail={"error": {"code": "INVALID_REQUEST",
                                  "message": "Limit must be between 1 and 100", "details": None}}
            )

        # Read image and run the onboarding fraud check
        image_data = await image.read()
        identity = applicant.model_dump()
        result = await dedup_service.onboarding_check(
            transaction_id=transaction_id,
            image_data=image_data,
            identity=identity,
            threshold=threshold,
            limit=limit,
            image_path=applicant.image_path or "",
            created_on=applicant.created_on,
        )

        matches = [MatchDetail(**m) for m in result["matches"]]
        best_match = MatchDetail(**result["best_match"]) if result.get("best_match") else None

        return CheckResponse(
            status="success",
            transaction_id=result["transaction_id"],
            verdict=result["verdict"],
            risk_score=result["risk_score"],
            reason_codes=result["reason_codes"],
            enrolled=result["enrolled"],
            query_identity=result["query_identity"],
            best_match=best_match,
            matches=matches,
            total_matches=result["total_matches"],
        )

    except DedupException as e:
        raise create_http_exception(e)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": {"code": "INTERNAL_ERROR",
                              "message": "Internal server error", "details": str(e)}}
        )


@router.post("/purge", response_model=PurgeResponse)
async def purge_record(
        request: PurgeRequest,
        token: str = Depends(verify_token)
):
    """
    Delete a customer record from the system.
    Removes both vector embeddings and metadata.
    """
    try:
        # Purge customer record
        await dedup_service.purge_customer(request.transaction_id)

        return PurgeResponse(
            status="success",
            transaction_id=request.transaction_id,
            message="Record purged successfully"
        )

    except DedupException as e:
        raise create_http_exception(e)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "Internal server error",
                    "details": str(e)
                }
            }
        )


def _probe_payloads() -> list:
    """Render the planted onboarding probes with base64 face images.

    Deterministic, so the UI can fetch/replay them any time without re-seeding.
    Prefers the curated PAN-card demo dataset; falls back to the legacy
    procedural sample gallery when the dataset is not bundled.
    """
    from app.demo import dataset
    if dataset.available():
        return [{
            "probe_id": p["probe_id"],
            "label": p["label"],
            "expected_verdict": p["expected_verdict"],
            "identity": p["identity"],
            "true_person": p["true_person"],
            "image_b64": base64.b64encode(p["face_bytes"]).decode("ascii"),
        } for p in dataset.probes()]

    from app.demo import sample_kyc
    plan = sample_kyc.build_demo_plan()
    by_id = {c["customer_id"]: c for c in sample_kyc.SAMPLE_CUSTOMERS}
    out = []
    for p in plan["probes"]:
        src = by_id[p["source_customer_id"]] if p["source_customer_id"] else plan["new_applicant"]
        img = sample_kyc.avatar_bytes(src)
        out.append({
            "label": p["label"],
            "expected_verdict": p["expected_verdict"],
            "identity": p["identity"],
            "true_person": p["true_person"],
            "image_b64": base64.b64encode(img).decode("ascii"),
        })
    return out


async def _seed_gallery_dataset(reset: bool):
    """Enrol the curated PAN-card gallery. Live-embeds locally, else inserts
    the precomputed ArcFace vectors directly (read-only / no-model path)."""
    from app.demo import dataset
    live = _use_live_inference()
    seeded = skipped = failed = 0
    errors: list = []
    for g in dataset.gallery():
        cid = g["customer_id"]
        img_path = ""
        try:
            img_path = os.path.join(settings.gallery_images_dir, f"{cid}.jpg")
            with open(img_path, "wb") as fh:
                fh.write(g["face_bytes"])
        except Exception:
            img_path = ""
        meta = StoreMetadata(
            created_on=datetime.utcnow().isoformat() + "Z",
            image_path=img_path,
            **{k: g.get(k) for k in _ID_FIELDS},
        )
        try:
            exists = await dedup_service.vector_store.customer_exists(cid)
            if reset and exists:
                await dedup_service.purge_customer(cid)
                exists = False
            if exists:
                skipped += 1
                continue
            if live:
                await dedup_service.store_customer(cid, g["face_bytes"], meta)
            else:
                await dedup_service.vector_store.store_vector(cid, g["vector"], meta)
            seeded += 1
        except DedupException as e:
            if getattr(e, "code", "") == "CUSTOMER_EXISTS":
                skipped += 1
            else:
                failed += 1
                if len(errors) < 5:
                    errors.append(f"{cid}: {e.message}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            if len(errors) < 5:
                errors.append(f"{cid}: {e}")
    return seeded, skipped, failed, errors


async def _seed_gallery_legacy(reset: bool):
    """Legacy procedural sample gallery (used only when no dataset is bundled)."""
    from app.demo import sample_kyc
    plan = sample_kyc.build_demo_plan()
    seeded = skipped = failed = 0
    errors: list = []
    for cust in plan["gallery"]:
        cid = cust["customer_id"]
        img = sample_kyc.avatar_bytes(cust)
        img_path = ""
        try:
            img_path = os.path.join(settings.gallery_images_dir, f"{cid}.jpg")
            with open(img_path, "wb") as fh:
                fh.write(img)
        except Exception:
            img_path = ""
        meta = StoreMetadata(
            created_on=datetime.utcnow().isoformat() + "Z",
            image_path=img_path,
            **sample_kyc.identity_of(cust),
        )
        try:
            if reset and await dedup_service.vector_store.customer_exists(cid):
                await dedup_service.purge_customer(cid)
            await dedup_service.store_customer(cid, img, meta)
            seeded += 1
        except DedupException as e:
            if getattr(e, "code", "") == "CUSTOMER_EXISTS":
                skipped += 1
            else:
                failed += 1
                if len(errors) < 5:
                    errors.append(f"{cid}: {e.message}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            if len(errors) < 5:
                errors.append(f"{cid}: {e}")
    return seeded, skipped, failed, errors


@router.post("/demo/seed")
async def demo_seed(
        reset: bool = Form(default=False, description="Purge & re-enrol existing sample customers"),
        token: str = Depends(verify_token),
):
    """One-click: enrol the curated demo gallery into the active vector DB.

    Uses the bundled PAN-card demo dataset (real faces + real OCR'd identities;
    fraud-applicant personas are fictional). Locally with the ArcFace model each
    face is embedded live; on a read-only deployment the precomputed embeddings
    are inserted directly (no model needed). Returns planted onboarding probes
    (base64 faces) the console replays for FRAUD / DUPLICATE / CLEAR verdicts.
    """
    from app.demo import dataset

    embedding_mode = dedup_service.embedding_service.get_health_status().get("embedding_mode")
    try:
        os.makedirs(settings.gallery_images_dir, exist_ok=True)
    except Exception:
        pass

    if dataset.available():
        seeded, skipped, failed, errors = await _seed_gallery_dataset(reset)
    else:
        seeded, skipped, failed, errors = await _seed_gallery_legacy(reset)

    gallery_size = await dedup_service.vector_store.count_records()
    backend = getattr(dedup_service.vector_store, "backend_name", settings.resolved_vector_backend)
    return {
        "status": "success",
        "seeded": seeded,
        "skipped_existing": skipped,
        "failed": failed,
        "errors": errors,
        "gallery_size": gallery_size,
        "vector_backend": backend,
        "embedding_mode": embedding_mode,
        "inference_enabled": settings.inference_enabled,
        "probes": _probe_payloads(),
    }


@router.get("/demo/probes")
async def demo_probes():
    """Return the planted onboarding probes (base64 avatars) for the console."""
    return {"probes": _probe_payloads()}


@router.post("/demo/check_probe")
async def demo_check_probe(
        probe_id: str = Form(..., description="Planted probe id, e.g. APP-900001"),
        limit: Optional[int] = Form(default=10, description="Maximum candidate matches"),
        token: str = Depends(verify_token),
):
    """Screen a *planted* probe against the gallery and return a fraud verdict.

    Model-free by default: screens the probe's precomputed ArcFace embedding, so
    it works on a read-only deployment with no model loaded. When live inference
    is enabled (local + ArcFace models present) the probe face is embedded fresh
    instead. Read-only: never enrols.
    """
    from app.demo import dataset
    if not dataset.available():
        raise HTTPException(
            status_code=503,
            detail={"error": {"code": "DATASET_UNAVAILABLE",
                              "message": "The bundled demo dataset is not present on this instance.",
                              "details": "app/demo/dataset/ (manifest.json + embeddings.npz) must be "
                                         "deployed for the read-only demo. Seed a gallery to use custom probes."}},
        )
    p = dataset.probe_by_id(probe_id)
    if not p:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "PROBE_NOT_FOUND", "message": f"Unknown probe id: {probe_id}"}},
        )
    identity = {k: p["identity"].get(k) for k in _ID_FIELDS}
    try:
        if _use_live_inference():
            embedding = await dedup_service.embedding_service.generate_embedding(
                p["face_bytes"], context="PROBE")
        else:
            embedding = p["vector"]
        result = await dedup_service.screen_embedding(
            p["probe_id"], embedding, identity, limit=limit, enroll=False)
        result["expected_verdict"] = p["expected_verdict"]
        result["true_person"] = p["true_person"]
        return result
    except DedupException as e:
        raise create_http_exception(e)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail={"error": {"code": "INTERNAL_ERROR",
                              "message": "Internal server error", "details": str(e)}},
        )


@router.get("/stats")
async def stats():
    """Lightweight gallery stats for dashboards / the review console."""
    backend = getattr(dedup_service.vector_store, "backend_name", settings.resolved_vector_backend)
    embedding_mode = dedup_service.embedding_service.get_health_status().get("embedding_mode")
    engine = _engine_label(embedding_mode)
    extra = {
        "embedding_mode": embedding_mode,
        "embedding_engine": engine,
        "inference_enabled": settings.inference_enabled,
        "live_inference": _use_live_inference(),   # true only when the real model runs
        "thresholds": {"t_candidate": settings.t_candidate,
                       "t_review": settings.t_review, "t_match": settings.t_match},
    }
    try:
        count = await dedup_service.vector_store.count_records()
        return {"gallery_size": count, "vector_backend": backend,
                "index_type": settings.index_type if backend == "redis" else backend, **extra}
    except Exception as e:
        return {"gallery_size": 0, "vector_backend": backend,
                "index_type": settings.index_type, **extra, "error": str(e)}


@router.get("/health")
async def health_check():
    """Check service health status"""
    try:
        health_status = await dedup_service.health_check()
        status_code = 200 if health_status["overall"] else 503

        return {
            "status": "healthy" if health_status["overall"] else "unhealthy",
            "services": health_status
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }