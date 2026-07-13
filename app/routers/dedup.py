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
    """Render the planted onboarding probes with base64 avatar images.

    Deterministic, so the UI can fetch/replay them any time without re-seeding.
    """
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


@router.post("/demo/seed")
async def demo_seed(
        reset: bool = Form(default=False, description="Purge & re-enrol existing sample customers"),
        token: str = Depends(verify_token),
):
    """One-click: enrol the fictional sample-KYC gallery into the active vector DB.

    Populates ~16 fictional IDBI customers (with procedurally-drawn avatar faces)
    and returns planted onboarding probes (as base64 images) the console can
    replay to demonstrate FRAUD / DUPLICATE / CLEAR verdicts. Designed for demo
    embedding mode; with the full face model, seed real faces via
    scripts/ingest_lfw.py instead.
    """
    from app.demo import sample_kyc

    embedding_mode = dedup_service.embedding_service.get_health_status().get("embedding_mode")
    plan = sample_kyc.build_demo_plan()
    try:
        os.makedirs(settings.gallery_images_dir, exist_ok=True)
    except Exception:
        pass

    seeded = skipped = failed = 0
    errors = []
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
        "probes": _probe_payloads(),
    }


@router.get("/demo/probes")
async def demo_probes():
    """Return the planted onboarding probes (base64 avatars) for the console."""
    return {"probes": _probe_payloads()}


@router.get("/stats")
async def stats():
    """Lightweight gallery stats for dashboards / the review console."""
    backend = getattr(dedup_service.vector_store, "backend_name", settings.resolved_vector_backend)
    embedding_mode = dedup_service.embedding_service.get_health_status().get("embedding_mode")
    try:
        count = await dedup_service.vector_store.count_records()
        return {
            "gallery_size": count,
            "vector_backend": backend,
            "index_type": settings.index_type if backend == "redis" else backend,
            "embedding_mode": embedding_mode,
        }
    except Exception as e:
        return {"gallery_size": 0, "vector_backend": backend,
                "index_type": settings.index_type, "embedding_mode": embedding_mode, "error": str(e)}


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