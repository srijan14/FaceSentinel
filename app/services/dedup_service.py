import logging
import os
from datetime import datetime
from typing import List, Dict, Optional
from app.config import settings
from app.models.requests import StoreMetadata
from app.models.responses import SearchResult
from app.services import fraud_decision
from app.services.fraud_decision import IDENTITY_FIELDS
from app.services.embedding import embedding_service
from app.services.vector_store import get_vector_store
from app.utils.exceptions import DedupException, CustomerNotFoundError, InvalidRequestError, VectorServiceError

logger = logging.getLogger(__name__)


class DeduplicationService:
    """Main service orchestrating the deduplication workflow"""

    def __init__(self):
        self.embedding_service = embedding_service
        self.vector_store = get_vector_store()

    async def store_customer(self, transaction_id: str, image_data: bytes, metadata: StoreMetadata) -> bool:
        """Store customer record with image embedding"""
        try:
            logger.info(f"[STORE] Processing store request for transaction_id: {transaction_id}")

            # Check if customer already exists
            if await self.vector_store.customer_exists(transaction_id):
                logger.warning(f"[STORE] Transaction ID already exists: {transaction_id}")
                raise DedupException(f"Transaction ID {transaction_id} already exists", "CUSTOMER_EXISTS")

            # Verify embedding service is ready (should be initialized at startup)
            if not self.embedding_service.is_ready():
                logger.error(f"[STORE] Embedding service not ready - this should not happen after startup")
                raise VectorServiceError("Face analysis service is not available")

            # Generate embedding from image with face detection
            embedding = await self.embedding_service.generate_embedding(image_data, context="STORE")

            # Store in Redis
            await self.vector_store.store_vector(transaction_id, embedding, metadata)

            logger.info(f"[STORE] Successfully stored customer record for transaction_id: {transaction_id}")
            return True

        except (DedupException, InvalidRequestError, VectorServiceError):
            raise
        except Exception as e:
            logger.error(f"[STORE] Unexpected error for transaction_id {transaction_id}: {str(e)}")
            raise DedupException(f"Failed to store customer: {str(e)}")

    async def search_similar_customers(self, transaction_id: str, image_data: bytes, query_metadata: Dict,
                                       threshold: float, limit: int) -> List[SearchResult]:
        """Search for similar customers based on image similarity"""
        try:
            logger.info(f"[SEARCH] Processing search request with threshold: {threshold}, limit: {limit}")

            # Verify embedding service is ready (should be initialized at startup)
            if not self.embedding_service.is_ready():
                logger.error(f"[SEARCH] Embedding service not ready - this should not happen after startup")
                raise VectorServiceError("Face analysis service is not available")

            # Generate embedding from query image with face detection
            query_embedding = await self.embedding_service.generate_embedding(image_data, context="SEARCH")

            # Search for similar vectors
            results = await self.vector_store.search_similar_vectors(
                query_embedding, query_metadata, threshold, limit
            )

            # Convert to SearchResult objects
            search_results = []
            for result in results:
                search_result = SearchResult(
                    similarity_score=result["similarity_score"],
                    metadata=StoreMetadata(**result["metadata"])
                )
                search_results.append(search_result)

            logger.info(
                f"[SEARCH] Search completed: {len(search_results)} matches found for transaction_id: {transaction_id}")
            return search_results

        except (InvalidRequestError, DedupException, VectorServiceError):
            raise
        except Exception as e:
            logger.error(f"[SEARCH] Unexpected error during search: {str(e)}")
            raise DedupException(f"Failed to search customers: {str(e)}")

    async def purge_customer(self, transaction_id: str) -> bool:
        """Delete customer record"""
        try:
            logger.info(f"[PURGE] Processing purge request for transaction_id: {transaction_id}")

            result = await self.vector_store.delete_customer(transaction_id)

            logger.info(f"[PURGE] Successfully purged transaction_id: {transaction_id}")
            return result

        except CustomerNotFoundError:
            logger.warning(f"[PURGE] Transaction ID not found: {transaction_id}")
            raise
        except Exception as e:
            logger.error(f"[PURGE] Unexpected error for transaction_id {transaction_id}: {str(e)}")
            raise DedupException(f"Failed to purge customer: {str(e)}")

    async def onboarding_check(self, transaction_id: str, image_data: bytes, identity: Dict,
                               threshold: Optional[float] = None, limit: Optional[int] = None,
                               image_path: str = "", created_on: Optional[str] = None) -> Dict:
        """Screen an applicant face against the gallery and return a fraud verdict.

        Retrieves near-duplicate faces (>= t_candidate), compares the applicant's
        identity to each match, and classifies the best match into
        CLEAR / REVIEW / DUPLICATE_SAME_IDENTITY / FRAUD_ALERT_DIFFERENT_IDENTITY
        with a 0-100 risk score and explainability reason codes. Enrols the face
        only when the verdict is CLEAR (or REVIEW when enroll_on_review is set).
        """
        logger.info(f"[CHECK] Onboarding check for transaction_id: {transaction_id}")

        if not self.embedding_service.is_ready():
            raise VectorServiceError("Face analysis service is not available")

        # 1) Embed the applicant face (raises InvalidRequestError if no face found)
        embedding = await self.embedding_service.generate_embedding(image_data, context="CHECK")

        # 2) Retrieve candidate near-duplicates from the gallery
        t_candidate = threshold if threshold is not None else settings.t_candidate
        search_limit = limit or 10
        raw = await self.vector_store.search_similar_vectors(
            embedding, {}, t_candidate, search_limit
        )

        # 3) Compare the applicant identity against every candidate
        matches = []
        for m in raw:
            md = m["metadata"]
            match_identity = {f: md.get(f) for f in IDENTITY_FIELDS}
            cmp = fraud_decision.compare_identity(identity, match_identity)
            matches.append({
                "transaction_id": m["transaction_id"],
                "similarity_score": round(float(m["similarity_score"]), 4),
                "identity_match": fraud_decision.is_same_identity(cmp),
                "field_diffs": cmp["diffs"],
                "identity": match_identity,
                "image_path": md.get("image_path", ""),
            })

        # 4) Classify the best (highest-similarity) match
        best = matches[0] if matches else None
        if best:
            best_cmp = fraud_decision.compare_identity(identity, best["identity"])
            best_sim = best["similarity_score"]
        else:
            best_cmp = {"diffs": [], "matches": []}
            best_sim = 0.0

        verdict, reason_codes = fraud_decision.classify(
            best_sim, best_cmp, has_match=bool(best),
            t_match=settings.t_match, t_review=settings.t_review,
        )
        if len(matches) > 1 and verdict != fraud_decision.CLEAR:
            reason_codes.append("MULTIPLE_MATCHES")
        risk = fraud_decision.risk_score(verdict, best_sim, best_cmp, settings.t_review)

        # 5) Store-if-clean policy (never auto-enrol a duplicate/fraud face)
        enrolled = False
        should_enrol = (verdict == fraud_decision.CLEAR) or (
            verdict == fraud_decision.REVIEW and settings.enroll_on_review)
        if should_enrol and not await self.vector_store.customer_exists(transaction_id):
            saved_path = image_path
            try:
                os.makedirs(settings.gallery_images_dir, exist_ok=True)
                saved_path = os.path.join(settings.gallery_images_dir, f"{transaction_id}.jpg")
                with open(saved_path, "wb") as fh:
                    fh.write(image_data)
            except Exception as e:
                logger.warning(f"[CHECK] Could not persist gallery image: {e}")
                saved_path = image_path
            meta = StoreMetadata(
                created_on=created_on or (datetime.utcnow().isoformat() + "Z"),
                image_path=saved_path or "",
                **{f: identity.get(f) for f in IDENTITY_FIELDS},
            )
            await self.vector_store.store_vector(transaction_id, embedding, meta)
            enrolled = True

        logger.info(f"[CHECK] transaction_id={transaction_id} verdict={verdict} "
                    f"risk={risk} enrolled={enrolled} matches={len(matches)}")

        return {
            "transaction_id": transaction_id,
            "verdict": verdict,
            "risk_score": risk,
            "reason_codes": reason_codes,
            "enrolled": enrolled,
            "query_identity": {f: identity.get(f) for f in IDENTITY_FIELDS},
            "best_match": best,
            "matches": matches,
            "total_matches": len(matches),
        }

    async def health_check(self) -> Dict:
        """Check health of all services"""
        try:
            redis_healthy = await self.vector_store.health_check()
            embedding_healthy = self.embedding_service.is_ready()

            return {
                "redis": redis_healthy,
                "embedding": embedding_healthy,
                "overall": redis_healthy and embedding_healthy
            }
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "redis": False,
                "embedding": False,
                "overall": False
            }


# Global instance
dedup_service = DeduplicationService()