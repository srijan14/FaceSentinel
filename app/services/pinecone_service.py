"""
Pinecone-backed vector store for FaceSentinel.

Implements the :class:`~app.services.vector_store.VectorStore` interface on top
of Pinecone serverless — a fully managed, zero-ops vector database. This is the
recommended backend for a hosted deployment: no Redis to run, scales to millions
of faces, and creates its own index on first boot.

Similarity space
----------------
Pinecone's ``cosine`` metric returns a similarity ``s`` in ``[-1, 1]``
(1 = identical). We rescale to the app's ``[0, 1]`` space with
``similarity = (1 + s) / 2`` so the fraud thresholds in ``app/config.py`` behave
identically to the Redis backend (which rescales COSINE *distance* the same way).
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import numpy as np

from app.config import settings
from app.models.requests import StoreMetadata
from app.services.fraud_decision import IDENTITY_FIELDS
from app.services.vector_store import VectorStore
from app.utils.exceptions import VectorServiceError, CustomerNotFoundError

logger = logging.getLogger(__name__)

# Metadata fields persisted alongside every vector.
_META_FIELDS = ["created_on", "image_path"] + IDENTITY_FIELDS


class PineconeVectorService(VectorStore):
    """Vector storage & 1:N similarity search backed by Pinecone serverless."""

    backend_name = "pinecone"

    def __init__(self):
        self.namespace = settings.pinecone_namespace or ""
        self.index_name = settings.pinecone_index_name
        self._index = None
        self._connect()

    # ------------------------------------------------------------------ #
    # Connection / index bootstrap
    # ------------------------------------------------------------------ #
    def _connect(self) -> None:
        if not settings.pinecone_api_key:
            raise VectorServiceError(
                "Pinecone backend selected but PINECONE_API_KEY is not set. "
                "Set it in your environment/.env, or use VECTOR_BACKEND=redis."
            )
        try:
            from pinecone import Pinecone, ServerlessSpec
        except ImportError as e:  # pragma: no cover - dependency guard
            raise VectorServiceError(
                "The 'pinecone' package is not installed. Run: pip install pinecone"
            ) from e

        try:
            self._pc = Pinecone(api_key=settings.pinecone_api_key)
            self._ensure_index(ServerlessSpec)
            self._index = self._pc.Index(self.index_name)
            logger.info("Connected to Pinecone index '%s' (namespace='%s')",
                        self.index_name, self.namespace)
        except VectorServiceError:
            raise
        except Exception as e:
            raise VectorServiceError(f"Failed to connect to Pinecone: {e}")

    def _index_exists(self) -> bool:
        try:
            # Newer SDKs expose has_index(); fall back to listing names.
            if hasattr(self._pc, "has_index"):
                return bool(self._pc.has_index(self.index_name))
            names = self._pc.list_indexes().names()
            return self.index_name in names
        except Exception:
            names = [getattr(ix, "name", ix.get("name") if isinstance(ix, dict) else None)
                     for ix in self._pc.list_indexes()]
            return self.index_name in names

    def _ensure_index(self, ServerlessSpec) -> None:
        if self._index_exists():
            return
        logger.info("Creating Pinecone index '%s' (dim=%d, metric=%s, %s/%s)...",
                    self.index_name, settings.vector_dimension, settings.pinecone_metric,
                    settings.pinecone_cloud, settings.pinecone_region)
        self._pc.create_index(
            name=self.index_name,
            dimension=settings.vector_dimension,
            metric=settings.pinecone_metric,
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )
        # Wait until the index reports ready (serverless is usually a few seconds).
        for _ in range(60):
            try:
                desc = self._pc.describe_index(self.index_name)
                ready = desc["status"]["ready"] if isinstance(desc, dict) else desc.status["ready"]
                if ready:
                    logger.info("Pinecone index '%s' is ready.", self.index_name)
                    return
            except Exception:
                pass
            time.sleep(1)
        logger.warning("Pinecone index '%s' not confirmed ready after 60s; continuing.",
                       self.index_name)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _clean_metadata(metadata: StoreMetadata) -> Dict[str, str]:
        """Pinecone metadata rejects None; coerce every field to a string."""
        md: Dict[str, str] = {
            "created_on": metadata.created_on or "",
            "image_path": metadata.image_path or "",
        }
        for f in IDENTITY_FIELDS:
            md[f] = getattr(metadata, f, None) or ""
        return md

    @staticmethod
    def _to_similarity(score: float) -> float:
        """Map Pinecone cosine similarity [-1, 1] -> app similarity [0, 1]."""
        return max(0.0, min(1.0, (1.0 + float(score)) / 2.0))

    @staticmethod
    def _match_fields(match) -> tuple:
        """Extract (id, score, metadata) from a Pinecone match (obj or dict)."""
        if isinstance(match, dict):
            return match.get("id"), match.get("score", 0.0), match.get("metadata") or {}
        return getattr(match, "id", None), getattr(match, "score", 0.0), getattr(match, "metadata", {}) or {}

    # ------------------------------------------------------------------ #
    # VectorStore interface
    # ------------------------------------------------------------------ #
    async def store_vector(self, transaction_id: str, vector: List[float],
                           metadata: StoreMetadata) -> bool:
        api_start = time.perf_counter()
        try:
            if len(vector) != settings.vector_dimension:
                raise VectorServiceError(
                    f"Vector dimension mismatch: expected {settings.vector_dimension}, got {len(vector)}")

            values = np.asarray(vector, dtype=np.float32).tolist()
            self._index.upsert(
                vectors=[{"id": transaction_id, "values": values,
                          "metadata": self._clean_metadata(metadata)}],
                namespace=self.namespace,
            )
            logger.info("[STORE] Upserted %s to Pinecone in %.3fs",
                        transaction_id, time.perf_counter() - api_start)
            return True
        except VectorServiceError:
            raise
        except Exception as e:
            logger.error("[STORE] Pinecone upsert failed for %s: %s", transaction_id, e)
            raise VectorServiceError(f"Failed to store vector: {e}")

    async def search_similar_vectors(self, query_vector: List[float], query_metadata: Dict,
                                     threshold: float, limit: Optional[int] = None) -> List[Dict]:
        api_start = time.perf_counter()
        try:
            if len(query_vector) != settings.vector_dimension:
                raise VectorServiceError(
                    f"Query vector dimension mismatch: expected {settings.vector_dimension}, got {len(query_vector)}")

            top_k = int(limit or settings.max_search_results)
            values = np.asarray(query_vector, dtype=np.float32).tolist()
            resp = self._index.query(
                vector=values, top_k=top_k, include_metadata=True, namespace=self.namespace,
            )
            matches = resp["matches"] if isinstance(resp, dict) else getattr(resp, "matches", [])

            results = []
            for m in matches or []:
                mid, score, md = self._match_fields(m)
                similarity = self._to_similarity(score)
                if similarity < threshold:
                    continue
                metadata = {"created_on": md.get("created_on", ""), "image_path": md.get("image_path", "")}
                for f in IDENTITY_FIELDS:
                    metadata[f] = md.get(f, "") or ""
                results.append({
                    "transaction_id": mid,
                    "similarity_score": similarity,
                    "metadata": metadata,
                })

            logger.info("[SEARCH] Pinecone returned %d/%d hits >= %.3f in %.3fs",
                        len(results), len(matches or []), threshold, time.perf_counter() - api_start)
            return sorted(results, key=lambda x: x["similarity_score"], reverse=True)
        except VectorServiceError:
            raise
        except Exception as e:
            logger.error("[SEARCH] Pinecone query failed: %s", e)
            raise VectorServiceError(f"Failed to search vectors: {e}")

    async def delete_customer(self, transaction_id: str) -> bool:
        try:
            if not await self.customer_exists(transaction_id):
                raise CustomerNotFoundError(transaction_id)
            self._index.delete(ids=[transaction_id], namespace=self.namespace)
            logger.info("[PURGE] Deleted %s from Pinecone", transaction_id)
            return True
        except CustomerNotFoundError:
            raise
        except Exception as e:
            logger.error("[PURGE] Pinecone delete failed for %s: %s", transaction_id, e)
            raise VectorServiceError(f"Failed to delete customer: {e}")

    async def customer_exists(self, transaction_id: str) -> bool:
        try:
            resp = self._index.fetch(ids=[transaction_id], namespace=self.namespace)
            vectors = resp["vectors"] if isinstance(resp, dict) else getattr(resp, "vectors", {})
            return transaction_id in (vectors or {})
        except Exception as e:
            logger.error("Failed to check existence for %s: %s", transaction_id, e)
            raise VectorServiceError(f"Failed to check customer existence: {e}")

    async def health_check(self) -> bool:
        try:
            self._index.describe_index_stats()
            return True
        except Exception as e:
            logger.error("Pinecone health check failed: %s", e)
            return False

    async def get_customer_metadata(self, transaction_id: str) -> Dict:
        try:
            resp = self._index.fetch(ids=[transaction_id], namespace=self.namespace)
            vectors = resp["vectors"] if isinstance(resp, dict) else getattr(resp, "vectors", {})
            if transaction_id not in (vectors or {}):
                raise CustomerNotFoundError(transaction_id)
            rec = vectors[transaction_id]
            md = rec["metadata"] if isinstance(rec, dict) else getattr(rec, "metadata", {})
            return {f: md.get(f) for f in _META_FIELDS if md.get(f) not in (None, "")}
        except CustomerNotFoundError:
            raise
        except Exception as e:
            logger.error("Failed to get metadata for %s: %s", transaction_id, e)
            raise VectorServiceError(f"Failed to retrieve customer metadata: {e}")

    async def count_records(self) -> int:
        try:
            stats = self._index.describe_index_stats()
            # Normalise the response (dict, dict-subclass, or object) to a dict.
            if hasattr(stats, "to_dict"):
                stats = stats.to_dict()
            elif not isinstance(stats, dict):
                stats = {"namespaces": getattr(stats, "namespaces", {}) or {},
                         "total_vector_count": getattr(stats, "total_vector_count", 0)}
            namespaces = stats.get("namespaces") or {}

            def _vc(ns) -> int:
                if isinstance(ns, dict):
                    return int(ns.get("vector_count", 0))
                return int(getattr(ns, "vector_count", 0))

            if self.namespace in namespaces:
                return _vc(namespaces[self.namespace])
            # Empty namespace requested -> aggregate across all namespaces.
            if self.namespace == "" and namespaces:
                return int(sum(_vc(ns) for ns in namespaces.values()))
            # Namespace not created yet (empty) -> fall back to the index total.
            return int(stats.get("total_vector_count", 0)) if not namespaces else 0
        except Exception as e:
            logger.warning("count_records failed: %s", e)
            return 0
