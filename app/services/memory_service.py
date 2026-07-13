"""
In-memory vector store — a zero-dependency backend for FaceSentinel.

Implements the :class:`~app.services.vector_store.VectorStore` interface with a
plain in-process dict + NumPy cosine search. It needs **no external service**
(no Redis, no Pinecone), so ``VECTOR_BACKEND=memory`` lets ``uvicorn app.main:app``
boot anywhere for a quick demo, local development, or tests.

Trade-off: it is single-process and non-persistent (the gallery is lost on
restart). Use Pinecone or Redis for anything you want to keep.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional

import numpy as np

from app.config import settings
from app.models.requests import StoreMetadata
from app.services.fraud_decision import IDENTITY_FIELDS
from app.services.vector_store import VectorStore
from app.utils.exceptions import VectorServiceError, CustomerNotFoundError

logger = logging.getLogger(__name__)

_META_FIELDS = ["created_on", "image_path"] + IDENTITY_FIELDS


class MemoryVectorService(VectorStore):
    """Non-persistent, in-process vector store (cosine similarity)."""

    backend_name = "memory"

    def __init__(self):
        self._lock = threading.RLock()
        self._vectors: Dict[str, np.ndarray] = {}   # unit-normalised
        self._meta: Dict[str, Dict[str, str]] = {}
        logger.info("In-memory vector store ready (non-persistent).")

    @staticmethod
    def _unit(vec: List[float]) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        return arr / norm if norm > 0 else arr

    @staticmethod
    def _clean_metadata(metadata: StoreMetadata) -> Dict[str, str]:
        md = {"created_on": metadata.created_on or "", "image_path": metadata.image_path or ""}
        for f in IDENTITY_FIELDS:
            md[f] = getattr(metadata, f, None) or ""
        return md

    async def store_vector(self, transaction_id: str, vector: List[float],
                           metadata: StoreMetadata) -> bool:
        if len(vector) != settings.vector_dimension:
            raise VectorServiceError(
                f"Vector dimension mismatch: expected {settings.vector_dimension}, got {len(vector)}")
        with self._lock:
            self._vectors[transaction_id] = self._unit(vector)
            self._meta[transaction_id] = self._clean_metadata(metadata)
        return True

    async def search_similar_vectors(self, query_vector: List[float], query_metadata: Dict,
                                     threshold: float, limit: Optional[int] = None) -> List[Dict]:
        if len(query_vector) != settings.vector_dimension:
            raise VectorServiceError(
                f"Query vector dimension mismatch: expected {settings.vector_dimension}, got {len(query_vector)}")
        top_k = int(limit or settings.max_search_results)
        q = self._unit(query_vector)
        with self._lock:
            items = list(self._vectors.items())
            meta_snapshot = dict(self._meta)
        results = []
        for tid, vec in items:
            cos = float(np.dot(q, vec))
            similarity = max(0.0, min(1.0, (1.0 + cos) / 2.0))
            if similarity < threshold:
                continue
            md = meta_snapshot.get(tid, {})
            metadata = {"created_on": md.get("created_on", ""), "image_path": md.get("image_path", "")}
            for f in IDENTITY_FIELDS:
                metadata[f] = md.get(f, "") or ""
            results.append({"transaction_id": tid, "similarity_score": similarity, "metadata": metadata})
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:top_k]

    async def delete_customer(self, transaction_id: str) -> bool:
        with self._lock:
            if transaction_id not in self._vectors:
                raise CustomerNotFoundError(transaction_id)
            self._vectors.pop(transaction_id, None)
            self._meta.pop(transaction_id, None)
        return True

    async def customer_exists(self, transaction_id: str) -> bool:
        with self._lock:
            return transaction_id in self._vectors

    async def health_check(self) -> bool:
        return True

    async def get_customer_metadata(self, transaction_id: str) -> Dict:
        with self._lock:
            if transaction_id not in self._meta:
                raise CustomerNotFoundError(transaction_id)
            md = self._meta[transaction_id]
            return {f: md.get(f) for f in _META_FIELDS if md.get(f) not in (None, "")}

    async def count_records(self) -> int:
        with self._lock:
            return len(self._vectors)
