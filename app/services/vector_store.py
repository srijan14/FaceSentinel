"""
Pluggable vector-store backend for FaceSentinel.

The de-duplication service talks to *this* interface, never to a concrete
database. Two backends implement it:

    * RedisVectorService     -> Redis Stack / RediSearch (self-hosted, FLAT/HNSW)
    * PineconeVectorService  -> Pinecone serverless (managed, zero-ops)

The active backend is chosen at runtime from ``settings.resolved_vector_backend``
(``auto`` picks Pinecone when ``PINECONE_API_KEY`` is set, else Redis). Only the
selected backend is imported/instantiated, so you never need Redis running to use
Pinecone (or vice-versa).

Every backend returns search hits in the app's rescaled ``[0, 1]`` similarity
space (1.0 = identical face, ~0.5 = unrelated) so the fraud thresholds in
``app/config.py`` are backend-independent.
"""
from __future__ import annotations

import abc
import logging
from typing import Dict, List, Optional

from app.config import settings
from app.models.requests import StoreMetadata

logger = logging.getLogger(__name__)


class VectorStore(abc.ABC):
    """Common interface every vector backend must implement."""

    #: Human-readable backend name, surfaced in /health and /stats.
    backend_name: str = "vector"

    @abc.abstractmethod
    async def store_vector(self, transaction_id: str, vector: List[float],
                           metadata: StoreMetadata) -> bool: ...

    @abc.abstractmethod
    async def search_similar_vectors(self, query_vector: List[float], query_metadata: Dict,
                                     threshold: float, limit: Optional[int] = None) -> List[Dict]: ...

    @abc.abstractmethod
    async def delete_customer(self, transaction_id: str) -> bool: ...

    @abc.abstractmethod
    async def customer_exists(self, transaction_id: str) -> bool: ...

    @abc.abstractmethod
    async def health_check(self) -> bool: ...

    @abc.abstractmethod
    async def get_customer_metadata(self, transaction_id: str) -> Dict: ...

    @abc.abstractmethod
    async def count_records(self) -> int: ...


_INSTANCE: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Return the process-wide vector store singleton for the configured backend.

    Instantiation is lazy so importing this module never opens a database
    connection, and only the selected backend's driver is imported.
    """
    global _INSTANCE
    if _INSTANCE is not None:
        return _INSTANCE

    backend = settings.resolved_vector_backend
    if backend == "pinecone":
        from app.services.pinecone_service import PineconeVectorService
        logger.info("Vector backend: Pinecone (index=%s, namespace=%s)",
                    settings.pinecone_index_name, settings.pinecone_namespace)
        _INSTANCE = PineconeVectorService()
    elif backend == "redis":
        from app.services.redis_service import RedisVectorService
        logger.info("Vector backend: Redis (%s, index=%s)",
                    settings.redis_url, str(settings.index_type).upper())
        _INSTANCE = RedisVectorService()
    elif backend == "memory":
        from app.services.memory_service import MemoryVectorService
        logger.info("Vector backend: in-memory (non-persistent, zero-dependency)")
        _INSTANCE = MemoryVectorService()
    else:
        raise ValueError(
            f"Unknown vector backend '{backend}'. Set VECTOR_BACKEND to "
            f"'auto', 'pinecone', 'redis', or 'memory'."
        )
    return _INSTANCE
