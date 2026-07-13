import redis
import redis.sentinel
import json
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from app.config import settings
from app.models.requests import StoreMetadata
from app.services.fraud_decision import IDENTITY_FIELDS
from app.services.vector_store import VectorStore
from app.utils.exceptions import VectorServiceError, CustomerNotFoundError
import time

logger = logging.getLogger(__name__)


class RedisVectorService(VectorStore):
    """Optimized Redis-based vector storage and similarity search service"""

    backend_name = "redis"

    def __init__(self):
        # Setup connection pool based on configuration
        if settings.redis_use_sentinel:
            self.sentinel = redis.sentinel.Sentinel(
                settings.redis_sentinel_hosts,
                socket_timeout=5.0,
                socket_connect_timeout=5.0
            )
            self.connection_pool = None
        else:
            # Single instance setup
            self.connection_pool = redis.ConnectionPool.from_url(
                settings.redis_url,
                password=settings.redis_password,
                db=settings.redis_db,
                max_connections=settings.redis_max_connections,
                retry_on_timeout=True
            )
            self.sentinel = None

        # RediSearch configuration
        self.vector_key_prefix = "vec:"
        self.index_name = "vector_index"
        self.vector_field = "vec"
        self.created_on_field = "created_on"

        self._setup_vector_index()

    def _get_connection(self) -> redis.Redis:
        """Get Redis connection from pool or sentinel"""
        if self.sentinel:
            # Use sentinel to get master connection
            conn = self.sentinel.master_for(
                settings.redis_sentinel_service_name,
                password=settings.redis_password,
                db=settings.redis_db,
                socket_timeout=5.0,
                socket_connect_timeout=5.0,
                decode_responses=False
            )
            total_keys = conn.dbsize()
            logger.debug(f"[DEBUG] App connected to Redis with {total_keys} keys")
            return conn
        else:
            # Use connection pool for single instance
            return redis.Redis(connection_pool=self.connection_pool, decode_responses=False)

    def _vector_schema_args(self) -> list:
        """Build the VECTOR field definition for FLAT (exact) or HNSW (approximate)."""
        common = [
            "TYPE", "FLOAT32",
            "DIM", settings.vector_dimension,
            "DISTANCE_METRIC", "COSINE",
        ]
        if str(settings.index_type).upper() == "HNSW":
            params = common + [
                "M", settings.hnsw_m,
                "EF_CONSTRUCTION", settings.hnsw_ef_construction,
                "EF_RUNTIME", settings.hnsw_ef_runtime,
            ]
            return ["VECTOR", "HNSW", str(len(params))] + params
        return ["VECTOR", "FLAT", str(len(common))] + common

    def _setup_vector_index(self) -> None:
        """Setup RediSearch vector index for fast similarity search"""
        conn = self._get_connection()
        try:
            conn.execute_command(
                "FT.CREATE", self.index_name,
                "ON", "HASH",
                "PREFIX", "1", self.vector_key_prefix,
                "SCHEMA",
                self.vector_field, "AS", self.vector_field,
                *self._vector_schema_args(),
                self.created_on_field, "AS", self.created_on_field, "TEXT", "NOSTEM"
            )
            logger.info(f"Created {str(settings.index_type).upper()} vector search index: {self.index_name}")
        except redis.exceptions.ResponseError as e:
            if "Index already exists" not in str(e):
                logger.error(f"Failed to create vector index: {e}")
                raise VectorServiceError(f"Index creation failed: {str(e)}")

    async def store_vector(self, transaction_id: str, vector: List[float], metadata: StoreMetadata) -> bool:
        """Store vector and metadata in Redis using optimized single-hash approach"""
        api_start = time.perf_counter()
        try:
            logger.info(f"[STORE] Initiating vector storage for transaction_id: {transaction_id}")

            # Validate vector dimension
            if len(vector) != settings.vector_dimension:
                raise VectorServiceError(
                    f"Vector dimension mismatch: expected {settings.vector_dimension}, got {len(vector)}")

            conn = self._get_connection()

            # Prepare vector for storage
            vector_array = np.array(vector, dtype=np.float32)

            # Single hash key for all data (optimized approach)
            vector_key = f"{self.vector_key_prefix}{transaction_id}"

            # Store everything in a single hash. Identity fields are coerced to ""
            # when absent (Redis hashes reject None values).
            hash_data = {
                self.vector_field: vector_array.tobytes(),
                self.created_on_field: metadata.created_on,
                "image_path": metadata.image_path,
            }
            for f in IDENTITY_FIELDS:
                hash_data[f] = getattr(metadata, f, None) or ""

            # Single HSET operation
            conn.hset(vector_key, mapping=hash_data)

            processing_time = time.perf_counter() - api_start
            logger.info(
                f"[STORE] Vector storage completed for transaction_id: {transaction_id} in {processing_time:.3f}s")
            return True

        except VectorServiceError:
            raise
        except Exception as e:
            logger.error(f"[STORE] Failed to store vector for {transaction_id}: {str(e)}")
            raise VectorServiceError(f"Failed to store vector: {str(e)}")

    async def search_similar_vectors(self, query_vector: List[float], query_metadata: Dict, threshold: float,
                                     limit: Optional[int] = None) -> List[Dict]:
        """Search for similar vectors using RediSearch with COSINE distance"""
        api_start = time.perf_counter()
        try:
            logger.debug(f"[SEARCH] Initiating similarity search with threshold: {threshold}, limit: {limit}")

            # Validate query vector
            if len(query_vector) != settings.vector_dimension:
                raise VectorServiceError(
                    f"Query vector dimension mismatch: expected {settings.vector_dimension}, got {len(query_vector)}")

            query_array = np.array(query_vector, dtype=np.float32)
            search_limit = limit or settings.max_search_results

            conn = self._get_connection()

            logger.debug(f"[SEARCH] Executing RediSearch KNN query")

            # KNN over the whole index; return the score + metadata fields per hit.
            return_fields = ["score", self.created_on_field, "image_path"] + IDENTITY_FIELDS
            result = conn.execute_command(
                "FT.SEARCH", self.index_name,
                f"*=>[KNN {search_limit} @{self.vector_field} $vec AS score]",
                "RETURN", str(len(return_fields)), *return_fields,
                "LIMIT", "0", search_limit,
                "PARAMS", "2", "vec", query_array.tobytes(),
                "DIALECT", "2"
            )

            filtered_results = []
            total_candidates = (len(result) - 1) // 2

            logger.debug(f"[SEARCH] Found {total_candidates} candidates from Redis")

            # Parse results - alternating [doc_key, fields_array, doc_key, fields_array, ...]
            for i in range(1, len(result), 2):
                if i + 1 >= len(result):
                    break

                doc_key = result[i].decode() if isinstance(result[i], bytes) else result[i]
                doc_fields = result[i + 1]

                if len(doc_fields) >= 2:
                    # Parse fields array into a dictionary
                    fields_dict = {}
                    for j in range(0, len(doc_fields), 2):
                        if j + 1 < len(doc_fields):
                            field_name = doc_fields[j].decode() if isinstance(doc_fields[j], bytes) else doc_fields[j]
                            field_value = doc_fields[j + 1]
                            fields_dict[field_name] = field_value

                    # Extract required fields
                    try:
                        # Get raw COSINE distance from Redis (range: 0-2)
                        cosine_distance = float(fields_dict.get('score', 0.0))

                        # Convert COSINE distance to similarity score
                        # COSINE distance range: [0, 2] where 0 = identical, 2 = opposite
                        # Similarity range: [0, 1] where 1 = perfect match, 0 = no match
                        similarity_score = (2.0 - cosine_distance) / 2.0

                        logger.debug(
                            f"[SEARCH] Candidate: distance={cosine_distance:.4f}, similarity={similarity_score:.4f}, threshold={threshold}")

                        if similarity_score >= threshold:
                            transaction_id = doc_key.replace(self.vector_key_prefix, "")
                            created_on = fields_dict.get(self.created_on_field, b'').decode() if isinstance(
                                fields_dict.get(self.created_on_field), bytes) else fields_dict.get(
                                self.created_on_field, '')

                            metadata = {
                                "created_on": created_on,
                                "image_path": fields_dict.get("image_path", b'').decode() if isinstance(
                                    fields_dict.get("image_path"), bytes) else fields_dict.get("image_path", '')
                            }
                            # Attach the identity fields (decoded, "" if absent)
                            for _f in IDENTITY_FIELDS:
                                _v = fields_dict.get(_f, '')
                                metadata[_f] = _v.decode() if isinstance(_v, bytes) else (_v or '')

                            filtered_results.append({
                                "transaction_id": transaction_id,
                                "similarity_score": similarity_score,
                                "metadata": metadata
                            })

                    except (ValueError, IndexError) as e:
                        logger.warning(f"[SEARCH] Error parsing search result: {e}")
                        continue

            processing_time = time.perf_counter() - api_start
            logger.info(
                f"[SEARCH] Search completed: {len(filtered_results)}/{total_candidates} matches above threshold {threshold} in {processing_time:.3f}s")

            return sorted(filtered_results, key=lambda x: x["similarity_score"], reverse=True)

        except VectorServiceError:
            raise
        except Exception as e:
            logger.error(f"[SEARCH] Vector search failed: {str(e)}")
            raise VectorServiceError(f"Failed to search vectors: {str(e)}")

    async def delete_customer(self, transaction_id: str) -> bool:
        """Delete customer vector and metadata"""
        api_start = time.perf_counter()
        try:
            logger.info(f"[PURGE] Initiating deletion for transaction_id: {transaction_id}")

            conn = self._get_connection()
            vector_key = f"{self.vector_key_prefix}{transaction_id}"

            if not conn.exists(vector_key):
                logger.warning(f"[PURGE] Transaction ID not found: {transaction_id}")
                raise CustomerNotFoundError(transaction_id)

            # Single delete operation
            result = conn.delete(vector_key)

            processing_time = time.perf_counter() - api_start
            logger.info(f"[PURGE] Deletion completed for transaction_id: {transaction_id} in {processing_time:.3f}s")

            return bool(result)

        except CustomerNotFoundError:
            raise
        except Exception as e:
            logger.error(f"[PURGE] Failed to delete transaction_id {transaction_id}: {str(e)}")
            raise VectorServiceError(f"Failed to delete customer: {str(e)}")

    async def customer_exists(self, transaction_id: str) -> bool:
        """Check if customer exists in the database"""
        try:
            conn = self._get_connection()
            vector_key = f"{self.vector_key_prefix}{transaction_id}"
            exists = bool(conn.exists(vector_key))
            logger.debug(f"Customer existence check for {transaction_id}: {exists}")
            return exists
        except Exception as e:
            logger.error(f"Failed to check customer existence for {transaction_id}: {str(e)}")
            raise VectorServiceError(f"Failed to check customer existence: {str(e)}")

    async def health_check(self) -> bool:
        """Check Redis connection health"""
        try:
            conn = self._get_connection()
            ping_result = conn.ping()
            logger.debug(f"Redis health check: {'OK' if ping_result else 'FAILED'}")
            return ping_result
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def get_customer_metadata(self, transaction_id: str) -> Dict:
        """Retrieve customer metadata for a given transaction_id"""
        try:
            conn = self._get_connection()
            vector_key = f"{self.vector_key_prefix}{transaction_id}"

            if not conn.exists(vector_key):
                raise CustomerNotFoundError(transaction_id)

            # Get all hash fields except the vector bytes
            metadata_fields = [
                self.created_on_field,
                "image_path",
            ] + IDENTITY_FIELDS

            result = conn.hmget(vector_key, metadata_fields)

            # Build metadata dictionary
            metadata = {}
            for i, field in enumerate(metadata_fields):
                if result[i] is not None:
                    metadata[field] = result[i].decode() if isinstance(result[i], bytes) else result[i]

            logger.debug(f"Retrieved metadata for {transaction_id}: {metadata}")
            return metadata

        except CustomerNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to get metadata for {transaction_id}: {str(e)}")
            raise VectorServiceError(f"Failed to retrieve customer metadata: {str(e)}")

    async def count_records(self) -> int:
        """Return the number of enrolled face records in the index."""
        try:
            conn = self._get_connection()
            res = conn.execute_command("FT.SEARCH", self.index_name, "*", "LIMIT", "0", "0")
            return int(res[0]) if res else 0
        except Exception as e:
            logger.warning(f"count_records failed: {e}")
            return 0

# NOTE: the backend singleton is created lazily by
# app.services.vector_store.get_vector_store() so importing this module never
# opens a Redis connection (important when the Pinecone backend is active).
