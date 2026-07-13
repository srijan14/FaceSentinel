import os
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import Optional, Set, List, Tuple

class Settings(BaseSettings):
    # API Configuration
    api_title: str = "FaceSentinel Deduplication API"
    api_version: str = "v1"
    api_description: str = "Facial de-duplication & synthetic-identity fraud screening for KYC onboarding"

    # ------------------------------------------------------------------ #
    # Vector backend selection: "auto" | "pinecone" | "redis" | "memory"
    #   auto   -> pinecone when PINECONE_API_KEY is set, otherwise redis
    #   memory -> in-process, non-persistent (zero external services)
    # ------------------------------------------------------------------ #
    vector_backend: str = "auto"

    # Redis Configuration (Local / self-hosted RediSearch)
    redis_use_sentinel: bool = False
    redis_url: str = "redis://localhost:6379/0"
    redis_password: Optional[str] = None
    redis_db: int = 0
    redis_max_connections: int = 20
    redis_sentinel_hosts: List[Tuple[str, int]] = []
    redis_sentinel_service_name: str = "mymaster"

    # Pinecone Configuration (managed, serverless vector DB)
    pinecone_api_key: Optional[str] = None
    pinecone_index_name: str = "facesentinel-kyc"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_namespace: str = "kyc"
    pinecone_metric: str = "cosine"

    # Vector Configuration
    vector_dimension: int = 512
    default_similarity_threshold: float = 0.6
    max_search_results: int = 500
    default_top_k: int = 500

    # Vector index (Redis only): "FLAT" (exact brute-force KNN) or "HNSW" (approximate, fast at scale)
    index_type: str = "FLAT"
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_runtime: int = 64

    # ------------------------------------------------------------------ #
    # Face-embedding backend: "auto" | "insightface" | "demo"
    #   auto  -> insightface when ONNX models are present in models/,
    #            otherwise a deterministic demo embedding (no model download).
    #   demo  -> force deterministic embeddings (great for a portable demo /
    #            lightweight deployment where the 200MB ArcFace model is absent).
    # ------------------------------------------------------------------ #
    embedding_mode: str = "auto"

    # ------------------------------------------------------------------ #
    # Hosted read-only demo switch.
    #   True  (default, local dev): live ArcFace inference + custom image
    #         uploads are enabled; the gallery is indexed by running the model.
    #   False (small hosted box, e.g. Linode): live inference and custom
    #         uploads are DISABLED. The app serves only the pre-indexed demo
    #         dataset and screens planted probes via *precomputed* embeddings,
    #         so no 200 MB model needs to load. Set INFERENCE_ENABLED=false.
    # ------------------------------------------------------------------ #
    inference_enabled: bool = True

    # Fraud-decision thresholds, in the rescaled [0,1] similarity space
    # (1.0 = identical face, ~0.5 = unrelated). Tuned via benchmarks/accuracy_lfw.py.
    t_candidate: float = 0.50   # retrieve neighbours at/above this (recall floor)
    t_review: float = 0.60      # below this = CLEAR; at/above = at least REVIEW
    t_match: float = 0.65       # at/above this = strong same-face match
    enroll_on_review: bool = False   # auto-enroll REVIEW verdicts (default: hold)

    # Where enrolled face images are saved (for review-console thumbnails)
    gallery_images_dir: str = "data/gallery_images"

    # Authentication (dev placeholder — override with the API_KEY env var in prod)
    api_key: str = "dev-local-key-change-me"

    # File Upload
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_image_types: str = "image/jpeg,image/png,image/jpg"

    @field_validator('allowed_image_types')
    @classmethod
    def parse_allowed_image_types(cls, v):
        if isinstance(v, str):
            return set(v.split(','))
        return v

    @property
    def allowed_image_types_set(self) -> Set[str]:
        """Get allowed image types as a set"""
        if isinstance(self.allowed_image_types, str):
            return set(self.allowed_image_types.split(','))
        return self.allowed_image_types

    @property
    def resolved_vector_backend(self) -> str:
        """Resolve the effective vector backend, honouring 'auto'."""
        backend = (self.vector_backend or "auto").strip().lower()
        if backend == "auto":
            return "pinecone" if self.pinecone_api_key else "redis"
        return backend

    def resolved_embedding_mode(self, models_present: bool) -> str:
        """Resolve the effective embedding mode, honouring 'auto'."""
        mode = (self.embedding_mode or "auto").strip().lower()
        if mode == "auto":
            return "insightface" if models_present else "demo"
        return mode

    class Config:
        env_file = ".env"

settings = Settings()
