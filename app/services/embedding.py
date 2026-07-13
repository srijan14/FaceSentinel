import numpy as np
import os
import logging
import time
from typing import List, Optional
from app.config import settings
from app.services.demo_embedding import demo_embedding_from_bytes
from app.utils.exceptions import VectorServiceError, InvalidRequestError

logger = logging.getLogger(__name__)

def get_max_area_detection(detections):
    """Get detection with maximum bounding box area"""
    def bbox_area(bbox):
        bbox = np.array(bbox, dtype=np.float32)
        x1, y1, x2, y2 = bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    return max(detections, key=lambda d: bbox_area(d['bbox']))

class EmbeddingService:
    """Production service for generating face embeddings using InsightFace models"""

    def __init__(self):
        self.dimension = settings.vector_dimension
        self.face_analysis = None
        self._initialized = False
        self._initialization_failed = False
        # Resolved at initialize(): "insightface" (real ArcFace) or "demo"
        # (deterministic embeddings, no model download).
        self._mode: Optional[str] = None

        # Model paths
        self.models_path = "models"

        logger.info("EmbeddingService instance created")

    def _check_models_exist(self) -> bool:
        """Check if required model files exist in models directory"""
        if not os.path.exists(self.models_path):
            return False

        # Check for any ONNX files in the models directory
        onnx_files = [f for f in os.listdir(self.models_path) if f.endswith(".onnx")]
        models_exist = len(onnx_files) > 0
        logger.debug(f"Models check: {models_exist}, found {len(onnx_files)} ONNX files")
        return models_exist

    async def initialize(self) -> bool:
        """Initialize the face analysis model (called once at startup)"""
        if self._initialized:
            logger.debug("EmbeddingService already initialized")
            return True

        if self._initialization_failed:
            logger.error("EmbeddingService initialization previously failed")
            return False

        # Resolve the embedding mode (auto -> insightface if models exist, else demo).
        self._mode = settings.resolved_embedding_mode(self._check_models_exist())
        if self._mode == "demo":
            self._initialized = True
            logger.warning(
                "EmbeddingService running in DEMO mode: deterministic embeddings "
                "(no ArcFace model). Set EMBEDDING_MODE=insightface with ONNX "
                "models in models/ for real face matching."
            )
            return True

        try:
            logger.debug("Starting EmbeddingService initialization...")
            logger.info("Models are ready, proceeding with FaceAnalysis initialization...")

            # Step 2: Import and initialize face analysis
            try:
                from app.core.src.face_analysis import FaceAnalysis
                logger.debug("FaceAnalysis import successful")
            except ImportError as e:
                logger.error(f"Failed to import FaceAnalysis: {str(e)}")
                self._initialization_failed = True
                return False

            # Step 3: Initialize the face analysis model
            logger.debug("Creating FaceAnalysis instance...")
            self.face_analysis = FaceAnalysis(
                allowed_modules=["detection", "recognition"]
            )

            logger.debug("Preparing FaceAnalysis model with GPU context...")
            self.face_analysis.prepare(ctx_id=0)

            # Step 4: Verify initialization
            if not self.face_analysis or not hasattr(self.face_analysis, 'models'):
                logger.error("FaceAnalysis initialization appears incomplete")
                self._initialization_failed = True
                return False

            self._initialized = True
            logger.info("EmbeddingService initialized successfully")
            
            # Log model details for debugging
            if hasattr(self.face_analysis, 'models'):
                model_count = len(self.face_analysis.models)
                logger.debug(f"Loaded {model_count} models: {list(self.face_analysis.models.keys())}")
            
            return True

        except Exception as e:
            logger.error(f"CRITICAL: Failed to initialize EmbeddingService: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            self._initialization_failed = True
            return False

    async def generate_embedding(self, image_data: bytes, context: str = "UNKNOWN") -> List[float]:
        """
        Generate face embedding from image data
        
        Args:
            image_data: Raw image bytes
            context: Context for logging (STORE/SEARCH)
            
        Returns:
            List[float]: Face embedding vector
            
        Raises:
            VectorServiceError: When model is not initialized
            InvalidRequestError: When no face is detected or image is invalid
        """
        api_start = time.perf_counter()

        try:
            logger.debug(f"[{context}] Starting face embedding generation")

            # Demo mode: deterministic embedding straight from the image bytes,
            # no model / no OpenCV required (validate it is a decodable image).
            if self._mode == "demo":
                self._validate_demo_image(image_data)
                embedding = demo_embedding_from_bytes(image_data, self.dimension)
                logger.info(f"[{context}] Demo embedding generated in "
                            f"{time.perf_counter() - api_start:.3f}s")
                return embedding

            # Check if service is initialized
            if not self._initialized or not self.face_analysis:
                logger.error(f"[{context}] Face analysis model not initialized")
                raise VectorServiceError("Face analysis model not available")

            # Convert bytes to OpenCV image
            image = self._bytes_to_cv_image(image_data)
            
            # Get face detections and embeddings
            detections = self.face_analysis.get(image)
            
            if len(detections) == 0:
                logger.warning(f"[{context}] No face detected in the provided image")
                raise InvalidRequestError("No face detected in the provided image")

            # If multiple faces, use the one with largest bounding box
            if len(detections) > 1:
                logger.debug(f"[{context}] Multiple faces detected ({len(detections)}), using largest")
                detection = get_max_area_detection(detections)
            else:
                detection = detections[0]

            embedding = detection.embedding
            
            processing_time = time.perf_counter() - api_start
            logger.info(f"[{context}] Face embedding generated successfully in {processing_time:.3f}s")
            
            return embedding.tolist()

        except InvalidRequestError:
            raise
        except Exception as e:
            logger.error(f"[{context}] Embedding generation failed: {str(e)}")
            raise VectorServiceError(f"Failed to generate embedding: {str(e)}")

    def _bytes_to_cv_image(self, image_data: bytes):
        """Convert image bytes to OpenCV format"""
        try:
            import cv2  # lazy import: not needed (or installed) in demo mode
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                raise InvalidRequestError("Failed to decode image data")

            return image

        except InvalidRequestError:
            raise
        except Exception as e:
            raise InvalidRequestError(f"Invalid image format: {str(e)}")

    def _validate_demo_image(self, image_data: bytes) -> None:
        """Cheaply verify the upload is a decodable image (demo mode)."""
        if not image_data:
            raise InvalidRequestError("Empty image payload")
        try:
            from io import BytesIO
            from PIL import Image
            Image.open(BytesIO(image_data)).verify()
        except InvalidRequestError:
            raise
        except Exception:
            # Fall back to a magic-byte sniff if Pillow is unavailable.
            if not (image_data[:3] == b"\xff\xd8\xff" or image_data[:8] == b"\x89PNG\r\n\x1a\n"):
                raise InvalidRequestError("Invalid image format: expected JPEG/PNG")

    def is_ready(self) -> bool:
        """Check if the service is ready to process requests"""
        if self._mode == "demo":
            return self._initialized
        return self._initialized and self.face_analysis is not None

    def get_health_status(self) -> dict:
        """Get detailed health status for diagnostics"""
        model_info = {}
        
        if self.face_analysis and hasattr(self.face_analysis, "models"):
            model_info = {
                "available_models": list(self.face_analysis.models.keys()),
                "model_count": len(self.face_analysis.models),
            }

        return {
            "service_ready": self.is_ready(),
            "initialized": self._initialized,
            "initialization_failed": self._initialization_failed,
            "embedding_mode": self._mode,
            "models_available": self._check_models_exist(),
            "models_path": self.models_path,
            "face_analysis_info": model_info,
        }


# Global instance
embedding_service = EmbeddingService()