from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import asyncio

from app.config import settings
from app.routers import dedup
from app.utils.exceptions import DedupException
from app.services.embedding import embedding_service
from app.services.vector_store import get_vector_store

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events"""
    
    # Startup
    logger.info("Starting application initialization...")
    
    try:
        # Initialize embedding service
        logger.info("Initializing embedding service...")
        embedding_initialized = await embedding_service.initialize()
        
        if not embedding_initialized:
            logger.error("Failed to initialize embedding service")
            raise RuntimeError("Embedding service initialization failed")
        
        # Connect to the configured vector backend (Redis or Pinecone)
        vector_store = get_vector_store()
        backend = getattr(vector_store, "backend_name", settings.resolved_vector_backend)
        vector_healthy = await vector_store.health_check()

        if vector_healthy:
            logger.info(f"Established vector-store connection (backend={backend})")
        else:
            logger.error(f"Vector-store connection failed (backend={backend})")
            raise RuntimeError(f"Vector-store connection failed (backend={backend})")

        logger.info("Application initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Application startup failed: {str(e)}")
        raise
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("Application shutdown initiated")

# Create FastAPI app with lifespan
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(dedup.router)

@app.exception_handler(DedupException)
async def dedup_exception_handler(request: Request, exc: DedupException):
    """Global exception handler for DedupException"""
    status_code_map = {
        "AUTHENTICATION_FAILED": 401,
        "CUSTOMER_NOT_FOUND": 404,
        "INVALID_REQUEST": 400,
        "CUSTOMER_EXISTS": 409,
        "VECTOR_SERVICE_ERROR": 500,
        "INTERNAL_ERROR": 500
    }

    status_code = status_code_map.get(exc.code, 500)

    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": None
            }
        }
    )

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": settings.api_title,
        "version": settings.api_version,
        "description": settings.api_description,
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.get("/health")
async def health():
    """Comprehensive health check endpoint"""
    try:
        # Check the configured vector backend
        vector_store = get_vector_store()
        backend = getattr(vector_store, "backend_name", settings.resolved_vector_backend)
        vector_healthy = await vector_store.health_check()

        # Check embedding service
        embedding_status = embedding_service.get_health_status()
        embedding_healthy = embedding_status["service_ready"]

        overall_healthy = vector_healthy and embedding_healthy

        return {
            "status": "healthy" if overall_healthy else "unhealthy",
            "services": {
                "vector_backend": backend,
                "vector_store": vector_healthy,
                # kept for backward compatibility with older clients:
                "redis": vector_healthy if backend == "redis" else None,
                "embedding": embedding_healthy,
                "embedding_mode": embedding_status.get("embedding_mode"),
                "overall": overall_healthy,
            },
            "embedding_details": embedding_status,
        }

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "services": {
                "vector_store": False,
                "embedding": False,
                "overall": False
            }
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)