"""
PCD-AI Service — Main Application
Panel Change Detection AI Service for C2Guard Platform

Three services:
  /pcd-ai/baseline  — Register and manage baseline reference images
  /pcd-ai/compare   — Run the comparison pipeline
  /pcd-ai/feedback  — Supervisor reviews and model accuracy tracking

Registers with Eureka for service discovery by Eden backend.
"""
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core import config
from app.core.database import init_db, close_db
from app.core.eureka import register as eureka_register, deregister as eureka_deregister
from app.api.baseline import router as baseline_router
from app.api.compare import router as compare_router
from app.api.feedback import router as feedback_router
from app.schemas.schemas import HealthResponse
import logging

# ── Logging ──
logging.basicConfig(
    level=logging.DEBUG if config.DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pcd-ai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info(f"Starting {config.APP_NAME} v{config.APP_VERSION}")

    # Create tables (dev mode — use Alembic migrations in production)
    await init_db()
    logger.info("Database initialized")

    # Ensure storage directories exist
    Path(config.BASELINE_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.HEATMAP_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.MODEL_DIR).mkdir(parents=True, exist_ok=True)
    logger.info("Storage directories ready")

    # Register with Eureka
    await eureka_register()

    yield

    # Shutdown
    await eureka_deregister()
    await close_db()
    logger.info("Shutdown complete")


# ── App ──
app = FastAPI(
    title=config.APP_NAME,
    version=config.APP_VERSION,
    description=(
        "AI service for detecting changes in electrical and fire suppression panels. "
        "Compares guard patrol photographs against stored baseline images using a "
        "dual-path pipeline: QR Detection → Validation (Structure + Color) → "
        "Fraud Detection → Alignment → Feature Extraction → SVM Classification."
    ),
    lifespan=lifespan,
    root_path=config.CONTEXT_PATH,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ──
app.include_router(baseline_router)
app.include_router(compare_router)
app.include_router(feedback_router)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Service health check."""
    from app.pipeline.runner import get_pipeline
    pipeline = get_pipeline()

    return HealthResponse(
        status="healthy",
        version=config.APP_VERSION,
        database="connected",
        models_loaded=pipeline._models_loaded,
    )


@app.get("/", tags=["Health"])
async def root():
    """Service info."""
    return {
        "service": config.APP_NAME,
        "version": config.APP_VERSION,
        "context_path": config.CONTEXT_PATH,
        "docs": f"{config.CONTEXT_PATH}/docs",
        "endpoints": {
            "baseline": {
                "POST /baseline/": "Register new baseline",
                "GET /baseline/{panel_id}": "Get active baseline",
                "PUT /baseline/{panel_id}": "Replace baseline",
            },
            "compare": {
                "POST /compare/": "Run comparison",
                "GET /compare/result/{id}": "Get comparison details",
                "GET /compare/result/{id}/heatmap": "Download heatmap",
            },
            "feedback": {
                "POST /feedback/": "Submit supervisor review",
                "GET /feedback/accuracy": "Model accuracy stats",
                "GET /feedback/stats": "Service statistics",
            },
        },
    }
