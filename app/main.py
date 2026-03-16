"""
PCD-AI Service — Main Application
Panel Change Detection AI Service for C2Guard Platform

Three services:
  /pcd-ai/baseline  — Register and manage baseline reference images
  /pcd-ai/compare   — Run the comparison pipeline
  /pcd-ai/feedback  — Supervisor reviews and model accuracy tracking

Registers with Eureka for service discovery by Eden backend.

Note: Gateway forwards full path (no StripPrefix), so all routes include /pcd-ai prefix.
      Do NOT use root_path — it would double-prefix Swagger docs.
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
from app.api.retrain import router as retrain_router
from app.schemas.schemas import HealthResponse
import logging

logging.basicConfig(
    level=logging.DEBUG if config.DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pcd-ai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {config.APP_NAME} v{config.APP_VERSION}")

    await init_db()
    logger.info("Database initialized")

    Path(config.BASELINE_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.HEATMAP_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.MODEL_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.RF_MODEL_DIR).mkdir(parents=True, exist_ok=True)
    logger.info("Storage directories ready")

    await eureka_register()

    yield

    await eureka_deregister()
    await close_db()
    logger.info("Shutdown complete")


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
    # No root_path — gateway does NOT strip /pcd-ai prefix, so routes include it directly.
    # Using root_path here would double-prefix Swagger try-it-out URLs.
    docs_url="/pcd-ai/docs",
    redoc_url="/pcd-ai/redoc",
    openapi_url="/pcd-ai/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes (prefixed with /pcd-ai to match gateway Path predicate) ──
app.include_router(baseline_router, prefix="/pcd-ai")
app.include_router(compare_router, prefix="/pcd-ai")
app.include_router(feedback_router, prefix="/pcd-ai")
app.include_router(retrain_router, prefix="/pcd-ai")


@app.get("/pcd-ai/health", response_model=HealthResponse, tags=["Health"])
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


@app.get("/pcd-ai/", tags=["Health"])
async def root():
    """Service info."""
    return {
        "service": config.APP_NAME,
        "version": config.APP_VERSION,
        "context_path": config.CONTEXT_PATH,
        "docs": "/pcd-ai/docs",
        "endpoints": {
            "baseline": {
                "POST /pcd-ai/baseline/": "Register new baseline",
                "GET /pcd-ai/baseline/": "Get active baseline",
                "PUT /pcd-ai/baseline/": "Replace baseline",
            },
            "compare": {
                "POST /pcd-ai/compare/": "Run comparison",
                "GET /pcd-ai/compare/result/{id}": "Get comparison details",
                "GET /pcd-ai/compare/result/{id}/heatmap": "Download heatmap",
            },
            "feedback": {
                "POST /pcd-ai/feedback/": "Submit supervisor review",
                "GET /pcd-ai/feedback/accuracy": "Model accuracy stats",
                "GET /pcd-ai/feedback/stats": "Service statistics",
            },
            "retrain": {
                "POST /pcd-ai/retrain/": "Retrain model from supervisor feedback",
            },
        },
    }
