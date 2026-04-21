"""
PCD-AI Service Configuration
All settings loaded from environment variables with sensible defaults.
Matches Eden service conventions (Eureka, context path, DB structure).
"""
import os

# ===============================
# Application
# ===============================
APP_NAME = "pcd-ai-service"
APP_VERSION = "1.0.0"
APP_PORT = int(os.getenv("APPLICATION_PORT", 8100))
CONTEXT_PATH = "/pcd-ai"
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# ===============================
# Database
# ===============================
DATABASE = {
    "host": os.getenv("DB_HOST", "10.80.3.231"),
    "port": int(os.getenv("DB_PORT", 5433)),
    "name": os.getenv("DB_NAME", "eden_crm_sec_crm"),
    "username": os.getenv("DB_USERNAME", "pcd_user"),
    "password": os.getenv("DB_PASSWORD", ""),
    "schema": os.getenv("DB_SCHEMA", "public"),
}

# SQLAlchemy connection strings
# Support both a full DATABASE_URL env var or individual DB_* vars
_db_url = os.getenv("DATABASE_URL")
if _db_url:
    # Ensure asyncpg driver for async usage
    DATABASE_URL = _db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    DATABASE_URL_SYNC = _db_url.replace("postgresql+asyncpg://", "postgresql://", 1)
else:
    DATABASE_URL = (
        f"postgresql+asyncpg://{DATABASE['username']}:{DATABASE['password']}"
        f"@{DATABASE['host']}:{DATABASE['port']}/{DATABASE['name']}"
    )
    DATABASE_URL_SYNC = (
        f"postgresql://{DATABASE['username']}:{DATABASE['password']}"
        f"@{DATABASE['host']}:{DATABASE['port']}/{DATABASE['name']}"
    )

# ===============================
# Eureka Service Discovery
# ===============================
EUREKA = {
    "enabled": os.getenv("EUREKA_ENABLED", "true").lower() == "true",
    "server_url": os.getenv("EUREKA_URL", "http://127.0.0.1:8761/eureka"),
    "instance_host": os.getenv("EUREKA_INSTANCE_HOST", ""),  # auto-detect if empty
    "prefer_ip": True,
}

# ===============================
# Storage Paths
# ===============================
STORAGE_ROOT = os.getenv("STORAGE_ROOT", "storage")
BASELINE_DIR = os.getenv("BASELINE_DIR", "storage/baselines")
HEATMAP_DIR = os.getenv("HEATMAP_DIR", "storage/heatmaps")
MODEL_DIR = os.getenv("MODEL_DIR", "ml_models")

# ===============================
# ML Model Files
# ===============================
SVM_MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
FEATURE_CONFIG_PATH = os.path.join(MODEL_DIR, "feature_config.json")
FRAUD_THRESHOLD_PATH = os.path.join(MODEL_DIR, "fraud_threshold.json")
RF_MODEL_DIR = os.path.join(MODEL_DIR, "rf_versions")
RF_MODEL_CURRENT = os.path.join(MODEL_DIR, "model_rf_current.joblib")

# ===============================
# Pipeline Thresholds
# ===============================
FRAUD_THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", 0.751))
VERIFY_THRESHOLD = float(os.getenv("VERIFY_THRESHOLD", 0.85))
VERIFY_THRESHOLD_DINO = float(os.getenv("VERIFY_THRESHOLD_DINO", 0.60))
SVM_THRESHOLD = float(os.getenv("SVM_THRESHOLD", 0.389))
ALIGNMENT_MIN_INLIERS = int(os.getenv("ALIGNMENT_MIN_INLIERS", 10))
BLUR_WARNING_THRESHOLD = float(os.getenv("BLUR_WARNING_THRESHOLD", 30.0))
BRIGHTNESS_WARNING_THRESHOLD = float(os.getenv("BRIGHTNESS_WARNING_THRESHOLD", 80.0))
DELTA_E_THRESHOLD = float(os.getenv("DELTA_E_THRESHOLD", 5.0))

# ===============================
# Oracle Cloud (Image Storage)
# ===============================
ORACLE_CLOUD = {
    "enabled": os.getenv("ORACLE_CLOUD_ENABLED", "false").lower() == "true",
    "namespace": os.getenv("ORACLE_CLOUD_NAMESPACE", ""),
    "bucket": os.getenv("ORACLE_CLOUD_BUCKET", ""),
    "region": os.getenv("ORACLE_CLOUD_REGION", "me-jeddah-1"),
    "access_key": os.getenv("ORACLE_CLOUD_ACCESS_KEY", ""),
    "secret_key": os.getenv("ORACLE_CLOUD_SECRET_KEY", ""),
    "par_base_url": os.getenv("ORACLE_CLOUD_PAR_BASE_URL", ""),
}

# ===============================
# Image Download
# ===============================
IMAGE_DOWNLOAD_TIMEOUT = int(os.getenv("IMAGE_DOWNLOAD_TIMEOUT", 30))
IMAGE_MAX_SIZE_MB = int(os.getenv("IMAGE_MAX_SIZE_MB", 20))

# ===============================
# CORS
# ===============================
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
