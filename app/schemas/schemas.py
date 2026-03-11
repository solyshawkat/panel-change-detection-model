"""
Pydantic schemas for API request/response validation.
Organized by service: Baseline, Comparison, Feedback.

Backend contract:
  - Baseline identifies by: task_location_checks_image_id
  - Comparison identifies by: taskcheck_execution_id + task_location_checks_image_id
  - Feedback: supervisor sets matching (true/false)
"""
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl
from typing import Optional


# ═══════════════════════════════════════════
#  BASELINE SERVICE
# ═══════════════════════════════════════════

class BaselineCreateRequest(BaseModel):
    """POST /baseline - Register a new baseline image."""
    task_location_checks_image_id: int = Field(..., description="Unique ID of the baseline image (from backend)")
    image_url: str = Field(..., description="URL to download the baseline image")

    model_config = {"json_schema_extra": {
        "example": {
            "task_location_checks_image_id": 5001,
            "image_url": "https://objectstorage.me-jeddah-1.oraclecloud.com/n/namespace/b/bucket/o/baseline_001.jpg"
        }
    }}


class BaselineResponse(BaseModel):
    """Response after baseline registration."""
    id: int
    task_location_checks_image_id: int
    is_active: bool
    blur_score: Optional[float] = None
    brightness: Optional[float] = None
    quality_warning: bool = False
    created_at: datetime

    model_config = {"from_attributes": True}


# ═══════════════════════════════════════════
#  COMPARISON ENGINE
# ═══════════════════════════════════════════

class CompareRequest(BaseModel):
    """POST /compare - Request a comparison between baseline and patrol image."""
    taskcheck_execution_id: int = Field(..., description="Comparison image unique ID (from backend)")
    task_location_checks_image_id: int = Field(..., description="Unique ID of the baseline image (looks up baseline)")
    image_url: str = Field(..., description="URL to download the patrol image")

    model_config = {"json_schema_extra": {
        "example": {
            "taskcheck_execution_id": 9001,
            "task_location_checks_image_id": 5001,
            "image_url": "https://objectstorage.me-jeddah-1.oraclecloud.com/n/namespace/b/bucket/o/patrol_5023.jpg"
        }
    }}


class CompareResponse(BaseModel):
    """Response after comparison completes. Supervisor sets matching via feedback."""
    id: int
    baseline_id: int
    taskcheck_execution_id: int
    status: str
    difference_percent: float = 0
    is_valid: Optional[bool] = None
    fraud_score: Optional[float] = None
    color_shift_detected: Optional[bool] = None
    processing_ms: Optional[int] = None
    heatmap_url: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class CompareDetailResponse(CompareResponse):
    """GET /result/{id} - Full comparison details including all feature scores."""
    # Structure features
    ssim_score: Optional[float] = None
    ms_ssim_score: Optional[float] = None
    edge_score: Optional[float] = None
    histogram_score: Optional[float] = None
    cluster_score: Optional[float] = None
    max_diff_area: Optional[float] = None
    stability_score: Optional[float] = None
    # Color features
    max_hue_shift: Optional[float] = None
    max_delta_e: Optional[float] = None
    # Alignment
    alignment_inliers: Optional[int] = None
    alignment_method: Optional[str] = None
    # Quality
    blur_score: Optional[float] = None
    # Supervisor feedback
    matching: Optional[bool] = None

    model_config = {"from_attributes": True}


# ═══════════════════════════════════════════
#  FEEDBACK SERVICE
# ═══════════════════════════════════════════

class FeedbackRequest(BaseModel):
    """POST /feedback - Supervisor says whether images match or not."""
    task_location_checks_image_id: int = Field(..., description="Baseline image ID")
    taskcheck_execution_id: int = Field(..., description="Comparison execution ID")
    matching: bool = Field(..., description="true = images match (no change), false = images differ (change detected)")

    model_config = {"json_schema_extra": {
        "example": {
            "task_location_checks_image_id": 5001,
            "taskcheck_execution_id": 9001,
            "matching": False
        }
    }}


class FeedbackResponse(BaseModel):
    """Response after feedback is recorded."""
    task_location_checks_image_id: int
    taskcheck_execution_id: int
    matching: bool
    recorded_at: datetime
    message: str


class AccuracyStats(BaseModel):
    """GET /accuracy - Model accuracy based on supervisor feedback."""
    total_reviewed: int
    true_positives: int   # model predicted change, supervisor confirmed (matching=false)
    false_positives: int  # model predicted change, supervisor said matching (matching=true)
    true_negatives: int   # model predicted normal, supervisor confirmed (matching=true)
    false_negatives: int  # model predicted normal, supervisor found change (matching=false)
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    last_updated: datetime


class ServiceStats(BaseModel):
    """GET /stats - General service statistics."""
    total_baselines: int
    active_baselines: int
    total_comparisons: int
    comparisons_today: int
    avg_processing_ms: Optional[float] = None
    fraud_detected_count: int
    pending_review_count: int


# ═══════════════════════════════════════════
#  HEALTH
# ═══════════════════════════════════════════

class HealthResponse(BaseModel):
    status: str
    version: str
    database: str
    models_loaded: bool
