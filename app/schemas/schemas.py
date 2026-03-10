"""
Pydantic schemas for API request/response validation.
Organized by service: Baseline, Comparison, Feedback.
"""
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl
from typing import Optional


# ═══════════════════════════════════════════
#  BASELINE SERVICE
# ═══════════════════════════════════════════

class BaselineCreateRequest(BaseModel):
    """POST /baseline - Register a new baseline image."""
    location_id: str = Field(..., description="Location identifier", max_length=50)
    taskcheck_id: int = Field(..., description="Task check ID (location_id + taskcheck_id = unique panel)")
    image_url: str = Field(..., description="URL to download the baseline image")

    model_config = {"json_schema_extra": {
        "example": {
            "location_id": "LOC-FLOOR3",
            "taskcheck_id": 1001,
            "image_url": "https://objectstorage.me-jeddah-1.oraclecloud.com/n/namespace/b/bucket/o/baseline_001.jpg"
        }
    }}


class BaselineUpdateRequest(BaseModel):
    """PUT /baseline - Replace baseline with new image."""
    image_url: str = Field(..., description="URL to the new baseline image")

    model_config = {"json_schema_extra": {
        "example": {
            "image_url": "https://objectstorage.me-jeddah-1.oraclecloud.com/n/namespace/b/bucket/o/baseline_002.jpg"
        }
    }}


class BaselineResponse(BaseModel):
    """Response after baseline registration."""
    id: int
    location_id: str
    taskcheck_id: int
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
    location_id: str = Field(..., description="Location identifier")
    taskcheck_id: int = Field(..., description="Task check ID (looks up active baseline)")
    image_url: str = Field(..., description="URL to download the patrol image")

    model_config = {"json_schema_extra": {
        "example": {
            "location_id": "LOC-FLOOR3",
            "taskcheck_id": 1001,
            "image_url": "https://objectstorage.me-jeddah-1.oraclecloud.com/n/namespace/b/bucket/o/patrol_5023.jpg"
        }
    }}


class CompareResponse(BaseModel):
    """Response after comparison completes."""
    id: int
    baseline_id: int
    status: str
    ratio: Optional[float] = None
    matching: Optional[bool] = None
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
    # Feedback
    supervisor_action: Optional[str] = None
    supervisor_notes: Optional[str] = None

    model_config = {"from_attributes": True}


# ═══════════════════════════════════════════
#  FEEDBACK SERVICE
# ═══════════════════════════════════════════

class FeedbackRequest(BaseModel):
    """POST /feedback - Supervisor submits a review."""
    comparison_id: int = Field(..., description="Comparison to review")
    action: str = Field(..., description="CONFIRM_CHANGE | REJECT_CHANGE | CONFIRM_NORMAL")
    notes: Optional[str] = Field(None, description="Optional supervisor notes")

    model_config = {"json_schema_extra": {
        "example": {
            "comparison_id": 42,
            "action": "CONFIRM_CHANGE",
            "notes": "Switch 3 in row B was flipped"
        }
    }}


class FeedbackResponse(BaseModel):
    """Response after feedback is recorded."""
    comparison_id: int
    action: str
    recorded_at: datetime
    message: str


class AccuracyStats(BaseModel):
    """GET /accuracy - Model accuracy based on supervisor feedback."""
    total_reviewed: int
    true_positives: int   # model said CHANGED, supervisor confirmed
    false_positives: int  # model said CHANGED, supervisor rejected
    true_negatives: int   # model said NORMAL, supervisor confirmed
    false_negatives: int  # model said NORMAL, supervisor said it changed
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
