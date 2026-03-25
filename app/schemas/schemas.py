"""
Pydantic schemas for API request/response validation.
Organized by service: Baseline, Comparison, Feedback.

Backend contract (Java sends camelCase):
  - Baseline: ReferenceImageUploadedRequest
  - Compare: ImageComparisonRequest
  - Feedback: MatchingFeedbackRequest
"""
from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field
from typing import Dict, List, Optional


# ═══════════════════════════════════════════
#  BASELINE SERVICE
# ═══════════════════════════════════════════

class BaselineCreateRequest(BaseModel):
    """POST /baseline - Register a new baseline image.
    Maps to Java: ReferenceImageUploadedRequest"""
    task_location_checks_image_id: int = Field(..., alias="taskLocationChecksImageId", description="Unique ID of the baseline image")
    reference_image_url: str = Field(..., alias="referenceImageUrl", description="URL to download the baseline image")

    model_config = ConfigDict(populate_by_name=True, json_schema_extra={
        "example": {
            "taskLocationChecksImageId": 5001,
            "referenceImageUrl": "https://objectstorage.me-jeddah-1.oraclecloud.com/n/namespace/b/bucket/o/baseline_001.jpg"
        }
    })


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
    """POST /compare - Request a comparison between baseline and patrol image.
    Maps to Java: ImageComparisonRequest"""
    task_check_execution_id: int = Field(..., alias="taskCheckExecutionId", description="Comparison image unique ID")
    task_location_checks_image_id: int = Field(..., alias="taskLocationChecksImageId", description="Baseline image ID (looks up baseline)")
    reference_image_path: str = Field(..., alias="referenceImagePath", description="Baseline image URL (fallback if cache missing)")
    evidence_image_path: str = Field(..., alias="evidenceImagePath", description="Patrol/evidence image URL to compare")

    model_config = ConfigDict(populate_by_name=True, json_schema_extra={
        "example": {
            "taskCheckExecutionId": 9001,
            "taskLocationChecksImageId": 5001,
            "referenceImagePath": "https://objectstorage.../baseline_001.jpg",
            "evidenceImagePath": "https://objectstorage.../patrol_5023.jpg"
        }
    })


class CompareResponse(BaseModel):
    """Response after comparison completes. Supervisor sets matching via feedback."""
    id: int
    baseline_id: int
    task_check_execution_id: int
    status: str
    ratio: float = 0
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
    """POST /feedback - Supervisor says whether images match or not.
    Maps to Java: MatchingFeedbackRequest"""
    task_check_execution_id: int = Field(..., alias="taskCheckExecutionId", description="Comparison execution ID")
    task_location_checks_image_id: int = Field(..., alias="taskLocationChecksImageId", description="Baseline image ID")
    matching: bool = Field(..., description="true = images match (no change), false = images differ (change detected)")

    model_config = ConfigDict(populate_by_name=True, json_schema_extra={
        "example": {
            "taskCheckExecutionId": 9001,
            "taskLocationChecksImageId": 5001,
            "matching": False
        }
    })


class FeedbackResponse(BaseModel):
    """Response after feedback is recorded."""
    task_location_checks_image_id: int
    task_check_execution_id: int
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
#  MODEL RETRAINING
# ═══════════════════════════════════════════

class PerClassMetrics(BaseModel):
    """Per-class precision, recall, F1."""
    precision: float
    recall: float
    f1: float
    support: int


class RetrainResponse(BaseModel):
    """POST /retrain - Model retraining results."""
    status: str = Field(..., description="'success', 'rejected', or 'error'")
    message: str
    total_samples: int = 0
    positive_samples: int = 0  # changed (matching=false)
    negative_samples: int = 0  # normal (matching=true)
    cv_accuracy: Optional[float] = None
    cv_std: Optional[float] = None
    cv_f1: Optional[float] = None
    cv_f1_std: Optional[float] = None
    per_class_metrics: Optional[Dict[str, PerClassMetrics]] = None
    feature_importance: Optional[Dict[str, float]] = None
    model_version: Optional[int] = None
    model_path: Optional[str] = None
    trained_at: Optional[datetime] = None


# ═══════════════════════════════════════════
#  OBJECT VERIFICATION
# ═══════════════════════════════════════════

class VerifyRequest(BaseModel):
    """POST /verify - Check if two images show the same object and photo quality."""
    image_base64_1: str = Field(..., alias="imageBase64_1", description="Baseline image as base64 string")
    image_base64_2: str = Field(..., alias="imageBase64_2", description="Patrol image as base64 string to check")

    model_config = ConfigDict(populate_by_name=True, json_schema_extra={
        "example": {
            "imageBase64_1": "<base64 encoded baseline image>",
            "imageBase64_2": "<base64 encoded patrol image>"
        }
    })


class VerifyResponse(BaseModel):
    """Response: same object check + photo quality indicators."""
    same_object: bool = Field(..., alias="sameObject", description="true if images show the same object")
    is_blurry: Optional[bool] = Field(None, alias="isBlurry", description="true if patrol photo is blurry. null if different object")
    is_bright: Optional[bool] = Field(None, alias="isBright", description="true if brightness is acceptable. null if different object")
    is_aligned: Optional[bool] = Field(None, alias="isAligned", description="true if angle/position is close enough. null if different object")

    model_config = ConfigDict(populate_by_name=True)


# ═══════════════════════════════════════════
#  HEALTH
# ═══════════════════════════════════════════

class HealthResponse(BaseModel):
    status: str
    version: str
    database: str
    models_loaded: bool
