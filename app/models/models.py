"""
Database models for PCD-AI Service.
baselines (1) -> comparisons (N)

Baseline identified by: task_location_checks_image_id (unique ID from backend).
Comparison identified by: taskcheck_execution_id (unique ID from backend).
Matching (true/false) is set by supervisor via feedback, NOT by the pipeline.
"""
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, Text,
    DateTime, ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.orm import relationship
from app.core.database import Base


class Baseline(Base):
    __tablename__ = "ai_baselines"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_location_checks_image_id = Column(Integer, nullable=False, unique=True, index=True)

    # Image storage
    image_url = Column(Text, nullable=False)
    enhanced_cache_path = Column(String(500), nullable=True)  # CLAHE preprocessed
    color_cache_path = Column(String(500), nullable=True)     # Color-preserved copy

    # Quality metrics (computed on registration)
    blur_score = Column(Float, nullable=True)
    brightness = Column(Float, nullable=True)
    quality_warning = Column(Boolean, default=False)

    # Lifecycle
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    comparisons = relationship("Comparison", back_populates="baseline", lazy="dynamic")

    def __repr__(self):
        return f"<Baseline id={self.id} img_id={self.task_location_checks_image_id} active={self.is_active}>"


class Comparison(Base):
    __tablename__ = "ai_comparisons"

    id = Column(Integer, primary_key=True, autoincrement=True)
    taskcheck_execution_id = Column(Integer, nullable=False, unique=True, index=True)

    # Relationships
    baseline_id = Column(Integer, ForeignKey("ai_baselines.id"), nullable=False, index=True)
    baseline = relationship("Baseline", back_populates="comparisons")

    # Input
    patrol_image_url = Column(Text, nullable=False)

    # -- Results --
    ratio = Column(Float, nullable=True)                # SVM probability (change %)
    similarity_percent = Column(Float, nullable=True)   # (1 - ratio) * 100
    matching = Column(Boolean, nullable=True)            # Set by supervisor via feedback, NOT pipeline
    status = Column(String(30), default="PENDING")       # PENDING | PROCESSING | COMPLETED | FAILED | FRAUD

    # -- Fraud Detection (M2) --
    is_valid = Column(Boolean, nullable=True)      # CLIP passed
    fraud_score = Column(Float, nullable=True)     # CLIP cosine similarity

    # -- AI Feature Scores (M4) --
    # Structure path
    ssim_score = Column(Float, nullable=True)
    ms_ssim_score = Column(Float, nullable=True)
    edge_score = Column(Float, nullable=True)
    histogram_score = Column(Float, nullable=True)
    cluster_score = Column(Float, nullable=True)
    max_diff_area = Column(Float, nullable=True)
    stability_score = Column(Float, nullable=True)
    # Color path
    max_hue_shift = Column(Float, nullable=True)
    max_delta_e = Column(Float, nullable=True)

    # -- Alignment (M3) --
    alignment_inliers = Column(Integer, nullable=True)
    alignment_method = Column(String(20), nullable=True)  # lightglue | orb

    # -- Quality --
    blur_score = Column(Float, nullable=True)
    heatmap_path = Column(String(500), nullable=True)
    processing_ms = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)

    # -- Supervisor Feedback --
    supervisor_reviewed_at = Column(DateTime, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_ai_comparisons_status", "status"),
        Index("idx_ai_comparisons_baseline_created", "baseline_id", "created_at"),
    )

    def __repr__(self):
        return f"<Comparison id={self.id} exec={self.taskcheck_execution_id} status={self.status}>"
