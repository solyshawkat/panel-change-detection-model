"""
Model Retraining API.
Trains a Random Forest classifier from supervisor-labeled comparisons.
Auto-replaces the pixel formula when accuracy is sufficient.
"""
import asyncio
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import config
from app.core.database import get_db
from app.models.models import Comparison
from app.schemas.schemas import RetrainResponse, PerClassMetrics

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/retrain", tags=["Model Retraining"])

FEATURE_COLUMNS = [
    "ssim_score", "edge_score", "histogram_score", "cluster_score",
    "max_diff_area", "stability_score", "max_hue_shift", "max_delta_e",
    "alignment_inliers", "dino_similarity",
]

MIN_TOTAL_SAMPLES = 200
MIN_PER_CLASS = 50
MIN_F1 = 0.80
MAX_CV_STD = 0.10
MAX_VERSIONS = 3


def _train_model(X, y):
    """Synchronous training + validation. Runs in a thread."""
    import numpy as np
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import cross_val_score, cross_val_predict
    from sklearn.metrics import precision_recall_fscore_support, f1_score

    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # 5-fold cross-validation
    cv_accuracy_scores = cross_val_score(rf, X, y, cv=5, scoring="accuracy")
    cv_f1_scores = cross_val_score(rf, X, y, cv=5, scoring="f1")
    cv_accuracy = float(np.mean(cv_accuracy_scores))
    cv_std = float(np.std(cv_accuracy_scores))
    cv_f1 = float(np.mean(cv_f1_scores))
    cv_f1_std = float(np.std(cv_f1_scores))

    # Per-class metrics via cross-validated predictions
    y_pred = cross_val_predict(rf, X, y, cv=5)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, y_pred, labels=[0, 1], zero_division=0
    )
    per_class = {
        "normal": PerClassMetrics(
            precision=round(float(precision[0]), 4),
            recall=round(float(recall[0]), 4),
            f1=round(float(f1[0]), 4),
            support=int(support[0]),
        ),
        "changed": PerClassMetrics(
            precision=round(float(precision[1]), 4),
            recall=round(float(recall[1]), 4),
            f1=round(float(f1[1]), 4),
            support=int(support[1]),
        ),
    }

    metrics = {
        "cv_accuracy": round(cv_accuracy, 4),
        "cv_std": round(cv_std, 4),
        "cv_f1": round(cv_f1, 4),
        "cv_f1_std": round(cv_f1_std, 4),
        "per_class": per_class,
    }

    # Gate 3: cv_std <= 0.10 (check variance first)
    if cv_std > MAX_CV_STD:
        return {
            "status": "rejected",
            "message": (
                f"Cross-validation variance too high: std={cv_std:.4f} "
                f"(max {MAX_CV_STD}). Need more diverse training data."
            ),
            **metrics,
        }

    # Gate 4: F1 >= 0.80
    if cv_f1 < MIN_F1:
        return {
            "status": "rejected",
            "message": (
                f"F1 score too low: {cv_f1:.4f} (min {MIN_F1}). "
                f"Need more or better labeled data."
            ),
            **metrics,
        }

    # All gates passed — calibrate and train on full data
    calibrated_rf = CalibratedClassifierCV(rf, cv=5, method="isotonic")
    calibrated_rf.fit(X, y)

    # Feature importance (from the base RF, not calibrated wrapper)
    rf.fit(X, y)
    importance = dict(zip(FEATURE_COLUMNS, rf.feature_importances_.tolist()))
    importance = {k: round(v, 4) for k, v in importance}

    # Versioning: find next version number
    rf_dir = Path(config.RF_MODEL_DIR)
    rf_dir.mkdir(parents=True, exist_ok=True)

    existing_versions = sorted([
        int(f.stem.split("_v")[-1])
        for f in rf_dir.glob("model_rf_v*.joblib")
        if f.stem.split("_v")[-1].isdigit()
    ])
    next_version = (existing_versions[-1] + 1) if existing_versions else 1

    # Save versioned model
    model_data = {
        "rf": calibrated_rf,
        "features": FEATURE_COLUMNS,
        "n_samples": len(y),
        "cv_accuracy": cv_accuracy,
        "cv_f1": cv_f1,
        "version": next_version,
        "trained_at": datetime.utcnow().isoformat(),
    }

    version_path = rf_dir / f"model_rf_v{next_version}.joblib"
    temp_path = str(version_path) + ".tmp"
    joblib.dump(model_data, temp_path)
    os.replace(temp_path, str(version_path))

    # Copy to current (atomic)
    current_temp = config.RF_MODEL_CURRENT + ".tmp"
    joblib.dump(model_data, current_temp)
    os.replace(current_temp, config.RF_MODEL_CURRENT)

    # Cleanup old versions (keep last MAX_VERSIONS)
    if len(existing_versions) >= MAX_VERSIONS:
        versions_to_delete = existing_versions[:-(MAX_VERSIONS - 1)]
        for v in versions_to_delete:
            old_path = rf_dir / f"model_rf_v{v}.joblib"
            if old_path.exists():
                old_path.unlink()
                logger.info(f"Deleted old model version: {old_path.name}")

    return {
        "status": "success",
        "message": (
            f"Model v{next_version} trained and deployed. "
            f"Accuracy={cv_accuracy:.4f}, F1={cv_f1:.4f}, "
            f"Samples={len(y)}"
        ),
        **metrics,
        "feature_importance": importance,
        "model_version": next_version,
        "model_path": str(version_path),
    }


@router.post("", response_model=RetrainResponse)
async def retrain_model(db: AsyncSession = Depends(get_db)):
    """Retrain the classification model from supervisor-labeled comparisons.

    Gates:
    1. Total samples >= 200
    2. Min 50 per class
    3. CV std <= 0.10 (variance check)
    4. F1 >= 0.80

    If all gates pass, trains a calibrated Random Forest and saves
    versioned model files. The pipeline auto-loads the new model.
    """
    import numpy as np

    # Query all labeled comparisons with complete features
    feature_filters = [
        getattr(Comparison, col).isnot(None) for col in FEATURE_COLUMNS
    ]
    query = select(Comparison).where(
        and_(
            Comparison.matching.isnot(None),
            *feature_filters,
        )
    )
    result = await db.execute(query)
    comparisons = result.scalars().all()

    total = len(comparisons)
    if total == 0:
        return RetrainResponse(
            status="error",
            message="No labeled comparisons found. Supervisors need to submit feedback first.",
            total_samples=0,
            positive_samples=0,
            negative_samples=0,
        )

    # Build feature matrix and labels
    X_list = []
    y_list = []
    for comp in comparisons:
        features = [getattr(comp, col) for col in FEATURE_COLUMNS]
        X_list.append(features)
        # matching=True → 0 (normal), matching=False → 1 (changed)
        y_list.append(0 if comp.matching else 1)

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.int32)

    n_normal = int(np.sum(y == 0))
    n_changed = int(np.sum(y == 1))

    # Gate 1: total >= 200
    if total < MIN_TOTAL_SAMPLES:
        return RetrainResponse(
            status="error",
            message=(
                f"Need at least {MIN_TOTAL_SAMPLES} labeled comparisons. "
                f"Currently have {total}. Need {MIN_TOTAL_SAMPLES - total} more."
            ),
            total_samples=total,
            positive_samples=n_changed,
            negative_samples=n_normal,
        )

    # Gate 2: min 50 per class
    if n_normal < MIN_PER_CLASS or n_changed < MIN_PER_CLASS:
        return RetrainResponse(
            status="error",
            message=(
                f"Need at least {MIN_PER_CLASS} samples per class. "
                f"Normal (matching=true): {n_normal}, "
                f"Changed (matching=false): {n_changed}."
            ),
            total_samples=total,
            positive_samples=n_changed,
            negative_samples=n_normal,
        )

    # Run training in thread to avoid blocking event loop
    train_result = await asyncio.to_thread(_train_model, X, y)

    # Invalidate pipeline's cached model if training succeeded
    if train_result["status"] == "success":
        try:
            from app.pipeline.runner import get_pipeline
            get_pipeline().invalidate_rf_model()
            logger.info("Pipeline RF model cache invalidated")
        except Exception as e:
            logger.warning(f"Failed to invalidate pipeline cache: {e}")

    return RetrainResponse(
        status=train_result["status"],
        message=train_result["message"],
        total_samples=total,
        positive_samples=n_changed,
        negative_samples=n_normal,
        cv_accuracy=train_result.get("cv_accuracy"),
        cv_std=train_result.get("cv_std"),
        cv_f1=train_result.get("cv_f1"),
        cv_f1_std=train_result.get("cv_f1_std"),
        per_class_metrics=train_result.get("per_class"),
        feature_importance=train_result.get("feature_importance"),
        model_version=train_result.get("model_version"),
        model_path=train_result.get("model_path"),
        trained_at=datetime.utcnow(),
    )
