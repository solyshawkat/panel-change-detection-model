"""
Feedback Service API Routes.
Supervisor provides matching (true/false) for each comparison.

Feedback contract:
  - task_location_checks_image_id: baseline image ID
  - taskcheck_execution_id: comparison execution ID
  - matching: true = images match (no change), false = change detected
"""
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, case
from app.core.database import get_db
from app.core import config
from app.models.models import Baseline, Comparison
from app.schemas.schemas import (
    FeedbackRequest, FeedbackResponse, AccuracyStats, ServiceStats
)
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/feedback", tags=["Feedback Service"])


@router.post("/", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Supervisor says whether the patrol image matches the baseline or not.

    - matching=true  -> images match, no change detected
    - matching=false -> images differ, change detected
    """
    # Find comparison by taskcheck_execution_id
    result = await db.execute(
        select(Comparison).where(
            Comparison.task_check_execution_id == request.task_check_execution_id,
        )
    )
    comparison = result.scalar_one_or_none()
    if not comparison:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Comparison not found for taskcheck_execution_id={request.task_check_execution_id}"
        )

    # Verify baseline matches
    baseline = await db.execute(
        select(Baseline).where(Baseline.id == comparison.baseline_id)
    )
    bl = baseline.scalar_one_or_none()
    if bl and bl.task_location_checks_image_id != request.task_location_checks_image_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"task_location_checks_image_id={request.task_location_checks_image_id} "
                   f"does not match comparison's baseline (expected {bl.task_location_checks_image_id})"
        )

    if comparison.status not in ("COMPLETED", "FRAUD"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot review comparison with status '{comparison.status}'"
        )

    # Record feedback
    comparison.matching = request.matching
    comparison.supervisor_reviewed_at = datetime.utcnow()
    await db.flush()

    logger.info(
        f"Feedback recorded: exec_id={request.task_check_execution_id} "
        f"img_id={request.task_location_checks_image_id} "
        f"matching={request.matching}"
    )

    return FeedbackResponse(
        task_location_checks_image_id=request.task_location_checks_image_id,
        task_check_execution_id=request.task_check_execution_id,
        matching=request.matching,
        recorded_at=comparison.supervisor_reviewed_at,
        message="Feedback recorded successfully",
    )


@router.get("/accuracy", response_model=AccuracyStats)
async def get_accuracy(
    db: AsyncSession = Depends(get_db),
):
    """
    Calculate model accuracy based on supervisor feedback.

    Model prediction derived from ratio vs threshold:
      - ratio >= threshold_pct -> model predicts CHANGED
      - ratio <  threshold_pct -> model predicts NORMAL (matching)

    Supervisor verdict:
      - matching=false -> supervisor says CHANGED
      - matching=true  -> supervisor says NORMAL

    TP: model=CHANGED + supervisor=CHANGED (matching=false)
    FP: model=CHANGED + supervisor=NORMAL  (matching=true)
    TN: model=NORMAL  + supervisor=NORMAL  (matching=true)
    FN: model=NORMAL  + supervisor=CHANGED (matching=false)
    """
    threshold_pct = getattr(config, "SVM_THRESHOLD", 0.389) * 100  # convert to percentage

    result = await db.execute(
        select(
            func.count().label("total"),
            # TP: model predicted change (diff >= threshold) AND supervisor confirmed change (matching=false)
            func.sum(case(
                (and_(Comparison.ratio >= threshold_pct, Comparison.matching == False), 1),
                else_=0
            )).label("tp"),
            # FP: model predicted change (diff >= threshold) AND supervisor said matching (matching=true)
            func.sum(case(
                (and_(Comparison.ratio >= threshold_pct, Comparison.matching == True), 1),
                else_=0
            )).label("fp"),
            # TN: model predicted normal (diff < threshold) AND supervisor confirmed matching (matching=true)
            func.sum(case(
                (and_(Comparison.ratio < threshold_pct, Comparison.matching == True), 1),
                else_=0
            )).label("tn"),
            # FN: model predicted normal (diff < threshold) AND supervisor found change (matching=false)
            func.sum(case(
                (and_(Comparison.ratio < threshold_pct, Comparison.matching == False), 1),
                else_=0
            )).label("fn"),
        ).where(Comparison.matching.isnot(None))
    )
    row = result.one()

    total = row.total or 0
    tp = row.tp or 0
    fp = row.fp or 0
    tn = row.tn or 0
    fn = row.fn or 0

    accuracy = (tp + tn) / total if total > 0 else None
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None

    return AccuracyStats(
        total_reviewed=total,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        last_updated=datetime.utcnow(),
    )


@router.get("/stats", response_model=ServiceStats)
async def get_stats(
    db: AsyncSession = Depends(get_db),
):
    """General service statistics."""
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    # Baseline counts
    baseline_result = await db.execute(
        select(
            func.count().label("total"),
            func.sum(case((Baseline.is_active == True, 1), else_=0)).label("active"),
        )
    )
    bl = baseline_result.one()

    # Comparison counts
    comp_result = await db.execute(
        select(
            func.count().label("total"),
            func.sum(case((Comparison.created_at >= today, 1), else_=0)).label("today"),
            func.avg(Comparison.processing_ms).label("avg_ms"),
            func.sum(case((Comparison.status == "FRAUD", 1), else_=0)).label("fraud"),
            func.sum(case(
                (and_(
                    Comparison.status == "COMPLETED",
                    Comparison.matching.is_(None)
                ), 1),
                else_=0
            )).label("pending"),
        )
    )
    cp = comp_result.one()

    return ServiceStats(
        total_baselines=bl.total or 0,
        active_baselines=bl.active or 0,
        total_comparisons=cp.total or 0,
        comparisons_today=cp.today or 0,
        avg_processing_ms=float(cp.avg_ms) if cp.avg_ms else None,
        fraud_detected_count=cp.fraud or 0,
        pending_review_count=cp.pending or 0,
    )
