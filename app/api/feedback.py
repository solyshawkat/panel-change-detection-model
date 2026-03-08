"""
Feedback Service API Routes.
Handles supervisor reviews and tracks model accuracy.
"""
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, case
from app.core.database import get_db
from app.models.models import Baseline, Comparison
from app.schemas.schemas import (
    FeedbackRequest, FeedbackResponse, AccuracyStats, ServiceStats
)
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/feedback", tags=["Feedback Service"])

VALID_ACTIONS = {"CONFIRM_CHANGE", "REJECT_CHANGE", "CONFIRM_NORMAL"}


@router.post("/", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Supervisor submits a review for a comparison result.
    
    Actions:
    - CONFIRM_CHANGE: Model said changed, supervisor agrees
    - REJECT_CHANGE: Model said changed, supervisor says it's normal (false positive)
    - CONFIRM_NORMAL: Model said normal, supervisor agrees
    
    Note: If model said normal but supervisor sees a change, they should
    use CONFIRM_CHANGE (this becomes a false negative for model tracking).
    """
    if request.action not in VALID_ACTIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid action. Must be one of: {', '.join(VALID_ACTIONS)}"
        )

    # Find comparison
    result = await db.execute(
        select(Comparison).where(Comparison.id == request.comparison_id)
    )
    comparison = result.scalar_one_or_none()
    if not comparison:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Comparison {request.comparison_id} not found"
        )

    if comparison.status not in ("COMPLETED", "FRAUD"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot review comparison with status '{comparison.status}'"
        )

    # Record feedback
    comparison.supervisor_action = request.action
    comparison.supervisor_notes = request.notes
    comparison.supervisor_reviewed_at = datetime.utcnow()
    await db.flush()

    logger.info(
        f"Feedback recorded: comparison={request.comparison_id} "
        f"action={request.action} model_said={'CHANGED' if not comparison.matching else 'NORMAL'}"
    )

    return FeedbackResponse(
        comparison_id=request.comparison_id,
        action=request.action,
        recorded_at=comparison.supervisor_reviewed_at,
        message="Feedback recorded successfully",
    )


@router.get("/accuracy", response_model=AccuracyStats)
async def get_accuracy(
    db: AsyncSession = Depends(get_db),
):
    """
    Calculate model accuracy based on all supervisor feedback.
    
    True Positive:  model=CHANGED + supervisor=CONFIRM_CHANGE
    False Positive: model=CHANGED + supervisor=REJECT_CHANGE  
    True Negative:  model=NORMAL  + supervisor=CONFIRM_NORMAL
    False Negative: model=NORMAL  + supervisor=CONFIRM_CHANGE (supervisor overrode)
    """
    result = await db.execute(
        select(
            func.count().label("total"),
            func.sum(case(
                (and_(Comparison.matching == False, Comparison.supervisor_action == "CONFIRM_CHANGE"), 1),
                else_=0
            )).label("tp"),
            func.sum(case(
                (and_(Comparison.matching == False, Comparison.supervisor_action == "REJECT_CHANGE"), 1),
                else_=0
            )).label("fp"),
            func.sum(case(
                (and_(Comparison.matching == True, Comparison.supervisor_action == "CONFIRM_NORMAL"), 1),
                else_=0
            )).label("tn"),
            func.sum(case(
                (and_(Comparison.matching == True, Comparison.supervisor_action == "CONFIRM_CHANGE"), 1),
                else_=0
            )).label("fn"),
        ).where(Comparison.supervisor_action.isnot(None))
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
                    Comparison.supervisor_action.is_(None)
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
