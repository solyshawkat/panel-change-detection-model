"""
Comparison Engine API Routes.
Runs the full pipeline and returns results.
"""
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pathlib import Path
from app.core.database import get_db
from app.models.models import Baseline, Comparison
from app.schemas.schemas import CompareRequest, CompareResponse, CompareDetailResponse
from app.utils.image_downloader import get_downloader, ImageDownloadError
from app.pipeline.runner import get_pipeline
import cv2
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/compare", tags=["Comparison Engine"])


@router.post("/", response_model=CompareResponse, status_code=status.HTTP_201_CREATED)
async def create_comparison(
    request: CompareRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Run a full comparison between a panel's active baseline and a patrol photo.
    
    Pipeline: M0 (QR) → M1a (Grayscale) + M1b (Color) → M2 (CLIP) → M3 (Align) → M4 (Features) → M5 (SVM)
    
    Can stop early at M2 (fraud) or M3 (alignment failure).
    """
    # Find active baseline for this panel
    result = await db.execute(
        select(Baseline).where(
            Baseline.panel_id == request.panel_id,
            Baseline.is_active == True,
        )
    )
    baseline = result.scalar_one_or_none()
    if not baseline:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active baseline for panel '{request.panel_id}'. Register one first via POST /baseline"
        )

    # Create comparison record (PENDING)
    comparison = Comparison(
        definition_id=request.definition_id,
        execution_id=request.execution_id,
        baseline_id=baseline.id,
        patrol_image_url=request.patrol_image_url,
        status="PROCESSING",
    )
    db.add(comparison)
    await db.flush()

    # Download patrol image
    try:
        downloader = get_downloader()
        patrol_image = await downloader.download(request.patrol_image_url)
    except ImageDownloadError as e:
        comparison.status = "FAILED"
        comparison.error_message = f"Image download failed: {str(e)}"
        await db.flush()
        await db.refresh(comparison)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to download patrol image: {str(e)}"
        )

    # Load cached baseline (enhanced + color)
    baseline_enhanced = None
    baseline_color = None
    if baseline.enhanced_cache_path and Path(baseline.enhanced_cache_path).exists():
        baseline_enhanced = cv2.imread(baseline.enhanced_cache_path, cv2.IMREAD_GRAYSCALE)
    if baseline.color_cache_path and Path(baseline.color_cache_path).exists():
        baseline_color = cv2.imread(baseline.color_cache_path)

    # If no cache, download baseline image
    if baseline_enhanced is None or baseline_color is None:
        try:
            baseline_image = await downloader.download(baseline.image_url)
            if baseline_color is None:
                baseline_color = baseline_image
        except ImageDownloadError as e:
            comparison.status = "FAILED"
            comparison.error_message = f"Baseline image unavailable: {str(e)}"
            await db.flush()
            await db.refresh(comparison)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Baseline image could not be loaded"
            )
    else:
        baseline_image = baseline_color

    # ── Run Pipeline ──
    pipeline = get_pipeline()
    pipeline_result = await pipeline.run(
        baseline_image=baseline_image,
        patrol_image=patrol_image,
        baseline_enhanced=baseline_enhanced,
        baseline_color=baseline_color,
    )

    # ── Update comparison record ──
    comparison.ratio = pipeline_result.ratio
    comparison.matching = pipeline_result.matching
    comparison.is_valid = pipeline_result.is_valid
    comparison.fraud_score = pipeline_result.fraud_score

    # Structure features
    comparison.ssim_score = pipeline_result.ssim_score
    comparison.ms_ssim_score = pipeline_result.ms_ssim_score
    comparison.edge_score = pipeline_result.edge_score
    comparison.histogram_score = pipeline_result.histogram_score
    comparison.cluster_score = pipeline_result.cluster_score
    comparison.max_diff_area = pipeline_result.max_diff_area
    comparison.stability_score = pipeline_result.stability_score

    # Color features
    comparison.max_hue_shift = pipeline_result.max_hue_shift
    comparison.max_delta_e = pipeline_result.max_delta_e

    # Alignment
    comparison.alignment_inliers = pipeline_result.alignment_inliers
    comparison.alignment_method = pipeline_result.alignment_method

    # Quality + output
    comparison.blur_score = pipeline_result.blur_score
    comparison.heatmap_path = pipeline_result.heatmap_path
    comparison.processing_ms = pipeline_result.processing_ms
    comparison.error_message = pipeline_result.error

    # Status
    if not pipeline_result.is_valid:
        comparison.status = "FRAUD"
    elif pipeline_result.success:
        comparison.status = "COMPLETED"
    else:
        comparison.status = "FAILED"

    comparison.completed_at = datetime.utcnow()
    await db.flush()
    await db.refresh(comparison)

    logger.info(
        f"Comparison complete: id={comparison.id} panel={request.panel_id} "
        f"ratio={comparison.ratio:.3f} matching={comparison.matching} "
        f"valid={comparison.is_valid} {comparison.processing_ms}ms"
    )

    # Build response
    response = CompareResponse(
        id=comparison.id,
        baseline_id=comparison.baseline_id,
        status=comparison.status,
        ratio=comparison.ratio,
        matching=comparison.matching,
        is_valid=comparison.is_valid,
        fraud_score=comparison.fraud_score,
        color_shift_detected=(
            comparison.max_hue_shift > 10 or comparison.max_delta_e > 5
            if comparison.max_hue_shift is not None else None
        ),
        processing_ms=comparison.processing_ms,
        heatmap_url=f"/compare/result/{comparison.id}/heatmap" if comparison.heatmap_path else None,
        error_message=comparison.error_message,
        created_at=comparison.created_at,
        completed_at=comparison.completed_at,
    )
    return response


@router.get("/result/{comparison_id}", response_model=CompareDetailResponse)
async def get_result(
    comparison_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get full comparison details including all feature scores."""
    result = await db.execute(
        select(Comparison).where(Comparison.id == comparison_id)
    )
    comparison = result.scalar_one_or_none()
    if not comparison:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Comparison {comparison_id} not found"
        )

    return CompareDetailResponse(
        id=comparison.id,
        baseline_id=comparison.baseline_id,
        status=comparison.status,
        ratio=comparison.ratio,
        matching=comparison.matching,
        is_valid=comparison.is_valid,
        fraud_score=comparison.fraud_score,
        color_shift_detected=(
            comparison.max_hue_shift > 10 or comparison.max_delta_e > 5
            if comparison.max_hue_shift is not None else None
        ),
        processing_ms=comparison.processing_ms,
        heatmap_url=f"/compare/result/{comparison.id}/heatmap" if comparison.heatmap_path else None,
        error_message=comparison.error_message,
        created_at=comparison.created_at,
        completed_at=comparison.completed_at,
        ssim_score=comparison.ssim_score,
        ms_ssim_score=comparison.ms_ssim_score,
        edge_score=comparison.edge_score,
        histogram_score=comparison.histogram_score,
        cluster_score=comparison.cluster_score,
        max_diff_area=comparison.max_diff_area,
        stability_score=comparison.stability_score,
        max_hue_shift=comparison.max_hue_shift,
        max_delta_e=comparison.max_delta_e,
        alignment_inliers=comparison.alignment_inliers,
        alignment_method=comparison.alignment_method,
        blur_score=comparison.blur_score,
        supervisor_action=comparison.supervisor_action,
        supervisor_notes=comparison.supervisor_notes,
    )


@router.get("/result/{comparison_id}/heatmap")
async def get_heatmap(
    comparison_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Download the change heatmap image for a comparison."""
    result = await db.execute(
        select(Comparison).where(Comparison.id == comparison_id)
    )
    comparison = result.scalar_one_or_none()
    if not comparison:
        raise HTTPException(status_code=404, detail="Comparison not found")

    if not comparison.heatmap_path or not Path(comparison.heatmap_path).exists():
        raise HTTPException(status_code=404, detail="Heatmap not available")

    return FileResponse(
        comparison.heatmap_path,
        media_type="image/jpeg",
        filename=f"heatmap_{comparison_id}.jpg",
    )
