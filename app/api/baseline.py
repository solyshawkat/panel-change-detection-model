"""
Baseline Service API Routes.
Manages baseline reference images for panels.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.database import get_db
from app.models.models import Baseline
from app.schemas.schemas import (
    BaselineCreateRequest, BaselineUpdateRequest, BaselineResponse
)
from app.utils.image_downloader import get_downloader, ImageDownloadError
from app.pipeline.runner import get_pipeline
import logging
import cv2
from pathlib import Path
from app.core import config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/baseline", tags=["Baseline Service"])


@router.post("/", response_model=BaselineResponse, status_code=status.HTTP_201_CREATED)
async def create_baseline(
    request: BaselineCreateRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Register a new baseline image for a panel.

    Panel identity = location_id + taskcheck_id.
    Downloads the image, runs quality checks (blur, brightness),
    pre-computes CLAHE enhanced version for faster comparisons.
    """
    # Check if active baseline already exists for this panel
    existing = await db.execute(
        select(Baseline).where(
            Baseline.location_id == request.location_id,
            Baseline.taskcheck_id == request.taskcheck_id,
            Baseline.is_active == True,
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Active baseline already exists for location_id='{request.location_id}' taskcheck_id={request.taskcheck_id}. Use PUT to replace."
        )

    # Download image
    try:
        downloader = get_downloader()
        image = await downloader.download(request.image_url)
    except ImageDownloadError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to download baseline image: {str(e)}"
        )

    # Run quality checks + CLAHE preprocessing
    pipeline = get_pipeline()
    _, enhanced, quality = pipeline._run_m1a_validation(image)

    # Save enhanced version to disk for fast comparison later
    baseline_dir = Path(config.BASELINE_DIR)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    file_prefix = f"{request.location_id}_{request.taskcheck_id}"
    enhanced_path = str(baseline_dir / f"{file_prefix}_enhanced.png")
    color_path = str(baseline_dir / f"{file_prefix}_color.jpg")
    cv2.imwrite(enhanced_path, enhanced)
    cv2.imwrite(color_path, image)

    # Create database record
    baseline = Baseline(
        location_id=request.location_id,
        taskcheck_id=request.taskcheck_id,
        image_url=request.image_url,
        enhanced_cache_path=enhanced_path,
        color_cache_path=color_path,
        blur_score=quality["blur"],
        brightness=quality["brightness"],
        quality_warning=quality["warning"],
    )
    db.add(baseline)
    await db.flush()
    await db.refresh(baseline)

    logger.info(
        f"Baseline registered: loc={request.location_id} task={request.taskcheck_id} "
        f"blur={quality['blur']:.1f} brightness={quality['brightness']:.1f} "
        f"warning={quality['warning']}"
    )

    return baseline


@router.get("/", response_model=BaselineResponse)
async def get_baseline(
    location_id: str,
    taskcheck_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get the active baseline for a panel (location_id + taskcheck_id)."""
    result = await db.execute(
        select(Baseline).where(
            Baseline.location_id == location_id,
            Baseline.taskcheck_id == taskcheck_id,
            Baseline.is_active == True,
        )
    )
    baseline = result.scalar_one_or_none()
    if not baseline:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active baseline for location_id='{location_id}' taskcheck_id={taskcheck_id}"
        )
    return baseline


@router.put("/", response_model=BaselineResponse)
async def update_baseline(
    request: BaselineCreateRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Replace the active baseline for a panel with a new image.

    Deactivates the current baseline and creates a new one.
    The old baseline is preserved (is_active=False) for audit history.
    """
    # Find current active baseline
    result = await db.execute(
        select(Baseline).where(
            Baseline.location_id == request.location_id,
            Baseline.taskcheck_id == request.taskcheck_id,
            Baseline.is_active == True,
        )
    )
    current = result.scalar_one_or_none()

    # Download new image
    try:
        downloader = get_downloader()
        image = await downloader.download(request.image_url)
    except ImageDownloadError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to download new baseline image: {str(e)}"
        )

    # Quality checks
    pipeline = get_pipeline()
    _, enhanced, quality = pipeline._run_m1a_validation(image)

    # Save enhanced version
    baseline_dir = Path(config.BASELINE_DIR)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    file_prefix = f"{request.location_id}_{request.taskcheck_id}"
    enhanced_path = str(baseline_dir / f"{file_prefix}_enhanced.png")
    color_path = str(baseline_dir / f"{file_prefix}_color.jpg")
    cv2.imwrite(enhanced_path, enhanced)
    cv2.imwrite(color_path, image)

    # Deactivate old baseline
    if current:
        current.is_active = False

    # Create new baseline
    new_baseline = Baseline(
        location_id=request.location_id,
        taskcheck_id=request.taskcheck_id,
        image_url=request.image_url,
        enhanced_cache_path=enhanced_path,
        color_cache_path=color_path,
        blur_score=quality["blur"],
        brightness=quality["brightness"],
        quality_warning=quality["warning"],
    )
    db.add(new_baseline)
    await db.flush()
    await db.refresh(new_baseline)

    logger.info(
        f"Baseline replaced: loc={request.location_id} task={request.taskcheck_id} "
        f"old_id={current.id if current else 'none'} new_id={new_baseline.id}"
    )

    return new_baseline
