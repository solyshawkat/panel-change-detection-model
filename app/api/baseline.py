"""
Baseline Service API Routes.
Manages baseline reference images for panels.

Baseline identified by task_location_checks_image_id (unique ID from backend).
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.database import get_db
from app.models.models import Baseline
from app.schemas.schemas import BaselineCreateRequest, BaselineResponse
from app.utils.image_downloader import get_downloader, ImageDownloadError
from app.pipeline.runner import get_pipeline
import logging
import cv2
from pathlib import Path
from app.core import config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/baseline", tags=["Baseline Service"])


@router.post("", response_model=BaselineResponse, status_code=status.HTTP_201_CREATED)
async def create_baseline(
    request: BaselineCreateRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Register a new baseline image.

    Identified by task_location_checks_image_id (unique from backend).
    Downloads the image, runs quality checks (blur, brightness),
    pre-computes CLAHE enhanced version for faster comparisons.
    """
    # Check if this baseline image ID already exists
    existing = await db.execute(
        select(Baseline).where(
            Baseline.task_location_checks_image_id == request.task_location_checks_image_id,
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Baseline already exists for task_location_checks_image_id={request.task_location_checks_image_id}."
        )

    # Download image
    try:
        downloader = get_downloader()
        image = await downloader.download(request.reference_image_url)
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
    file_prefix = f"baseline_{request.task_location_checks_image_id}"
    enhanced_path = str(baseline_dir / f"{file_prefix}_enhanced.png")
    color_path = str(baseline_dir / f"{file_prefix}_color.jpg")
    cv2.imwrite(enhanced_path, enhanced)
    cv2.imwrite(color_path, image)

    # Create database record
    baseline = Baseline(
        task_location_checks_image_id=request.task_location_checks_image_id,
        image_url=request.reference_image_url,
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
        f"Baseline registered: img_id={request.task_location_checks_image_id} "
        f"blur={quality['blur']:.1f} brightness={quality['brightness']:.1f} "
        f"warning={quality['warning']}"
    )

    return baseline


@router.get("", response_model=BaselineResponse)
async def get_baseline(
    task_location_checks_image_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get a baseline by its task_location_checks_image_id."""
    result = await db.execute(
        select(Baseline).where(
            Baseline.task_location_checks_image_id == task_location_checks_image_id,
            Baseline.is_active == True,
        )
    )
    baseline = result.scalar_one_or_none()
    if not baseline:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active baseline for task_location_checks_image_id={task_location_checks_image_id}"
        )
    return baseline


@router.put("", response_model=BaselineResponse)
async def update_baseline(
    request: BaselineCreateRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Replace a baseline image.

    Deactivates the current baseline and creates a new one with the same
    task_location_checks_image_id. Old baseline preserved for audit history.
    """
    # Find current active baseline
    result = await db.execute(
        select(Baseline).where(
            Baseline.task_location_checks_image_id == request.task_location_checks_image_id,
            Baseline.is_active == True,
        )
    )
    current = result.scalar_one_or_none()

    # Download new image
    try:
        downloader = get_downloader()
        image = await downloader.download(request.reference_image_url)
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
    file_prefix = f"baseline_{request.task_location_checks_image_id}"
    enhanced_path = str(baseline_dir / f"{file_prefix}_enhanced.png")
    color_path = str(baseline_dir / f"{file_prefix}_color.jpg")
    cv2.imwrite(enhanced_path, enhanced)
    cv2.imwrite(color_path, image)

    # Deactivate old baseline
    if current:
        current.is_active = False
        # Remove unique constraint conflict by clearing the image_id on old record
        current.task_location_checks_image_id = -current.id  # negative = deactivated

    # Create new baseline
    new_baseline = Baseline(
        task_location_checks_image_id=request.task_location_checks_image_id,
        image_url=request.reference_image_url,
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
        f"Baseline replaced: img_id={request.task_location_checks_image_id} "
        f"old_id={current.id if current else 'none'} new_id={new_baseline.id}"
    )

    return new_baseline
