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
    
    Downloads the image, runs quality checks (blur, brightness),
    pre-computes CLAHE enhanced version for faster comparisons,
    and stores everything in the database.
    """
    # Check if processing_id already exists
    existing = await db.execute(
        select(Baseline).where(Baseline.processing_id == request.processing_id)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Baseline with processing_id '{request.processing_id}' already exists"
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
    enhanced_path = str(baseline_dir / f"{request.processing_id}_enhanced.png")
    color_path = str(baseline_dir / f"{request.processing_id}_color.jpg")
    cv2.imwrite(enhanced_path, enhanced)
    cv2.imwrite(color_path, image)

    # Create database record
    baseline = Baseline(
        processing_id=request.processing_id,
        panel_id=request.panel_id,
        site_id=request.site_id,
        location_id=request.location_id,
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
        f"Baseline registered: panel={request.panel_id} "
        f"blur={quality['blur']:.1f} brightness={quality['brightness']:.1f} "
        f"warning={quality['warning']}"
    )

    return baseline


@router.get("/{panel_id}", response_model=BaselineResponse)
async def get_baseline(
    panel_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get the active baseline for a panel."""
    result = await db.execute(
        select(Baseline).where(
            Baseline.panel_id == panel_id,
            Baseline.is_active == True,
        )
    )
    baseline = result.scalar_one_or_none()
    if not baseline:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active baseline found for panel '{panel_id}'"
        )
    return baseline


@router.put("/{panel_id}", response_model=BaselineResponse)
async def update_baseline(
    panel_id: str,
    request: BaselineUpdateRequest,
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
            Baseline.panel_id == panel_id,
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
    enhanced_path = str(baseline_dir / f"{request.processing_id}_enhanced.png")
    color_path = str(baseline_dir / f"{request.processing_id}_color.jpg")
    cv2.imwrite(enhanced_path, enhanced)
    cv2.imwrite(color_path, image)

    # Deactivate old baseline
    if current:
        current.is_active = False
        current.replaced_by = request.processing_id

    # Create new baseline
    new_baseline = Baseline(
        processing_id=request.processing_id,
        panel_id=panel_id,
        site_id=current.site_id if current else None,
        location_id=current.location_id if current else None,
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

    logger.info(f"Baseline replaced: panel={panel_id} old={current.processing_id if current else 'none'} new={request.processing_id}")

    return new_baseline
