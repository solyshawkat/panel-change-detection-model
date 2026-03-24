"""
Object Verification API.
Pre-check: same object? Photo quality? Alignment?
Uses CLIP + pipeline checks — no DB writes, stateless inference.
"""
import logging
import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, status
from app.core import config
from app.schemas.schemas import VerifyRequest, VerifyResponse
from app.utils.image_downloader import get_downloader, ImageDownloadError
from app.pipeline.runner import get_pipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/verify", tags=["Object Verification"])


@router.post("", response_model=VerifyResponse, status_code=status.HTTP_200_OK)
async def verify_same_object(request: VerifyRequest):
    """
    Pre-check before comparison: same object, photo quality, and alignment.

    - sameObject: CLIP similarity >= 0.85
    - isBlurry: Laplacian variance below threshold (null if different object)
    - isBright: brightness within acceptable range (null if different object)
    - isAligned: enough keypoint matches for good comparison (null if different object)

    No database writes — purely stateless inference.
    """
    downloader = get_downloader()

    # Download both images
    try:
        image1 = await downloader.download(request.image_url_1)
    except ImageDownloadError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to download imageUrl1: {str(e)}"
        )

    try:
        image2 = await downloader.download(request.image_url_2)
    except ImageDownloadError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to download imageUrl2: {str(e)}"
        )

    # Step 1: CLIP same-object check
    pipeline = get_pipeline()
    pipeline._ensure_models()
    fraud_result = pipeline._run_m2_fraud(image1, image2)

    score = fraud_result["score"]
    same = score >= config.VERIFY_THRESHOLD

    logger.info(f"Verify: similarity={score:.4f} sameObject={same}")

    # If different object, return nulls for quality checks
    if not same:
        return VerifyResponse(same_object=False)

    # Step 2: Blur check on patrol image (image2)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray2, cv2.CV_64F).var()
    is_blurry = blur_score < config.BLUR_WARNING_THRESHOLD

    # Step 3: Brightness check on patrol image (image2)
    brightness = float(gray2.mean())
    is_bright = brightness >= config.BRIGHTNESS_WARNING_THRESHOLD

    # Step 4: Alignment check (quick keypoint matching)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced1 = clahe.apply(cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY))
    enhanced2 = clahe.apply(gray2)
    alignment_result = pipeline._run_m3_alignment(enhanced1, enhanced2)
    inliers = alignment_result["inliers"]
    is_aligned = inliers >= 30

    logger.info(
        f"Verify quality: blur={blur_score:.0f} brightness={brightness:.0f} "
        f"inliers={inliers} blurry={is_blurry} bright={is_bright} aligned={is_aligned}"
    )

    return VerifyResponse(
        same_object=True,
        is_blurry=is_blurry,
        is_bright=is_bright,
        is_aligned=is_aligned,
    )
