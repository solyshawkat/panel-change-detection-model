"""
Object Verification API.
Quick check: do two images show the same object?
Uses CLIP cosine similarity — no DB writes, stateless inference.
"""
import logging
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
    Check if two images show the same object using CLIP similarity.

    Returns sameObject: true if same object, false if different.
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

    # Run CLIP similarity (reuses already-loaded model)
    pipeline = get_pipeline()
    pipeline._ensure_models()
    fraud_result = pipeline._run_m2_fraud(image1, image2)

    score = fraud_result["score"]
    same = score >= config.VERIFY_THRESHOLD

    logger.info(f"Verify: similarity={score:.4f} sameObject={same}")

    return VerifyResponse(same_object=same)
