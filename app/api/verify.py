"""
Object Verification API.
Pre-check: same object? Photo quality? Alignment?
Uses DINOv2 for same-object check — no DB writes, stateless inference.
"""
import base64
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


def _decode_base64_image(b64_string: str, label: str) -> np.ndarray:
    """Decode a base64 string to an OpenCV BGR image."""
    try:
        # Strip data URI prefix if present (e.g., "data:image/jpeg;base64,...")
        if "," in b64_string:
            b64_string = b64_string.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_string)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("cv2.imdecode returned None")
        return image
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to decode {label}: {str(e)}"
        )


@router.post("", response_model=VerifyResponse, status_code=status.HTTP_200_OK)
async def verify_same_object(request: VerifyRequest):
    """
    Pre-check before comparison: same object, photo quality, and alignment.

    Uses DINOv2 semantic similarity for same-object detection.

    - sameObject: DINOv2 similarity >= threshold
    - isBlurry: Laplacian variance below threshold (null if different object)
    - isBright: brightness within acceptable range (null if different object)
    - isAligned: enough keypoint matches for good comparison (null if different object)

    No database writes — purely stateless inference.
    """
    # Download baseline image from URL
    downloader = get_downloader()
    try:
        image1 = await downloader.download(request.image_url_1)
    except ImageDownloadError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to download baseline image: {str(e)}"
        )

    # Decode patrol image from base64
    image2 = _decode_base64_image(request.image_base64_2, "imageBase64_2")

    pipeline = get_pipeline()
    pipeline._ensure_models()

    # ── DINOv2 same-object check ──
    if pipeline._dino_model is None:
        logger.error("DINOv2 not loaded — verify returning sameObject=false as safety default")
        return VerifyResponse(same_object=False)

    score = pipeline._compute_dino_similarity(image1, image2)
    same = score >= config.VERIFY_THRESHOLD_DINO

    logger.info(f"Verify: DINOv2 similarity={score:.4f} sameObject={same}")

    if not same:
        return VerifyResponse(same_object=False)

    # ── Quality checks (only if same object) ──

    # Blur check on patrol image
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray2, cv2.CV_64F).var()
    is_blurry = blur_score < config.BLUR_WARNING_THRESHOLD

    # Brightness check on patrol image
    brightness = float(gray2.mean())
    is_bright = brightness >= config.BRIGHTNESS_WARNING_THRESHOLD

    # Alignment check (quick keypoint matching)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced1 = clahe.apply(cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY))
    enhanced2 = clahe.apply(gray2)
    alignment_result = pipeline._run_m3_alignment(enhanced1, enhanced2)
    inliers = alignment_result["inliers"]
    scale = alignment_result.get("scale", 1.0)
    is_aligned = inliers >= 8 and 0.15 <= scale <= 6.0

    logger.info(
        f"Verify quality: blur={blur_score:.0f} brightness={brightness:.0f} "
        f"inliers={inliers} scale={scale:.2f} blurry={is_blurry} bright={is_bright} aligned={is_aligned}"
    )

    return VerifyResponse(
        same_object=True,
        is_blurry=is_blurry,
        is_bright=is_bright,
        is_aligned=is_aligned,
    )
