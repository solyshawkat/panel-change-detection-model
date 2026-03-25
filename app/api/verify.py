"""
Object Verification API.
Pre-check: same object? Photo quality? Alignment?
Uses CLIP + pipeline checks — no DB writes, stateless inference.
"""
import base64
import logging
import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, status
from app.core import config
from app.schemas.schemas import VerifyRequest, VerifyResponse
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

    Accepts base64-encoded images (with or without data URI prefix).

    - sameObject: CLIP similarity >= 0.85
    - isBlurry: Laplacian variance below threshold (null if different object)
    - isBright: brightness within acceptable range (null if different object)
    - isAligned: enough keypoint matches for good comparison (null if different object)

    No database writes — purely stateless inference.
    """
    # Decode base64 images
    image1 = _decode_base64_image(request.image_base64_1, "imageBase64_1")
    image2 = _decode_base64_image(request.image_base64_2, "imageBase64_2")

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
