"""
Pipeline Runner — Orchestrates M0 through M5.

Each module has a clear contract:
- Input: what it receives from the previous step
- Output: what it passes to the next step
- Can Fail: whether it can stop the pipeline

The runner tracks timing and collects all results into a PipelineResult.
"""
import time
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from app.core import config

logger = logging.getLogger(__name__)



@dataclass
class PipelineResult:
    """Collects all outputs from the pipeline."""
    # Status
    success: bool = False
    error: str = None
    stopped_at: str = None  # module name that stopped pipeline

    # M0: QR Detection
    qr_panel_id: Optional[str] = None

    # M1a: Quality
    blur_score: float = 0.0
    brightness: float = 0.0
    quality_warning: bool = False

    # M2: Fraud
    is_valid: bool = True
    fraud_score: float = 0.0

    # M3: Alignment
    alignment_inliers: int = 0
    alignment_method: str = ""

    # M4: Structure features
    ssim_score: float = 0.0
    ms_ssim_score: float = 0.0
    edge_score: float = 0.0
    histogram_score: float = 0.0
    cluster_score: float = 0.0
    max_diff_area: float = 0.0
    stability_score: float = 0.0

    # M4: Color features
    max_hue_shift: float = 0.0
    max_delta_e: float = 0.0

    # M5: Classification
    ratio: float = 0.0
    matching: bool = True

    # Output
    heatmap_path: Optional[str] = None
    processing_ms: int = 0


class PipelineRunner:
    """
    Runs the full image comparison pipeline.
    
    Pipeline flow:
        M0 (QR) → M1a (Grayscale+CLAHE) → M2 (CLIP Fraud) → M3 (Align) → M4 (Features) → M5 (SVM)
                   M1b (Color HSV+LAB) ──────────────────────────────────↗
    """

    def __init__(self):
        # Lazy-load heavy modules only when first comparison runs
        self._models_loaded = False

    def _ensure_models(self):
        """Load ML models on first use (CLIP, SuperPoint, LightGlue, SVM)."""
        if self._models_loaded:
            return

        logger.info("Loading ML models (first comparison)...")
        start = time.time()

        # TODO: Load actual models here
        # self.clip_model = load_clip()
        # self.superpoint = load_superpoint()
        # self.lightglue = load_lightglue()
        # self.svm_model = load_svm(config.SVM_MODEL_PATH)

        self._models_loaded = True
        logger.info(f"Models loaded in {time.time() - start:.1f}s")

    async def run(
        self,
        baseline_image: np.ndarray,
        patrol_image: np.ndarray,
        baseline_enhanced: Optional[np.ndarray] = None,
        baseline_color: Optional[np.ndarray] = None,
    ) -> PipelineResult:
        """
        Run the full pipeline on a baseline/patrol image pair.
        
        Args:
            baseline_image: Original baseline image (BGR)
            patrol_image: Patrol photo to compare (BGR)
            baseline_enhanced: Pre-cached CLAHE grayscale (skip M1a for baseline)
            baseline_color: Pre-cached color copy (skip download for color path)
        
        Returns:
            PipelineResult with all scores and outputs
        """
        self._ensure_models()
        result = PipelineResult()
        start_time = time.time()

        try:
            # ── M0: QR Detection (optional) ──
            logger.debug("M0: QR Detection")
            result.qr_panel_id = self._run_m0_qr(patrol_image)

            # ── M1a: Image Validation + CLAHE ──
            logger.debug("M1a: Image Validation (Structure Path)")
            patrol_gray, patrol_enhanced, quality = self._run_m1a_validation(patrol_image)
            result.blur_score = quality["blur"]
            result.brightness = quality["brightness"]
            result.quality_warning = quality["warning"]

            # Use cached baseline if available, otherwise process
            if baseline_enhanced is None:
                _, baseline_enhanced, _ = self._run_m1a_validation(baseline_image)

            # ── M1b: Color Path (parallel) ──
            logger.debug("M1b: Color Path (HSV + LAB)")
            # Color analysis runs on original color images, not grayscale
            # Results feed into M4 features

            # ── M2: Fraud Detection (CLIP) ──
            logger.debug("M2: Fraud Detection (CLIP)")
            fraud_result = self._run_m2_fraud(baseline_image, patrol_image)
            result.is_valid = fraud_result["is_valid"]
            result.fraud_score = fraud_result["score"]

            if not result.is_valid:
                result.stopped_at = "M2_FRAUD"
                result.error = f"Fraud detected: CLIP score {result.fraud_score:.3f} below threshold {config.FRAUD_THRESHOLD}"
                result.processing_ms = int((time.time() - start_time) * 1000)
                return result

            # ── M3: Alignment ──
            logger.debug("M3: Image Alignment")
            alignment_result = self._run_m3_alignment(baseline_enhanced, patrol_enhanced)
            result.alignment_inliers = alignment_result["inliers"]
            result.alignment_method = alignment_result["method"]

            if alignment_result["inliers"] < config.ALIGNMENT_MIN_INLIERS:
                result.stopped_at = "M3_ALIGNMENT"
                result.error = f"Alignment failed: only {alignment_result['inliers']} inliers (min {config.ALIGNMENT_MIN_INLIERS})"
                result.processing_ms = int((time.time() - start_time) * 1000)
                return result

            patrol_warped = alignment_result["warped"]

            # ── M4: Feature Extraction ──
            logger.debug("M4: Feature Extraction (9 signals)")

            # Structure features (grayscale path)
            structure_features = self._run_m4_structure(baseline_enhanced, patrol_warped)
            result.ssim_score = structure_features["ssim"]
            result.ms_ssim_score = structure_features["ms_ssim"]
            result.edge_score = structure_features["edge"]
            result.histogram_score = structure_features["histogram"]
            result.cluster_score = structure_features["cluster"]
            result.max_diff_area = structure_features["max_diff_area"]
            result.stability_score = structure_features["stability"]

            # Color features (color path)
            color_features = self._run_m4_color(
                baseline_color if baseline_color is not None else baseline_image,
                patrol_image
            )
            result.max_hue_shift = color_features["max_hue_shift"]
            result.max_delta_e = color_features["max_delta_e"]

            # Generate heatmap
            heatmap_path = self._generate_heatmap(baseline_enhanced, patrol_warped)
            result.heatmap_path = heatmap_path

            # ── M5: SVM Classification ──
            logger.debug("M5: SVM Classification")
            feature_vector = [
                result.max_diff_area,
                result.ssim_score,
                result.stability_score,
                result.ms_ssim_score,
                result.cluster_score,
                result.edge_score,
                result.histogram_score,
                result.max_hue_shift,
                result.max_delta_e,
            ]
            classification = self._run_m5_classify(feature_vector)
            result.ratio = classification["probability"]
            result.matching = classification["probability"] < config.SVM_THRESHOLD

            result.success = True

        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}", exc_info=True)
            result.error = str(e)

        result.processing_ms = int((time.time() - start_time) * 1000)
        return result

    # ─────────────────────────────────────────
    #  MODULE IMPLEMENTATIONS
    #  TODO: Replace stubs with real implementations
    # ─────────────────────────────────────────

    def _run_m0_qr(self, image: np.ndarray) -> Optional[str]:
        """M0: Detect QR code and return panel_id. Returns None if no QR found."""
        # TODO: Implement QR detection
        # detector = cv2.QRCodeDetector()
        # data, bbox, _ = detector.detectAndDecode(image)
        return None

    def _run_m1a_validation(self, image: np.ndarray) -> tuple:
        """M1a: EXIF fix, grayscale, CLAHE enhancement, quality checks."""
        import cv2

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = float(gray.mean())
        warning = blur < config.BLUR_WARNING_THRESHOLD or brightness < config.BRIGHTNESS_WARNING_THRESHOLD

        quality = {"blur": blur, "brightness": brightness, "warning": warning}
        return gray, enhanced, quality

    def _run_m2_fraud(self, baseline: np.ndarray, patrol: np.ndarray) -> dict:
        """M2: CLIP-based fraud detection."""
        # TODO: Implement CLIP comparison
        # embeddings_a = self.clip_model.encode(baseline)
        # embeddings_b = self.clip_model.encode(patrol)
        # score = cosine_similarity(embeddings_a, embeddings_b)
        score = 0.95  # STUB: assume valid
        return {
            "is_valid": score >= config.FRAUD_THRESHOLD,
            "score": score,
        }

    def _run_m3_alignment(self, baseline_enhanced: np.ndarray, patrol_enhanced: np.ndarray) -> dict:
        """M3: SuperPoint + LightGlue alignment."""
        # TODO: Implement real alignment
        # kp1 = self.superpoint.detect(baseline_enhanced)
        # kp2 = self.superpoint.detect(patrol_enhanced)
        # matches = self.lightglue.match(kp1, kp2)
        # H, inliers = cv2.findHomography(...)
        # warped = cv2.warpPerspective(patrol_enhanced, H, ...)
        return {
            "warped": patrol_enhanced,  # STUB: return unwarped
            "inliers": 388,             # STUB
            "method": "lightglue",
        }

    def _run_m4_structure(self, baseline: np.ndarray, patrol: np.ndarray) -> dict:
        """M4: Structure feature extraction (7 features from grayscale path)."""
        # TODO: Implement real feature extraction
        # ssim_score, ssim_diff = ssim(baseline, patrol, full=True)
        # edges_base = cv2.Canny(baseline, 50, 150)
        # ...
        return {
            "ssim": 0.0,
            "ms_ssim": 0.0,
            "edge": 0.0,
            "histogram": 0.0,
            "cluster": 0.0,
            "max_diff_area": 0.0,
            "stability": 0.0,
        }

    def _run_m4_color(self, baseline_color: np.ndarray, patrol_color: np.ndarray) -> dict:
        """M4: Color feature extraction (HSV hue delta + LAB delta-E)."""
        import cv2
        # HSV hue comparison
        hsv_base = cv2.cvtColor(baseline_color, cv2.COLOR_BGR2HSV)
        hsv_patrol = cv2.cvtColor(patrol_color, cv2.COLOR_BGR2HSV)
        hue_diff = cv2.absdiff(hsv_base[:, :, 0].astype(np.int16), hsv_patrol[:, :, 0].astype(np.int16))
        hue_diff = np.minimum(np.abs(hue_diff), 180 - np.abs(hue_diff)).astype(np.float32)
        max_hue_shift = float(np.percentile(hue_diff, 95))

        # LAB delta-E
        lab_base = cv2.cvtColor(baseline_color, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_patrol = cv2.cvtColor(patrol_color, cv2.COLOR_BGR2LAB).astype(np.float32)
        delta_e = np.sqrt(np.sum((lab_base - lab_patrol) ** 2, axis=2))
        max_delta_e = float(np.percentile(delta_e, 95))

        return {
            "max_hue_shift": max_hue_shift,
            "max_delta_e": max_delta_e,
        }

    def _generate_heatmap(self, baseline: np.ndarray, patrol: np.ndarray) -> Optional[str]:
        """Generate SSIM diff heatmap and save to disk."""
        import cv2
        from pathlib import Path

        try:
            diff = cv2.absdiff(baseline, patrol)
            heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(
                cv2.cvtColor(baseline, cv2.COLOR_GRAY2BGR), 0.5,
                heatmap, 0.5, 0
            )

            heatmap_dir = Path(config.HEATMAP_DIR)
            heatmap_dir.mkdir(parents=True, exist_ok=True)
            filename = f"heatmap_{int(time.time() * 1000)}.jpg"
            path = str(heatmap_dir / filename)
            cv2.imwrite(path, overlay)
            return path
        except Exception as e:
            logger.warning(f"Heatmap generation failed: {e}")
            return None

    def _run_m5_classify(self, feature_vector: list) -> dict:
        """M5: SVM classification."""
        # TODO: Implement real SVM prediction
        # features = np.array(feature_vector).reshape(1, -1)
        # features_scaled = self.scaler.transform(features)
        # probability = self.svm_model.predict_proba(features_scaled)[0][1]
        probability = 0.15  # STUB: assume normal
        return {
            "probability": probability,
        }


# Singleton
_pipeline = None

def get_pipeline() -> PipelineRunner:
    global _pipeline
    if _pipeline is None:
        _pipeline = PipelineRunner()
    return _pipeline
