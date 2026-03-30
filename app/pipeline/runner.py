"""
Pipeline Runner — Orchestrates M0 through M5.

Each module has a clear contract:
- Input: what it receives from the previous step
- Output: what it passes to the next step
- Can Fail: whether it can stop the pipeline

The runner tracks timing and collects all results into a PipelineResult.

UPGRADE PATH — DINOv2 Embedding Replacement (Phase 2):
    When 200+ labeled comparisons are collected and RF accuracy plateaus,
    replace the 9 handcrafted pixel features (M4) with DINOv2 embeddings:

    1. Load DINOv2 ViT-S/14 (facebook/dinov2-small, ~85MB) or
       ViT-B/14 (facebook/dinov2-base, ~300MB) via torch.hub or transformers
    2. For each image: resize to 224x224 → normalize → forward pass → 384/768-dim vector
    3. Compute difference vector: abs(embedding_base - embedding_patrol)
    4. Feed difference vector (384/768 features) into the RF instead of the 9 pixel features
    5. Retrain RF on same labeled data — no new labels needed

    Why: DINOv2 understands semantic structure (shape, texture, spatial layout)
    not just pixel values. This handles lighting changes, angle differences,
    and same-brand-different-unit cases that pixel features struggle with.

    Runtime: +200-500ms on CPU. No GPU required. CLIP stays for fraud detection + categories.
    The RF training infrastructure, retrain endpoint, and supervisor feedback loop
    all remain unchanged — only the feature extraction step changes.
"""
import json
import time
import logging
import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from skimage.metrics import structural_similarity as ssim
from app.core import config

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────
# Constants (from session1_validate.py)
# ─────────────────────────────────────────────────
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID = (8, 8)
CANNY_LOW = 50
CANNY_HIGH = 150
_QR_PAD_PX = 10
_MULTISCALE_FACTORS = [1.0, 0.5, 2.0]


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

    # DINOv2 similarity
    dino_similarity: float = 0.0

    # M5: Classification
    ratio: float = 0.0

    # Object category (CLIP auto-detected)
    object_category: Optional[str] = None

    # Output
    heatmap_path: Optional[str] = None
    processing_ms: int = 0


class PipelineRunner:
    """
    Runs the full image comparison pipeline.

    Pipeline flow:
        M0 (QR) -> M1a (Grayscale+CLAHE) -> M2 (CLIP Fraud) -> M3 (Align) -> M4 (Features) -> M5 (SVM)
                   M1b (Color HSV+LAB) ───────────────────────────────────/
    """

    def __init__(self):
        import threading
        # Lazy-load heavy modules only when first comparison runs
        self._models_loaded = False
        self._clip_model = None
        self._clip_processor = None
        self._dino_model = None
        self._lg_extractor = None
        self._lg_matcher = None
        self._svm_model = None
        self._feature_config = None
        self._fraud_config = None
        self._device = "cpu"
        # Random Forest model (auto-retrain)
        self._rf_classifier = None
        self._rf_feature_names = None
        self._rf_loaded = False
        self._rf_lock = threading.Lock()

    def _ensure_models(self):
        """Load ML models on first use (CLIP, SuperPoint, LightGlue, SVM)."""
        if self._models_loaded:
            return

        logger.info("Loading ML models (first comparison)...")
        start = time.time()

        import torch

        # Detect best available device
        if torch.backends.mps.is_available():
            self._device = "mps"
        elif torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"
        logger.info(f"Using device: {self._device}")

        # ── Load CLIP (M2: Fraud Detection) ──
        try:
            from transformers import CLIPModel, CLIPProcessor
            clip_model_name = "openai/clip-vit-base-patch32"
            self._clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
            self._clip_model = CLIPModel.from_pretrained(clip_model_name)
            self._clip_model.eval()
            logger.info("CLIP ViT-B/32 loaded")
        except Exception as e:
            logger.warning(f"CLIP load failed (M2 will use fallback): {e}")

        # ── Load DINOv2 (Verify: Same-object check + Compare: RF feature) ──
        try:
            import timm
            self._dino_model = timm.create_model(
                "vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0
            )
            self._dino_model.eval().to(self._device)
            logger.info(f"DINOv2 ViT-S/14 loaded on {self._device}")
        except Exception as e:
            logger.warning(f"DINOv2 load failed: {e}")

        # ── Load SuperPoint + LightGlue (M3: Alignment) ──
        try:
            from lightglue import LightGlue, SuperPoint
            self._lg_extractor = SuperPoint(max_num_keypoints=2048).eval().to(self._device)
            self._lg_matcher = LightGlue(features="superpoint").eval().to(self._device)
            logger.info(f"SuperPoint + LightGlue loaded on {self._device}")
        except Exception as e:
            logger.warning(f"LightGlue load failed (M3 will use ORB fallback): {e}")

        # ── Load SVM model (M5: Classification) ──
        self._svm_scaler = None
        self._svm_classifier = None
        try:
            import joblib
            model_data = joblib.load(config.SVM_MODEL_PATH)
            # model.joblib is a dict: {'scaler': StandardScaler, 'svm': SVC, ...}
            if isinstance(model_data, dict):
                self._svm_classifier = model_data["svm"]
                self._svm_scaler = model_data["scaler"]
                self._svm_model = model_data  # keep full dict for reference
                logger.info(f"SVM model loaded (dict format) from {config.SVM_MODEL_PATH}")
            else:
                # Fallback: if model is a plain sklearn object
                self._svm_classifier = model_data
                self._svm_model = model_data
                logger.info(f"SVM model loaded (plain) from {config.SVM_MODEL_PATH}")
        except Exception as e:
            logger.error(f"SVM model load failed: {e}")

        # ── Load feature config ──
        try:
            with open(config.FEATURE_CONFIG_PATH) as f:
                self._feature_config = json.load(f)
            logger.info(f"Feature config loaded: {self._feature_config['n_features']} features")
        except Exception as e:
            logger.warning(f"Feature config load failed: {e}")

        # ── Load fraud threshold config ──
        try:
            with open(config.FRAUD_THRESHOLD_PATH) as f:
                self._fraud_config = json.load(f)
            logger.info(f"Fraud config loaded: threshold={self._fraud_config.get('fraud_threshold')}")
        except Exception as e:
            logger.warning(f"Fraud config load failed: {e}")

        self._models_loaded = True
        logger.info(f"All models loaded in {time.time() - start:.1f}s")

    def _load_rf_model(self):
        """Thread-safe lazy load of Random Forest model."""
        if self._rf_loaded:
            return
        with self._rf_lock:
            if self._rf_loaded:  # double-check after acquiring lock
                return
            try:
                rf_path = config.RF_MODEL_CURRENT
                if Path(rf_path).exists():
                    import joblib
                    model_data = joblib.load(rf_path)
                    self._rf_classifier = model_data["rf"]
                    self._rf_feature_names = model_data["features"]
                    logger.info(
                        f"Random Forest v{model_data.get('version', '?')} loaded: "
                        f"{model_data.get('n_samples')} samples, "
                        f"CV accuracy={model_data.get('cv_accuracy', '?')}"
                    )
                else:
                    logger.info("No RF model found, using pixel formula")
            except Exception as e:
                logger.warning(f"RF model load failed (using pixel formula): {e}")
                self._rf_classifier = None
            self._rf_loaded = True

    def invalidate_rf_model(self):
        """Force RF model reload on next comparison. Called after retraining."""
        with self._rf_lock:
            self._rf_loaded = False
            self._rf_classifier = None
            self._rf_feature_names = None
            logger.info("RF model cache invalidated")

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
                result.object_category = self._classify_object(baseline_image)
                result.processing_ms = int((time.time() - start_time) * 1000)
                return result

            # ── Object Classification (reuses CLIP, no extra model load) ──
            result.object_category = self._classify_object(baseline_image)

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
            warp_mask = alignment_result["mask"]

            # Ensure patrol_warped matches baseline dimensions (safety resize)
            bh, bw = baseline_enhanced.shape[:2]
            if patrol_warped.shape[:2] != (bh, bw):
                logger.debug(f"Size mismatch: baseline={bh}x{bw}, warped={patrol_warped.shape[0]}x{patrol_warped.shape[1]}. Resizing.")
                patrol_warped = cv2.resize(patrol_warped, (bw, bh))
                if warp_mask is not None:
                    warp_mask = cv2.resize(warp_mask, (bw, bh))

            # ── M4: Feature Extraction ──
            logger.debug("M4: Feature Extraction (9 signals)")

            # Structure features (grayscale path)
            structure_features = self._run_m4_structure(
                baseline_enhanced, patrol_warped, warp_mask
            )
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

            # ── DINOv2 Similarity (semantic feature for RF) ──
            result.dino_similarity = self._compute_dino_similarity(
                baseline_color if baseline_color is not None else baseline_image,
                patrol_image
            )
            logger.info(f"DINOv2 similarity: {result.dino_similarity:.4f}")

            # ── M5: Classification ──
            self._load_rf_model()

            if self._rf_classifier is not None:
                # Path A: Trained Random Forest model
                logger.debug("M5: Using Random Forest model")
                feature_values = {
                    "ssim_score": structure_features["ssim"],
                    "edge_score": structure_features["edge"],
                    "histogram_score": structure_features["histogram"],
                    "cluster_score": structure_features["cluster"],
                    "max_diff_area": structure_features["max_diff_area"],
                    "stability_score": structure_features["stability"],
                    "max_hue_shift": color_features["max_hue_shift"],
                    "max_delta_e": color_features["max_delta_e"],
                    "alignment_inliers": float(result.alignment_inliers or 0),
                    "dino_similarity": float(result.dino_similarity),
                }
                feature_vector = [feature_values[f] for f in self._rf_feature_names]
                features_array = np.array(feature_vector).reshape(1, -1)
                probability = float(self._rf_classifier.predict_proba(features_array)[0][1])
                ratio = round(min(max(probability, 0.0), 1.0), 4)

                # CLIP floor safety net: if CLIP says truly different
                # object, enforce minimum ratio even after RF prediction
                clip_sim = result.fraud_score
                if clip_sim and clip_sim < 0.80:
                    clip_floor = 0.50
                    if ratio < clip_floor:
                        logger.debug(
                            f"CLIP floor override: RF={ratio}, "
                            f"CLIP sim={clip_sim:.3f}, floor={clip_floor}"
                        )
                        ratio = clip_floor

                result.ratio = ratio
                logger.info(f"M5 RF: ratio={result.ratio}")
            else:
                # Path B: Alignment-aware formula with DINOv2 (no trained model yet)
                # DINOv2 provides semantic similarity: high = same object, low = different
                # dino_change = 1 - dino_sim: 0 = identical, 1 = completely different
                logger.debug("M5: Using alignment-aware pixel formula + DINOv2")
                inliers = result.alignment_inliers or 0

                ssim = structure_features["ssim"]
                edge = structure_features["edge"]
                hist = structure_features["histogram"]
                cluster = structure_features["cluster"]
                max_diff = structure_features["max_diff_area"]
                dino_change = 1.0 - (result.dino_similarity or 0.0)

                if inliers >= 30:
                    # Good alignment: pixel metrics + DINOv2 semantic check
                    ratio = (
                        0.35 * ssim
                        + 0.20 * edge
                        + 0.20 * hist
                        + 0.25 * dino_change
                    )
                else:
                    # Poor alignment: histogram-dominant + DINOv2 semantic check
                    ratio = (
                        0.30 * hist
                        + 0.15 * cluster
                        + 0.15 * max_diff
                        + 0.10 * ssim
                        + 0.30 * dino_change
                    )

                # CLIP floor: if CLIP says truly different object,
                # enforce minimum ratio regardless of pixel metrics.
                clip_sim = result.fraud_score
                if clip_sim and clip_sim < 0.80:
                    clip_floor = 0.50
                    ratio = max(ratio, clip_floor)
                    logger.debug(
                        f"CLIP floor: sim={clip_sim:.3f}, "
                        f"floor={clip_floor}, ratio={ratio:.4f}"
                    )

                result.ratio = round(min(max(ratio, 0.0), 1.0), 4)
                logger.info(
                    f"M5 pixel: ratio={result.ratio} inliers={inliers} "
                    f"ssim={ssim:.3f} edge={edge:.3f} hist={hist:.3f}"
                )

            # Generate heatmap only when ratio >= 10% (skip for clearly identical panels)
            if result.ratio >= 0.10:
                heatmap_path = self._generate_heatmap(
                    baseline_enhanced, patrol_warped,
                    structure_features.get("ssim_diff"),
                )
                result.heatmap_path = heatmap_path
            else:
                logger.debug(f"Skipping heatmap: ratio={result.ratio} < 0.10")

            result.success = True

        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}", exc_info=True)
            result.error = str(e)

        result.processing_ms = int((time.time() - start_time) * 1000)
        return result

    # ─────────────────────────────────────────
    #  M0: QR Detection (from m00_qr_detection.py)
    # ─────────────────────────────────────────

    def _run_m0_qr(self, image: np.ndarray) -> Optional[str]:
        """M0: Detect QR code, extract panel_id, mask QR region.

        Uses multi-scale detection (1x, 0.5x, 2x) with both the standard
        OpenCV QR detector and the ArUco-based fallback.

        Returns panel_id string or None if no QR found.
        """
        try:
            if image is None or image.size == 0:
                return None

            qr_info = self._detect_qr_multiscale(image)
            if qr_info["found"]:
                logger.info(f"QR detected: panel_id={qr_info['panel_id']}")
                return qr_info["panel_id"]

            logger.debug("No QR code detected")
            return None
        except Exception as e:
            logger.warning(f"QR detection error: {e}")
            return None

    def _detect_qr(self, image_bgr: np.ndarray) -> dict:
        """Detect and decode a QR code in a BGR image."""
        if image_bgr is None or image_bgr.size == 0:
            return {"found": False, "panel_id": None, "bbox": None, "raw_data": None}

        # Attempt 1: standard detector
        detector_std = cv2.QRCodeDetector()
        found, data, points = self._try_qr_detect(detector_std, image_bgr)

        # Attempt 2: ArUco-based fallback
        if not found:
            try:
                detector_aruco = cv2.QRCodeDetectorAruco()
                found, data, points = self._try_qr_detect(detector_aruco, image_bgr)
            except AttributeError:
                pass

        if not found:
            return {"found": False, "panel_id": None, "bbox": None, "raw_data": None}

        pts = np.array(points, dtype=np.float32)
        if pts.ndim == 3:
            pts = pts.reshape(-1, 2)
        bbox = np.rint(pts).astype(np.int32)
        panel_id = data.strip() if data else None

        return {
            "found": True,
            "panel_id": panel_id,
            "bbox": bbox.tolist(),
            "raw_data": data,
        }

    def _detect_qr_multiscale(self, image_bgr: np.ndarray) -> dict:
        """Try QR detection at 1x, 0.5x, 2x scales."""
        if image_bgr is None or image_bgr.size == 0:
            return {"found": False, "panel_id": None, "bbox": None, "raw_data": None}

        for scale in _MULTISCALE_FACTORS:
            if scale == 1.0:
                scaled = image_bgr
            else:
                interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                scaled = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=interp)

            result = self._detect_qr(scaled)
            if result["found"]:
                if scale != 1.0 and result["bbox"] is not None:
                    bbox_arr = np.array(result["bbox"], dtype=np.float64)
                    bbox_arr /= scale
                    result["bbox"] = np.rint(bbox_arr).astype(int).tolist()
                return result

        return {"found": False, "panel_id": None, "bbox": None, "raw_data": None}

    @staticmethod
    def _try_qr_detect(detector, image: np.ndarray) -> Tuple[bool, str, np.ndarray]:
        """Run detector.detectAndDecode and return (found, data, points)."""
        data, points, _ = detector.detectAndDecode(image)
        found = data is not None and len(data) > 0 and points is not None
        if not found:
            data = ""
            points = np.array([])
        return found, data, points

    # ─────────────────────────────────────────
    #  M1a: Image Validation + CLAHE (already implemented)
    # ─────────────────────────────────────────

    def _run_m1a_validation(self, image: np.ndarray) -> tuple:
        """M1a: EXIF fix, grayscale, CLAHE enhancement, quality checks."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
        enhanced = clahe.apply(gray)

        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = float(gray.mean())
        warning = blur < config.BLUR_WARNING_THRESHOLD or brightness < config.BRIGHTNESS_WARNING_THRESHOLD

        quality = {"blur": blur, "brightness": brightness, "warning": warning}
        return gray, enhanced, quality

    # ─────────────────────────────────────────
    #  M2: Fraud Detection (CLIP ViT-B/32)
    # ─────────────────────────────────────────

    def _run_m2_fraud(self, baseline: np.ndarray, patrol: np.ndarray) -> dict:
        """M2: CLIP-based fraud detection.

        Computes cosine similarity of CLIP ViT-B/32 image embeddings.
        If similarity < fraud_threshold, the patrol photo is rejected as
        fraudulent or from the wrong panel.
        """
        if self._clip_model is None or self._clip_processor is None:
            logger.warning("CLIP not loaded, skipping fraud detection (assuming valid)")
            return {"is_valid": True, "score": 1.0}

        try:
            import torch
            from PIL import Image

            pil_base = Image.fromarray(cv2.cvtColor(baseline, cv2.COLOR_BGR2RGB))
            pil_patrol = Image.fromarray(cv2.cvtColor(patrol, cv2.COLOR_BGR2RGB))

            inputs = self._clip_processor(
                images=[pil_base, pil_patrol], return_tensors="pt", padding=True
            )
            with torch.no_grad():
                # Run vision model explicitly to get pooled output, then project
                # This works identically across all transformers versions
                vision_outputs = self._clip_model.vision_model(
                    pixel_values=inputs.get("pixel_values")
                )
                # Get pooled output (CLS token)
                pooled = vision_outputs.pooler_output  # shape: (batch, hidden_dim)
                # Apply visual projection to get CLIP embedding space
                features = self._clip_model.visual_projection(pooled)  # shape: (batch, projection_dim)
                features = features / torch.norm(features, dim=-1, keepdim=True)

            score = float(torch.cosine_similarity(features[0:1], features[1:2]).item())
            threshold = config.FRAUD_THRESHOLD

            logger.debug(f"CLIP similarity: {score:.4f} (threshold: {threshold})")
            return {
                "is_valid": score >= threshold,
                "score": round(score, 6),
            }
        except Exception as e:
            logger.error(f"CLIP fraud detection error: {e}")
            return {"is_valid": True, "score": 1.0}

    # ─────────────────────────────────────────
    #  DINOv2 Similarity (Verify + Compare)
    # ─────────────────────────────────────────

    def _compute_dino_similarity(self, image1: np.ndarray, image2: np.ndarray) -> float:
        """Compute DINOv2 cosine similarity between two images.

        Used by:
        - Verify endpoint: same-object check (threshold-based)
        - Compare pipeline: RF feature (dino_similarity)

        Returns cosine similarity (0.0 to 1.0). Higher = more similar.
        """
        if self._dino_model is None:
            logger.warning("DINOv2 not loaded, returning 0.0")
            return 0.0

        try:
            import torch
            from PIL import Image
            from torchvision import transforms

            # DINOv2 ViT-S/14 expects 518x518 normalized images
            transform = transforms.Compose([
                transforms.Resize((518, 518)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

            pil1 = Image.fromarray(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
            pil2 = Image.fromarray(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))

            t1 = transform(pil1).unsqueeze(0).to(self._device)
            t2 = transform(pil2).unsqueeze(0).to(self._device)

            with torch.no_grad():
                feat1 = self._dino_model(t1)
                feat2 = self._dino_model(t2)

            similarity = float(torch.nn.functional.cosine_similarity(feat1, feat2).item())
            logger.debug(f"DINOv2 similarity: {similarity:.4f}")
            return round(similarity, 6)

        except Exception as e:
            logger.error(f"DINOv2 similarity error: {e}")
            return 0.0

    # ─────────────────────────────────────────
    #  Object Category Classification (CLIP)
    # ─────────────────────────────────────────

    # Categories that CLIP will classify against
    _OBJECT_CATEGORIES = [
        "electrical panel",
        "fire alarm panel",
        "fire extinguisher",
        "control panel",
        "meter box",
        "circuit breaker",
        "generator",
        "air conditioning unit",
        "security camera",
        "server rack",
        "vehicle",
        "door or gate",
        "valve or pipe",
        "water tank",
        "pump",
        "boiler or heater",
        "elevator or lift",
        "emergency exit sign",
        "sprinkler system",
        "fuel storage",
        "transformer",
        "solar panel",
        "fence or barrier",
        "safe or vault",
        "toolbox or cabinet",
        "fire hose",
        "gas cylinder",
        "roof or ceiling",
        "parking lot",
        "stairwell",
        "window or glass",
        "other equipment",
    ]

    def _classify_object(self, image: np.ndarray) -> Optional[str]:
        """Classify the object in the image using CLIP zero-shot classification."""
        if self._clip_model is None or self._clip_processor is None:
            return None

        try:
            import torch
            from PIL import Image

            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            text_labels = [f"a photo of a {cat}" for cat in self._OBJECT_CATEGORIES]

            inputs = self._clip_processor(
                text=text_labels, images=pil_img,
                return_tensors="pt", padding=True
            )
            with torch.no_grad():
                outputs = self._clip_model(**inputs)
                logits = outputs.logits_per_image[0]
                probs = logits.softmax(dim=0)

            best_idx = int(probs.argmax())
            best_prob = float(probs[best_idx])
            category = self._OBJECT_CATEGORIES[best_idx]

            logger.debug(f"Object classification: {category} ({best_prob:.2f})")
            return category

        except Exception as e:
            logger.warning(f"Object classification failed: {e}")
            return None

    # ─────────────────────────────────────────
    #  M3: Alignment (SuperPoint + LightGlue)
    # ─────────────────────────────────────────

    def _run_m3_alignment(
        self, baseline_enhanced: np.ndarray, patrol_enhanced: np.ndarray
    ) -> dict:
        """M3: SuperPoint + LightGlue alignment with ORB fallback.

        Returns dict with 'warped', 'inliers', 'method', 'mask'.
        """
        # Try LightGlue first
        warped, inliers, method, mask = self._align_lightglue(
            baseline_enhanced, patrol_enhanced
        )
        if warped is not None:
            return {"warped": warped, "inliers": inliers, "method": method, "mask": mask}

        # Fallback to ORB
        warped, inliers, method, mask = self._align_orb(
            baseline_enhanced, patrol_enhanced
        )
        if warped is not None:
            return {"warped": warped, "inliers": inliers, "method": method, "mask": mask}

        # Worst case: just resize patrol to match baseline
        h, w = baseline_enhanced.shape[:2]
        resized = cv2.resize(patrol_enhanced, (w, h))
        mask = np.ones((h, w), dtype=np.uint8) * 255
        return {"warped": resized, "inliers": 0, "method": "resize_only", "mask": mask}

    def _align_lightglue(
        self, ref_enhanced: np.ndarray, patrol_enhanced: np.ndarray
    ) -> Tuple[Optional[np.ndarray], int, str, Optional[np.ndarray]]:
        """Align patrol to reference using LightGlue + SuperPoint."""
        if self._lg_extractor is None or self._lg_matcher is None:
            logger.debug("LightGlue not loaded, skipping")
            return None, 0, "lightglue_not_loaded", None

        try:
            import torch
            from lightglue.utils import numpy_image_to_torch, rbd

            h, w = ref_enhanced.shape[:2]
            patrol_resized = cv2.resize(patrol_enhanced, (w, h))

            ref_t = numpy_image_to_torch(ref_enhanced).to(self._device)
            pat_t = numpy_image_to_torch(patrol_resized).to(self._device)

            with torch.no_grad():
                feats0 = self._lg_extractor.extract(ref_t)
                feats1 = self._lg_extractor.extract(pat_t)
                matches01 = self._lg_matcher({"image0": feats0, "image1": feats1})

            feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
            kpts0 = feats0["keypoints"].cpu().numpy()
            kpts1 = feats1["keypoints"].cpu().numpy()
            matches = matches01["matches"].cpu().numpy()
            mkpts0 = kpts0[matches[:, 0]]
            mkpts1 = kpts1[matches[:, 1]]
            n_inliers = len(mkpts0)

            if n_inliers >= 10:
                H, ransac_mask = cv2.findHomography(mkpts1, mkpts0, cv2.RANSAC, 5.0)
                if H is not None:
                    warped = cv2.warpPerspective(patrol_resized, H, (w, h))
                    ransac_inliers = int(ransac_mask.sum()) if ransac_mask is not None else n_inliers
                    mask = self._create_warp_mask(warped)
                    return warped, ransac_inliers, "lightglue", mask

            return None, n_inliers, "lightglue_failed", None
        except Exception as e:
            logger.warning(f"LightGlue alignment error: {e}")
            return None, 0, "lightglue_error", None

    def _align_orb(
        self, ref_enhanced: np.ndarray, patrol_enhanced: np.ndarray
    ) -> Tuple[Optional[np.ndarray], int, str, Optional[np.ndarray]]:
        """ORB-based alignment fallback."""
        try:
            h, w = ref_enhanced.shape[:2]
            patrol_resized = cv2.resize(patrol_enhanced, (w, h))

            orb = cv2.ORB_create(nfeatures=2000)
            kp1, des1 = orb.detectAndCompute(ref_enhanced, None)
            kp2, des2 = orb.detectAndCompute(patrol_resized, None)

            if des1 is None or des2 is None:
                return None, 0, "orb_failed", None

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            good = [m for m in matches if m.distance < 50]

            if len(good) >= 4:
                src = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                dst = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                H, ransac_mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
                if H is not None:
                    warped = cv2.warpPerspective(patrol_resized, H, (w, h))
                    inliers = int(ransac_mask.sum()) if ransac_mask is not None else len(good)
                    mask = self._create_warp_mask(warped)
                    return warped, inliers, "orb", mask

            return None, len(good), "orb_failed", None
        except Exception as e:
            logger.warning(f"ORB alignment error: {e}")
            return None, 0, "orb_error", None

    @staticmethod
    def _create_warp_mask(warped: np.ndarray) -> np.ndarray:
        """Create a mask of valid (non-black-border) pixels after warping."""
        mask = (warped > 5).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask = cv2.erode(mask, kernel, iterations=1)
        return mask

    # ─────────────────────────────────────────
    #  M4: Structure Feature Extraction
    # ─────────────────────────────────────────

    def _run_m4_structure(
        self,
        baseline: np.ndarray,
        patrol_warped: np.ndarray,
        warp_mask: Optional[np.ndarray] = None,
    ) -> dict:
        """M4: Structure feature extraction (7 features from grayscale path).

        Extracts SSIM, multi-scale SSIM, edge, histogram, cluster,
        max_diff_area, and stability scores from the aligned pair.
        All metrics are computed within the valid warp mask with a
        central ROI (center 70%) to reduce edge alignment artifacts.
        """
        h, w = baseline.shape[:2]

        # Build combined mask: warp mask AND central ROI (center 70%)
        if warp_mask is None:
            warp_mask = np.ones((h, w), dtype=np.uint8) * 255

        roi_mask = np.zeros_like(warp_mask)
        my, mx = int(h * 0.15), int(w * 0.15)
        roi_mask[my:h - my, mx:w - mx] = 255
        mask = cv2.bitwise_and(warp_mask, roi_mask)

        # ── SSIM ──
        ssim_change, ssim_diff = self._compute_ssim(baseline, patrol_warped, mask)

        # ── Multi-scale SSIM ──
        ms_ssim_change = self._compute_multiscale_ssim(baseline, patrol_warped, mask)

        # ── Edge change ──
        edge_change = self._compute_edge_change(baseline, patrol_warped, mask)

        # ── Histogram change ──
        histogram_change = self._compute_histogram_change(baseline, patrol_warped, mask)

        # ── Concentrated change (cluster) ──
        conc = self._compute_concentrated_change(ssim_diff, mask)

        # ── Top-percentile change ──
        top5pct = self._compute_top_percentile_change(baseline, patrol_warped, mask)

        # ── Change peakedness ──
        peakedness = self._compute_change_peakedness(baseline, patrol_warped, mask)

        # ── Diff image features ──
        diff_features = self._compute_diff_image_features(baseline, patrol_warped, mask)

        # ── Stability weighted (simplified: use SSIM weighted by inverse variance) ──
        # With a single baseline, stability_weighted approximates ssim_change
        stability_weighted = ssim_change

        # ── Ratio of top5 to SSIM (engineered feature) ──
        ratio_top5_to_ssim = top5pct / max(ssim_change, 0.001)

        return {
            "ssim": round(float(ssim_change), 6),
            "ms_ssim": round(float(ms_ssim_change), 6),
            "edge": round(float(edge_change), 6),
            "histogram": round(float(histogram_change), 6),
            "cluster": round(float(conc["cluster_score"]), 6),
            "max_diff_area": round(float(diff_features["diff_connected_max_area"]), 6),
            "stability": round(float(stability_weighted), 6),
            # Extra features needed by SVM
            "top5pct_change": round(float(top5pct), 6),
            "change_peakedness": round(float(peakedness), 6),
            "ratio_top5_to_ssim": round(float(ratio_top5_to_ssim), 6),
            "ssim_diff": ssim_diff,  # kept for heatmap generation
        }

    @staticmethod
    def _compute_ssim(
        ref: np.ndarray, aligned: np.ndarray, mask: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Compute SSIM change score and diff map within the valid mask."""
        score, diff = ssim(ref, aligned, full=True)
        diff_normalized = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
        mask_bool = mask > 127
        if mask_bool.sum() > 0:
            masked_mean = diff[mask_bool].mean()
            ssim_change = 1.0 - masked_mean
        else:
            ssim_change = 1.0 - score
        return ssim_change, diff_normalized

    @staticmethod
    def _compute_multiscale_ssim(
        ref: np.ndarray, aligned: np.ndarray, mask: np.ndarray
    ) -> float:
        """SSIM at 1/4 resolution for noise robustness."""
        scale = 0.25
        h, w = ref.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)

        ref_small = cv2.resize(ref, (new_w, new_h), interpolation=cv2.INTER_AREA)
        aligned_small = cv2.resize(aligned, (new_w, new_h), interpolation=cv2.INTER_AREA)
        mask_small = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        win = min(7, new_h - 1, new_w - 1)
        if win % 2 == 0:
            win -= 1
        win = max(3, win)

        score, diff = ssim(ref_small, aligned_small, full=True, win_size=win)
        mask_bool = mask_small > 127
        if mask_bool.sum() > 0:
            return 1.0 - float(diff[mask_bool].mean())
        return 1.0 - score

    @staticmethod
    def _compute_edge_change(
        ref: np.ndarray, aligned: np.ndarray, mask: np.ndarray
    ) -> float:
        """Canny edge change with blur and dilation tolerance."""
        ref_blur = cv2.GaussianBlur(ref, (5, 5), 1.0)
        pat_blur = cv2.GaussianBlur(aligned, (5, 5), 1.0)

        edges_ref = cv2.Canny(ref_blur, CANNY_LOW, CANNY_HIGH)
        edges_pat = cv2.Canny(pat_blur, CANNY_LOW, CANNY_HIGH)

        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        edges_ref_dilated = cv2.dilate(edges_ref, dilate_kernel, iterations=1)
        edges_pat_dilated = cv2.dilate(edges_pat, dilate_kernel, iterations=1)

        new_edges = cv2.bitwise_and(edges_pat, cv2.bitwise_not(edges_ref_dilated))
        lost_edges = cv2.bitwise_and(edges_ref, cv2.bitwise_not(edges_pat_dilated))
        edge_diff = cv2.bitwise_or(new_edges, lost_edges)

        mask_bool = mask > 127
        ref_in_mask = np.count_nonzero(edges_ref[mask_bool]) if mask_bool.any() else 0
        pat_in_mask = np.count_nonzero(edges_pat[mask_bool]) if mask_bool.any() else 0
        diff_in_mask = np.count_nonzero(edge_diff[mask_bool]) if mask_bool.any() else 0

        total = max(ref_in_mask, pat_in_mask, 1)
        return min(diff_in_mask / total, 1.0)

    @staticmethod
    def _compute_histogram_change(
        ref: np.ndarray, aligned: np.ndarray, mask: np.ndarray
    ) -> float:
        """Histogram correlation change within the valid mask."""
        mask_u8 = (mask > 127).astype(np.uint8) * 255
        hist1 = cv2.calcHist([ref], [0], mask_u8, [256], [0, 256])
        hist2 = cv2.calcHist([aligned], [0], mask_u8, [256], [0, 256])
        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return 1.0 - max(correlation, 0)

    @staticmethod
    def _compute_concentrated_change(ssim_diff: np.ndarray, mask: np.ndarray) -> dict:
        """Detect concentrated clusters of change in the SSIM diff map."""
        mask_bool = mask > 127
        change_map = 1.0 - ssim_diff
        change_masked = np.zeros_like(change_map)
        change_masked[mask_bool] = change_map[mask_bool]

        valid_pixels = mask_bool.sum()
        if valid_pixels == 0:
            return {"cluster_score": 0, "hot_pixel_ratio": 0, "max_local_change": 0, "num_clusters": 0}

        # Hot pixel ratio
        hot_threshold = 0.55
        hot_pixels = (change_masked > hot_threshold).sum()
        hot_pixel_ratio = hot_pixels / valid_pixels

        # Find concentrated change clusters
        binary_change = (change_masked > 0.45).astype(np.uint8) * 255
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        binary_clean = cv2.morphologyEx(binary_change, cv2.MORPH_CLOSE, kernel_close)
        binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_OPEN, kernel_open)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_clean)

        min_cluster_area = 400
        significant_clusters = []
        total_cluster_area = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_cluster_area:
                significant_clusters.append(area)
                total_cluster_area += area

        cluster_score = total_cluster_area / valid_pixels

        # Max local change
        kernel_size = 51
        local_mean = cv2.blur(change_masked.astype(np.float32), (kernel_size, kernel_size))
        local_mean[~mask_bool] = 0
        max_local_change = float(local_mean.max())

        return {
            "cluster_score": round(float(cluster_score), 6),
            "hot_pixel_ratio": round(float(hot_pixel_ratio), 6),
            "max_local_change": round(float(max_local_change), 6),
            "num_clusters": len(significant_clusters),
        }

    @staticmethod
    def _compute_top_percentile_change(
        ref: np.ndarray, aligned: np.ndarray, mask: np.ndarray,
        block_size: int = 32, percentile: float = 95,
    ) -> float:
        """Change level at the 95th percentile of most-changed blocks."""
        _, diff = ssim(ref, aligned, full=True)
        change_map = 1.0 - diff
        h, w = change_map.shape
        mask_bool = mask > 127

        block_scores = []
        for y in range(0, h - block_size + 1, block_size):
            for x in range(0, w - block_size + 1, block_size):
                block_change = change_map[y:y + block_size, x:x + block_size]
                block_mask = mask_bool[y:y + block_size, x:x + block_size]
                valid_count = block_mask.sum()
                if valid_count > block_size * block_size * 0.5:
                    block_scores.append(float(block_change[block_mask].mean()))

        if not block_scores:
            return 0.0
        return round(float(np.percentile(block_scores, percentile)), 6)

    @staticmethod
    def _compute_change_peakedness(
        ref: np.ndarray, aligned: np.ndarray, mask: np.ndarray,
        block_size: int = 32,
    ) -> float:
        """Ratio of max-block-change to median-block-change (log-scaled)."""
        _, diff = ssim(ref, aligned, full=True)
        change_map = 1.0 - diff
        h, w = change_map.shape
        mask_bool = mask > 127

        block_scores = []
        for y in range(0, h - block_size + 1, block_size):
            for x in range(0, w - block_size + 1, block_size):
                block = change_map[y:y + block_size, x:x + block_size]
                bm = mask_bool[y:y + block_size, x:x + block_size]
                if bm.sum() > block_size * block_size * 0.5:
                    block_scores.append(float(block[bm].mean()))

        if len(block_scores) < 3:
            return 0.0

        median_val = float(np.median(block_scores))
        max_val = float(np.max(block_scores))
        if median_val < 0.01:
            return round(max_val * 10, 6)

        ratio = max_val / median_val
        return round(float(np.log1p(ratio - 1)), 6)

    @staticmethod
    def _compute_diff_image_features(
        ref: np.ndarray, aligned: np.ndarray, mask: np.ndarray,
    ) -> dict:
        """Extract features from the difference image itself."""
        diff = cv2.absdiff(ref, aligned)
        diff_masked = diff.copy()
        mask_bool = mask > 127
        diff_masked[~mask_bool] = 0

        valid_count = mask_bool.sum()
        if valid_count < 100:
            return {
                "diff_energy": 0, "diff_peak_ratio": 0,
                "diff_spatial_concentration": 0,
                "diff_above_2sigma": 0, "diff_connected_max_area": 0,
            }

        valid_vals = diff_masked[mask_bool].astype(np.float32)
        mean_val = valid_vals.mean()
        std_val = valid_vals.std()

        # Energy
        diff_energy = float(mean_val / 255.0)

        # Peak ratio
        block_sz = 32
        h, w = diff.shape
        block_means = []
        for y in range(0, h - block_sz + 1, block_sz // 2):
            for x in range(0, w - block_sz + 1, block_sz // 2):
                bm = mask_bool[y:y + block_sz, x:x + block_sz]
                if bm.sum() > block_sz * block_sz * 0.5:
                    bv = diff_masked[y:y + block_sz, x:x + block_sz][bm].mean()
                    block_means.append(float(bv))

        if block_means and mean_val > 1:
            diff_peak_ratio = max(block_means) / mean_val
        else:
            diff_peak_ratio = 1.0

        # Spatial concentration
        if len(block_means) > 5:
            sorted_blocks = sorted(block_means, reverse=True)
            top_5pct = sorted_blocks[:max(1, len(sorted_blocks) // 20)]
            total_energy = sum(block_means)
            top_energy = sum(top_5pct)
            diff_spatial_concentration = top_energy / max(total_energy, 1e-8)
        else:
            diff_spatial_concentration = 0.0

        # Fraction of pixels above 2-sigma
        if std_val > 1:
            threshold_2s = mean_val + 2 * std_val
            diff_above_2sigma = float((valid_vals > threshold_2s).sum()) / valid_count
        else:
            diff_above_2sigma = 0.0

        # Connected component analysis
        binary = (diff_masked > max(mean_val + std_val, 25)).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        max_area = 0
        for i in range(1, num_labels):
            max_area = max(max_area, stats[i, cv2.CC_STAT_AREA])
        diff_connected_max_area = max_area / valid_count

        return {
            "diff_energy": round(float(diff_energy), 6),
            "diff_peak_ratio": round(float(diff_peak_ratio), 6),
            "diff_spatial_concentration": round(float(diff_spatial_concentration), 6),
            "diff_above_2sigma": round(float(diff_above_2sigma), 6),
            "diff_connected_max_area": round(float(diff_connected_max_area), 6),
        }

    # ─────────────────────────────────────────
    #  M4: Color Feature Extraction (already implemented)
    # ─────────────────────────────────────────

    def _run_m4_color(self, baseline_color: np.ndarray, patrol_color: np.ndarray) -> dict:
        """M4: Color feature extraction (HSV hue delta + LAB delta-E)."""
        # Resize patrol to match baseline dimensions for color comparison
        h, w = baseline_color.shape[:2]
        patrol_resized = cv2.resize(patrol_color, (w, h))

        # HSV hue comparison
        hsv_base = cv2.cvtColor(baseline_color, cv2.COLOR_BGR2HSV)
        hsv_patrol = cv2.cvtColor(patrol_resized, cv2.COLOR_BGR2HSV)
        hue_diff = cv2.absdiff(
            hsv_base[:, :, 0].astype(np.int16),
            hsv_patrol[:, :, 0].astype(np.int16),
        )
        hue_diff = np.minimum(np.abs(hue_diff), 180 - np.abs(hue_diff)).astype(np.float32)
        max_hue_shift = float(np.percentile(hue_diff, 95))

        # LAB delta-E
        lab_base = cv2.cvtColor(baseline_color, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_patrol = cv2.cvtColor(patrol_resized, cv2.COLOR_BGR2LAB).astype(np.float32)
        delta_e = np.sqrt(np.sum((lab_base - lab_patrol) ** 2, axis=2))
        max_delta_e = float(np.percentile(delta_e, 95))

        return {
            "max_hue_shift": max_hue_shift,
            "max_delta_e": max_delta_e,
        }

    # ─────────────────────────────────────────
    #  Heatmap Generation
    # ─────────────────────────────────────────

    def _generate_heatmap(
        self,
        baseline: np.ndarray,
        patrol: np.ndarray,
        ssim_diff_map: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """Generate SSIM-based diff heatmap and save to disk."""
        try:
            if ssim_diff_map is not None:
                # Use SSIM diff map for better heatmap quality
                change_map = 1.0 - ssim_diff_map
                change_uint8 = (change_map * 255).astype(np.uint8)
                heatmap = cv2.applyColorMap(change_uint8, cv2.COLORMAP_JET)
            else:
                diff = cv2.absdiff(baseline, patrol)
                heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)

            baseline_bgr = cv2.cvtColor(baseline, cv2.COLOR_GRAY2BGR)
            overlay = cv2.addWeighted(baseline_bgr, 0.5, heatmap, 0.5, 0)

            heatmap_dir = Path(config.HEATMAP_DIR)
            heatmap_dir.mkdir(parents=True, exist_ok=True)
            filename = f"heatmap_{int(time.time() * 1000)}.jpg"
            path = str(heatmap_dir / filename)
            cv2.imwrite(path, overlay)
            return path
        except Exception as e:
            logger.warning(f"Heatmap generation failed: {e}")
            return None

    # ─────────────────────────────────────────
    #  M5: SVM Classification
    # ─────────────────────────────────────────

    def _run_m5_classify(self, structure_features: dict) -> dict:
        """M5: SVM classification using model.joblib.

        model.joblib is a dict containing:
            - 'svm': fitted SVC with predict_proba
            - 'scaler': fitted StandardScaler
            - 'features': ordered feature names

        Uses the 7 features specified in feature_config.json:
            1. max_diff_connected_max_area
            2. ratio_top5_to_ssim
            3. stability_weighted (stability)
            4. max_ms_ssim_change (ms_ssim)
            5. cluster_score
            6. min_top5pct_change (top5pct_change — single ref, so min == value)
            7. max_change_peakedness (change_peakedness — single ref, so max == value)
        """
        if self._svm_classifier is None:
            logger.error("SVM classifier not loaded — cannot classify")
            return {"probability": 0.15, "error": "SVM not loaded"}

        try:
            # Map feature_config names to structure_features keys
            feature_map = {
                "max_diff_connected_max_area": structure_features.get("max_diff_area", 0),
                "ratio_top5_to_ssim": structure_features.get("ratio_top5_to_ssim", 0),
                "stability_weighted": structure_features.get("stability", 0),
                "max_ms_ssim_change": structure_features.get("ms_ssim", 0),
                "cluster_score": structure_features.get("cluster", 0),
                "min_top5pct_change": structure_features.get("top5pct_change", 0),
                "max_change_peakedness": structure_features.get("change_peakedness", 0),
            }

            # Build feature vector in the order specified by feature_config
            if self._feature_config is not None:
                feature_names = self._feature_config["features"]
            else:
                feature_names = [
                    "max_diff_connected_max_area",
                    "ratio_top5_to_ssim",
                    "stability_weighted",
                    "max_ms_ssim_change",
                    "cluster_score",
                    "min_top5pct_change",
                    "max_change_peakedness",
                ]

            feature_vector = [feature_map.get(name, 0.0) for name in feature_names]
            features_array = np.array(feature_vector).reshape(1, -1)

            logger.info(f"M5 raw features: {dict(zip(feature_names, feature_vector))}")

            # Scale using the model's fitted StandardScaler
            if self._svm_scaler is not None:
                features_scaled = self._svm_scaler.transform(features_array)
            elif self._feature_config and "scaler_params" in self._feature_config:
                # Fallback: manual scaling from feature_config.json
                scaler = self._feature_config["scaler_params"]
                mean = np.array(scaler["mean"])
                scale = np.array(scaler["scale"])
                features_scaled = (features_array - mean) / scale
            else:
                features_scaled = features_array

            # Predict probability using the SVC classifier
            if hasattr(self._svm_classifier, "predict_proba"):
                probability = float(self._svm_classifier.predict_proba(features_scaled)[0][1])
            else:
                decision = float(self._svm_classifier.decision_function(features_scaled)[0])
                probability = 1.0 / (1.0 + np.exp(-decision))

            logger.info(
                f"M5 SVM prediction: probability={probability:.6f}, "
                f"ratio={probability * 100:.2f}%"
            )
            return {"probability": round(probability, 6)}
        except Exception as e:
            logger.error(f"SVM classification error: {e}", exc_info=True)
            return {"probability": 0.15, "error": str(e)}


# Singleton
_pipeline = None


def get_pipeline() -> PipelineRunner:
    global _pipeline
    if _pipeline is None:
        _pipeline = PipelineRunner()
    return _pipeline
