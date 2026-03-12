"""
Image downloader utility.
Supports Oracle Cloud Object Storage URLs and generic HTTP URLs.
Mock mode available when Oracle credentials aren't configured.
"""
import httpx
import numpy as np
import cv2
from pathlib import Path
from app.core import config
import logging

logger = logging.getLogger(__name__)



class ImageDownloader:
    """Downloads images from Oracle Cloud or any HTTP URL."""

    def __init__(self):
        self.timeout = config.IMAGE_DOWNLOAD_TIMEOUT
        self.max_size = config.IMAGE_MAX_SIZE_MB * 1024 * 1024

    async def download(self, url: str) -> np.ndarray:
        """
        Download image from URL and return as OpenCV numpy array (BGR).
        
        Supports:
        - Oracle Cloud pre-authenticated request URLs
        - Oracle Cloud with API key auth (when configured)
        - Any public HTTP/HTTPS image URL
        - Local file paths (for testing)
        
        Returns:
            np.ndarray: Image in BGR format (OpenCV standard)
        
        Raises:
            ImageDownloadError: If download or decode fails
        """
        # Local file path (for testing)
        if url.startswith("file://") or url.startswith("/"):
            return self._load_local(url)

        # HTTP download
        try:
            headers = self._build_headers(url)
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()

                content_length = len(response.content)
                if content_length > self.max_size:
                    raise ImageDownloadError(
                        f"Image too large: {content_length / 1024 / 1024:.1f}MB "
                        f"(max {config.IMAGE_MAX_SIZE_MB}MB)"
                    )

                return self._decode_image(response.content)

        except httpx.HTTPStatusError as e:
            raise ImageDownloadError(f"HTTP {e.response.status_code}: {url}") from e
        except httpx.RequestError as e:
            raise ImageDownloadError(f"Request failed: {url} - {str(e)}") from e

    def _build_headers(self, url: str) -> dict:
        """
        Build auth headers for Oracle Cloud if needed.
        Pre-authenticated request URLs don't need auth.
        
        TODO: Implement Oracle Cloud API key signing when credentials are provided.
        Currently supports:
        - Pre-authenticated request URLs (no auth needed)
        - Public URLs (no auth needed)
        """
        headers = {}

        if config.ORACLE_CLOUD["enabled"] and "oraclecloud.com" in url:
            # If it's a PAR URL, no auth needed
            if "/p/" in url:
                logger.debug("Oracle Cloud PAR URL detected, no auth needed")
                return headers

            # TODO: Implement Oracle Cloud API key signing
            # This requires the OCI SDK or manual request signing
            # For now, log a warning
            logger.warning(
                "Oracle Cloud URL detected but API key signing not yet implemented. "
                "Use pre-authenticated request URLs or configure PAR_BASE_URL."
            )

        return headers

    def _load_local(self, path: str) -> np.ndarray:
        """Load image from local file path."""
        clean_path = path.replace("file://", "")
        img = cv2.imread(clean_path)
        if img is None:
            raise ImageDownloadError(f"Cannot read local image: {clean_path}")
        return img

    def _decode_image(self, content: bytes) -> np.ndarray:
        """Decode image bytes to OpenCV numpy array."""
        img_array = np.frombuffer(content, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ImageDownloadError("Failed to decode image bytes")
        return img


class ImageDownloadError(Exception):
    """Raised when image download or decode fails."""
    pass


# Singleton
_downloader = None

def get_downloader() -> ImageDownloader:
    global _downloader
    if _downloader is None:
        _downloader = ImageDownloader()
    return _downloader
