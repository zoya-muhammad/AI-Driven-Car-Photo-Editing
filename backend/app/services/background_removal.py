"""RMBG-1.4 background removal service using Hugging Face transformers."""

import io
import logging
from pathlib import Path
from typing import Optional

from PIL import Image

from app.config import RMBG_MODEL

logger = logging.getLogger(__name__)


class BackgroundRemovalService:
    """Service for removing backgrounds using RMBG-1.4 model."""

    _instance: Optional["BackgroundRemovalService"] = None
    _pipeline = None

    def __new__(cls) -> "BackgroundRemovalService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _get_pipeline(self):
        """Lazy-load the pipeline to avoid startup delay."""
        if self._pipeline is None:
            logger.info("Loading RMBG-1.4 model...")
            from transformers import pipeline

            self._pipeline = pipeline(
                "image-segmentation",
                model=RMBG_MODEL,
                trust_remote_code=True,
            )
            logger.info("RMBG-1.4 model loaded successfully")
        return self._pipeline

    def remove_background(
        self,
        image_input: bytes | Path,
        output_format: str = "png",
        background_color: tuple[int, int, int] = (255, 255, 255),
    ) -> bytes:
        """
        Remove background from image using RMBG-1.4.
        Returns image bytes with white background (studio-style for car photos).
        """
        pipe = self._get_pipeline()

        if isinstance(image_input, Path):
            image = Image.open(image_input).convert("RGB")
        else:
            image = Image.open(io.BytesIO(image_input)).convert("RGB")

        # Run inference - returns PIL image with mask applied
        result = pipe(image)

        # pipe() returns the composited image (foreground + transparent bg)
        # We need to replace transparency with white for car studio look
        if isinstance(result, Image.Image):
            no_bg_image = result
        else:
            # Handle list/tuple output
            no_bg_image = result[0] if isinstance(result, (list, tuple)) else result

        if no_bg_image.mode == "RGBA":
            # Composite onto white background
            background = Image.new("RGB", no_bg_image.size, background_color)
            background.paste(no_bg_image, mask=no_bg_image.split()[3])
            no_bg_image = background

        output_buffer = io.BytesIO()
        fmt = output_format.upper()
        if fmt == "PNG":
            no_bg_image.save(output_buffer, format=fmt, compress_level=6)
        else:
            no_bg_image.save(output_buffer, format=fmt, quality=95)
        output_buffer.seek(0)
        return output_buffer.read()


# Singleton instance
background_removal_service = BackgroundRemovalService()
