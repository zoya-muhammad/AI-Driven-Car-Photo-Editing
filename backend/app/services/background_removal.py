"""RMBG-1.4 background removal service using Hugging Face transformers."""

import io
import logging
from pathlib import Path
from typing import Optional

from PIL import Image

from app.config import RMBG_MODEL
from app.services.image_utils import load_image

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
        keep_transparent: bool = False,
        filename: str | None = None,
    ) -> bytes:
        """
        Remove background from image using RMBG-1.4.
        Returns image bytes. With keep_transparent=False, composites onto background_color.
        """
        pipe = self._get_pipeline()

        if isinstance(image_input, Path):
            data = image_input.read_bytes()
            fname = image_input.name
        else:
            data = image_input
            fname = filename or "image.jpg"
        image = load_image(data, fname)

        result = pipe(image)
        if isinstance(result, Image.Image):
            no_bg_image = result
        else:
            no_bg_image = result[0] if isinstance(result, (list, tuple)) else result

        if no_bg_image.mode == "RGBA" and not keep_transparent:
            background = Image.new("RGB", no_bg_image.size, background_color)
            background.paste(no_bg_image, mask=no_bg_image.split()[3])
            no_bg_image = background

        output_buffer = io.BytesIO()
        fmt = output_format.upper()
        if fmt == "PNG":
            no_bg_image.save(output_buffer, format="PNG", compress_level=6)
        elif fmt in ("JPEG", "JPG"):
            if no_bg_image.mode == "RGBA":
                background = Image.new("RGB", no_bg_image.size, background_color)
                background.paste(no_bg_image, mask=no_bg_image.split()[3])
                no_bg_image = background
            no_bg_image.save(output_buffer, format="JPEG", quality=95)
        else:
            no_bg_image.save(output_buffer, format=fmt, quality=95)
        output_buffer.seek(0)
        return output_buffer.read()


# Singleton instance
background_removal_service = BackgroundRemovalService()
