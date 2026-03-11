"""Image loading utilities including RAW (NEF) support."""

import io
import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

RAW_EXTENSIONS = {".nef", ".nrw", ".arw", ".cr2", ".dng"}


def load_image(data: bytes, filename: str) -> Image.Image:
    """
    Load image from bytes. Handles JPEG, PNG, WebP, and RAW (NEF) formats.
    Returns RGB PIL Image.
    """
    ext = Path(filename).suffix.lower()

    if ext in RAW_EXTENSIONS:
        return _load_raw(data, filename)

    return Image.open(io.BytesIO(data)).convert("RGB")


def _load_raw(data: bytes, filename: str) -> Image.Image:
    """Decode Nikon/other RAW (NEF) to RGB PIL Image."""
    try:
        import rawpy
        import numpy as np
    except ImportError:
        raise ValueError(
            "RAW (NEF) support requires rawpy. Install with: pip install rawpy"
        )

    with rawpy.imread(io.BytesIO(data)) as raw:
        rgb = raw.postprocess(use_camera_wb=True)
    return Image.fromarray(rgb)
