"""
Gemini API service for car image processing.

Uses gemini-3.1-flash-image-preview model to process car photos:
- Remove reflections
- Clean the floor
- Maintain natural car color
- Keep walls and floor intact
"""

import base64
import io
import logging
import time
from PIL import Image

from app.config import GEMINI_API_KEY

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-3.1-flash-image-preview"
# Image generation can take 2-5 min; 3 min caused ReadTimeout on slow responses
REQUEST_TIMEOUT_MS = 360_000  # 6 minutes
MAX_INPUT_SIZE = 2048  # Resize large images to max 2048x2048 before sending to API
MAX_RETRIES = 3  # Retry up to 3 times on server errors
RETRY_DELAY_SECONDS = 10  # Wait 10 seconds between retries

ENHANCE_PROMPT = (
    "You are a professional automotive photo editor. "
    "Edit this car dealership photo with these exact instructions:\n\n"
    "BACKGROUND: Remove all studio lights, light fixtures, cables, and equipment visible in the background. "
    "Clean the walls to be smooth and uniform white. Keep the wall corner visible and natural. "
    "Do not replace or remove the walls and floor completely.\n\n"
    "FLOOR: Clean all dirt, dust, marks, tire tracks and uneven patches from the floor. "
    "Keep the same floor color and texture, just make it look professionally cleaned.\n\n"
    "REFLECTIONS: Aggressively remove all white light reflections from the car hood, roof, doors and fenders. "
    "Remove all glare from windows and windshield. Windows should look dark and clear with no glare.\n\n"
    "CAR COLOR: Keep the exact original car color. Do not darken or lighten the car paint. "
    "Do not change the hue.\n\n"
    "TIRES: Make tires deep black. Remove all dust and discoloration.\n\n"
    "OVERALL: The final image should look like a professional showroom photo. "
    "The editing should be clearly visible and significant compared to the original. "
    "Do not make subtle changes, make professional quality edits.\n\n"
    "Return only the edited image with no text or watermarks."
)

BACKGROUND_REMOVAL_PROMPT = (
    "Remove the background from this car photo. "
    "Replace the background with a clean solid white background. "
    "Keep the car exactly as it is - preserve all details, colors, and reflections. "
    "Return only the edited image, no text."
)

BACKGROUND_REMOVAL_TRANSPARENT_PROMPT = (
    "Remove the background from this car photo. "
    "Make the background fully transparent. "
    "Keep the car exactly as it is - preserve all details, colors, and reflections. "
    "Return only the edited image with transparent background, no text."
)


# Gemini 3.1 Flash Image only accepts these aspect ratios
_ALLOWED_ASPECT_RATIOS = (
    "1:1", "1:4", "1:8", "2:3", "3:2", "3:4", "4:1", "4:3", "4:5", "5:4",
    "8:1", "9:16", "16:9", "21:9",
)


def _resize_for_api(pil_img: Image.Image, max_side: int = MAX_INPUT_SIZE) -> Image.Image:
    """Resize image so longest side <= max_side. Reduces payload and processing time."""
    w, h = pil_img.size
    if max(w, h) <= max_side:
        return pil_img
    if w >= h:
        new_w, new_h = max_side, int(h * max_side / w)
    else:
        new_w, new_h = int(w * max_side / h), max_side
    return pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def _aspect_ratio_str(w: int, h: int) -> str:
    """Map image dimensions to nearest allowed Gemini aspect ratio."""
    if h == 0:
        return "1:1"
    actual = w / h
    best_ratio = "1:1"
    best_diff = float("inf")
    for ratio in _ALLOWED_ASPECT_RATIOS:
        num, den = map(int, ratio.split(":"))
        target = num / den
        diff = abs(actual - target)
        if diff < best_diff:
            best_diff = diff
            best_ratio = ratio
    return best_ratio


def _get_client():
    """Lazy-load Gemini client."""
    from google import genai

    api_key = GEMINI_API_KEY
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is required. Set it in backend/.env"
        )
    return genai.Client(api_key=api_key)


def process_car_image(
    image_data: bytes,
    filename: str,
    mode: str = "enhance-preserve",
    output_format: str = "png",
    background: str = "white",
) -> bytes:
    """
    Process car image using Gemini API.

    Args:
        image_data: Raw image bytes (JPEG, PNG, WebP, NEF)
        filename: Original filename (for format detection)
        mode: "enhance-preserve" or "standard"
        output_format: "png", "jpg", or "webp"

    Returns:
        Processed image as bytes
    """
    from app.services.image_utils import load_image

    pil_img = load_image(image_data, filename)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    # Resize large images (e.g. NEF 6016x4016) to max 2048x2048
    pil_img = _resize_for_api(pil_img)
    w, h = pil_img.size
    aspect = _aspect_ratio_str(w, h)

    if mode == "enhance-preserve":
        prompt = ENHANCE_PROMPT
    elif background == "transparent":
        prompt = BACKGROUND_REMOVAL_TRANSPARENT_PROMPT
    else:
        prompt = BACKGROUND_REMOVAL_PROMPT

    client = _get_client()

    # Convert PIL to high-quality JPEG before sending to API
    # This prevents checkerboard glitches from sending raw/PNG bytes directly
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=95)
    img_bytes = buf.getvalue()

    # Use types for config
    from google.genai import types

    # Retry logic for 503 and other server errors
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info("Gemini API attempt %d/%d for %s", attempt, MAX_RETRIES, filename)
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    prompt,
                    types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                ],
                config=types.GenerateContentConfig(
                    temperature=0.4,
                    top_p=0.9,
                    response_modalities=["TEXT", "IMAGE"],
                    image_config=types.ImageConfig(aspect_ratio=aspect, image_size="1K"),
                    http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
                ),
            )
            # If we get here, the request succeeded — break out of retry loop
            break
        except Exception as e:
            last_error = e
            error_str = str(e)
            is_server_error = any(code in error_str for code in ("503", "500", "502", "504", "UNAVAILABLE", "RESOURCE_EXHAUSTED"))
            if is_server_error and attempt < MAX_RETRIES:
                logger.warning(
                    "AI server busy (attempt %d/%d) for %s: %s — retrying in %ds...",
                    attempt, MAX_RETRIES, filename, error_str, RETRY_DELAY_SECONDS,
                )
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            # Non-server error or final attempt — raise user-friendly message
            logger.error("Gemini API failed after %d attempts for %s: %s", attempt, filename, error_str)
            raise RuntimeError(
                "Processing failed due to high server demand. Please try again in a few minutes."
            ) from e

    # Extract image bytes from response (genai returns custom Image type, not PIL)
    result_bytes = None
    parts = getattr(response, "parts", None) or (
        response.candidates[0].content.parts if response.candidates else []
    )
    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            data = inline.data
            result_bytes = data if isinstance(data, bytes) else base64.b64decode(data)
            break
    if result_bytes is None:
        raise RuntimeError("Gemini did not return an image")
    result_pil = Image.open(io.BytesIO(result_bytes)).convert("RGB")

    # Convert to requested format if needed
    out_buf = io.BytesIO()
    if output_format.lower() == "png":
        result_pil.save(out_buf, format="PNG")
    elif output_format.lower() in ("jpg", "jpeg"):
        result_pil.save(out_buf, format="JPEG", quality=95)
    elif output_format.lower() == "webp":
        result_pil.save(out_buf, format="WEBP", quality=95)
    else:
        result_pil.save(out_buf, format="PNG")

    return out_buf.getvalue()
