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
from PIL import Image

from app.config import GEMINI_API_KEY

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-3.1-flash-image-preview"
# Image generation can take 2-5 min; 3 min caused ReadTimeout on slow responses
REQUEST_TIMEOUT_MS = 360_000  # 6 minutes
MAX_INPUT_SIZE = 1024  # Resize large images to reduce payload and processing time

ENHANCE_PROMPT = """Edit this car photo for a professional automotive listing. Apply these changes precisely:

CRITICAL — CAR COLOR (do not change):
- Preserve the original car paint color exactly. Do not darken, lighten, or alter the hue.
- The car must look the same color as the input — neither too dark nor too light.
- Do not change original color. This is mandatory.

REFLECTIONS:
- Remove all light reflections from the upper body (hood, roof, fenders) where studio lights appear.
- Remove all reflections and lights from the windshield and side windows. Windows should look clear/tinted with no glare.

WALLS:
- Walls must be clean and uniform. Remove any visible lights, fixtures, doors, or door outlines from the walls.
- Use the same clean wall appearance throughout.

FLOOR:
- Clean the floor of all dirt, dust, marks, and uneven patches.
- Maintain a consistent dark tiled texture. Floor should look professionally cleaned.

TIRES:
- Tires must look deep black. Remove any dust, grime, or discoloration from the rubber.

Output only the edited image. Do not add text."""

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

    # Resize large images (e.g. NEF 6016x4016) to reduce payload and avoid 503 timeout
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

    # Convert PIL to bytes for API (PNG for quality)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    # Use types for config
    from google.genai import types

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            prompt,
            types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
        ],
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            image_config=types.ImageConfig(aspect_ratio=aspect, image_size="1K"),
            http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
        ),
    )

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
