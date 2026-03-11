"""Hugging Face Inference API — SDXL/SD2 inpainting for car reflection removal.

Primary:  diffusers/stable-diffusion-xl-1.0-inpainting-0.1  (768×768, higher quality)
Fallback: stabilityai/stable-diffusion-2-inpainting          (512×512, more reliable free-tier)

Both run via HF serverless inference — no local model download.

Requirements:
  - HUGGINGFACE_HUB_TOKEN set in backend/.env
  - Free tier: ~300 calls/day; Pro tier: higher limits
  - Gracefully falls back to local Tier 1/2 if token missing or both APIs fail.
"""

import base64
import io
import logging
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── Models (tried in order) ───────────────────────────────────────────────────

# FLUX.1-Fill: Black Forest Labs inpainting — on HF's new router
FLUX_MODEL = "black-forest-labs/FLUX.1-Fill-dev"
FLUX_SIZE = 512

# SD2: fallback
SD2_MODEL = "stabilityai/stable-diffusion-2-inpainting"
SD2_SIZE = 512

HF_API_BASE = "https://router.huggingface.co/hf-inference/models"

# ── Prompts ───────────────────────────────────────────────────────────────────

_POS_PROMPT = (
    "{color} car paint surface, automotive studio photography, "
    "clean smooth paint, no reflections, no glare, no bright spots, "
    "realistic car body texture, professional product photograph, showroom quality"
)
_NEG_PROMPT = (
    "reflection, glare, bright spot, overexposed, white spot, light streak, "
    "blurry, artifacts, distortion, noise, watermark, text, logo, wrong color, "
    "different shade, color mismatch"
)


# ── Public API ────────────────────────────────────────────────────────────────


def inpaint_reflections_hf(
    image: Image.Image,
    refl_mask: np.ndarray,
    hf_token: str,
    car_color: str = "gray",
) -> Optional[Image.Image]:
    """
    Remove reflections from masked areas using HF inpainting API.

    Tries SDXL inpainting first (higher quality at 768px).
    Falls back to SD2 (more reliable on free tier at 512px) if SDXL fails.

    Args:
        image:      RGB PIL image (after sky removal, before car enhancement).
        refl_mask:  Boolean or uint8 array — True/255 = area to fill.
        hf_token:   HuggingFace Hub API token.
        car_color:  Human-readable colour description used in the prompt.

    Returns:
        Inpainted PIL image (same size as input), or None on failure.
    """
    if not hf_token:
        return None
    if not np.any(refl_mask):
        return image
    if image.mode != "RGB":
        image = image.convert("RGB")

    prompt = _POS_PROMPT.format(color=car_color)
    orig_size = image.size

    # ── Try FLUX.1-Fill first ─────────────────────────────────────────────────
    result = _call_inpaint_api(
        None, image, refl_mask, orig_size,
        model=FLUX_MODEL, api_size=FLUX_SIZE,
        prompt=prompt, hf_token=hf_token,
        steps=30, guidance=30.0, strength=0.85,
    )
    if result is not None:
        print(f"[HF] FLUX.1-Fill inpainting success  color={car_color}")
        return result

    # ── Fallback to SD2 ───────────────────────────────────────────────────────
    logger.info("[HF] FLUX unavailable — falling back to SD2")
    result = _call_inpaint_api(
        None, image, refl_mask, orig_size,
        model=SD2_MODEL, api_size=SD2_SIZE,
        prompt=prompt, hf_token=hf_token,
        steps=25, guidance=9.0, strength=0.75,
    )
    if result is not None:
        print(f"[HF] SD2 inpainting success (fallback)  color={car_color}")
        return result

    logger.warning("[HF] Both FLUX and SD2 inpainting failed — using local Tier 1/2")
    return None


# ── Core API caller ───────────────────────────────────────────────────────────


def _call_inpaint_api(
    requests_mod,  # kept for API compat, unused
    image: Image.Image,
    refl_mask: np.ndarray,
    orig_size: tuple,
    model: str,
    api_size: int,
    prompt: str,
    hf_token: str,
    steps: int,
    guidance: float,
    strength: float,
) -> Optional[Image.Image]:
    """
    Call a single HF inpainting model endpoint via raw requests.

    SD inpainting models (SDXL, SD2) have pipeline_tag="text-to-image" on HF Hub.
    Correct format: prompt as `inputs` (string), image+mask inside `parameters`.
    Using raw requests bypasses InferenceClient's model_info() validation which
    requires Hub read permission that fine-grained Inference Provider tokens lack.
    """
    import requests

    img_sq, mask_sq, pad = _resize_with_pad(image, refl_mask, api_size)

    # Text-to-image inpainting format: prompt as inputs, image/mask in parameters
    payload = {
        "inputs": prompt,
        "parameters": {
            "image": _pil_to_b64(img_sq),
            "mask_image": _pil_to_b64(mask_sq),
            "negative_prompt": _NEG_PROMPT,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "strength": strength,
        },
    }
    url = f"{HF_API_BASE}/{model}"
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }

    mask_px = int(np.sum(refl_mask > 0))
    logger.info("[HF] %s  size=%d  mask=%d px", model.split("/")[-1], api_size, mask_px)
    print(f"[HF] → {model}  size={api_size}  mask_px={mask_px}")

    timeout = 120 if api_size >= 768 else 90
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    except requests.exceptions.Timeout:
        logger.warning("[HF] %s timed out (%ds)", model.split("/")[-1], timeout)
        return None
    except requests.exceptions.RequestException as exc:
        logger.warning("[HF] %s request error: %s", model.split("/")[-1], exc)
        return None

    if resp.status_code != 200:
        try:
            err = resp.json()
        except Exception:
            err = resp.text[:300]
        logger.warning("[HF] %s error %d: %s", model.split("/")[-1], resp.status_code, err)
        return None

    try:
        result_sq = Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as exc:
        logger.warning("[HF] Could not decode response: %s", exc)
        return None

    return _unpad_and_resize(result_sq, orig_size, pad)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _resize_with_pad(image: Image.Image, mask_np: np.ndarray, size: int):
    """Letterbox image + mask to a square canvas of `size` pixels."""
    w, h = image.size
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)

    img_r = image.resize((new_w, new_h), Image.LANCZOS)
    mask_r = Image.fromarray(
        (mask_np > 0).astype(np.uint8) * 255, mode="L"
    ).resize((new_w, new_h), Image.NEAREST)

    img_sq = Image.new("RGB", (size, size), (0, 0, 0))
    mask_sq = Image.new("L", (size, size), 0)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    img_sq.paste(img_r, (pad_x, pad_y))
    mask_sq.paste(mask_r, (pad_x, pad_y))

    return img_sq, mask_sq, (pad_x, pad_y, new_w, new_h)


def _unpad_and_resize(result_sq: Image.Image, orig_size: tuple, pad_info: tuple) -> Image.Image:
    """Remove letterbox padding and resize to original dimensions."""
    pad_x, pad_y, new_w, new_h = pad_info
    cropped = result_sq.crop((pad_x, pad_y, pad_x + new_w, pad_y + new_h))
    return cropped.resize(orig_size, Image.LANCZOS)


def detect_car_color(arr: np.ndarray, car_mask: np.ndarray, refl_mask: np.ndarray) -> str:
    """
    Detect dominant car paint colour from non-reflection body pixels.
    Returns a descriptive string used in the inpainting prompt.
    """
    body = arr[car_mask & ~refl_mask]
    if body.size == 0:
        return "gray"

    dom = np.median(body, axis=0)  # [R, G, B] in 0–1
    r, g, b = float(dom[0]), float(dom[1]), float(dom[2])
    brightness = max(r, g, b)

    if brightness < 0.20:
        return "black"
    if min(r, g, b) > 0.72:
        return "white"
    if r > g + 0.12 and r > b + 0.12:
        return "red"
    if b > r + 0.10 and b > g + 0.05:
        return "blue"
    if g > r + 0.08 and g > b + 0.05:
        return "green"
    if r > g + 0.05 and g > b + 0.03:
        return "champagne gold"
    if brightness < 0.35:
        return "dark gray"
    if brightness < 0.55:
        return "gray"
    return "silver gray"
