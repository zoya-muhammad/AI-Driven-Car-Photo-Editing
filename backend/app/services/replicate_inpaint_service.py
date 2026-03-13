"""Replicate API — FLUX.1-Fill-dev inpainting for car reflection removal.

Sends the image + reflection mask to Replicate's FLUX.1-Fill-dev model,
which generates a clean car surface in place of the reflections.

Requirements:
  - REPLICATE_API_TOKEN set in backend/.env
  - Pay-per-use (~$0.03/image)
  - Falls back to local Tier 1/2 if token missing or API fails.
"""

import base64
import io
import logging
import time
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

FLUX_FILL_MODEL = "black-forest-labs/flux-fill-dev"

# Body paint prompt — avoid specific colors; let FLUX infer from context
_BODY_PROMPT = (
    "smooth automotive paint surface, professional studio lighting, "
    "clean metallic finish, no bright spots, high quality car photograph"
)
# Glass/window prompt — specific description for FLUX inpainting
_GLASS_PROMPT = (
    "Clean dark tinted car window glass, no reflections, smooth dark surface, "
    "natural looking"
)
# Legacy (deprecated — was causing "dark gray" flat patches)
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


def inpaint_reflections_replicate(
    image: Image.Image,
    refl_mask: np.ndarray,
    replicate_token: str,
    car_color: str = "gray",
    prompt: str | None = None,
) -> Optional[Image.Image]:
    """
    Remove reflections via Replicate FLUX.1-Fill-dev inpainting.

    Args:
        image:           RGB PIL image.
        refl_mask:       Boolean or uint8 array — True/255 = area to fill.
        replicate_token: Replicate API token.
        car_color:       Descriptive colour (only used if prompt is None).
        prompt:          Override prompt. Use _GLASS_PROMPT for glass, _BODY_PROMPT for body.
                        If None, uses legacy color-based prompt (avoid for body).

    Returns:
        Inpainted PIL image (same size as input), or None on failure.
    """
    if not replicate_token:
        return None
    if not np.any(refl_mask):
        return image
    if image.mode != "RGB":
        image = image.convert("RGB")

    import requests

    prompt = prompt or _POS_PROMPT.format(color=car_color)
    orig_size = image.size

    # Resize to max 1024px for API (faster + cheaper)
    api_size = 1024
    img_sq, mask_sq, pad = _resize_with_pad(image, refl_mask, api_size)

    img_b64 = _pil_to_data_uri(img_sq)
    mask_b64 = _pil_to_data_uri(mask_sq)

    mask_px = int(np.sum(refl_mask > 0))
    print(f"[REPLICATE] FLUX.1-Fill-dev  size={api_size}  mask_px={mask_px}  color={car_color}")
    logger.info("[REPLICATE] FLUX.1-Fill-dev  size=%d  mask=%d px", api_size, mask_px)

    # Create prediction
    headers = {
        "Authorization": f"Bearer {replicate_token}",
        "Content-Type": "application/json",
        "Prefer": "wait",
    }
    payload = {
        "input": {
            "image": img_b64,
            "mask": mask_b64,
            "prompt": prompt,
            "num_inference_steps": 28,
            "guidance_scale": 30,
            "strength": 0.85,
        },
    }

    create_url = f"https://api.replicate.com/v1/models/{FLUX_FILL_MODEL}/predictions"

    try:
        resp = requests.post(create_url, headers=headers, json=payload, timeout=180)
    except requests.exceptions.RequestException as exc:
        logger.warning("[REPLICATE] Request failed: %s", exc)
        return None

    if resp.status_code not in (200, 201):
        try:
            err = resp.json()
        except Exception:
            err = resp.text[:300]
        logger.warning("[REPLICATE] Create error %d: %s", resp.status_code, err)
        return None

    prediction = resp.json()

    # Poll for completion if not using Prefer: wait or if still processing
    poll_url = prediction.get("urls", {}).get("get") or f"https://api.replicate.com/v1/predictions/{prediction['id']}"
    status = prediction.get("status", "starting")
    output = prediction.get("output")

    poll_headers = {"Authorization": f"Bearer {replicate_token}"}
    max_wait = 180
    waited = 0

    while status not in ("succeeded", "failed", "canceled") and waited < max_wait:
        time.sleep(3)
        waited += 3
        try:
            poll_resp = requests.get(poll_url, headers=poll_headers, timeout=30)
            if poll_resp.status_code == 200:
                prediction = poll_resp.json()
                status = prediction.get("status", "starting")
                output = prediction.get("output")
        except requests.exceptions.RequestException:
            continue

    if status != "succeeded" or not output:
        logger.warning("[REPLICATE] Prediction %s: status=%s", prediction.get("id", "?"), status)
        return None

    # Download result image
    result_url = output if isinstance(output, str) else output[0] if isinstance(output, list) else None
    if not result_url:
        logger.warning("[REPLICATE] No output URL in response")
        return None

    try:
        img_resp = requests.get(result_url, timeout=60)
        if img_resp.status_code != 200:
            logger.warning("[REPLICATE] Failed to download result: %d", img_resp.status_code)
            return None
        result_sq = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
    except Exception as exc:
        logger.warning("[REPLICATE] Failed to decode result: %s", exc)
        return None

    result = _unpad_and_resize(result_sq, orig_size, pad)
    print(f"[REPLICATE] FLUX.1-Fill-dev inpainting SUCCESS  color={car_color}")
    return result


# ── Helpers ───────────────────────────────────────────────────────────────────


def _pil_to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


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
    """Detect dominant car BODY color — exclude windows, tires, shadows."""
    import cv2

    arr_f = arr.astype(np.float32) / 255.0 if arr.max() > 1.0 else arr.astype(np.float32)
    arr_u8 = (np.clip(arr_f, 0, 1) * 255).astype(np.uint8)
    arr_bgr = cv2.cvtColor(arr_u8, cv2.COLOR_RGB2BGR)
    hsv_full = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv_full[:, :, 0].astype(np.float32), hsv_full[:, :, 1].astype(np.float32), hsv_full[:, :, 2].astype(np.float32)

    car_bool = (car_mask > 0) if np.issubdtype(car_mask.dtype, np.integer) else car_mask
    refl_bool = (refl_mask > 0) if np.issubdtype(refl_mask.dtype, np.integer) else refl_mask

    # Body panels only: exclude dark (windows, tires), bright (reflections)
    body_mask = (
        car_bool
        & ~refl_bool
        & (v > 80)
        & (v < 220)
        & (s < 150)
    )
    if np.sum(body_mask) < 1000:
        return "metallic silver"

    s_mean = float(np.mean(s[body_mask]))
    v_mean = float(np.mean(v[body_mask]))
    h_mean = float(np.mean(h[body_mask]))

    if s_mean < 20:
        if v_mean > 180:
            return "bright silver metallic"
        elif v_mean > 120:
            return "medium silver metallic"
        else:
            return "charcoal metallic"
    elif s_mean < 50:
        if v_mean > 150:
            return "light metallic silver"
        else:
            return "metallic gray"
    else:
        if h_mean < 15 or h_mean > 165:
            return "metallic red"
        elif h_mean < 35:
            return "metallic orange"
        elif h_mean < 75:
            return "metallic green"
        elif h_mean < 130:
            return "metallic blue"
        else:
            return "metallic purple"
