"""Enhance car photos while preserving floor and walls.

- Removes sky/ceiling only (keeps floor & walls)
- Removes white light reflections from ALL car surfaces
  (windshield, hood, bumper, doors, roof, chrome)
- Enhances car (sharpening, contrast)
- Adjusts lighting
"""

import io
import logging
from typing import Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from app.services.image_utils import load_image

logger = logging.getLogger(__name__)

# ADE20K class IDs (SegFormer trained on this)
ADE_SKY = 2
ADE_CEILING = 6
ADE_FLOOR = 5
ADE_WALL = 0

# Classes to remove (replace with inpainting)
REMOVE_CLASSES = [ADE_SKY, ADE_CEILING]


class EnhancePreserveService:
    """Enhance car photos: remove sky/ceiling, remove reflections, keep floor/walls, enhance car."""

    _instance: Optional["EnhancePreserveService"] = None
    _seg_model = None
    _seg_processor = None

    def __new__(cls) -> "EnhancePreserveService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # ─────────────────────────────────────────────
    # Model loading
    # ─────────────────────────────────────────────

    def _get_segmentation(self):
        """Lazy-load SegFormer for semantic segmentation."""
        if self._seg_model is None:
            logger.info("Loading SegFormer (ADE20K) for scene segmentation...")
            from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
            import torch

            self._seg_processor = AutoImageProcessor.from_pretrained(
                "nvidia/segformer-b0-finetuned-ade-512-512"
            )
            self._seg_model = AutoModelForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b0-finetuned-ade-512-512"
            )
            self._seg_model.eval()
            self._device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
            self._seg_model.to(self._device)
            logger.info("SegFormer loaded successfully")
        return self._seg_processor, self._seg_model

    # ─────────────────────────────────────────────
    # Mask helpers
    # ─────────────────────────────────────────────

    def _get_remove_mask(self, image: Image.Image) -> np.ndarray:
        """Get binary mask of sky+ceiling to remove."""
        import torch

        processor, model = self._get_segmentation()
        orig_size = image.size  # (W, H)

        img_resized = image.resize((512, 512), Image.BILINEAR)
        inputs = processor(images=img_resized, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

        from skimage.transform import resize

        pred_full = resize(
            pred.astype(np.float32),
            (orig_size[1], orig_size[0]),
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(np.int64)

        mask = np.isin(pred_full, REMOVE_CLASSES)
        return mask.astype(np.uint8)

    def _get_car_mask(self, image_data: bytes, filename: str, target_size: tuple) -> np.ndarray:
        """Get boolean car mask from RMBG model (computed ONCE, reused everywhere)."""
        from app.services.background_removal import background_removal_service

        orig_img = load_image(image_data, filename)
        if orig_img.mode != "RGB":
            orig_img = orig_img.convert("RGB")

        result = background_removal_service._get_pipeline()(orig_img)
        if isinstance(result, Image.Image):
            no_bg = result
        else:
            no_bg = result[0] if isinstance(result, (list, tuple)) else result

        # Extract alpha channel — handle RGBA and plain mask (mode "L") both
        if hasattr(no_bg, "mode"):
            if no_bg.mode == "RGBA":
                alpha = np.array(no_bg.split()[3])
            elif no_bg.mode == "L":
                alpha = np.array(no_bg)
            else:
                logger.warning("RMBG returned mode=%s, using brightness fallback", no_bg.mode)
                alpha = None
        else:
            logger.warning("RMBG result has no .mode attribute (type=%s)", type(no_bg))
            alpha = None

        if alpha is None:
            # Fallback: use brightness to separate car from white background
            arr_f = np.array(orig_img).astype(np.float32) / 255.0
            luma = 0.299 * arr_f[:, :, 0] + 0.587 * arr_f[:, :, 1] + 0.114 * arr_f[:, :, 2]
            # Car is darker than the white background
            bg_thresh = np.percentile(luma, 85)
            alpha = ((luma < bg_thresh) * 255).astype(np.uint8)
            logger.info("Car mask via brightness fallback (bg_thresh=%.2f)", bg_thresh)

        if alpha.shape[:2] != (target_size[1], target_size[0]):
            alpha_pil = Image.fromarray(alpha).convert("L")
            alpha_pil = alpha_pil.resize(target_size, Image.BILINEAR)
            alpha = np.array(alpha_pil)

        mask = alpha > 10
        coverage = mask.mean()
        print(f"[CAR MASK] coverage={coverage:.3f} ({mask.sum()} px) target={target_size}")
        logger.info("Car mask coverage: %.1f%%", coverage * 100)

        # If mask is inverted (covers >80% of image) or empty, fix it
        if coverage > 0.80:
            logger.warning("Car mask >80%% — likely inverted, flipping")
            mask = ~mask
        elif coverage < 0.01:
            logger.warning("Car mask <1%% — using full image")
            mask = np.ones((target_size[1], target_size[0]), dtype=bool)

        return mask

    # ─────────────────────────────────────────────
    # Main pipeline
    # ─────────────────────────────────────────────

    def process(
        self,
        image_data: bytes,
        filename: str,
        output_format: str = "png",
        remove_sky_ceiling: bool = True,
        enhance_car: bool = True,
        lighting_boost: float = 1.0,
        car_sharpness: float = 1.08,
        car_contrast: float = 1.0,
    ) -> bytes:
        """
        Full pipeline:
          1. Remove sky/ceiling via inpainting
          2. Get car mask (ONCE)
          3. Remove white light reflections from all car surfaces
          4. Enhance car region (sharpen + contrast)
          5. Darken tires
          6. Global lighting adjustment
        """
        img = load_image(image_data, filename)
        if img.mode != "RGB":
            img = img.convert("RGB")

        print(f"[ENHANCE] START {filename} size={img.size} lighting={lighting_boost}")
        arr = np.array(img)

        # 1. Remove sky/ceiling  (cv2.inpaint — fast, no black-image artifacts)
        if remove_sky_ceiling:
            try:
                import cv2

                sky_mask = self._get_remove_mask(img)
                if sky_mask.any():
                    sky_u8 = sky_mask.astype(np.uint8) * 255
                    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                    arr_bgr = cv2.inpaint(arr_bgr, sky_u8, 25, cv2.INPAINT_TELEA)
                    arr = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB)
                    logger.info("Sky/ceiling removed (%d px, TELEA)", sky_mask.sum())
            except Exception as e:
                logger.warning("Sky/ceiling removal failed, skipping: %s", e)

        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

        # 2. Get car mask ONCE (expensive RMBG call - reused by steps 3, 4, 5)
        try:
            car_mask = self._get_car_mask(image_data, filename, img.size)
        except Exception as e:
            logger.warning("Car mask extraction failed: %s", e)
            car_mask = np.ones((img.height, img.width), dtype=bool)

        # 3. Remove white light reflections from entire car surface
        #    Tier 0: Replicate FLUX.1-Fill-dev AI inpainting (best quality)
        #    Tier 1/2: Local HSV + OpenCV fallback
        try:
            img = self._remove_reflections(img, car_mask, image_data=image_data, filename=filename)
        except Exception as e:
            logger.warning("Reflection removal failed, skipping: %s", e)

        # 4. Car-only enhancement
        if enhance_car:
            try:
                img = self._enhance_car_region(img, car_mask, car_sharpness, car_contrast)
            except Exception as e:
                logger.warning("Car enhancement failed, applying global: %s", e)
                enh = ImageEnhance.Sharpness(img)
                img = enh.enhance(car_sharpness)
                enh = ImageEnhance.Contrast(img)
                img = enh.enhance(car_contrast)

        # 5. Darken tires
        try:
            img = self._darken_tires(img, car_mask)
        except Exception as e:
            logger.warning("Tire darkening failed, skipping: %s", e)

        # 6. Global lighting
        if lighting_boost != 1.0:
            enh = ImageEnhance.Brightness(img)
            img = enh.enhance(lighting_boost)

        # Save
        print(f"[ENHANCE] DONE {filename}, saving as {output_format.upper()}")
        buf = io.BytesIO()
        fmt = output_format.upper() if output_format.lower() != "jpg" else "JPEG"
        if fmt in ("JPEG", "WEBP"):
            img.save(buf, format=fmt, quality=95)
        else:
            img.save(buf, format=fmt)  # PNG: no quality param
        buf.seek(0)
        return buf.read()

    # ─────────────────────────────────────────────
    # Step 3: Reflection removal (the big fix)
    # ─────────────────────────────────────────────

    def _remove_reflections(
        self, img: Image.Image, car_mask: np.ndarray,
        image_data: bytes = b"", filename: str = "",
    ) -> Image.Image:
        """
        Remove white light reflections with strict zone separation:
        - GLASS (windshield, windows): aggressive hotspot removal, never inpaint (avoids black)
        - BODY UPPER (hood, doors, roof): conservative HSV correction only
        - BODY LOWER (wheel wells, bumper, side skirts): EXCLUDED — prevents black fade
        """
        from scipy.ndimage import gaussian_filter, binary_closing
        from skimage.color import rgb2hsv, hsv2rgb

        arr = np.array(img).astype(np.float32) / 255.0
        h, w, _ = arr.shape
        yy = np.arange(h, dtype=np.float32)[:, np.newaxis]

        hsv = rgb2hsv(arr)
        h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        car_v = v_ch[car_mask]
        if car_v.size == 0:
            return img
        median_v = np.median(car_v)
        median_s = np.median(s_ch[car_mask])
        logger.info("Car baseline: median_v=%.3f, median_s=%.3f", median_v, median_s)

        # ── ZONE DEFINITIONS (root cause: lower body correction → black fade) ──
        # glass_zone: windshield, windows, headlights (y 2–72%)
        # lower_body: wheel wells, bumper, side skirts (y 58%+) — NEVER correct
        # body_upper: hood, roof, doors (y 0–58%) — conservative correction only
        y_gt, y_gb = int(h * 0.02), int(h * 0.72)
        glass_zone = (yy >= y_gt) & (yy < y_gb) & car_mask
        lower_body = (yy >= int(h * 0.58)) & car_mask
        body_upper = car_mask & (yy < int(h * 0.58))

        glass_v = v_ch[glass_zone] if glass_zone.any() else np.array([0.3])
        glass_s = s_ch[glass_zone] if glass_zone.any() else np.array([0.05])

        # ── GLASS REFLECTION DETECTION (aggressive — client: "no lights on windshield") ──
        glass_refl = np.zeros((h, w), dtype=bool)
        if glass_zone.any():
            gz_float = glass_zone.astype(np.float32)
            gz_w = np.maximum(gaussian_filter(gz_float, sigma=20), 1e-6)
            local_gv = gaussian_filter(v_ch * gz_float, sigma=20) / gz_w
            local_gs = gaussian_filter(s_ch * gz_float, sigma=20) / gz_w
            gv_excess = v_ch - local_gv
            v_glass_p85 = np.percentile(glass_v, 85)
            glass_refl |= glass_zone & (v_ch > max(v_glass_p85, 0.48)) & (s_ch < 0.35)
            glass_refl |= glass_zone & (gv_excess > 0.04) & (s_ch < 0.35)
            glass_refl |= glass_zone & (v_ch > 0.58) & (s_ch < 0.20)
            glass_refl = binary_closing(glass_refl, structure=np.ones((9, 9), dtype=bool))
            glass_refl &= glass_zone

        # ── BODY UPPER REFLECTION DETECTION (strict — avoid false positives → black/green) ──
        body_refl = np.zeros((h, w), dtype=bool)
        body_refl_zone = body_upper
        if body_refl_zone.any():
            bu_float = body_refl_zone.astype(np.float32)
            bu_w = np.maximum(gaussian_filter(bu_float, sigma=25), 1e-6)
            local_bv = gaussian_filter(v_ch * bu_float, sigma=25) / bu_w
            bv_excess = v_ch - local_bv
            body_refl |= body_refl_zone & (v_ch > 0.84) & (s_ch < 0.12)
            body_refl |= body_refl_zone & (bv_excess > 0.18) & (v_ch > 0.72) & (s_ch < 0.18)
            body_refl = binary_closing(body_refl, structure=np.ones((5, 5), dtype=bool))
            body_refl &= body_refl_zone
            body_refl &= (v_ch > 0.60)
            body_refl &= ~glass_refl

        refl_mask = glass_refl | body_refl
        if not refl_mask.any():
            logger.info("No reflections detected")
            return img

        coverage = refl_mask.sum() / max(car_mask.sum(), 1)
        print(f"[REFL] glass={glass_refl.sum()} body_upper={body_refl.sum()} (lower_body EXCLUDED)")
        logger.info("Reflection: %.1f%% of car (%d px)", coverage * 100, refl_mask.sum())

        # ── GLASS: STRONG HOTSPOT REMOVAL (no inpaint, never create black) ──
        non_refl_glass = glass_zone & ~glass_refl
        if glass_refl.any() and non_refl_glass.any():
            nrg = non_refl_glass.astype(np.float32)
            nrg_w = np.maximum(gaussian_filter(nrg, sigma=18), 1e-6)
            target_gv = gaussian_filter(v_ch * nrg, sigma=18) / nrg_w
            target_gs = gaussian_filter(s_ch * nrg, sigma=18) / nrg_w
            target_gv = np.clip(target_gv, 0.22, 0.92)
            target_gs = np.clip(target_gs, 0, 0.5)
            v_excess_g = np.clip(v_ch - target_gv, 0, None)
            s_deficit_g = np.clip(target_gs - s_ch, 0, None)
            gain_g = np.clip(np.maximum(v_excess_g * 3.2, s_deficit_g * 1.5), 0, 1.0)
            gain_g[~glass_refl] = 0
            gain_g = gaussian_filter(gain_g, sigma=1.2)
            gain_g[~glass_refl] = 0
            gain_g = np.minimum(gain_g, 0.95)
            new_v = np.clip(v_ch - v_excess_g * gain_g, 0.20, 1.0)
            new_s = np.clip(s_ch + s_deficit_g * gain_g * 0.5, 0, 0.4)
            glass_apply = gaussian_filter(glass_refl.astype(np.float32), sigma=2)
            glass_apply = np.clip(glass_apply, 0, 1)[:, :, np.newaxis]
            corrected_glass_rgb = hsv2rgb(np.stack([h_ch, new_s, new_v], axis=-1))
            corrected_glass_rgb = np.clip(corrected_glass_rgb, 0, 1)
            arr = arr * (1 - glass_apply) + corrected_glass_rgb * glass_apply
            arr = np.clip(arr, 0, 1)
            hsv = rgb2hsv(arr)
            h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
            logger.info("Glass reflection removal applied (%d px)", glass_refl.sum())

        # ── BODY UPPER: BRIGHTNESS REDUCTION ONLY (no S boost — avoids black/green artifacts) ──
        if body_refl.any():
            non_refl_body = body_refl_zone & ~body_refl
            if non_refl_body.any():
                nrb = non_refl_body.astype(np.float32)
                nrb_w = np.maximum(gaussian_filter(nrb, sigma=18), 1e-6)
                target_bv = gaussian_filter(v_ch * nrb, sigma=18) / nrb_w
                target_bv = np.clip(target_bv, 0.38, 0.90)
                v_excess_b = np.clip(v_ch - target_bv, 0, None)
                gain_b = np.clip(v_excess_b * 1.5, 0, 0.38)
                gain_b[~body_refl] = 0
                gain_b = gaussian_filter(gain_b, sigma=1.2)
                gain_b[~body_refl] = 0
                new_v = np.clip(v_ch - v_excess_b * gain_b, 0.42, 1.0)
                new_s = s_ch
                corrected_body = hsv2rgb(np.stack([h_ch, new_s, new_v], axis=-1))
                corrected_body = np.clip(corrected_body, 0, 1)
                body_apply = gaussian_filter(body_refl.astype(np.float32), sigma=2)
                body_apply = np.clip(body_apply, 0, 1)[:, :, np.newaxis]
                arr = arr * (1 - body_apply) + corrected_body * body_apply
                arr = np.clip(arr, 0, 1)
                logger.info("Body upper reflection removal applied (%d px)", body_refl.sum())

        logger.info("Reflection removal complete")
        return Image.fromarray((arr * 255).astype(np.uint8))

    # ─────────────────────────────────────────────
    # Step 4: Car enhancement
    # ─────────────────────────────────────────────

    def _enhance_car_region(
        self,
        img: Image.Image,
        car_mask: np.ndarray,
        sharpness: float,
        contrast: float,
    ) -> Image.Image:
        """Enhance only the car region using pre-computed car mask."""
        from scipy.ndimage import gaussian_filter

        # Smooth mask edges for natural blending (avoid hard car boundary)
        car_mask_soft = gaussian_filter(car_mask.astype(np.float32), sigma=2)
        car_mask_3ch = np.stack([car_mask_soft] * 3, axis=-1)

        # Subtle enhancement only — strong values cause high contrast and color shift
        img_enhanced = img.filter(ImageFilter.UnsharpMask(radius=1, percent=80, threshold=3))
        enh = ImageEnhance.Contrast(img_enhanced)
        img_enhanced = enh.enhance(contrast)
        enh2 = ImageEnhance.Sharpness(img_enhanced)
        img_enhanced = enh2.enhance(sharpness)

        # Blend: car region gets enhanced, rest stays original
        arr_orig = np.array(img).astype(np.float32)
        arr_enh = np.array(img_enhanced).astype(np.float32)
        blended = arr_orig * (1 - car_mask_3ch) + arr_enh * car_mask_3ch
        return Image.fromarray(blended.astype(np.uint8))

    # ─────────────────────────────────────────────
    # Step 5: Tire cleaning (dust removal + gentle darkening)
    # ─────────────────────────────────────────────

    def _darken_tires(self, img: Image.Image, car_mask: np.ndarray) -> Image.Image:
        """
        Clean tire surfaces:
          Step A — Detect dust (bright spots inside tire area)
          Step B — Remove dust via cv2.inpaint (reconstructs tire texture from surroundings)
          Step C — Gentle darkening to deepen tire blacks (0.90 multiplier — secondary, not primary)
          Step D — Feathered blend so edges look natural
        """
        import cv2
        from scipy.ndimage import gaussian_filter, binary_opening, label
        from skimage.color import rgb2hsv

        arr = np.array(img).astype(np.float32) / 255.0
        h, w, _ = arr.shape

        hsv = rgb2hsv(arr)
        v_ch = hsv[:, :, 2]
        s_ch = hsv[:, :, 1]
        luma = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

        # ── Tire region detection ──────────────────────────────────────────────
        y_tire_start = int(h * 0.55)
        lower_zone = np.zeros((h, w), dtype=bool)
        lower_zone[y_tire_start:, :] = True

        side_band = int(w * 0.30)
        tire_zone = np.zeros((h, w), dtype=bool)
        tire_zone[y_tire_start:, :side_band] = True
        tire_zone[y_tire_start:, w - side_band:] = True

        tire_candidate = car_mask & tire_zone & (v_ch < 0.35) & (s_ch < 0.48)
        tire_candidate = binary_opening(tire_candidate, structure=np.ones((5, 5), dtype=bool))

        tire_mask = np.zeros((h, w), dtype=bool)
        for x_start, x_end in [(0, w // 2), (w // 2, w)]:
            region = np.zeros((h, w), dtype=bool)
            region[:, x_start:x_end] = True
            reg = tire_candidate & region
            if reg.sum() > 200:
                tire_mask |= reg

        if not tire_mask.any():
            # Fallback: darkest 20% inside side bands only (never center bottom).
            lower_car = car_mask & tire_zone
            if lower_car.any():
                thresh = min(np.percentile(luma[lower_car], 20), 0.28)
                tire_mask = lower_car & (luma < thresh)

        if not tire_mask.any():
            return img

        # ── Step A: Detect dust pixels (bright local hot-spots on tire) ──────
        tire_float = tire_mask.astype(np.float32)
        tire_smooth = np.maximum(gaussian_filter(tire_float, sigma=15), 1e-6)
        local_luma = gaussian_filter(luma * tire_float, sigma=15) / tire_smooth

        dust_excess = luma - local_luma
        dust_mask = tire_mask & (dust_excess > 0.04) & (luma > 0.12)
        dust_mask = binary_opening(dust_mask, structure=np.ones((3, 3), dtype=bool))

        lbl, num = label(dust_mask)
        if num > 0:
            filtered = np.zeros_like(dust_mask)
            for idx in range(1, num + 1):
                comp = lbl == idx
                area = int(comp.sum())
                if 10 <= area <= 8000:
                    filtered |= comp
            dust_mask = filtered

        # ── Step B: Remove dust via cv2.inpaint ───────────────────────────────
        # cv2.inpaint reconstructs each dust pixel from the surrounding clean
        # tire texture (dark rubber), which is far more accurate than averaging.
        if dust_mask.any():
            arr_u8 = np.clip(arr * 255, 0, 255).astype(np.uint8)
            dust_u8 = dust_mask.astype(np.uint8) * 255
            arr_bgr = cv2.cvtColor(arr_u8, cv2.COLOR_RGB2BGR)
            inpainted_bgr = cv2.inpaint(arr_bgr, dust_u8, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            arr_inpainted = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            # Apply inpainting ONLY to tire region (don't touch anything outside)
            dust_3ch = np.stack([dust_mask] * 3, axis=-1)
            arr = np.where(dust_3ch, arr_inpainted, arr)
            logger.info("Tire dust removed via cv2.inpaint: %d px", dust_mask.sum())
            print(f"[TIRE] Dust inpainted: {dust_mask.sum()} px")

        # ── Step C: Darken tires to clean black (client: "tires should look black")
        arr[tire_mask] = np.clip(arr[tire_mask] * 0.82, 0.0, 1.0)

        # ── Step D: Feathered blend (small sigma to avoid bottom blur) ─────────
        tire_soft = gaussian_filter(tire_mask.astype(np.float32), sigma=2)
        tire_soft = np.clip(tire_soft, 0, 1)[:, :, np.newaxis]
        orig = np.array(img).astype(np.float32) / 255.0
        arr = orig * (1 - tire_soft) + arr * tire_soft

        return Image.fromarray((arr * 255.0).astype(np.uint8))


enhance_preserve_service = EnhancePreserveService()
