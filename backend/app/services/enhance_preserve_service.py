"""Enhance car photos — protection-first architecture.

Walls and floor are NEVER modified. Processing applies to car region only.
Final step composites untouched floor/wall from original.
"""

import io
import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from app.config import REPLICATE_API_TOKEN
from app.services.image_utils import load_image
from app.services.replicate_inpaint_service import (
    _GLASS_PROMPT,
    inpaint_reflections_replicate,
)

logger = logging.getLogger(__name__)

# Safety caps — abort step if mask exceeds these
TIRE_MAX_PX = 500_000
GLASS_REFL_MAX_PX = 100_000
BODY_REFL_MAX_PX = 1_000_000
DUST_MAX_PX = 50_000
CEILING_MAX_PX = 500_000


class EnhancePreserveService:
    """Enhance car photos: ceiling, reflections, tires. Floor and walls protected."""

    _instance: Optional["EnhancePreserveService"] = None

    def __new__(cls) -> "EnhancePreserveService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # ─────────────────────────────────────────────
    # Step 1: Car mask
    # ─────────────────────────────────────────────

    def _get_car_mask(self, image: Image.Image) -> np.ndarray:
        """Get binary car mask from RMBG-1.4. 255=car, 0=not car."""
        from app.services.background_removal import background_removal_service

        if image.mode != "RGB":
            image = image.convert("RGB")

        result = background_removal_service._get_pipeline()(image)
        no_bg = result if isinstance(result, Image.Image) else result[0]

        if hasattr(no_bg, "mode"):
            if no_bg.mode == "RGBA":
                alpha = np.array(no_bg.split()[3])
            elif no_bg.mode == "L":
                alpha = np.array(no_bg)
            else:
                alpha = None
        else:
            alpha = None

        if alpha is None:
            arr_f = np.array(image).astype(np.float32) / 255.0
            luma = 0.299 * arr_f[:, :, 0] + 0.587 * arr_f[:, :, 1] + 0.114 * arr_f[:, :, 2]
            bg_thresh = np.percentile(luma, 85)
            alpha = ((luma < bg_thresh) * 255).astype(np.uint8)

        mask = (alpha > 128).astype(np.uint8) * 255
        if alpha.shape[:2] != (image.height, image.width):
            alpha_pil = Image.fromarray(alpha).convert("L").resize(
                (image.width, image.height), Image.BILINEAR
            )
            mask = (np.array(alpha_pil) > 128).astype(np.uint8) * 255

        kernel = np.ones((5, 5), np.uint8)
        mask = np.where(
            cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) > 0, 255, 0
        ).astype(np.uint8)

        coverage = np.mean(mask > 0)
        if coverage > 0.80:
            mask = (255 - mask)
        elif coverage < 0.01:
            mask = np.ones((image.height, image.width), dtype=np.uint8) * 255

        print(f"[CAR MASK] coverage={coverage:.3f}")
        return mask

    # ─────────────────────────────────────────────
    # Step 2: Protection masks
    # ─────────────────────────────────────────────

    def _get_protection_masks(
        self, arr: np.ndarray, car_mask: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Create immutable floor, wall, above_car masks.
        Floor: dark tiles in bottom 35% + below car. Wall: beside car. Ceiling NOT protected."""
        h, w = arr.shape[:2]
        car_bool = car_mask > 0 if car_mask.dtype == bool else car_mask > 127

        car_rows = np.where(np.any(car_bool, axis=1))[0]
        car_cols = np.where(np.any(car_bool, axis=0))[0]
        if len(car_rows) == 0 or len(car_cols) == 0:
            return {
                "floor_mask": np.zeros((h, w), dtype=np.uint8),
                "wall_mask": np.zeros((h, w), dtype=np.uint8),
                "above_car_mask": np.zeros((h, w), dtype=np.uint8),
                "combined_protection": np.zeros((h, w), dtype=np.uint8),
            }

        car_top = int(car_rows[0])
        car_bottom = int(car_rows[-1])

        result_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2GRAY)

        floor_region_top = int(h * 0.65)
        floor_mask = np.zeros((h, w), dtype=np.uint8)
        floor_condition = (
            (gray < 130)
            & (~car_bool)
            & (np.arange(h)[:, np.newaxis] >= floor_region_top)
        )
        floor_mask[floor_condition] = 255

        below_car = np.zeros((h, w), dtype=np.uint8)
        below_car[car_bottom:, :] = 255
        below_car[car_bool] = 0
        floor_mask = cv2.bitwise_or(floor_mask, below_car)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_CLOSE, kernel)

        above_car_mask = np.zeros((h, w), dtype=np.uint8)
        above_car_mask[:car_top, :] = 255

        wall_mask = np.zeros((h, w), dtype=np.uint8)
        wall_mask[:, :] = 255
        wall_mask[car_bool] = 0
        wall_mask[floor_mask > 0] = 0
        wall_mask[above_car_mask > 0] = 0

        combined = cv2.bitwise_or(floor_mask, wall_mask)
        print(f"[MASKS] floor={np.sum(floor_mask>0)} px wall={np.sum(wall_mask>0)} px above_car={np.sum(above_car_mask>0)} px")
        return {
            "floor_mask": floor_mask,
            "wall_mask": wall_mask,
            "above_car_mask": above_car_mask,
            "combined_protection": combined,
        }

    # ─────────────────────────────────────────────
    # Step 3: Ceiling lights (above car only, discrete objects)
    # ─────────────────────────────────────────────

    def _remove_ceiling_lights(
        self,
        arr: np.ndarray,
        above_car_mask: np.ndarray,
        car_mask: np.ndarray,
    ) -> np.ndarray:
        """Remove ceiling lights, pipes, fixtures — above car only. Fill with white."""
        result = arr.copy()
        if result.dtype != np.uint8:
            result = np.clip(result, 0, 255).astype(np.uint8)
        h, w = result.shape[:2]
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2GRAY)

        try:
            if np.count_nonzero(above_car_mask) == 0:
                print("[CEILING] No above-car region, skipping")
                return result

            ceiling_objects = np.zeros((h, w), dtype=np.uint8)
            dark_on_ceiling = (
                (gray < 180)
                & (above_car_mask > 0)
                & (car_mask == 0)
            )
            ceiling_objects[dark_on_ceiling] = 255

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            ceiling_objects = cv2.morphologyEx(ceiling_objects, cv2.MORPH_CLOSE, kernel)
            ceiling_objects = cv2.morphologyEx(ceiling_objects, cv2.MORPH_OPEN, kernel)
            ceiling_objects = cv2.dilate(ceiling_objects, kernel, iterations=2)

            removed_px = int(np.count_nonzero(ceiling_objects))
            print(f"[CEILING] Objects detected: {removed_px} px")

            if removed_px < 100:
                return result

            if removed_px > CEILING_MAX_PX:
                print(f"[CEILING] WARNING: mask too large ({removed_px} px), SKIPPING")
                return result

            result_bgr[ceiling_objects > 0] = [255, 255, 255]

            edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            dilated = cv2.dilate(ceiling_objects, edge_kernel, iterations=1)
            edge_mask = cv2.subtract(dilated, ceiling_objects)
            blurred = cv2.GaussianBlur(result_bgr, (5, 5), 2)
            edge_float = edge_mask.astype(np.float32) / 255.0
            for c in range(3):
                result_bgr[:, :, c] = (
                    blurred[:, :, c] * edge_float
                    + result_bgr[:, :, c] * (1 - edge_float)
                ).astype(np.uint8)

            print(f"[CEILING] Removed {removed_px} px (above car only)")
            return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"[CEILING] ERROR: {e}")
            import traceback
            traceback.print_exc()
            return result

    # ─────────────────────────────────────────────
    # Step 4: Body reflections (specular detection, stronger)
    # ─────────────────────────────────────────────

    def _remove_body_reflections(
        self, arr: np.ndarray, car_mask: np.ndarray, strength: float = 0.6
    ) -> np.ndarray:
        """Subtly reduce harsh specular highlights. diff>50, 35% correction, heavy smoothing."""
        result = arr.copy()
        if result.dtype != np.uint8:
            result = np.clip(result, 0, 255).astype(np.uint8)
        try:
            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            lab = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2LAB)
            l_ch = lab[:, :, 0].astype(np.float32)

            l_blur = cv2.GaussianBlur(l_ch, (0, 0), sigmaX=60)
            diff = l_ch - l_blur
            car_px = car_mask > 0 if car_mask.dtype == bool else car_mask > 127

            highlight = (diff > 50) & car_px
            affected_px = int(np.sum(highlight))
            if affected_px > BODY_REFL_MAX_PX:
                print(f"[BODY REFL] WARNING: mask too large ({affected_px} px), SKIPPING")
                return result

            strength_map = np.zeros_like(l_ch)
            if np.any(highlight):
                d_vals = diff[highlight]
                strength_map[highlight] = (d_vals - 50) / (float(np.max(d_vals)) + 1e-6)
            strength_map = cv2.GaussianBlur(strength_map, (31, 31), 10)

            correction = diff * strength_map * 0.35
            l_corrected = l_ch - correction
            l_corrected = np.clip(l_corrected, 0, 255).astype(np.uint8)

            lab_out = lab.copy()
            lab_out[:, :, 0][car_px] = l_corrected[car_px]

            result_bgr = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
            result = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
            print(f"[BODY REFL] Reduced {affected_px} px of specular highlights")
            return result
        except Exception as e:
            print(f"[BODY REFL] ERROR: {e}")
            return result

    # ─────────────────────────────────────────────
    # Step 5: Glass reflections (adaptive detection + FLUX)
    # ─────────────────────────────────────────────

    def _detect_and_remove_glass_reflections(
        self, img: Image.Image, car_mask: np.ndarray
    ) -> Image.Image:
        """Remove bright reflection spots from car windows. Full image, multi-path detection."""
        import traceback

        arr = np.array(img)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        h, w = arr.shape[:2]
        result = arr.copy()
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        print("[GLASS] Starting glass reflection detection...")
        try:
            car_bool = car_mask > 0 if car_mask.dtype == bool else car_mask > 127
            car_rows = np.where(np.any(car_bool, axis=1))[0]
            car_cols = np.where(np.any(car_bool, axis=0))[0]
            if len(car_rows) == 0 or len(car_cols) == 0:
                print("[GLASS] No car mask found, skipping")
                return img

            car_top = int(car_rows[0])
            car_bottom = int(car_rows[-1])
            car_left = int(car_cols[0])
            car_right = int(car_cols[-1])
            car_height = car_bottom - car_top

            window_bottom = car_top + int(car_height * 0.50)
            window_region_mask = np.zeros((h, w), dtype=np.uint8)
            window_region_mask[car_top:window_bottom, car_left:car_right] = 255

            gray = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2HSV)

            glass_mask = np.zeros((h, w), dtype=np.uint8)
            glass_condition = (
                (gray < 120) & (car_mask > 0) & (window_region_mask > 0)
            )
            glass_mask[glass_condition] = 255

            glass_px = int(np.count_nonzero(glass_mask))
            print(f"[GLASS] Glass area detected: {glass_px} px")

            if glass_px < 1000:
                print("[GLASS] No significant glass area found, skipping")
                return img

            gray_float = gray.astype(np.float32)
            local_mean = cv2.GaussianBlur(gray_float, (51, 51), 15)
            brightness_diff = gray_float - local_mean

            reflection_mask = np.zeros((h, w), dtype=np.uint8)
            adaptive_reflections = (
                (brightness_diff > 25) & (gray > 100) & (glass_mask > 0)
            )
            reflection_mask[adaptive_reflections] = 255

            very_bright = (
                (gray > 180) & (hsv[:, :, 1] < 50) & (glass_mask > 0)
            )
            reflection_mask[very_bright] = 255

            medium_bright_on_dark = (
                (gray > 90) & (local_mean < 60) & (glass_mask > 0)
            )
            reflection_mask[medium_bright_on_dark] = 255

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            reflection_mask = cv2.morphologyEx(reflection_mask, cv2.MORPH_CLOSE, kernel)
            reflection_mask = cv2.morphologyEx(reflection_mask, cv2.MORPH_OPEN, kernel)
            reflection_mask = cv2.dilate(reflection_mask, kernel, iterations=2)

            reflection_px = int(np.count_nonzero(reflection_mask))
            print(f"[GLASS] Reflection spots detected: {reflection_px} px")

            if reflection_px < 50:
                print("[GLASS] No significant reflections found, skipping")
                return img

            if reflection_px > GLASS_REFL_MAX_PX:
                print(f"[GLASS] WARNING: mask large ({reflection_px} px), reducing to bright spots only")
                reflection_mask = np.zeros((h, w), dtype=np.uint8)
                reflection_mask[very_bright] = 255
                reflection_mask = cv2.dilate(reflection_mask, kernel, iterations=2)
                reflection_px = int(np.count_nonzero(reflection_mask))
                print(f"[GLASS] Reduced to {reflection_px} px")

            if reflection_px < 50:
                return img

            token = REPLICATE_API_TOKEN or ""
            if reflection_px > 30000 and token:
                out = inpaint_reflections_replicate(
                    img, reflection_mask, token, car_color="gray", prompt=_GLASS_PROMPT
                )
                if out is not None:
                    result_bgr = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
                    print(f"[GLASS] Removed {reflection_px} px via FLUX")
                else:
                    result_bgr = cv2.inpaint(result_bgr, reflection_mask, 7, cv2.INPAINT_NS)
                    print(f"[GLASS] Removed {reflection_px} px via cv2.inpaint (FLUX failed)")
            else:
                result_bgr = cv2.inpaint(result_bgr, reflection_mask, 7, cv2.INPAINT_NS)
                print(f"[GLASS] Removed {reflection_px} px via cv2.inpaint")

            return Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))
        except Exception as e:
            print(f"[GLASS] ERROR: {e}")
            traceback.print_exc()
            return img

    # ─────────────────────────────────────────────
    # Step 6: Dust on car only (small radius)
    # ─────────────────────────────────────────────

    def _clean_dust_on_car(
        self,
        arr: np.ndarray,
        car_mask: np.ndarray,
        floor_mask: np.ndarray,
    ) -> np.ndarray:
        """Remove dust on car body only. Radius 2-3 to prevent color bleeding."""
        result = arr.copy()
        if result.dtype != np.uint8:
            result = np.clip(result, 0, 255).astype(np.uint8)
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2GRAY)

        work_mask = np.where((car_mask > 0) & (floor_mask == 0), 255, 0).astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        combined = cv2.add(tophat, blackhat)

        _, dust_mask = cv2.threshold(combined, 30, 255, cv2.THRESH_BINARY)
        dust_mask = cv2.bitwise_and(dust_mask, work_mask)

        contours, _ = cv2.findContours(dust_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered = np.zeros_like(dust_mask)
        for c in contours:
            if cv2.contourArea(c) < 30:
                cv2.drawContours(filtered, [c], -1, 255, -1)

        filtered = cv2.dilate(filtered, np.ones((2, 2), np.uint8))
        dust_px = int(np.sum(filtered > 0))
        if dust_px > DUST_MAX_PX:
            print(f"[DUST] WARNING: dust mask too large ({dust_px} px), SKIPPING")
        elif dust_px > 0:
            result_bgr = cv2.inpaint(result_bgr, filtered, 2, cv2.INPAINT_TELEA)
            print(f"[DUST] Removed {dust_px} px from car")

        return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    # ─────────────────────────────────────────────
    # Step 7: Clean and blacken tires
    # ─────────────────────────────────────────────

    def _clean_and_blacken_tires(self, arr: np.ndarray, car_mask: np.ndarray) -> np.ndarray:
        """Clean dust from tires and darken rubber. Preserve rims. Process even if count high."""
        import traceback

        result = arr.copy()
        if result.dtype != np.uint8:
            result = np.clip(result, 0, 255).astype(np.uint8)
        h, w = result.shape[:2]
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        print("[TIRE] Starting tire detection...")
        try:
            car_bool = car_mask > 0 if car_mask.dtype == bool else car_mask > 127
            car_rows = np.where(np.any(car_bool, axis=1))[0]
            car_cols = np.where(np.any(car_bool, axis=0))[0]
            if len(car_rows) == 0 or len(car_cols) == 0:
                return result

            car_top = int(car_rows[0])
            car_bottom = int(car_rows[-1])
            car_left = int(car_cols[0])
            car_right = int(car_cols[-1])
            car_height = car_bottom - car_top
            car_width = car_right - car_left

            tire_region_top = car_bottom - int(car_height * 0.25)
            zones = [
                (tire_region_top, car_bottom, car_left, car_left + int(car_width * 0.30)),
                (tire_region_top, car_bottom, car_right - int(car_width * 0.30), car_right),
            ]

            total_tire_px = 0

            for zone_idx, (zt, zb, zl, zr) in enumerate(zones):
                zt, zb = max(0, zt), min(h, zb)
                zl, zr = max(0, zl), min(w, zr)

                zone = result_bgr[zt:zb, zl:zr].copy()
                zone_car = car_mask[zt:zb, zl:zr]

                if zone.size == 0:
                    continue

                hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
                gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)

                tire_mask_zone = np.zeros(zone.shape[:2], dtype=np.uint8)
                tire_cond = (
                    (hsv[:, :, 2] < 50)
                    & (hsv[:, :, 1] < 70)
                    & (gray < 55)
                    & (zone_car > 0)
                )
                tire_mask_zone[tire_cond] = 255

                rim_exclude = (
                    (hsv[:, :, 2] > 70) | (gray > 80) | (hsv[:, :, 1] > 100)
                )
                tire_mask_zone[rim_exclude] = 0

                zone_h = zb - zt
                arch_cutoff = int(zone_h * 0.40)
                tire_mask_zone[:arch_cutoff, :] = 0

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                tire_mask_zone = cv2.morphologyEx(tire_mask_zone, cv2.MORPH_OPEN, kernel)
                tire_mask_zone = cv2.morphologyEx(tire_mask_zone, cv2.MORPH_CLOSE, kernel)

                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(tire_mask_zone)
                filtered = np.zeros_like(tire_mask_zone)
                for i in range(1, num_labels):
                    area = stats[i, cv2.CC_STAT_AREA]
                    if 3000 < area < 200000:
                        filtered[labels == i] = 255

                tire_mask_zone = filtered
                tire_px = int(np.count_nonzero(tire_mask_zone))
                total_tire_px += tire_px

                print(f"[TIRE] Zone {zone_idx}: {tire_px} px detected")

                if tire_px < 2000:
                    continue

                dust_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, dust_kernel)
                dust = ((tophat > 15) & (tire_mask_zone > 0)).astype(np.uint8) * 255
                if np.count_nonzero(dust) > 0:
                    zone = cv2.inpaint(zone, dust, 2, cv2.INPAINT_TELEA)

                lab = cv2.cvtColor(zone, cv2.COLOR_BGR2LAB)
                l_ch = lab[:, :, 0].astype(np.float32)
                tire_px_mask = tire_mask_zone > 0
                l_ch[tire_px_mask] = l_ch[tire_px_mask] * 0.45
                l_ch = np.clip(l_ch, 0, 255).astype(np.uint8)
                lab[:, :, 0] = l_ch
                darkened = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

                feathered = cv2.GaussianBlur(tire_mask_zone, (9, 9), 3).astype(np.float32) / 255.0
                for c in range(3):
                    zone[:, :, c] = (
                        darkened[:, :, c] * feathered + zone[:, :, c] * (1 - feathered)
                    ).astype(np.uint8)

                result_bgr[zt:zb, zl:zr] = zone

            if total_tire_px > 400000:
                print(f"[TIRE] WARNING: total tire pixels high ({total_tire_px}), processed with strict masks")

            print(f"[TIRE] Cleaned and darkened {total_tire_px} px total")
            return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"[TIRE] ERROR: {e}")
            traceback.print_exc()
            return result

    # ─────────────────────────────────────────────
    # Step 8: Enhance car region
    # ─────────────────────────────────────────────

    def _enhance_car_region(
        self, arr: np.ndarray, car_mask: np.ndarray
    ) -> np.ndarray:
        """UnsharpMask + contrast + saturation on car only."""
        from scipy.ndimage import gaussian_filter

        result = arr.copy()
        if result.dtype != np.uint8:
            result = np.clip(result, 0, 255).astype(np.uint8)
        img = Image.fromarray(result)
        img_enhanced = img.filter(ImageFilter.UnsharpMask(radius=2, percent=30, threshold=3))
        enh = ImageEnhance.Contrast(img_enhanced)
        img_enhanced = enh.enhance(1.05)
        hsv = cv2.cvtColor(
            cv2.cvtColor(np.array(img_enhanced), cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV
        ).astype(np.float32)
        car_float = gaussian_filter(
            (np.where(car_mask > 0, 255, 0) / 255.0).astype(np.float32), sigma=2
        )[:, :, np.newaxis]
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + car_float[:, :, 0] * 0.08), 0, 255)
        result_enh = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        result_enh = cv2.cvtColor(result_enh, cv2.COLOR_BGR2RGB)
        blended = result.astype(np.float32) * (1 - car_float) + result_enh.astype(np.float32) * car_float
        return np.clip(blended, 0, 255).astype(np.uint8)

    # ─────────────────────────────────────────────
    # Step 9: Contrast restoration (car only)
    # ─────────────────────────────────────────────

    def _restore_contrast(self, arr: np.ndarray, car_mask: np.ndarray) -> np.ndarray:
        """CLAHE + saturation on car only."""
        from scipy.ndimage import gaussian_filter

        result = arr.copy()
        if result.dtype != np.uint8:
            result = np.clip(result, 0, 255).astype(np.uint8)
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2LAB)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enh = clahe.apply(lab[:, :, 0])
        car_float = gaussian_filter(
            (np.where(car_mask > 0, 255, 0) / 255.0).astype(np.float32), sigma=2
        )
        lab[:, :, 0] = (lab[:, :, 0].astype(np.float32) * (1 - car_float * 0.4) + l_enh.astype(np.float32) * car_float * 0.4).astype(np.uint8)

        hsv = cv2.cvtColor(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + car_float * 0.05), 0, 255)
        result_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    # ─────────────────────────────────────────────
    # Step 10: Lighting adjustment (global)
    # ─────────────────────────────────────────────

    def _apply_lighting(self, arr: np.ndarray, lighting_boost: float) -> np.ndarray:
        """Global brightness. lighting_boost 1.0=no change, 1.5=+50 to L."""
        result = arr.copy()
        if result.dtype != np.uint8:
            result = np.clip(result, 0, 255).astype(np.uint8)
        additive = int((lighting_boost - 1.0) * 100)
        if additive == 0:
            return result
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = np.clip(lab[:, :, 0].astype(np.int32) + additive, 0, 255).astype(np.uint8)
        return cv2.cvtColor(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)

    # ─────────────────────────────────────────────
    # Final: Composite — restore floor and walls from original
    # ─────────────────────────────────────────────

    def _final_composite(
        self,
        processed: np.ndarray,
        original: np.ndarray,
        floor_mask: np.ndarray,
        wall_mask: np.ndarray,
        car_mask: np.ndarray,
    ) -> np.ndarray:
        """Restore ONLY floor and wall from original. Ceiling stays processed."""
        result = processed.copy().astype(np.float32)
        protection = cv2.bitwise_or(floor_mask, wall_mask)

        car_rows = np.where(car_mask > 0)
        if len(car_rows[0]) > 0:
            car_top = int(car_rows[0].min())
            above_car_protected = int(np.count_nonzero(protection[:car_top, :]))
            print(f"[COMPOSITE] Pixels protected ABOVE car top: {above_car_protected} (should be 0)")
        protected_px = int(np.count_nonzero(protection))
        print(f"[COMPOSITE] Total protected pixels: {protected_px}")

        feathered = cv2.GaussianBlur(protection.astype(np.float32), (11, 11), 3)
        alpha = np.clip(feathered / 255.0, 0, 1)[:, :, np.newaxis]
        for c in range(3):
            result[:, :, c] = (
                original[:, :, c].astype(np.float32) * alpha[:, :, 0]
                + result[:, :, c] * (1 - alpha[:, :, 0])
            )
        return np.clip(result, 0, 255).astype(np.uint8)

    # ─────────────────────────────────────────────
    # Main process
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
        """Full pipeline with protection-first architecture."""
        img = load_image(image_data, filename)
        if img.mode != "RGB":
            img = img.convert("RGB")
        original = np.array(img)
        arr = original.copy()

        print(f"[ENHANCE] START {filename} size={img.size} lighting={lighting_boost}")

        try:
            car_mask = self._get_car_mask(img)
        except Exception as e:
            logger.warning("Car mask failed: %s", e)
            car_mask = np.ones((img.height, img.width), dtype=np.uint8) * 255

        masks = self._get_protection_masks(arr, car_mask)

        if remove_sky_ceiling:
            try:
                arr = self._remove_ceiling_lights(
                    arr, masks["above_car_mask"], car_mask
                )
            except Exception as e:
                logger.warning("Ceiling removal failed: %s", e)

        try:
            arr = self._remove_body_reflections(arr, car_mask, strength=0.6)
        except Exception as e:
            logger.warning("Body reflections failed: %s", e)

        img = Image.fromarray(arr)
        try:
            img = self._detect_and_remove_glass_reflections(img, car_mask)
        except Exception as e:
            logger.warning("Glass reflections failed: %s", e)
        arr = np.array(img)

        try:
            arr = self._clean_dust_on_car(arr, car_mask, masks["floor_mask"])
        except Exception as e:
            logger.warning("Dust cleanup failed: %s", e)

        try:
            arr = self._clean_and_blacken_tires(arr, car_mask)
        except Exception as e:
            logger.warning("Tire cleanup failed: %s", e)

        if enhance_car:
            try:
                arr = self._enhance_car_region(arr, car_mask)
            except Exception as e:
                logger.warning("Car enhancement failed: %s", e)

        try:
            arr = self._restore_contrast(arr, car_mask)
        except Exception as e:
            logger.warning("Contrast restoration failed: %s", e)

        try:
            arr = self._apply_lighting(arr, lighting_boost)
        except Exception as e:
            logger.warning("Lighting failed: %s", e)

        arr = self._final_composite(
            arr, original, masks["floor_mask"], masks["wall_mask"], car_mask
        )

        print(f"[ENHANCE] DONE {filename}")
        buf = io.BytesIO()
        fmt = output_format.upper() if output_format.lower() != "jpg" else "JPEG"
        if fmt in ("JPEG", "WEBP"):
            Image.fromarray(arr).save(buf, format=fmt, quality=95)
        else:
            Image.fromarray(arr).save(buf, format=fmt)
        buf.seek(0)
        return buf.read()


enhance_preserve_service = EnhancePreserveService()
