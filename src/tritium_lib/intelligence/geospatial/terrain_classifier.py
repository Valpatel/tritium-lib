# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Terrain classifier — maps image segments to TerrainType.

Two strategies:
1. Color heuristic (always available, numpy only): HSV analysis of
   average segment color to classify terrain.
2. LLM-assisted via llama-server: sends segment image crop to a
   vision model for classification.

Color heuristic is the default and handles satellite imagery well.
LLM classification is optional for ambiguous segments.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from tritium_lib.intelligence.geospatial._deps import HAS_NUMPY
from tritium_lib.intelligence.geospatial.models import SegmentationConfig
from tritium_lib.models.terrain import TerrainType

logger = logging.getLogger(__name__)

# HSV ranges for satellite imagery terrain classification
# H: 0-180 (OpenCV convention), S: 0-255, V: 0-255
# Tuned for Esri World Imagery at zoom 16-17.
# The gray zone (S<40) is disambiguated by texture/shape features, not just HSV.
_COLOR_RULES: list[tuple[str, dict[str, tuple[float, float]]]] = [
    # Water (clear/ocean): blue hue, moderate-high saturation
    ("water", {"h": (85, 145), "s": (35, 255), "v": (20, 210)}),
    # Water (dark/murky): low saturation, dark, slightly blue-shifted
    # Lady Bird Lake, reservoirs, rivers — appear as dark gray with blue cast
    ("water_dark", {"h": (90, 150), "s": (5, 50), "v": (20, 100)}),
    # Vegetation: green hue, moderate saturation (wider V for shadows)
    ("vegetation", {"h": (25, 95), "s": (15, 255), "v": (15, 230)}),
    # Vegetation in shadow: dark, slightly green-shifted
    ("vegetation_shadow", {"h": (100, 150), "s": (5, 30), "v": (20, 70)}),
    # Barren/dirt: warm hue (brown/tan), low-moderate saturation
    ("barren", {"h": (8, 35), "s": (30, 160), "v": (60, 210)}),
    # Road: gray with low saturation — conservative to avoid over-detection
    # Real roads have H=20-32, S=24-54 but that overlaps with vegetation shadow
    ("road", {"h": (0, 180), "s": (0, 30), "v": (90, 170)}),
    # Building: very bright (white/light roofs)
    ("building", {"h": (0, 180), "s": (0, 25), "v": (180, 255)}),
    # Building: dark roofs — warm-shifted gray (NOT blue-shifted)
    ("building_dark", {"h": (0, 40), "s": (0, 35), "v": (10, 70)}),
    # Parking: similar to road but slightly warmer
    ("parking", {"h": (0, 25), "s": (5, 50), "v": (100, 165)}),
    # Sidewalk: very narrow band — light concrete
    ("sidewalk", {"h": (0, 180), "s": (0, 20), "v": (195, 245)}),
]

# Map rule names to TerrainType
_RULE_TO_TERRAIN: dict[str, TerrainType] = {
    "water": TerrainType.WATER,
    "water_dark": TerrainType.WATER,
    "vegetation": TerrainType.VEGETATION,
    "vegetation_shadow": TerrainType.VEGETATION,
    "road": TerrainType.ROAD,
    "building": TerrainType.BUILDING,
    "building_dark": TerrainType.BUILDING,
    "parking": TerrainType.PARKING,
    "barren": TerrainType.BARREN,
    "sidewalk": TerrainType.SIDEWALK,
}



class TerrainClassifier:
    """Classifies image segments into TerrainType categories.

    Primary method: HSV color analysis of segment pixel statistics.
    Optional: LLM-assisted classification via llama-server.
    """

    def __init__(self, config: Optional[SegmentationConfig] = None) -> None:
        self.config = config or SegmentationConfig()

    def classify_segment(
        self,
        image: Any,
        mask: Any,
    ) -> tuple[TerrainType, float]:
        """Classify a single segment by its pixel content.

        Uses color heuristic first. If confidence is below 0.5 and
        llm_classify is enabled, falls back to llama-server for
        a second opinion using the segment's color statistics.

        Args:
            image: RGB image as numpy array (H, W, 3)
            mask: Binary mask as numpy array (H, W) — True where segment is

        Returns:
            (terrain_type, confidence) tuple
        """
        if not HAS_NUMPY:
            return (TerrainType.UNKNOWN, 0.0)

        terrain, conf = self._classify_by_color(image, mask)

        # If color heuristic is low-confidence and LLM is enabled, ask LLM
        if conf < 0.5 and self.config.llm_classify:
            llm_result = self._classify_by_llm(image, mask, terrain, conf)
            if llm_result is not None:
                return llm_result

        return (terrain, conf)

    def classify_segments(
        self,
        image: Any,
        segments: list[dict],
    ) -> list[tuple[TerrainType, float]]:
        """Classify multiple segments from the same image.

        Pre-computes a road probability mask using Hough line detection
        and morphological operations, then uses it to boost road
        confidence for blocks in road-like areas.

        Args:
            image: RGB image as numpy array (H, W, 3)
            segments: list of dicts with "mask" key (binary numpy arrays)

        Returns:
            list of (terrain_type, confidence) tuples
        """
        # Pre-compute road probability mask for the full image
        road_mask = None
        try:
            from tritium_lib.intelligence.geospatial.road_detector import detect_road_mask
            road_mask = detect_road_mask(image)
        except Exception:
            pass

        results = []
        for seg in segments:
            terrain, conf = self.classify_segment(image, seg["mask"])

            # Boost road confidence using the pre-computed road mask
            if road_mask is not None and HAS_NUMPY:
                import numpy as np
                from tritium_lib.intelligence.geospatial.road_detector import road_probability_for_block
                mask = seg["mask"]
                ys, xs = np.nonzero(mask)
                if len(ys) > 0:
                    road_prob = road_probability_for_block(
                        road_mask, ys.min(), xs.min(), ys.max() + 1, xs.max() + 1,
                    )
                    # Only boost to road if detector is very confident AND
                    # the block is currently classified as barren (generic gray)
                    if road_prob > 0.7 and terrain == TerrainType.BARREN:
                        terrain = TerrainType.ROAD
                        conf = max(conf, 0.6 + road_prob * 0.2)

            results.append((terrain, conf))
        return results

    def _classify_by_color(
        self,
        image: Any,
        mask: Any,
    ) -> tuple[TerrainType, float]:
        """Classify segment using HSV color statistics."""
        import numpy as np

        # Extract pixels within the mask
        if mask.sum() == 0:
            return (TerrainType.UNKNOWN, 0.0)

        # Get masked pixels
        pixels = image[mask]  # (N, 3) RGB

        # Convert to HSV
        hsv = self._rgb_to_hsv(pixels)

        # Compute statistics
        h_mean = np.mean(hsv[:, 0])
        s_mean = np.mean(hsv[:, 1])
        v_mean = np.mean(hsv[:, 2])
        h_std = np.std(hsv[:, 0])
        s_std = np.std(hsv[:, 1])
        v_std = np.std(hsv[:, 2])

        # Score each rule
        best_type = TerrainType.UNKNOWN
        best_score = 0.0

        for rule_name, ranges in _COLOR_RULES:
            score = self._score_rule(
                h_mean, s_mean, v_mean,
                h_std, s_std, v_std,
                ranges,
            )
            if score > best_score:
                best_score = score
                best_type = _RULE_TO_TERRAIN[rule_name]

        # Apply contextual adjustments using texture and shape
        confidence = min(best_score, 1.0)

        # Texture variance — key disambiguator for the gray zone
        # Roads are smooth (low RGB variance), buildings have edges (high variance)
        r_std = np.std(pixels[:, 0].astype(np.float32))
        g_std = np.std(pixels[:, 1].astype(np.float32))
        b_std = np.std(pixels[:, 2].astype(np.float32))
        rgb_variance = (r_std + g_std + b_std) / 3.0

        # Segment area and shape (from mask bounding box)
        area = mask.sum()
        ys_nz, xs_nz = np.nonzero(mask)
        if len(ys_nz) > 0:
            bbox_w = xs_nz.max() - xs_nz.min() + 1
            bbox_h = ys_nz.max() - ys_nz.min() + 1
            aspect = max(bbox_w, bbox_h) / max(min(bbox_w, bbox_h), 1)
            fill_ratio = area / max(bbox_w * bbox_h, 1)
        else:
            aspect = 1.0
            fill_ratio = 1.0

        # --- Edge density: strong linear edges indicate roads or buildings ---
        edge_density = self._compute_edge_density(image, mask)

        # --- Gray zone disambiguation (S < 50) ---
        # Real satellite imagery has S~40 even for roads (warm asphalt tone).
        # Use edge density and shape to distinguish road from building/parking.
        if s_mean < 50:
            # Buildings: very bright or very dark, compact shape
            if (v_mean > 200 or v_mean < 50) and aspect < 3.0:
                best_type = TerrainType.BUILDING
                confidence = max(confidence, 0.55)
            # Buildings: compact with high texture (HVAC, roof detail)
            elif rgb_variance > 60 and aspect < 2.0 and fill_ratio > 0.7:
                best_type = TerrainType.BUILDING
                confidence = max(confidence, 0.50)
            # Roads: very desaturated (S<20) + strong edges only.
            # Despite ground truth showing S~38, we can't use that because
            # it overlaps too heavily with vegetation shadow. Only claim
            # road for truly gray blocks with clear linear edges.
            elif (edge_density > 0.3 and s_mean < 20 and
                  90 < v_mean < 175):
                best_type = TerrainType.ROAD
                confidence = max(confidence, 0.60)

        # Strong signals that override HSV ambiguity
        # Deep blue with low variance = confident water
        if best_type == TerrainType.WATER and s_std < 25 and h_std < 20:
            confidence = min(confidence + 0.15, 1.0)

        # Very uniform green = strong vegetation signal
        if best_type == TerrainType.VEGETATION and h_std < 20 and s_mean > 30:
            confidence = min(confidence + 0.1, 1.0)

        # Vegetation in shadow (dark green) — rescue from misclass as building_dark
        if best_type == TerrainType.BUILDING and 25 < h_mean < 95 and s_mean > 20:
            best_type = TerrainType.VEGETATION
            confidence = 0.50

        return (best_type, confidence)

    def _score_rule(
        self,
        h_mean: float,
        s_mean: float,
        v_mean: float,
        h_std: float,
        s_std: float,
        v_std: float,
        ranges: dict[str, tuple[float, float]],
    ) -> float:
        """Score how well HSV stats match a classification rule."""
        score = 1.0

        for channel, (lo, hi) in ranges.items():
            if channel == "h":
                mean = h_mean
            elif channel == "s":
                mean = s_mean
            else:
                mean = v_mean

            if lo <= mean <= hi:
                # Inside range: score based on how centered the value is
                mid = (lo + hi) / 2
                span = (hi - lo) / 2
                if span > 0:
                    dist = abs(mean - mid) / span
                    score *= 1.0 - 0.3 * dist  # up to 30% penalty for being near edge
            else:
                # Outside range: sharp penalty
                if mean < lo:
                    dist = lo - mean
                else:
                    dist = mean - hi
                score *= max(0.0, 1.0 - dist / 50.0)

        return score

    @staticmethod
    def _compute_edge_density(image: Any, mask: Any) -> float:
        """Compute edge density within a segment using Canny edge detection.

        Roads have strong linear edges (lane markings, curbs, shoulders).
        Buildings have rectangular edges. Vegetation has scattered edges.

        Returns fraction of edge pixels within the masked area (0.0-1.0).
        """
        from tritium_lib.intelligence.geospatial._deps import HAS_CV2

        if not HAS_CV2:
            return 0.0

        import cv2
        import numpy as np

        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            return 0.0

        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1

        roi = image[y0:y1, x0:x1]
        roi_mask = mask[y0:y1, x0:x1]

        if roi.ndim == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        else:
            gray = roi

        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150)  # tighter thresholds for satellite

        edge_pixels = (edges > 0) & roi_mask
        total_pixels = roi_mask.sum()

        if total_pixels == 0:
            return 0.0

        density = edge_pixels.sum() / total_pixels

        # Use Hough lines to detect road-like linear structures.
        # Roads produce long parallel lines; vegetation produces short scattered edges.
        if density > 0.05 and roi.shape[0] >= 16 and roi.shape[1] >= 16:
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180,
                threshold=15,
                minLineLength=max(roi.shape[0] // 3, 8),
                maxLineGap=5,
            )
            if lines is not None and len(lines) >= 2:
                # Boost density for regions with strong linear features
                line_boost = min(len(lines) / 20.0, 0.3)  # up to 0.3 boost
                density = min(density + line_boost, 1.0)

        return density

    def _classify_by_llm(
        self,
        image: Any,
        mask: Any,
        heuristic_type: TerrainType,
        heuristic_conf: float,
    ) -> Optional[tuple[TerrainType, float]]:
        """Classify segment using llama-server text inference.

        Sends color statistics to the LLM and asks it to classify
        the terrain type. This is a text-only approach — works with
        any model, not just vision models.
        """
        import numpy as np

        if mask.sum() == 0:
            return None

        try:
            import requests
        except ImportError:
            return None

        # Build color statistics description
        pixels = image[mask]
        r_mean, g_mean, b_mean = pixels.mean(axis=0)
        r_std, g_std, b_std = pixels.std(axis=0)

        hsv = self._rgb_to_hsv(pixels)
        h_mean, s_mean, v_mean = hsv.mean(axis=0)

        area_px = int(mask.sum())
        h, w = mask.shape
        ys, xs = np.nonzero(mask)
        bbox_w = int(xs.max() - xs.min()) if len(xs) > 0 else 0
        bbox_h = int(ys.max() - ys.min()) if len(ys) > 0 else 0
        aspect = bbox_w / max(bbox_h, 1)

        prompt = (
            f"Classify this satellite image segment as one of: "
            f"building, road, water, vegetation, parking, sidewalk, bridge, rail, barren.\n\n"
            f"Color stats (RGB mean): R={r_mean:.0f} G={g_mean:.0f} B={b_mean:.0f}\n"
            f"Color stats (RGB std): R={r_std:.0f} G={g_std:.0f} B={b_std:.0f}\n"
            f"HSV: H={h_mean:.0f} S={s_mean:.0f} V={v_mean:.0f}\n"
            f"Area: {area_px} pixels, aspect ratio: {aspect:.2f}\n"
            f"Bounding box: {bbox_w}x{bbox_h} pixels in a {w}x{h} image\n"
            f"Color heuristic guessed: {heuristic_type.value} (confidence: {heuristic_conf:.2f})\n\n"
            f"Respond with ONLY the terrain type name, nothing else."
        )

        try:
            endpoint = self.config.llm_endpoint.rstrip("/")
            resp = requests.post(
                f"{endpoint}/v1/chat/completions",
                json={
                    "model": "any",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 20,
                    "temperature": 0.1,
                },
                timeout=5,
            )
            if resp.status_code != 200:
                return None

            data = resp.json()
            answer = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip().lower()

            # Parse the LLM's answer
            type_map = {
                "building": TerrainType.BUILDING,
                "road": TerrainType.ROAD,
                "water": TerrainType.WATER,
                "vegetation": TerrainType.VEGETATION,
                "parking": TerrainType.PARKING,
                "sidewalk": TerrainType.SIDEWALK,
                "bridge": TerrainType.BRIDGE,
                "rail": TerrainType.RAIL,
                "barren": TerrainType.BARREN,
            }

            for key, ttype in type_map.items():
                if key in answer:
                    # LLM agreed or overrode — boost confidence
                    new_conf = 0.7 if ttype != heuristic_type else max(heuristic_conf + 0.2, 0.7)
                    logger.debug(
                        "LLM classified segment as %s (heuristic was %s)",
                        ttype.value, heuristic_type.value,
                    )
                    return (ttype, min(new_conf, 1.0))

            return None

        except Exception as e:
            logger.debug("LLM classification failed: %s", e)
            return None

    @staticmethod
    def _rgb_to_hsv(rgb: Any) -> Any:
        """Convert RGB pixel array to HSV (OpenCV convention: H 0-180, S 0-255, V 0-255)."""
        import numpy as np

        rgb = rgb.astype(np.float32) / 255.0
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

        v = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)
        delta = v - min_rgb
        # Avoid division by zero for black pixels
        v_safe = np.where(v > 0, v, 1.0)
        s = np.where(v > 0, delta / v_safe, 0.0)

        h = np.zeros_like(v)
        # Red is max
        mask_r = (v == r) & (delta > 0)
        h[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
        # Green is max
        mask_g = (v == g) & (delta > 0)
        h[mask_g] = 60 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
        # Blue is max
        mask_b = (v == b) & (delta > 0)
        h[mask_b] = 60 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)

        # Convert to OpenCV convention
        h = h / 2.0  # 0-180
        s = s * 255.0
        v = v * 255.0

        return np.stack([h, s, v], axis=1)
