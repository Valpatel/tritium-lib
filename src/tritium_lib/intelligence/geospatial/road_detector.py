# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Road detector — specialized pre-pass for road identification.

Uses OpenCV morphological operations and line detection to find
road-like structures in satellite imagery BEFORE the general
classifier runs. Roads have specific structural properties:

1. Linear: long, thin, continuous features
2. Gray: low saturation, moderate brightness
3. Edges: sharp boundaries with adjacent terrain
4. Connected: form a network, not isolated patches

Strategy:
    1. Convert to grayscale
    2. Extract gray-band pixels (road candidates)
    3. Apply morphological closing to connect nearby road pixels
    4. Use Hough line transform to find linear structures
    5. Dilate detected lines to form road corridors
    6. Return a road probability mask (0.0-1.0 per pixel)

The road probability mask is used by the TerrainClassifier to
boost road confidence for blocks that overlap high-probability areas.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from tritium_lib.intelligence.geospatial._deps import HAS_CV2, HAS_NUMPY

logger = logging.getLogger(__name__)


def detect_road_mask(image: Any) -> Optional[Any]:
    """Detect road-like structures and return a probability mask.

    Args:
        image: RGB numpy array (H, W, 3)

    Returns:
        Probability mask (H, W) with values 0.0-1.0 where 1.0 = definite road.
        Returns None if OpenCV is not available.
    """
    if not HAS_CV2 or not HAS_NUMPY:
        return None

    import cv2
    import numpy as np

    h, w = image.shape[:2]

    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Step 2: Extract gray-band pixels (roads are gray with low saturation)
    # Convert to HSV for saturation check
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # Road candidates: very low saturation AND moderate brightness
    road_candidates = (
        (saturation < 35) &  # very low color saturation (true gray)
        (value > 80) &       # not too dark
        (value < 180)        # not too bright
    ).astype(np.uint8) * 255

    # Step 3: Morphological operations
    # Close small gaps (connect broken road segments)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(road_candidates, cv2.MORPH_CLOSE, kernel_close)

    # Remove noise (small isolated gray patches that aren't roads)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

    # Step 4: Edge detection on original image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)

    # Step 5: Hough line detection for strong linear features
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=30,
        minLineLength=max(w // 10, 20),  # at least 10% of image width
        maxLineGap=15,
    )

    # Create line mask from detected lines
    line_mask = np.zeros((h, w), dtype=np.uint8)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 5)  # 5px wide corridor

    # Step 6: Combine evidence
    # Road = gray-band pixels that are near detected lines
    # Use distance transform from line mask
    _, line_dist = cv2.threshold(line_mask, 0, 255, cv2.THRESH_BINARY)

    # Dilate lines to create narrow corridors (typical road width ~3-5 pixels at zoom 16)
    road_corridor = cv2.dilate(line_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))

    # Final road probability: intersection of gray-band and line-corridor
    road_prob = np.zeros((h, w), dtype=np.float32)

    # High probability: gray candidate pixels within a line corridor
    high_prob = (cleaned > 0) & (road_corridor > 0)
    road_prob[high_prob] = 0.8

    # Medium probability: gray candidate pixels near lines but not in corridor
    near_lines = cv2.dilate(line_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21)))
    med_prob = (cleaned > 0) & (near_lines > 0) & ~high_prob
    road_prob[med_prob] = 0.4

    # Low probability: gray candidates far from any line
    low_prob = (cleaned > 0) & ~high_prob & ~med_prob
    road_prob[low_prob] = 0.1

    line_count = len(lines) if lines is not None else 0
    road_pct = (road_prob > 0.3).sum() / (h * w) * 100
    logger.debug(
        "Road detection: %d Hough lines, %.1f%% road coverage",
        line_count, road_pct,
    )

    return road_prob


def road_probability_for_block(
    road_mask: Any,
    y0: int, x0: int,
    y1: int, x1: int,
) -> float:
    """Get the mean road probability for a block region.

    Returns 0.0-1.0 where >0.5 strongly suggests road.
    """
    if road_mask is None:
        return 0.0

    block = road_mask[y0:y1, x0:x1]
    if block.size == 0:
        return 0.0

    return float(block.mean())
