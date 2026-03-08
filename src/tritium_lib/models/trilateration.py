# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""BLE trilateration — estimate device position from multi-node RSSI sightings.

Uses the log-distance path loss model to convert RSSI to distance, then
weighted centroid to estimate position. Weighted centroid is preferred over
least-squares for noisy BLE RSSI data because it degrades gracefully with
bad readings rather than diverging.
"""

import math
from typing import Optional

from pydantic import BaseModel, Field


class AnchorPoint(BaseModel):
    """A reference node with known position and observed RSSI/distance."""
    node_id: str
    lat: float
    lon: float
    rssi: float
    distance: float = 0.0


class PositionEstimate(BaseModel):
    """Estimated position of a BLE device with confidence metadata."""
    lat: float
    lon: float
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="0.0 = no confidence, 1.0 = high confidence",
    )
    anchors_used: int
    method: str = "weighted_centroid"


def rssi_to_distance(
    rssi: float,
    tx_power: float = -59.0,
    path_loss_exponent: float = 2.5,
) -> float:
    """Convert RSSI to estimated distance using the log-distance path loss model.

    Formula: d = 10 ^ ((tx_power - rssi) / (10 * n))

    Args:
        rssi: Received signal strength in dBm (negative).
        tx_power: RSSI at 1 meter reference distance (typically -59 dBm for BLE).
        path_loss_exponent: Environment factor (2.0 = free space, 2.5-3.0 = indoor,
            4.0 = obstructed). Default 2.5 for typical indoor BLE.

    Returns:
        Estimated distance in meters. Always >= 0.1 (clamped).
    """
    if path_loss_exponent <= 0:
        raise ValueError("path_loss_exponent must be positive")

    exponent = (tx_power - rssi) / (10.0 * path_loss_exponent)
    distance = 10.0 ** exponent

    # Clamp minimum distance — sub-10cm readings are noise
    return max(distance, 0.1)


def trilaterate_2d(
    anchors: list[tuple[float, float, float]],
) -> Optional[tuple[float, float]]:
    """Estimate 2D position from anchor points using inverse-distance weighted centroid.

    Each anchor is (x, y, distance). Closer anchors (smaller distance) get
    higher weight via 1/d^2 weighting, which naturally emphasizes the
    nearest nodes.

    Args:
        anchors: List of (x, y, distance) tuples. Needs at least 2 anchors.

    Returns:
        (x, y) estimated position, or None if fewer than 2 anchors.
    """
    if len(anchors) < 2:
        return None

    total_weight = 0.0
    wx = 0.0
    wy = 0.0

    for x, y, d in anchors:
        # Inverse-square distance weighting
        d_clamped = max(d, 0.1)
        w = 1.0 / (d_clamped ** 2)
        wx += x * w
        wy += y * w
        total_weight += w

    if total_weight == 0:
        return None

    return (wx / total_weight, wy / total_weight)


def _compute_confidence(
    anchors: list[AnchorPoint],
    distances: list[float],
) -> float:
    """Compute confidence score [0, 1] based on anchor geometry and quality.

    Factors:
    - Anchor count: 2 anchors = low, 3 = moderate, 4+ = high
    - Geometric spread: wider spatial distribution = better triangulation
    - Distance uncertainty: closer anchors have lower relative error
    """
    n = len(anchors)
    if n < 2:
        return 0.0

    # --- Factor 1: Anchor count (0.3 - 1.0) ---
    # 2 anchors -> 0.3, 3 -> 0.6, 4 -> 0.8, 5+ -> 1.0
    count_score = min(1.0, 0.3 + (n - 2) * 0.25)

    # --- Factor 2: Geometric spread (0.0 - 1.0) ---
    # Compute bounding box diagonal of anchor positions relative to mean distance
    lats = [a.lat for a in anchors]
    lons = [a.lon for a in anchors]
    lat_spread = max(lats) - min(lats)
    lon_spread = max(lons) - min(lons)
    diagonal = math.sqrt(lat_spread ** 2 + lon_spread ** 2)

    # Compare spread to mean distance — if anchors are spread out relative
    # to the distances being measured, geometry is good
    mean_dist = sum(distances) / len(distances) if distances else 1.0
    # Convert diagonal from degrees to rough meters (1 deg ~ 111km at equator,
    # but for relative comparison this ratio still works)
    # For local coordinates, diagonal is already meaningful
    if mean_dist > 0:
        spread_ratio = diagonal / (mean_dist * 0.00001)  # rough deg-to-meter factor
        # Cap at reasonable values for lat/lon inputs
        # For lat/lon: spread of 0.001 deg ~ 111m, which is reasonable for indoor
        spread_ratio_direct = diagonal * 111_000 / max(mean_dist, 0.1)
        geometry_score = min(1.0, spread_ratio_direct / 5.0)
    else:
        geometry_score = 0.5

    # --- Factor 3: Distance quality (0.0 - 1.0) ---
    # Closer anchors have more reliable RSSI. Penalize if all anchors are far.
    close_count = sum(1 for d in distances if d < 10.0)
    distance_score = min(1.0, close_count / max(n * 0.5, 1.0))

    # Weighted combination
    confidence = (
        0.40 * count_score
        + 0.35 * geometry_score
        + 0.25 * distance_score
    )

    return round(min(1.0, max(0.0, confidence)), 3)


def estimate_position(
    sightings: list[dict],
    node_positions: dict[str, tuple[float, float]],
    tx_power: float = -59.0,
    path_loss_exponent: float = 2.5,
) -> Optional[dict]:
    """Estimate BLE device position from multi-node sightings.

    End-to-end pipeline: RSSI -> distance -> weighted centroid -> confidence.

    Args:
        sightings: List of dicts with 'node_id' and 'ble_rssi' keys.
        node_positions: Map of node_id -> (lat, lon).
        tx_power: RSSI at 1m reference distance.
        path_loss_exponent: Path loss exponent for environment.

    Returns:
        Dict with position estimate, or None if insufficient data.
        Keys: lat, lon, confidence, anchors_used, method, anchors.
    """
    anchors: list[AnchorPoint] = []

    for s in sightings:
        node_id = s.get("node_id")
        rssi = s.get("ble_rssi")

        if node_id is None or rssi is None:
            continue

        pos = node_positions.get(node_id)
        if pos is None:
            continue

        distance = rssi_to_distance(rssi, tx_power, path_loss_exponent)
        anchors.append(AnchorPoint(
            node_id=node_id,
            lat=pos[0],
            lon=pos[1],
            rssi=rssi,
            distance=distance,
        ))

    if len(anchors) < 2:
        return None

    # Build trilateration input
    anchor_tuples = [(a.lat, a.lon, a.distance) for a in anchors]
    result = trilaterate_2d(anchor_tuples)

    if result is None:
        return None

    distances = [a.distance for a in anchors]
    confidence = _compute_confidence(anchors, distances)

    estimate = PositionEstimate(
        lat=round(result[0], 8),
        lon=round(result[1], 8),
        confidence=confidence,
        anchors_used=len(anchors),
        method="weighted_centroid",
    )

    return estimate.model_dump()
