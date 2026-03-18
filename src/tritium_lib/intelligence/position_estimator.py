# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Position estimation from detection edges and trust anchors.

Simple position estimation that places non-GPS devices near the anchors
that detect them, weighted by signal strength. This is the foundation —
graph-based and ML-enhanced estimation layers on top of this later.
"""

from __future__ import annotations

import math
import time

from tritium_lib.models.position_anchor import (
    DetectionEdge,
    FusedPositionEstimate,
    PositionAnchor,
)


def rssi_to_distance(
    rssi_dbm: float,
    tx_power: float = -40.0,
    path_loss_exp: float = 2.5,
) -> float:
    """Convert RSSI to estimated distance using the log-distance path loss model.

    Formula: d = 10 ^ ((tx_power - rssi) / (10 * n))

    Args:
        rssi_dbm: Received signal strength in dBm (negative value).
        tx_power: RSSI at 1 meter reference distance. Default -40 dBm
            (typical for LoRa/mesh — higher power than BLE).
        path_loss_exp: Path loss exponent. 2.0 = free space, 2.5 = light
            indoor, 3.0 = dense indoor, 4.0 = heavily obstructed.

    Returns:
        Estimated distance in meters, clamped to minimum 0.5m.
    """
    if path_loss_exp <= 0:
        raise ValueError("path_loss_exp must be positive")

    exponent = (tx_power - rssi_dbm) / (10.0 * path_loss_exp)
    distance = 10.0 ** exponent

    # Clamp: sub-half-meter readings are noise at mesh/WiFi scale
    return max(distance, 0.5)


def estimate_from_single_anchor(
    anchor: PositionAnchor,
    detection: DetectionEdge,
) -> FusedPositionEstimate:
    """Place detected device near an anchor, offset by RSSI-estimated distance.

    With only one anchor, direction is unknown — we place the target at
    the anchor's position with accuracy radius equal to the estimated
    distance. This is "proximity" estimation: we know it's nearby, but
    not which direction.

    Args:
        anchor: The GPS-anchored device that detected the target.
        detection: The detection edge with signal strength data.

    Returns:
        Position estimate centered on the anchor with accuracy radius
        based on RSSI distance estimate.
    """
    if detection.distance_estimate_m is not None:
        distance_m = detection.distance_estimate_m
    elif detection.rssi is not None:
        distance_m = rssi_to_distance(detection.rssi)
    else:
        # No signal data — use large default radius
        distance_m = 100.0

    # Confidence: anchor confidence * detection confidence * penalty for single anchor
    confidence = anchor.confidence * detection.confidence * 0.5

    return FusedPositionEstimate(
        target_id=detection.detected_id,
        lat=anchor.lat,
        lng=anchor.lng,
        accuracy_m=distance_m,
        method="proximity",
        anchor_count=1,
        confidence=round(min(1.0, confidence), 3),
        timestamp=detection.timestamp,
    )


def estimate_from_multiple_anchors(
    anchors: list[PositionAnchor],
    detections: list[DetectionEdge],
) -> FusedPositionEstimate | None:
    """Weighted centroid of anchors that detected the target.

    Each anchor's contribution is weighted by inverse RSSI distance —
    closer anchors (stronger signal) pull harder. With 2+ anchors we
    get a centroid that converges on the true position.

    Args:
        anchors: List of GPS-anchored devices that detected the target.
        detections: Corresponding detection edges (must match anchors by
            detector_id). Only detections whose detector_id matches an
            anchor's anchor_id are used.

    Returns:
        Weighted centroid position estimate, or None if no valid pairs.
    """
    if not anchors or not detections:
        return None

    # Build anchor lookup
    anchor_map = {a.anchor_id: a for a in anchors}

    # Match detections to anchors
    pairs: list[tuple[PositionAnchor, DetectionEdge, float]] = []
    for det in detections:
        anchor = anchor_map.get(det.detector_id)
        if anchor is None:
            continue

        if det.distance_estimate_m is not None:
            dist = max(det.distance_estimate_m, 0.5)
        elif det.rssi is not None:
            dist = rssi_to_distance(det.rssi)
        else:
            dist = 100.0

        pairs.append((anchor, det, dist))

    if not pairs:
        return None

    if len(pairs) == 1:
        return estimate_from_single_anchor(pairs[0][0], pairs[0][1])

    # Weighted centroid: weight = 1 / distance^2
    total_weight = 0.0
    w_lat = 0.0
    w_lng = 0.0

    for anchor, det, dist in pairs:
        weight = 1.0 / (dist ** 2)
        w_lat += anchor.lat * weight
        w_lng += anchor.lng * weight
        total_weight += weight

    if total_weight == 0:
        return None

    est_lat = w_lat / total_weight
    est_lng = w_lng / total_weight

    # Accuracy: weighted average of distances (closer anchors dominate)
    w_dist = sum(d * (1.0 / (d ** 2)) for _, _, d in pairs) / total_weight

    # Confidence factors: anchor count, average anchor confidence, detection confidence
    avg_anchor_conf = sum(a.confidence for a, _, _ in pairs) / len(pairs)
    avg_det_conf = sum(d.confidence for _, d, _ in pairs) / len(pairs)
    count_factor = min(1.0, len(pairs) / 4.0)  # 4+ anchors = full count bonus
    confidence = avg_anchor_conf * avg_det_conf * (0.5 + 0.5 * count_factor)

    # Determine method based on anchor count
    method = "centroid" if len(pairs) >= 3 else "centroid"

    # Get target_id from first detection
    target_id = pairs[0][1].detected_id

    return FusedPositionEstimate(
        target_id=target_id,
        lat=round(est_lat, 8),
        lng=round(est_lng, 8),
        accuracy_m=round(w_dist, 1),
        method=method,
        anchor_count=len(pairs),
        confidence=round(min(1.0, max(0.0, confidence)), 3),
        timestamp=max(d.timestamp for _, d, _ in pairs),
    )
