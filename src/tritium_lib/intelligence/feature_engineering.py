# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Feature engineering functions for correlation learners.

Reusable feature computation for target correlation scoring.
These functions extract meaningful signals from target pairs to
improve the RL correlation model beyond the baseline 6 features.

Features:
    device_type_match   — semantic match between asset types across sensors
    co_movement_score   — co-movement duration from position history
    time_similarity     — time-of-day similarity between detections
    source_diversity    — diversity of sensor sources observing the pair
    wifi_probe_temporal — temporal correlation between WiFi probe and BLE
"""

from __future__ import annotations

import math
from typing import Any, Protocol, Sequence


class PositionLike(Protocol):
    """Protocol for objects with position and timing data."""

    @property
    def position(self) -> tuple[float, float]: ...

    @property
    def last_seen(self) -> float: ...


# --- Cross-sensor device type compatibility ---

# Semantic compatibility between asset types from different sensors.
# BLE "phone" + camera "person" is a strong match (person carries phone).
# BLE "watch" + camera "person" is also strong.
# BLE "laptop" + camera "person" is moderate.
_TYPE_COMPATIBILITY: dict[frozenset[str], float] = {
    frozenset({"phone", "person"}): 1.0,
    frozenset({"watch", "person"}): 0.95,
    frozenset({"headphones", "person"}): 0.9,
    frozenset({"fitness_tracker", "person"}): 0.9,
    frozenset({"tablet", "person"}): 0.8,
    frozenset({"laptop", "person"}): 0.7,
    frozenset({"phone", "vehicle"}): 0.6,  # driver has phone
    frozenset({"laptop", "vehicle"}): 0.5,
    frozenset({"beacon", "vehicle"}): 0.4,
    frozenset({"iot_device", "vehicle"}): 0.3,
    frozenset({"speaker", "person"}): 0.3,
}


def device_type_match(
    type_a: str,
    type_b: str,
    source_a: str = "",
    source_b: str = "",
) -> float:
    """Compute semantic device type match score.

    Args:
        type_a: Asset type of first target (e.g., "phone", "person").
        type_b: Asset type of second target.
        source_a: Source sensor of first target (e.g., "ble", "yolo").
        source_b: Source sensor of second target.

    Returns:
        Score from 0.0 (no match) to 1.0 (strong semantic match).
    """
    if not type_a or not type_b:
        return 0.0

    ta = type_a.lower().strip()
    tb = type_b.lower().strip()

    # Same type from different sensors is a moderate match
    if ta == tb and ta != "unknown":
        # Cross-sensor same type (e.g., two BLE phones seen together)
        if source_a != source_b:
            return 0.6
        return 0.3

    # Check compatibility table
    pair = frozenset({ta, tb})
    compat = _TYPE_COMPATIBILITY.get(pair)
    if compat is not None:
        return compat

    # Both unknown
    if ta == "unknown" or tb == "unknown":
        return 0.1

    # No known compatibility
    return 0.0


def co_movement_score(
    trail_a: Sequence[tuple[float, float, float]],
    trail_b: Sequence[tuple[float, float, float]],
    *,
    max_distance: float = 5.0,
    min_overlap_seconds: float = 5.0,
) -> float:
    """Compute co-movement duration score from position trails.

    Measures how long two targets have been moving together by examining
    temporal overlap where both are within max_distance of each other.

    Args:
        trail_a: List of (x, y, timestamp) tuples for target A.
        trail_b: List of (x, y, timestamp) tuples for target B.
        max_distance: Maximum distance to consider "together".
        min_overlap_seconds: Minimum co-located duration for any score.

    Returns:
        Score from 0.0 (never co-located) to 1.0 (long co-movement).
    """
    if len(trail_a) < 2 or len(trail_b) < 2:
        return 0.0

    # Find temporal overlap region
    t_start = max(trail_a[0][2], trail_b[0][2])
    t_end = min(trail_a[-1][2], trail_b[-1][2])

    if t_end - t_start < min_overlap_seconds:
        return 0.0

    # Sample positions at common timestamps
    co_located_time = 0.0
    total_time = t_end - t_start

    # Interpolate trail B at trail A timestamps
    b_idx = 0
    prev_co_located = False
    prev_time = t_start

    for ax, ay, at in trail_a:
        if at < t_start or at > t_end:
            continue

        # Advance B index to bracket this time
        while b_idx < len(trail_b) - 1 and trail_b[b_idx + 1][2] <= at:
            b_idx += 1

        if b_idx >= len(trail_b) - 1:
            break

        # Linear interpolation of B position at time at
        b1 = trail_b[b_idx]
        b2 = trail_b[min(b_idx + 1, len(trail_b) - 1)]

        if b2[2] - b1[2] > 0:
            frac = (at - b1[2]) / (b2[2] - b1[2])
            bx = b1[0] + frac * (b2[0] - b1[0])
            by = b1[1] + frac * (b2[1] - b1[1])
        else:
            bx, by = b1[0], b1[1]

        dist = math.hypot(ax - bx, ay - by)
        is_close = dist <= max_distance

        if is_close:
            co_located_time += at - prev_time

        prev_time = at

    if total_time <= 0:
        return 0.0

    # Normalize: 30 seconds of co-movement = 1.0
    duration_score = min(1.0, co_located_time / 30.0)
    # Also factor in fraction of overlap spent co-located
    fraction_score = co_located_time / total_time

    return min(1.0, 0.6 * duration_score + 0.4 * fraction_score)


def time_similarity(
    last_seen_a: float,
    last_seen_b: float,
    *,
    time_of_day_a: float | None = None,
    time_of_day_b: float | None = None,
) -> float:
    """Compute time-of-day similarity between two detections.

    Targets seen at similar times of day across sessions are more likely
    to be the same entity (e.g., a commuter's phone seen every morning).

    Args:
        last_seen_a: Unix timestamp of target A's last detection.
        last_seen_b: Unix timestamp of target B's last detection.
        time_of_day_a: Optional explicit time-of-day in hours (0-24).
        time_of_day_b: Optional explicit time-of-day in hours (0-24).

    Returns:
        Score from 0.0 (opposite times) to 1.0 (same time of day).
    """
    import time as _time

    if time_of_day_a is None:
        try:
            time_of_day_a = _time.localtime(last_seen_a).tm_hour + \
                           _time.localtime(last_seen_a).tm_min / 60.0
        except (OSError, ValueError, OverflowError):
            time_of_day_a = 0.0

    if time_of_day_b is None:
        try:
            time_of_day_b = _time.localtime(last_seen_b).tm_hour + \
                           _time.localtime(last_seen_b).tm_min / 60.0
        except (OSError, ValueError, OverflowError):
            time_of_day_b = 0.0

    # Circular difference (wraps around 24 hours)
    diff = abs(time_of_day_a - time_of_day_b)
    if diff > 12.0:
        diff = 24.0 - diff

    # Score: 1.0 at same time, 0.0 at 12 hours apart
    return max(0.0, 1.0 - diff / 12.0)


def source_diversity(
    sources_a: Sequence[str],
    sources_b: Sequence[str],
) -> float:
    """Compute source diversity score for a target pair.

    Higher diversity means more different sensor types have seen
    these targets, increasing confidence in their correlation.

    Args:
        sources_a: List of source types that detected target A.
        sources_b: List of source types that detected target B.

    Returns:
        Score from 0.0 (single source) to 1.0 (many diverse sources).
    """
    set_a = set(s.lower().strip() for s in sources_a if s)
    set_b = set(s.lower().strip() for s in sources_b if s)

    all_sources = set_a | set_b
    if len(all_sources) <= 1:
        return 0.0

    # More source types = higher score
    # 2 sources = 0.4, 3 = 0.6, 4 = 0.8, 5+ = 1.0
    count = len(all_sources)
    score = min(1.0, (count - 1) * 0.2 + 0.2)

    # Bonus for cross-category diversity (RF + visual is better than RF + RF)
    rf_sources = {"ble", "wifi", "wifi_probe", "meshtastic", "lora"}
    visual_sources = {"yolo", "camera", "thermal"}
    acoustic_sources = {"acoustic", "audio"}

    categories = set()
    for s in all_sources:
        if s in rf_sources:
            categories.add("rf")
        elif s in visual_sources:
            categories.add("visual")
        elif s in acoustic_sources:
            categories.add("acoustic")
        else:
            categories.add("other")

    if len(categories) >= 2:
        score = min(1.0, score + 0.2)

    return score


def wifi_probe_temporal_correlation(
    ble_last_seen: float,
    wifi_probe_last_seen: float,
    *,
    same_observer: bool = False,
    max_window_s: float = 10.0,
) -> float:
    """Score temporal correlation between WiFi probe and BLE detection.

    When a device sends WiFi probe requests and BLE advertisements
    from the same hardware, the timing is very close. A phone's WiFi
    radio probes on a schedule, and its BLE radio advertises on a
    different schedule, but they originate from the same physical device.

    Args:
        ble_last_seen: Timestamp of BLE detection.
        wifi_probe_last_seen: Timestamp of WiFi probe detection.
        same_observer: Whether both were captured by the same edge node.
        max_window_s: Maximum time window for correlation.

    Returns:
        Score from 0.0 (no correlation) to 1.0 (strong temporal match).
    """
    time_diff = abs(ble_last_seen - wifi_probe_last_seen)

    if time_diff > max_window_s:
        return 0.0

    # Base score from temporal proximity
    score = 1.0 - (time_diff / max_window_s)

    # Same observer bonus: much stronger signal
    if same_observer:
        score = min(1.0, score * 1.3)

    return max(0.0, min(1.0, score))


# --- Feature vector builder ---

EXTENDED_FEATURE_NAMES = [
    # Original 6
    "distance",
    "rssi_delta",
    "co_movement",
    "device_type_match",
    "time_gap",
    "signal_pattern",
    # New 4
    "co_movement_duration",
    "time_of_day_similarity",
    "source_diversity_score",
    "wifi_probe_correlation",
]


def build_extended_features(
    *,
    distance: float = 0.0,
    rssi_delta: float = 0.0,
    co_movement: float = 0.0,
    device_type_match_score: float = 0.0,
    time_gap: float = 0.0,
    signal_pattern: float = 0.0,
    co_movement_duration: float = 0.0,
    time_of_day_similarity: float = 0.0,
    source_diversity_score: float = 0.0,
    wifi_probe_correlation: float = 0.0,
) -> dict[str, float]:
    """Build a complete extended feature dict for correlation scoring.

    Constructs all 10 features (original 6 + new 4) into a single dict.
    """
    return {
        "distance": distance,
        "rssi_delta": rssi_delta,
        "co_movement": co_movement,
        "device_type_match": device_type_match_score,
        "time_gap": time_gap,
        "signal_pattern": signal_pattern,
        "co_movement_duration": co_movement_duration,
        "time_of_day_similarity": time_of_day_similarity,
        "source_diversity_score": source_diversity_score,
        "wifi_probe_correlation": wifi_probe_correlation,
    }
