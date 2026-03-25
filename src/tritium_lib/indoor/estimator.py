# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Position estimation from live RSSI using k-NN fingerprint matching.

Given a :class:`FingerprintDB` of reference fingerprints and a live RSSI
observation, the :class:`PositionEstimator` finds the *k* most similar
reference fingerprints and computes a weighted centroid as the estimated
position.

Weighting scheme: inverse RSSI-distance — closer matches in RSSI-space
pull harder on the centroid. This gives sub-room accuracy in well-surveyed
buildings (typically 2-5 m with 20+ reference points per floor).

Pure Python — no numpy required.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

from .fingerprint import Fingerprint, FingerprintDB


# ---------------------------------------------------------------------------
# Position result
# ---------------------------------------------------------------------------

@dataclass
class PositionResult:
    """Estimated indoor position with metadata.

    Attributes:
        x: Estimated X-coordinate in local metres.
        y: Estimated Y-coordinate in local metres.
        floor: Estimated floor level.
        confidence: Confidence score 0.0-1.0 (higher = better).
        accuracy_m: Estimated accuracy radius in metres.
        k_used: Number of neighbours that contributed.
        method: Estimation method label.
        nearest_label: Label of the single closest reference fingerprint.
        timestamp: When the estimate was produced (epoch seconds).
    """
    x: float
    y: float
    floor: int = 0
    confidence: float = 0.0
    accuracy_m: float = float("inf")
    k_used: int = 0
    method: str = "knn_fingerprint"
    nearest_label: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        """Export as a JSON-serialisable dict."""
        return {
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "floor": self.floor,
            "confidence": round(self.confidence, 4),
            "accuracy_m": round(self.accuracy_m, 2),
            "k_used": self.k_used,
            "method": self.method,
            "nearest_label": self.nearest_label,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# PositionEstimator
# ---------------------------------------------------------------------------

class PositionEstimator:
    """Estimate indoor position from live RSSI via k-NN fingerprint matching.

    Args:
        db: Reference fingerprint database.
        k: Number of nearest neighbours to use (default 3).
        use_weighted_distance: If True, penalise missing APs in distance
            calculation. Default False (only common APs contribute).
        min_common_aps: Minimum number of common APs required for a match
            to be considered valid. Default 2.
        max_rssi_distance: Maximum RSSI-space distance to accept a match.
            Default inf (no limit).
    """

    def __init__(
        self,
        db: FingerprintDB,
        k: int = 3,
        use_weighted_distance: bool = False,
        min_common_aps: int = 2,
        max_rssi_distance: float = float("inf"),
    ) -> None:
        self._db = db
        self._k = max(1, k)
        self._use_weighted = use_weighted_distance
        self._min_common_aps = max(1, min_common_aps)
        self._max_distance = max_rssi_distance

    # -- Configuration ------------------------------------------------------

    @property
    def k(self) -> int:
        return self._k

    @k.setter
    def k(self, value: int) -> None:
        self._k = max(1, value)

    @property
    def db(self) -> FingerprintDB:
        return self._db

    @db.setter
    def db(self, value: FingerprintDB) -> None:
        self._db = value

    # -- Estimation ---------------------------------------------------------

    def estimate(
        self,
        live_rssi: dict[str, float],
        floor: Optional[int] = None,
    ) -> PositionResult:
        """Estimate position from a live RSSI observation.

        Args:
            live_rssi: Current RSSI mapping {ap_id: rssi_dbm}.
            floor: If set, restrict matching to this floor.

        Returns:
            :class:`PositionResult` with estimated coordinates. If no valid
            matches are found, returns (0, 0) with confidence 0.
        """
        if not live_rssi or self._db.count == 0:
            return PositionResult(x=0.0, y=0.0, confidence=0.0)

        # Pre-filter: only consider fingerprints sharing enough APs
        neighbours = self._db.find_nearest(
            live_rssi,
            k=self._k,
            floor=floor,
            use_weighted=self._use_weighted,
            max_distance=self._max_distance,
        )

        # Filter by minimum common APs
        valid: list[tuple[Fingerprint, float]] = []
        for fp, dist in neighbours:
            common = len(set(fp.rssi.keys()) & set(live_rssi.keys()))
            if common >= self._min_common_aps:
                valid.append((fp, dist))

        if not valid:
            return PositionResult(x=0.0, y=0.0, confidence=0.0)

        # Single match — return it directly
        if len(valid) == 1:
            fp, dist = valid[0]
            conf = _distance_to_confidence(dist)
            return PositionResult(
                x=fp.x,
                y=fp.y,
                floor=fp.floor,
                confidence=conf,
                accuracy_m=_estimate_accuracy(valid),
                k_used=1,
                nearest_label=fp.label,
            )

        # Weighted centroid from top-k matches
        return self._weighted_centroid(valid)

    def estimate_floor(self, live_rssi: dict[str, float]) -> Optional[int]:
        """Estimate which floor the device is on.

        Runs k-NN across all floors, then returns the floor that appears
        most often among the top matches.

        Args:
            live_rssi: Current RSSI mapping.

        Returns:
            Most likely floor level, or None if no matches.
        """
        if not live_rssi or self._db.count == 0:
            return None

        neighbours = self._db.find_nearest(
            live_rssi,
            k=self._k,
            floor=None,  # search all floors
            use_weighted=self._use_weighted,
            max_distance=self._max_distance,
        )

        if not neighbours:
            return None

        # Vote by inverse distance
        floor_scores: dict[int, float] = {}
        for fp, dist in neighbours:
            weight = 1.0 / (dist + 1e-6)
            floor_scores[fp.floor] = floor_scores.get(fp.floor, 0.0) + weight

        return max(floor_scores, key=floor_scores.get)  # type: ignore[arg-type]

    # -- Internal -----------------------------------------------------------

    def _weighted_centroid(
        self,
        matches: list[tuple[Fingerprint, float]],
    ) -> PositionResult:
        """Compute weighted centroid from k-NN matches.

        Weight = 1 / (distance^2 + epsilon). Squaring the distance makes
        closer matches dominate more strongly, which improves accuracy
        when the closest match is significantly better than the rest.
        """
        eps = 1e-6
        total_weight = 0.0
        wx = 0.0
        wy = 0.0
        floor_weights: dict[int, float] = {}

        for fp, dist in matches:
            weight = 1.0 / (dist ** 2 + eps)
            wx += fp.x * weight
            wy += fp.y * weight
            total_weight += weight
            floor_weights[fp.floor] = floor_weights.get(fp.floor, 0.0) + weight

        est_x = wx / total_weight
        est_y = wy / total_weight

        # Floor: highest weighted vote
        est_floor = max(floor_weights, key=floor_weights.get)  # type: ignore[arg-type]

        # Confidence based on best match distance and match count
        best_dist = matches[0][1]
        conf = _distance_to_confidence(best_dist)
        # Bonus for more matches (up to 20%)
        count_bonus = min(0.2, 0.05 * len(matches))
        conf = min(1.0, conf + count_bonus)

        # Nearest label
        nearest_label = matches[0][0].label

        return PositionResult(
            x=est_x,
            y=est_y,
            floor=est_floor,
            confidence=round(conf, 4),
            accuracy_m=_estimate_accuracy(matches),
            k_used=len(matches),
            nearest_label=nearest_label,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _distance_to_confidence(rssi_distance: float) -> float:
    """Map RSSI-space distance to a confidence score [0.0, 1.0].

    Uses a sigmoid-like decay: confidence drops from ~0.95 at distance 0
    to ~0.10 at distance 50 dBm.

    The formula: conf = 1 / (1 + (d / 15)^2)
    """
    if rssi_distance <= 0:
        return 0.95
    return 1.0 / (1.0 + (rssi_distance / 15.0) ** 2)


def _estimate_accuracy(
    matches: list[tuple[Fingerprint, float]],
) -> float:
    """Estimate position accuracy in metres from the spatial spread of matches.

    Computes the weighted standard deviation of match positions around
    the centroid. If matches are physically close together, the estimate
    is tight; if they are spread out, accuracy is poor.
    """
    if len(matches) <= 1:
        return 5.0  # default 5m for single match

    eps = 1e-6
    total_weight = 0.0
    wx = 0.0
    wy = 0.0

    for fp, dist in matches:
        w = 1.0 / (dist ** 2 + eps)
        wx += fp.x * w
        wy += fp.y * w
        total_weight += w

    cx = wx / total_weight
    cy = wy / total_weight

    # Weighted variance of physical positions
    var_sum = 0.0
    for fp, dist in matches:
        w = 1.0 / (dist ** 2 + eps)
        var_sum += w * ((fp.x - cx) ** 2 + (fp.y - cy) ** 2)
    variance = var_sum / total_weight

    # Accuracy ~ standard deviation, with minimum 1m
    accuracy = max(1.0, math.sqrt(variance))
    return round(accuracy, 2)
