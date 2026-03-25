# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Advanced threat assessment engine — multi-source intelligence fusion.

Combines multiple intelligence indicators (signal, movement, temporal,
association, history) into a unified threat score for targets and areas.
Builds on top of :mod:`~tritium_lib.intelligence.threat_model` (signal-level
scoring) and :mod:`~tritium_lib.intelligence.anomaly_engine` (behavioral baselines)
to provide higher-level threat assessment with predictive capability.

Key classes:

  - **ThreatIndicator** — a single indicator (behavioral, signal, temporal,
    association, or history) with a typed score.
  - **ThreatMatrix** — aggregates multiple indicators into a weighted threat
    score matrix.  Supports per-category weights and cross-indicator boosting.
  - **ThreatAssessmentEngine** — the top-level engine that runs ``assess_target``,
    ``assess_area``, and ``predict_threat``.

Indicator categories:

  1. **signal_anomaly** — unusual BLE/WiFi patterns (new devices, RSSI spikes,
     probe-request bursts, OUI mismatches).
  2. **movement_anomaly** — unusual speed, route deviation, dwell, stop-and-go.
  3. **temporal_anomaly** — unusual time-of-day activity, out-of-schedule
     appearances, abnormal inter-visit intervals.
  4. **association_anomaly** — unusual co-location patterns, device pairing
     anomalies, convoy detection.
  5. **history_factor** — previous threat level, repeat visits, prior alerts,
     escalation trajectory.

Usage::

    from tritium_lib.intelligence.threat_assessment import (
        ThreatAssessmentEngine,
        ThreatIndicator,
        ThreatMatrix,
    )

    engine = ThreatAssessmentEngine()

    # Feed target data
    engine.update_target("ble_aa:bb:cc", position=(50.0, 120.0), speed=1.2,
                         signal_strength=-65, hour_of_day=3,
                         co_located_targets=["wifi_dd:ee:ff"])

    # Full assessment
    result = engine.assess_target("ble_aa:bb:cc")
    print(result.composite_score, result.threat_level)

    # Area assessment
    area_result = engine.assess_area((0, 0, 200, 200))

    # Prediction
    prediction = engine.predict_threat("ble_aa:bb:cc", hours=4)
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from tritium_lib.intelligence.threat_model import (
    ThreatLevel,
    score_to_threat_level,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Indicator category names
INDICATOR_SIGNAL = "signal_anomaly"
INDICATOR_MOVEMENT = "movement_anomaly"
INDICATOR_TEMPORAL = "temporal_anomaly"
INDICATOR_ASSOCIATION = "association_anomaly"
INDICATOR_HISTORY = "history_factor"

ALL_INDICATOR_CATEGORIES = [
    INDICATOR_SIGNAL,
    INDICATOR_MOVEMENT,
    INDICATOR_TEMPORAL,
    INDICATOR_ASSOCIATION,
    INDICATOR_HISTORY,
]

# Default weights for indicator categories
DEFAULT_INDICATOR_WEIGHTS: dict[str, float] = {
    INDICATOR_SIGNAL: 0.25,
    INDICATOR_MOVEMENT: 0.25,
    INDICATOR_TEMPORAL: 0.15,
    INDICATOR_ASSOCIATION: 0.20,
    INDICATOR_HISTORY: 0.15,
}

# Cross-indicator boost: when multiple categories are elevated, overall
# threat is amplified.  The boost factor is applied per additional active
# category beyond the first.
CROSS_INDICATOR_BOOST = 0.10

# Minimum indicator score to consider "active" for cross-indicator boost
ACTIVE_INDICATOR_THRESHOLD = 0.3

# History decay half-life in seconds (how quickly old assessment influence fades)
HISTORY_DECAY_HALF_LIFE = 3600.0  # 1 hour

# Normal speed range for pedestrians / vehicles (m/s)
NORMAL_SPEED_PEDESTRIAN = (0.0, 2.5)
NORMAL_SPEED_VEHICLE = (0.0, 35.0)

# Normal hours of activity (6am - 10pm)
NORMAL_HOURS = set(range(6, 22))

# Co-location suspicion threshold: seeing the same pair together this many
# times in different zones becomes suspicious
COLOCATION_SUSPICION_THRESHOLD = 3


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class IndicatorCategory(str, Enum):
    """Typed indicator categories."""
    SIGNAL = INDICATOR_SIGNAL
    MOVEMENT = INDICATOR_MOVEMENT
    TEMPORAL = INDICATOR_TEMPORAL
    ASSOCIATION = INDICATOR_ASSOCIATION
    HISTORY = INDICATOR_HISTORY


@dataclass(slots=True)
class ThreatIndicator:
    """A single threat indicator contributing to a target's assessment.

    Attributes:
        category: Indicator category (signal_anomaly, movement_anomaly, etc).
        score: Indicator score 0.0 (benign) to 1.0 (maximum threat).
        source: What produced this indicator (detector name, algorithm, etc).
        detail: Human-readable explanation.
        confidence: How confident the detector is in this indicator (0.0-1.0).
        timestamp: When the indicator was generated (epoch seconds).
        target_id: Target this indicator applies to.
        raw_data: Optional dict of underlying data for debugging / audit.
    """
    category: str = ""
    score: float = 0.0
    source: str = ""
    detail: str = ""
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    target_id: str = ""
    raw_data: dict[str, Any] = field(default_factory=dict)

    def effective_score(self) -> float:
        """Score weighted by confidence."""
        return self.score * self.confidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "score": round(self.score, 4),
            "source": self.source,
            "detail": self.detail,
            "confidence": round(self.confidence, 4),
            "timestamp": self.timestamp,
            "target_id": self.target_id,
        }


@dataclass
class ThreatMatrix:
    """Aggregates indicators into a weighted threat score matrix.

    The matrix maintains per-category scores and computes a composite
    score using configurable weights and cross-indicator boosting.

    Attributes:
        target_id: The assessed target.
        category_scores: Per-category aggregated scores.
        category_indicators: Per-category list of contributing indicators.
        composite_score: Weighted aggregate with cross-indicator boost.
        threat_level: Derived discrete level.
        active_categories: Number of categories above the active threshold.
        cross_boost_applied: The cross-indicator boost that was applied.
        assessed_at: When the matrix was computed.
    """
    target_id: str = ""
    category_scores: dict[str, float] = field(default_factory=dict)
    category_indicators: dict[str, list[ThreatIndicator]] = field(
        default_factory=lambda: defaultdict(list)
    )
    composite_score: float = 0.0
    threat_level: ThreatLevel = ThreatLevel.GREEN
    active_categories: int = 0
    cross_boost_applied: float = 0.0
    assessed_at: float = field(default_factory=time.time)

    def compute(
        self,
        weights: dict[str, float] | None = None,
        boost_factor: float = CROSS_INDICATOR_BOOST,
        active_threshold: float = ACTIVE_INDICATOR_THRESHOLD,
    ) -> None:
        """Compute composite score from category scores.

        For each category, the score is the maximum effective score among
        its indicators.  The composite is a weighted average with a
        cross-indicator boost applied when multiple categories are elevated.
        """
        w = weights or DEFAULT_INDICATOR_WEIGHTS

        # Compute per-category scores (max effective score)
        for cat in ALL_INDICATOR_CATEGORIES:
            indicators = self.category_indicators.get(cat, [])
            if indicators:
                self.category_scores[cat] = max(
                    ind.effective_score() for ind in indicators
                )
            else:
                self.category_scores[cat] = 0.0

        # Count active categories
        self.active_categories = sum(
            1 for score in self.category_scores.values()
            if score >= active_threshold
        )

        # Weighted sum
        total_weight = 0.0
        weighted_sum = 0.0
        for cat, score in self.category_scores.items():
            weight = w.get(cat, 0.0)
            weighted_sum += score * weight
            total_weight += weight

        # Also account for categories with no indicators
        for cat, weight in w.items():
            if cat not in self.category_scores:
                total_weight += weight

        base_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Cross-indicator boost
        extra_categories = max(0, self.active_categories - 1)
        self.cross_boost_applied = boost_factor * extra_categories
        boosted = base_score + self.cross_boost_applied

        self.composite_score = max(0.0, min(1.0, boosted))
        self.threat_level = score_to_threat_level(self.composite_score)
        self.assessed_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "category_scores": {
                k: round(v, 4) for k, v in self.category_scores.items()
            },
            "composite_score": round(self.composite_score, 4),
            "threat_level": self.threat_level.value,
            "active_categories": self.active_categories,
            "cross_boost_applied": round(self.cross_boost_applied, 4),
            "assessed_at": self.assessed_at,
        }


@dataclass(slots=True)
class AreaAssessment:
    """Threat assessment for a geographic area.

    Attributes:
        bounds: (min_x, min_y, max_x, max_y) bounding box.
        composite_score: Average threat score across all targets in the area.
        max_score: Maximum threat score among targets.
        threat_level: Derived from composite score.
        target_count: Number of targets in the area.
        threat_distribution: Count of targets per threat level.
        highest_threat_targets: Top N target IDs by score.
        assessed_at: Timestamp.
    """
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    composite_score: float = 0.0
    max_score: float = 0.0
    threat_level: ThreatLevel = ThreatLevel.GREEN
    target_count: int = 0
    threat_distribution: dict[str, int] = field(default_factory=dict)
    highest_threat_targets: list[dict[str, Any]] = field(default_factory=list)
    assessed_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bounds": list(self.bounds),
            "composite_score": round(self.composite_score, 4),
            "max_score": round(self.max_score, 4),
            "threat_level": self.threat_level.value,
            "target_count": self.target_count,
            "threat_distribution": self.threat_distribution,
            "highest_threat_targets": self.highest_threat_targets,
            "assessed_at": self.assessed_at,
        }


@dataclass(slots=True)
class ThreatPrediction:
    """Predicted future threat level for a target.

    Attributes:
        target_id: The target.
        current_score: Current composite threat score.
        predicted_score: Predicted threat score at the horizon.
        predicted_level: Predicted threat level.
        hours_ahead: Prediction horizon in hours.
        trend: "escalating", "de-escalating", or "stable".
        trend_slope: Rate of score change per hour.
        confidence: Confidence in the prediction (0.0-1.0).
        contributing_factors: Key factors driving the prediction.
    """
    target_id: str = ""
    current_score: float = 0.0
    predicted_score: float = 0.0
    predicted_level: ThreatLevel = ThreatLevel.GREEN
    hours_ahead: float = 0.0
    trend: str = "stable"
    trend_slope: float = 0.0
    confidence: float = 0.0
    contributing_factors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "current_score": round(self.current_score, 4),
            "predicted_score": round(self.predicted_score, 4),
            "predicted_level": self.predicted_level.value,
            "hours_ahead": self.hours_ahead,
            "trend": self.trend,
            "trend_slope": round(self.trend_slope, 6),
            "confidence": round(self.confidence, 4),
            "contributing_factors": self.contributing_factors,
        }


# ---------------------------------------------------------------------------
# Internal: target observation record
# ---------------------------------------------------------------------------

@dataclass
class _TargetObservation:
    """Internal snapshot of a target's state at a point in time."""
    target_id: str = ""
    position: tuple[float, float] = (0.0, 0.0)
    speed: float = 0.0
    signal_strength: float = 0.0
    hour_of_day: int = 0
    co_located_targets: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    zone_id: str = ""


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _mean_std(values: list[float]) -> tuple[float, float]:
    """Compute mean and population std. Returns (0, 0) if empty."""
    if not values:
        return 0.0, 0.0
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return mean, math.sqrt(variance) if variance > 0 else 0.0


def _linear_regression_slope(values: list[float]) -> float:
    """Compute slope of a simple linear regression on values indexed 0..n-1.

    Returns the slope (change per step). Returns 0 if fewer than 2 values.
    """
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    if abs(denominator) < 1e-12:
        return 0.0
    return numerator / denominator


# ---------------------------------------------------------------------------
# ThreatAssessmentEngine
# ---------------------------------------------------------------------------

class ThreatAssessmentEngine:
    """Advanced threat assessment combining multiple intelligence sources.

    Maintains observation history for each target and computes indicators
    across five categories: signal, movement, temporal, association, and
    history.  Provides three assessment modes:

    - ``assess_target`` — full assessment for a single target.
    - ``assess_area`` — overall threat level for a geographic bounding box.
    - ``predict_threat`` — predict future threat level for a target.

    Thread-safe.  All public methods acquire the internal lock as needed.

    Parameters
    ----------
    weights:
        Custom indicator category weights.
    max_observations:
        Maximum observations retained per target.
    baseline_min_observations:
        Minimum observations before anomaly indicators activate.
    normal_hours:
        Set of hour-of-day values considered "normal activity hours".
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        max_observations: int = 500,
        baseline_min_observations: int = 5,
        normal_hours: set[int] | None = None,
    ) -> None:
        self._weights = weights or dict(DEFAULT_INDICATOR_WEIGHTS)
        self._max_observations = max_observations
        self._min_baseline = baseline_min_observations
        self._normal_hours = normal_hours if normal_hours is not None else set(NORMAL_HOURS)
        self._lock = threading.Lock()

        # Per-target observation history: target_id -> list[_TargetObservation]
        self._observations: dict[str, list[_TargetObservation]] = defaultdict(list)

        # Assessment history for prediction: target_id -> list[(timestamp, score)]
        self._assessment_history: dict[str, list[tuple[float, float]]] = defaultdict(list)
        self._max_assessment_history = 200

        # Co-location tracker: (target_a, target_b) -> list[zone_ids]
        self._co_locations: dict[tuple[str, str], list[str]] = defaultdict(list)

        # Previous assessment cache: target_id -> ThreatMatrix
        self._cache: dict[str, ThreatMatrix] = {}

        # Counters
        self._total_assessments = 0
        self._total_updates = 0

    # ------------------------------------------------------------------
    # Target data ingestion
    # ------------------------------------------------------------------

    def update_target(
        self,
        target_id: str,
        *,
        position: tuple[float, float] = (0.0, 0.0),
        speed: float = 0.0,
        signal_strength: float = 0.0,
        hour_of_day: int | None = None,
        co_located_targets: list[str] | None = None,
        timestamp: float | None = None,
        zone_id: str = "",
    ) -> None:
        """Record a target observation.

        Call this for each target update to build the assessment baselines.
        """
        ts = timestamp or time.time()
        if hour_of_day is None:
            hour_of_day = time.localtime(ts).tm_hour

        obs = _TargetObservation(
            target_id=target_id,
            position=position,
            speed=speed,
            signal_strength=signal_strength,
            hour_of_day=hour_of_day,
            co_located_targets=list(co_located_targets) if co_located_targets else [],
            timestamp=ts,
            zone_id=zone_id,
        )

        with self._lock:
            history = self._observations[target_id]
            history.append(obs)
            if len(history) > self._max_observations:
                self._observations[target_id] = history[-self._max_observations:]

            # Track co-locations
            for other_id in obs.co_located_targets:
                pair = tuple(sorted([target_id, other_id]))
                zone_key = zone_id or f"pos_{position[0]:.0f}_{position[1]:.0f}"
                coloc_list = self._co_locations[pair]
                coloc_list.append(zone_key)
                if len(coloc_list) > 100:
                    self._co_locations[pair] = coloc_list[-100:]

            # Invalidate cache
            self._cache.pop(target_id, None)
            self._total_updates += 1

    # ------------------------------------------------------------------
    # assess_target
    # ------------------------------------------------------------------

    def assess_target(self, target_id: str) -> ThreatMatrix:
        """Full threat assessment for a single target.

        Computes indicators across all five categories and combines them
        into a ThreatMatrix with composite score and threat level.

        Returns a ThreatMatrix (GREEN with zero scores if no data).
        """
        with self._lock:
            cached = self._cache.get(target_id)
            if cached is not None:
                return cached
            observations = list(self._observations.get(target_id, []))

        matrix = ThreatMatrix(target_id=target_id)

        if not observations:
            matrix.compute(weights=self._weights)
            return matrix

        # 1. Signal anomaly indicators
        signal_indicators = self._compute_signal_indicators(target_id, observations)
        for ind in signal_indicators:
            matrix.category_indicators[INDICATOR_SIGNAL].append(ind)

        # 2. Movement anomaly indicators
        movement_indicators = self._compute_movement_indicators(target_id, observations)
        for ind in movement_indicators:
            matrix.category_indicators[INDICATOR_MOVEMENT].append(ind)

        # 3. Temporal anomaly indicators
        temporal_indicators = self._compute_temporal_indicators(target_id, observations)
        for ind in temporal_indicators:
            matrix.category_indicators[INDICATOR_TEMPORAL].append(ind)

        # 4. Association anomaly indicators
        association_indicators = self._compute_association_indicators(target_id, observations)
        for ind in association_indicators:
            matrix.category_indicators[INDICATOR_ASSOCIATION].append(ind)

        # 5. History factor indicators
        history_indicators = self._compute_history_indicators(target_id)
        for ind in history_indicators:
            matrix.category_indicators[INDICATOR_HISTORY].append(ind)

        matrix.compute(weights=self._weights)

        # Store assessment in history and cache
        now = time.time()
        with self._lock:
            self._cache[target_id] = matrix
            self._total_assessments += 1
            ah = self._assessment_history[target_id]
            ah.append((now, matrix.composite_score))
            if len(ah) > self._max_assessment_history:
                self._assessment_history[target_id] = ah[-self._max_assessment_history:]

        return matrix

    # ------------------------------------------------------------------
    # assess_area
    # ------------------------------------------------------------------

    def assess_area(
        self,
        bounds: tuple[float, float, float, float],
    ) -> AreaAssessment:
        """Assess overall threat level for a geographic area.

        Parameters
        ----------
        bounds:
            (min_x, min_y, max_x, max_y) bounding box.

        Returns
        -------
        AreaAssessment
            Aggregated assessment over all targets within bounds.
        """
        min_x, min_y, max_x, max_y = bounds

        with self._lock:
            target_ids = list(self._observations.keys())

        # Find targets within bounds
        targets_in_area: list[str] = []
        for tid in target_ids:
            with self._lock:
                obs_list = self._observations.get(tid, [])
                if not obs_list:
                    continue
                latest = obs_list[-1]

            px, py = latest.position
            if min_x <= px <= max_x and min_y <= py <= max_y:
                targets_in_area.append(tid)

        if not targets_in_area:
            return AreaAssessment(
                bounds=bounds,
                assessed_at=time.time(),
            )

        # Assess each target
        matrices: list[ThreatMatrix] = []
        for tid in targets_in_area:
            matrices.append(self.assess_target(tid))

        scores = [m.composite_score for m in matrices]
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)

        # Threat level distribution
        distribution: dict[str, int] = {}
        for m in matrices:
            level = m.threat_level.value
            distribution[level] = distribution.get(level, 0) + 1

        # Top threats
        sorted_matrices = sorted(matrices, key=lambda m: m.composite_score, reverse=True)
        top_threats = [
            {"target_id": m.target_id, "score": round(m.composite_score, 4),
             "level": m.threat_level.value}
            for m in sorted_matrices[:5]
        ]

        return AreaAssessment(
            bounds=bounds,
            composite_score=avg_score,
            max_score=max_score,
            threat_level=score_to_threat_level(avg_score),
            target_count=len(targets_in_area),
            threat_distribution=distribution,
            highest_threat_targets=top_threats,
            assessed_at=time.time(),
        )

    # ------------------------------------------------------------------
    # predict_threat
    # ------------------------------------------------------------------

    def predict_threat(
        self,
        target_id: str,
        hours: float = 4.0,
    ) -> ThreatPrediction:
        """Predict future threat level for a target.

        Uses historical assessment scores to extrapolate a trend line.
        Also considers temporal patterns (time-of-day) and assessment
        velocity (how fast the score has been changing).

        Parameters
        ----------
        target_id:
            Target to predict for.
        hours:
            How many hours into the future to predict.

        Returns
        -------
        ThreatPrediction
        """
        hours = max(0.1, min(48.0, hours))

        # Get current assessment
        current_matrix = self.assess_target(target_id)
        current_score = current_matrix.composite_score

        with self._lock:
            history = list(self._assessment_history.get(target_id, []))
            observations = list(self._observations.get(target_id, []))

        if len(history) < 2:
            # Not enough history to predict — return stable at current level
            return ThreatPrediction(
                target_id=target_id,
                current_score=current_score,
                predicted_score=current_score,
                predicted_level=score_to_threat_level(current_score),
                hours_ahead=hours,
                trend="stable",
                trend_slope=0.0,
                confidence=0.1,
                contributing_factors=["insufficient_history"],
            )

        # Extract scores from history for trend analysis
        scores = [score for _, score in history]

        # Compute trend slope (score change per observation step)
        slope = _linear_regression_slope(scores)

        # Convert slope to per-hour rate
        # Estimate time span per step from history timestamps
        if len(history) >= 2:
            total_time = history[-1][0] - history[0][0]
            steps = len(history) - 1
            seconds_per_step = total_time / steps if steps > 0 else 3600.0
            slope_per_hour = slope / (seconds_per_step / 3600.0) if seconds_per_step > 0 else 0.0
        else:
            slope_per_hour = 0.0

        # Project forward
        projected = current_score + slope_per_hour * hours
        projected = max(0.0, min(1.0, projected))

        # Temporal factor: if predicted hours land in unusual time,
        # add a small boost
        now = time.time()
        future_hour = time.localtime(now + hours * 3600).tm_hour
        temporal_boost = 0.0
        contributing_factors: list[str] = []

        if future_hour not in self._normal_hours:
            temporal_boost = 0.05
            contributing_factors.append("predicted_time_outside_normal_hours")

        # Activity rate factor: recent observations more frequent = higher confidence
        recent_obs = [o for o in observations if (now - o.timestamp) < 3600]
        if len(recent_obs) > 10:
            contributing_factors.append("high_recent_activity")
        elif len(recent_obs) == 0:
            contributing_factors.append("no_recent_observations")

        projected_with_temporal = max(0.0, min(1.0, projected + temporal_boost))

        # Determine trend
        if slope_per_hour > 0.01:
            trend = "escalating"
            contributing_factors.append("score_trending_up")
        elif slope_per_hour < -0.01:
            trend = "de-escalating"
            contributing_factors.append("score_trending_down")
        else:
            trend = "stable"

        # Confidence based on data quality
        data_points = len(history)
        confidence = min(1.0, data_points / 20.0)
        # Reduce confidence for distant predictions
        if hours > 8:
            confidence *= 0.7
        elif hours > 4:
            confidence *= 0.85

        return ThreatPrediction(
            target_id=target_id,
            current_score=current_score,
            predicted_score=projected_with_temporal,
            predicted_level=score_to_threat_level(projected_with_temporal),
            hours_ahead=hours,
            trend=trend,
            trend_slope=slope_per_hour,
            confidence=confidence,
            contributing_factors=contributing_factors,
        )

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_all_assessments(self) -> list[ThreatMatrix]:
        """Assess all known targets. Returns sorted by score descending."""
        with self._lock:
            target_ids = list(self._observations.keys())

        matrices = [self.assess_target(tid) for tid in target_ids]
        matrices.sort(key=lambda m: m.composite_score, reverse=True)
        return matrices

    def get_targets_by_level(self, level: ThreatLevel) -> list[str]:
        """Return target IDs matching a specific threat level."""
        return [
            m.target_id for m in self.get_all_assessments()
            if m.threat_level == level
        ]

    def get_stats(self) -> dict[str, Any]:
        """Return engine statistics."""
        with self._lock:
            target_count = len(self._observations)
            total_obs = sum(len(v) for v in self._observations.values())
        return {
            "target_count": target_count,
            "total_observations": total_obs,
            "total_assessments": self._total_assessments,
            "total_updates": self._total_updates,
            "weights": dict(self._weights),
            "baseline_min_observations": self._min_baseline,
        }

    def clear(self, target_id: str | None = None) -> None:
        """Clear data. If target_id given, clear only that target."""
        with self._lock:
            if target_id:
                self._observations.pop(target_id, None)
                self._assessment_history.pop(target_id, None)
                self._cache.pop(target_id, None)
                # Clean co-locations involving this target
                keys_to_remove = [
                    k for k in self._co_locations if target_id in k
                ]
                for k in keys_to_remove:
                    del self._co_locations[k]
            else:
                self._observations.clear()
                self._assessment_history.clear()
                self._cache.clear()
                self._co_locations.clear()
                self._total_assessments = 0
                self._total_updates = 0

    # ------------------------------------------------------------------
    # Indicator computation: Signal Anomaly
    # ------------------------------------------------------------------

    def _compute_signal_indicators(
        self,
        target_id: str,
        observations: list[_TargetObservation],
    ) -> list[ThreatIndicator]:
        """Detect signal anomalies from RSSI / signal strength patterns."""
        indicators: list[ThreatIndicator] = []
        if len(observations) < self._min_baseline:
            return indicators

        # Gather signal strengths
        strengths = [o.signal_strength for o in observations if o.signal_strength != 0.0]
        if len(strengths) < self._min_baseline:
            return indicators

        mean_str, std_str = _mean_std(strengths)
        latest = observations[-1]

        if latest.signal_strength != 0.0 and std_str > 0:
            deviation = abs(latest.signal_strength - mean_str) / std_str
            if deviation >= 2.0:
                score = min(1.0, deviation / 5.0)
                indicators.append(ThreatIndicator(
                    category=INDICATOR_SIGNAL,
                    score=score,
                    source="signal_strength_anomaly",
                    detail=(
                        f"Signal strength {latest.signal_strength:.1f} deviates "
                        f"{deviation:.1f} sigma from baseline {mean_str:.1f}"
                    ),
                    confidence=min(1.0, len(strengths) / 20.0),
                    timestamp=latest.timestamp,
                    target_id=target_id,
                    raw_data={
                        "observed": latest.signal_strength,
                        "mean": mean_str,
                        "std": std_str,
                        "deviation_sigma": deviation,
                    },
                ))

        # Signal variability: high variance in recent readings is suspicious
        if len(strengths) >= 10:
            recent = strengths[-10:]
            _, recent_std = _mean_std(recent)
            if std_str > 0 and recent_std > std_str * 2.0:
                score = min(1.0, (recent_std / std_str) / 5.0)
                indicators.append(ThreatIndicator(
                    category=INDICATOR_SIGNAL,
                    score=score,
                    source="signal_variability",
                    detail=(
                        f"Recent signal variability (std={recent_std:.2f}) "
                        f"is {recent_std/std_str:.1f}x baseline ({std_str:.2f})"
                    ),
                    confidence=0.7,
                    timestamp=latest.timestamp,
                    target_id=target_id,
                ))

        return indicators

    # ------------------------------------------------------------------
    # Indicator computation: Movement Anomaly
    # ------------------------------------------------------------------

    def _compute_movement_indicators(
        self,
        target_id: str,
        observations: list[_TargetObservation],
    ) -> list[ThreatIndicator]:
        """Detect movement anomalies from speed, route, and dwell patterns."""
        indicators: list[ThreatIndicator] = []
        if len(observations) < self._min_baseline:
            return indicators

        speeds = [o.speed for o in observations]
        mean_spd, std_spd = _mean_std(speeds)
        latest = observations[-1]

        # Speed anomaly
        if std_spd > 0:
            deviation = abs(latest.speed - mean_spd) / std_spd
            if deviation >= 2.5:
                score = min(1.0, deviation / 5.0)
                indicators.append(ThreatIndicator(
                    category=INDICATOR_MOVEMENT,
                    score=score,
                    source="speed_anomaly",
                    detail=(
                        f"Speed {latest.speed:.2f} m/s deviates "
                        f"{deviation:.1f} sigma from baseline {mean_spd:.2f}"
                    ),
                    confidence=min(1.0, len(speeds) / 20.0),
                    timestamp=latest.timestamp,
                    target_id=target_id,
                    raw_data={
                        "observed_speed": latest.speed,
                        "mean_speed": mean_spd,
                        "std_speed": std_spd,
                        "deviation_sigma": deviation,
                    },
                ))

        # Dwell detection: very low speed (near zero) for extended periods
        if len(observations) >= 5:
            recent = observations[-5:]
            stationary_count = sum(1 for o in recent if o.speed < 0.3)
            if stationary_count >= 4:
                # Calculate dwell duration
                dwell_start = recent[0].timestamp
                dwell_duration = latest.timestamp - dwell_start
                if dwell_duration > 300:  # 5+ minutes
                    score = min(1.0, dwell_duration / 3600.0)
                    indicators.append(ThreatIndicator(
                        category=INDICATOR_MOVEMENT,
                        score=score,
                        source="dwell_anomaly",
                        detail=(
                            f"Target stationary for {dwell_duration/60:.1f} min "
                            f"({stationary_count}/5 recent observations near-zero speed)"
                        ),
                        confidence=0.8,
                        timestamp=latest.timestamp,
                        target_id=target_id,
                        raw_data={
                            "dwell_seconds": dwell_duration,
                            "stationary_ratio": stationary_count / len(recent),
                        },
                    ))

        # Route deviation: position jumps (teleportation)
        if len(observations) >= 2:
            prev = observations[-2]
            dx = latest.position[0] - prev.position[0]
            dy = latest.position[1] - prev.position[1]
            jump_distance = math.sqrt(dx * dx + dy * dy)
            dt = max(0.1, latest.timestamp - prev.timestamp)
            implied_speed = jump_distance / dt

            # If implied speed is much higher than observed speeds
            if mean_spd > 0 and implied_speed > mean_spd * 5.0:
                score = min(1.0, implied_speed / (mean_spd * 10.0))
                indicators.append(ThreatIndicator(
                    category=INDICATOR_MOVEMENT,
                    score=score,
                    source="position_jump",
                    detail=(
                        f"Position jumped {jump_distance:.1f}m in {dt:.1f}s "
                        f"(implied {implied_speed:.1f} m/s vs avg {mean_spd:.1f})"
                    ),
                    confidence=0.6,
                    timestamp=latest.timestamp,
                    target_id=target_id,
                    raw_data={
                        "jump_distance": jump_distance,
                        "implied_speed": implied_speed,
                        "dt_seconds": dt,
                    },
                ))

        return indicators

    # ------------------------------------------------------------------
    # Indicator computation: Temporal Anomaly
    # ------------------------------------------------------------------

    def _compute_temporal_indicators(
        self,
        target_id: str,
        observations: list[_TargetObservation],
    ) -> list[ThreatIndicator]:
        """Detect temporal anomalies — unusual time-of-day activity."""
        indicators: list[ThreatIndicator] = []
        if not observations:
            return indicators

        latest = observations[-1]

        # Activity outside normal hours
        if latest.hour_of_day not in self._normal_hours:
            # Score based on how deep into abnormal hours
            # Midnight-4am is more suspicious than 10pm-midnight
            if 0 <= latest.hour_of_day <= 4:
                score = 0.6
            elif 22 <= latest.hour_of_day <= 23:
                score = 0.3
            else:
                score = 0.4

            indicators.append(ThreatIndicator(
                category=INDICATOR_TEMPORAL,
                score=score,
                source="off_hours_activity",
                detail=(
                    f"Activity at hour {latest.hour_of_day:02d}:00 "
                    f"(outside normal hours {min(self._normal_hours)}-{max(self._normal_hours)})"
                ),
                confidence=0.9,
                timestamp=latest.timestamp,
                target_id=target_id,
            ))

        # Hour distribution anomaly: target usually active at certain hours
        if len(observations) >= self._min_baseline:
            hour_counts: dict[int, int] = defaultdict(int)
            for obs in observations:
                hour_counts[obs.hour_of_day] += 1

            total = len(observations)
            latest_hour_count = hour_counts.get(latest.hour_of_day, 0)
            hour_frequency = latest_hour_count / total

            # If this hour accounts for less than 2% of observations and
            # we have enough data, it is unusual for this target
            if hour_frequency < 0.02 and total >= 20:
                score = min(1.0, (0.05 - hour_frequency) / 0.05)
                indicators.append(ThreatIndicator(
                    category=INDICATOR_TEMPORAL,
                    score=score,
                    source="unusual_hour_for_target",
                    detail=(
                        f"Hour {latest.hour_of_day:02d}:00 is unusual for this target "
                        f"({hour_frequency*100:.1f}% of observations, {latest_hour_count}/{total})"
                    ),
                    confidence=min(1.0, total / 50.0),
                    timestamp=latest.timestamp,
                    target_id=target_id,
                ))

        # Inter-visit interval anomaly
        if len(observations) >= 3:
            intervals = []
            for i in range(1, len(observations)):
                intervals.append(observations[i].timestamp - observations[i - 1].timestamp)

            mean_interval, std_interval = _mean_std(intervals)
            if len(intervals) >= 2 and std_interval > 0:
                latest_interval = intervals[-1]
                deviation = abs(latest_interval - mean_interval) / std_interval
                if deviation >= 3.0:
                    score = min(1.0, deviation / 5.0)
                    direction = "longer" if latest_interval > mean_interval else "shorter"
                    indicators.append(ThreatIndicator(
                        category=INDICATOR_TEMPORAL,
                        score=score,
                        source="inter_visit_interval",
                        detail=(
                            f"Inter-observation interval {latest_interval:.0f}s is "
                            f"{deviation:.1f} sigma {direction} than normal "
                            f"({mean_interval:.0f}s +/- {std_interval:.0f}s)"
                        ),
                        confidence=min(1.0, len(intervals) / 20.0),
                        timestamp=latest.timestamp if observations else time.time(),
                        target_id=target_id,
                    ))

        return indicators

    # ------------------------------------------------------------------
    # Indicator computation: Association Anomaly
    # ------------------------------------------------------------------

    def _compute_association_indicators(
        self,
        target_id: str,
        observations: list[_TargetObservation],
    ) -> list[ThreatIndicator]:
        """Detect association anomalies — unusual co-location patterns."""
        indicators: list[ThreatIndicator] = []

        with self._lock:
            # Find all co-location pairs involving this target
            relevant_pairs: dict[str, list[str]] = {}
            for (a, b), zones in self._co_locations.items():
                if a == target_id:
                    relevant_pairs[b] = list(zones)
                elif b == target_id:
                    relevant_pairs[a] = list(zones)

        for other_id, zone_list in relevant_pairs.items():
            unique_zones = len(set(zone_list))
            total_sightings = len(zone_list)

            # Suspicious if seen together in multiple distinct zones
            if unique_zones >= COLOCATION_SUSPICION_THRESHOLD:
                score = min(1.0, unique_zones / 10.0)
                indicators.append(ThreatIndicator(
                    category=INDICATOR_ASSOCIATION,
                    score=score,
                    source="co_location_pattern",
                    detail=(
                        f"Co-located with {other_id} in {unique_zones} distinct zones "
                        f"({total_sightings} total sightings)"
                    ),
                    confidence=min(1.0, total_sightings / 10.0),
                    timestamp=time.time(),
                    target_id=target_id,
                    raw_data={
                        "other_target": other_id,
                        "unique_zones": unique_zones,
                        "total_sightings": total_sightings,
                    },
                ))

        # New association: target suddenly appearing with many others it
        # has not been seen with before
        if observations:
            latest = observations[-1]
            if len(latest.co_located_targets) >= 3:
                known_associates = set()
                for obs in observations[:-1]:
                    known_associates.update(obs.co_located_targets)

                new_associates = [
                    t for t in latest.co_located_targets
                    if t not in known_associates
                ]
                if len(new_associates) >= 2:
                    score = min(1.0, len(new_associates) / 5.0)
                    indicators.append(ThreatIndicator(
                        category=INDICATOR_ASSOCIATION,
                        score=score,
                        source="new_group_formation",
                        detail=(
                            f"Appeared with {len(new_associates)} previously unseen "
                            f"targets: {new_associates[:3]}"
                        ),
                        confidence=0.6,
                        timestamp=latest.timestamp,
                        target_id=target_id,
                    ))

        return indicators

    # ------------------------------------------------------------------
    # Indicator computation: History Factor
    # ------------------------------------------------------------------

    def _compute_history_indicators(
        self,
        target_id: str,
    ) -> list[ThreatIndicator]:
        """Compute history-based indicators from prior assessments."""
        indicators: list[ThreatIndicator] = []

        with self._lock:
            history = list(self._assessment_history.get(target_id, []))

        if len(history) < 2:
            return indicators

        scores = [score for _, score in history]
        timestamps = [ts for ts, _ in history]

        # Previous peak threat level (worst-case memory)
        peak_score = max(scores)
        if peak_score >= 0.4:
            # History of elevated threat persists
            recency = time.time() - timestamps[scores.index(peak_score)]
            decay = 0.5 ** (recency / HISTORY_DECAY_HALF_LIFE)
            decayed_score = peak_score * decay

            if decayed_score >= 0.1:
                indicators.append(ThreatIndicator(
                    category=INDICATOR_HISTORY,
                    score=decayed_score,
                    source="historical_peak",
                    detail=(
                        f"Historical peak threat {peak_score:.2f} "
                        f"({recency/3600:.1f}h ago, decayed to {decayed_score:.2f})"
                    ),
                    confidence=0.8,
                    timestamp=time.time(),
                    target_id=target_id,
                ))

        # Escalation trajectory: score trending upward
        if len(scores) >= 3:
            recent_scores = scores[-min(10, len(scores)):]
            slope = _linear_regression_slope(recent_scores)
            if slope > 0.02:
                score = min(1.0, slope * 10)
                indicators.append(ThreatIndicator(
                    category=INDICATOR_HISTORY,
                    score=score,
                    source="escalation_trajectory",
                    detail=(
                        f"Threat score trending up (slope={slope:.4f}/step "
                        f"over {len(recent_scores)} assessments)"
                    ),
                    confidence=min(1.0, len(recent_scores) / 10.0),
                    timestamp=time.time(),
                    target_id=target_id,
                ))

        # Repeat visit indicator: many assessments = persistent presence
        if len(history) >= 10:
            time_span = timestamps[-1] - timestamps[0]
            if time_span > 0:
                rate = len(history) / (time_span / 3600.0)  # assessments per hour
                if rate > 5.0:
                    score = min(1.0, rate / 20.0)
                    indicators.append(ThreatIndicator(
                        category=INDICATOR_HISTORY,
                        score=score,
                        source="persistent_presence",
                        detail=(
                            f"Assessed {len(history)} times in "
                            f"{time_span/3600:.1f}h ({rate:.1f}/hour)"
                        ),
                        confidence=0.7,
                        timestamp=time.time(),
                        target_id=target_id,
                    ))

        return indicators
