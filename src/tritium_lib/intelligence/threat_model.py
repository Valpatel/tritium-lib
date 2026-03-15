# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Unified threat assessment engine — ThreatModel.

Combines multiple threat signals into a composite threat score per target:

  1. Behavioral patterns — loitering, probing, revisiting, casing
  2. Threat feed matches — known-bad indicators (MAC, SSID, OUI)
  3. Device classification — unknown/suspicious device type scoring
  4. Zone violations — restricted zone entry, geofence breaches
  5. Operator feedback — manual confirm/deny adjustments

Each signal produces a sub-score (0.0-1.0). The ThreatModel aggregates
them with configurable weights into a final composite score and threat
level (GREEN/YELLOW/ORANGE/RED/CRITICAL).

Usage::

    from tritium_lib.intelligence.threat_model import ThreatModel, ThreatSignal

    model = ThreatModel()
    model.add_signal(ThreatSignal(
        signal_type="behavior",
        score=0.7,
        source="loiter_detector",
        detail="Target loitered 12min in restricted zone",
    ))
    model.add_signal(ThreatSignal(
        signal_type="threat_feed",
        score=0.9,
        source="stix_feed",
        detail="MAC matches known surveillance device",
    ))
    assessment = model.assess("ble_aa:bb:cc:dd:ee:ff")
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ThreatLevel(str, Enum):
    """Discrete threat levels derived from composite score."""
    GREEN = "GREEN"        # 0.0 - 0.2  — No threat
    YELLOW = "YELLOW"      # 0.2 - 0.4  — Low concern
    ORANGE = "ORANGE"      # 0.4 - 0.6  — Moderate threat
    RED = "RED"            # 0.6 - 0.8  — High threat
    CRITICAL = "CRITICAL"  # 0.8 - 1.0  — Immediate action required


# Default weights for each signal type
DEFAULT_SIGNAL_WEIGHTS: dict[str, float] = {
    "behavior": 0.25,
    "threat_feed": 0.30,
    "classification": 0.15,
    "zone_violation": 0.20,
    "operator_feedback": 0.10,
}

# Threat level thresholds
THREAT_THRESHOLDS: list[tuple[float, ThreatLevel]] = [
    (0.8, ThreatLevel.CRITICAL),
    (0.6, ThreatLevel.RED),
    (0.4, ThreatLevel.ORANGE),
    (0.2, ThreatLevel.YELLOW),
    (0.0, ThreatLevel.GREEN),
]


@dataclass(slots=True)
class ThreatSignal:
    """A single threat signal contributing to a target's assessment.

    Attributes:
        signal_type: Category — behavior, threat_feed, classification,
                     zone_violation, or operator_feedback.
        score: Signal score 0.0 (benign) to 1.0 (maximum threat).
        source: What produced this signal (detector name, feed ID, etc).
        detail: Human-readable explanation.
        timestamp: When the signal was generated (epoch seconds).
        ttl_seconds: How long this signal remains valid (0 = forever).
        target_id: Target this signal applies to.
    """
    signal_type: str = ""
    score: float = 0.0
    source: str = ""
    detail: str = ""
    timestamp: float = field(default_factory=time.time)
    ttl_seconds: float = 0.0
    target_id: str = ""

    def is_expired(self, now: Optional[float] = None) -> bool:
        """Check if this signal has expired based on TTL."""
        if self.ttl_seconds <= 0:
            return False
        now = now or time.time()
        return (now - self.timestamp) > self.ttl_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_type": self.signal_type,
            "score": round(self.score, 4),
            "source": self.source,
            "detail": self.detail,
            "timestamp": self.timestamp,
            "ttl_seconds": self.ttl_seconds,
            "target_id": self.target_id,
        }


@dataclass(slots=True)
class ThreatAssessment:
    """Result of a threat assessment for a single target.

    Attributes:
        target_id: The assessed target.
        composite_score: Weighted aggregate score (0.0 - 1.0).
        threat_level: Derived discrete level.
        sub_scores: Per-signal-type aggregated scores.
        signal_count: Total number of active signals.
        top_signals: The highest-scoring signals driving the assessment.
        assessed_at: When this assessment was computed.
    """
    target_id: str = ""
    composite_score: float = 0.0
    threat_level: ThreatLevel = ThreatLevel.GREEN
    sub_scores: dict[str, float] = field(default_factory=dict)
    signal_count: int = 0
    top_signals: list[dict[str, Any]] = field(default_factory=list)
    assessed_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "composite_score": round(self.composite_score, 4),
            "threat_level": self.threat_level.value,
            "sub_scores": {k: round(v, 4) for k, v in self.sub_scores.items()},
            "signal_count": self.signal_count,
            "top_signals": self.top_signals,
            "assessed_at": self.assessed_at,
        }


class ThreatModel:
    """Unified threat assessment engine.

    Maintains a per-target signal buffer. On ``assess(target_id)``, it
    aggregates active (non-expired) signals by type, applies configurable
    weights, and returns a ThreatAssessment with composite score and
    threat level.

    Thread-safe via a simple lock (the primary consumers are single-threaded
    event loops, but plugin threads may call ``add_signal`` concurrently).
    """

    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
        decay_enabled: bool = True,
        max_signals_per_target: int = 100,
    ) -> None:
        self._weights = weights or dict(DEFAULT_SIGNAL_WEIGHTS)
        self._decay_enabled = decay_enabled
        self._max_signals = max_signals_per_target
        # target_id -> list of ThreatSignal
        self._signals: dict[str, list[ThreatSignal]] = {}
        self._lock = __import__("threading").Lock()
        # Assessment cache
        self._cache: dict[str, ThreatAssessment] = {}
        self._assessment_count: int = 0

    def add_signal(self, signal: ThreatSignal) -> None:
        """Add a threat signal for a target.

        Clamps score to [0.0, 1.0]. Validates signal_type against
        known weight keys (unknown types are accepted but get weight 0.0).
        """
        signal.score = max(0.0, min(1.0, signal.score))
        if not signal.target_id:
            return

        with self._lock:
            if signal.target_id not in self._signals:
                self._signals[signal.target_id] = []

            signals = self._signals[signal.target_id]
            signals.append(signal)

            # Trim old signals if over limit
            if len(signals) > self._max_signals:
                signals.sort(key=lambda s: s.timestamp)
                self._signals[signal.target_id] = signals[-self._max_signals:]

            # Invalidate cache
            self._cache.pop(signal.target_id, None)

    def assess(self, target_id: str) -> ThreatAssessment:
        """Compute a threat assessment for a target.

        Aggregates all active (non-expired) signals by type, computes
        weighted sub-scores, and derives the composite score and threat level.

        For each signal type, the sub-score is the maximum score among
        active signals of that type (worst-case approach).
        """
        with self._lock:
            # Check cache
            cached = self._cache.get(target_id)
            if cached is not None:
                return cached

            signals = self._signals.get(target_id, [])

        now = time.time()

        # Filter expired signals
        active_signals = [s for s in signals if not s.is_expired(now)]

        # Prune expired signals from storage
        if len(active_signals) < len(signals):
            with self._lock:
                self._signals[target_id] = active_signals

        if not active_signals:
            assessment = ThreatAssessment(
                target_id=target_id,
                composite_score=0.0,
                threat_level=ThreatLevel.GREEN,
                sub_scores={},
                signal_count=0,
                top_signals=[],
                assessed_at=now,
            )
            with self._lock:
                self._cache[target_id] = assessment
            return assessment

        # Group signals by type and compute sub-scores (max per type)
        type_signals: dict[str, list[ThreatSignal]] = {}
        for sig in active_signals:
            if sig.signal_type not in type_signals:
                type_signals[sig.signal_type] = []
            type_signals[sig.signal_type].append(sig)

        sub_scores: dict[str, float] = {}
        for sig_type, sigs in type_signals.items():
            # Apply time decay: signals lose potency as they age
            if self._decay_enabled:
                scores = []
                for s in sigs:
                    age = now - s.timestamp
                    # Half-life decay: score halves every 3600 seconds (1 hour)
                    decay = 0.5 ** (age / 3600.0) if age > 0 else 1.0
                    scores.append(s.score * decay)
                sub_scores[sig_type] = max(scores) if scores else 0.0
            else:
                sub_scores[sig_type] = max(s.score for s in sigs)

        # Compute weighted composite score
        total_weight = 0.0
        weighted_sum = 0.0
        for sig_type, sub_score in sub_scores.items():
            weight = self._weights.get(sig_type, 0.0)
            weighted_sum += sub_score * weight
            total_weight += weight

        # Also account for weight types that have no signals (they contribute 0)
        for sig_type, weight in self._weights.items():
            if sig_type not in sub_scores:
                total_weight += weight

        composite = weighted_sum / total_weight if total_weight > 0 else 0.0
        composite = max(0.0, min(1.0, composite))

        # Determine threat level
        threat_level = ThreatLevel.GREEN
        for threshold, level in THREAT_THRESHOLDS:
            if composite >= threshold:
                threat_level = level
                break

        # Top signals (sorted by score descending, top 5)
        sorted_signals = sorted(active_signals, key=lambda s: s.score, reverse=True)
        top_signals = [s.to_dict() for s in sorted_signals[:5]]

        assessment = ThreatAssessment(
            target_id=target_id,
            composite_score=composite,
            threat_level=threat_level,
            sub_scores=sub_scores,
            signal_count=len(active_signals),
            top_signals=top_signals,
            assessed_at=now,
        )

        with self._lock:
            self._cache[target_id] = assessment
            self._assessment_count += 1

        return assessment

    def assess_all(self) -> list[ThreatAssessment]:
        """Assess all known targets. Returns list sorted by composite score descending."""
        with self._lock:
            target_ids = list(self._signals.keys())

        assessments = [self.assess(tid) for tid in target_ids]
        assessments.sort(key=lambda a: a.composite_score, reverse=True)
        return assessments

    def clear_signals(self, target_id: str) -> int:
        """Clear all signals for a target. Returns count removed."""
        with self._lock:
            signals = self._signals.pop(target_id, [])
            self._cache.pop(target_id, None)
            return len(signals)

    def clear_all(self) -> int:
        """Clear all signals for all targets. Returns total count removed."""
        with self._lock:
            total = sum(len(sigs) for sigs in self._signals.values())
            self._signals.clear()
            self._cache.clear()
            return total

    def get_signals(self, target_id: str, signal_type: Optional[str] = None) -> list[ThreatSignal]:
        """Get active signals for a target, optionally filtered by type."""
        with self._lock:
            signals = list(self._signals.get(target_id, []))

        now = time.time()
        active = [s for s in signals if not s.is_expired(now)]

        if signal_type:
            active = [s for s in active if s.signal_type == signal_type]

        return active

    def get_targets_above(self, threshold: float) -> list[str]:
        """Get target IDs with composite score above threshold."""
        assessments = self.assess_all()
        return [a.target_id for a in assessments if a.composite_score >= threshold]

    def get_stats(self) -> dict[str, Any]:
        """Return model statistics."""
        with self._lock:
            total_targets = len(self._signals)
            total_signals = sum(len(sigs) for sigs in self._signals.values())

        assessments = self.assess_all()
        level_counts: dict[str, int] = {}
        for a in assessments:
            level_counts[a.threat_level.value] = level_counts.get(a.threat_level.value, 0) + 1

        return {
            "total_targets": total_targets,
            "total_signals": total_signals,
            "total_assessments": self._assessment_count,
            "level_distribution": level_counts,
            "weights": dict(self._weights),
            "decay_enabled": self._decay_enabled,
            "max_signals_per_target": self._max_signals,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the model state for API responses."""
        return {
            "stats": self.get_stats(),
            "assessments": [a.to_dict() for a in self.assess_all()],
        }


def score_to_threat_level(score: float) -> ThreatLevel:
    """Convert a numeric score (0.0-1.0) to a ThreatLevel."""
    for threshold, level in THREAT_THRESHOLDS:
        if score >= threshold:
            return level
    return ThreatLevel.GREEN
