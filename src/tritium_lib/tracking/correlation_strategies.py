# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Correlation strategies for multi-factor identity resolution.

Each strategy evaluates a pair of tracked targets and produces a score
from 0.0 (no correlation) to 1.0 (definite same entity). The correlator
combines strategy scores with configurable weights.

Strategies:
  - SpatialStrategy: distance-based proximity
  - TemporalStrategy: co-movement detection from position history
  - SignalPatternStrategy: appearance/disappearance timing correlation
  - DossierStrategy: known prior associations from DossierStore
  - WiFiProbeStrategy: WiFi probe + BLE correlation
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .target_history import TargetHistory
from .target_tracker import TrackedTarget


@dataclass(slots=True)
class StrategyScore:
    """Result of a single strategy evaluation."""

    strategy_name: str
    score: float  # 0.0 to 1.0
    detail: str  # human-readable explanation


class CorrelationStrategy(ABC):
    """Abstract base class for correlation strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short name identifying this strategy."""

    @abstractmethod
    def evaluate(
        self,
        target_a: TrackedTarget,
        target_b: TrackedTarget,
    ) -> StrategyScore:
        """Evaluate correlation strength between two targets."""


class SpatialStrategy(CorrelationStrategy):
    """Distance-based spatial proximity scoring."""

    def __init__(self, radius: float = 5.0) -> None:
        self.radius = radius

    @property
    def name(self) -> str:
        return "spatial"

    def evaluate(
        self,
        target_a: TrackedTarget,
        target_b: TrackedTarget,
    ) -> StrategyScore:
        dx = target_a.position[0] - target_b.position[0]
        dy = target_a.position[1] - target_b.position[1]
        dist = math.hypot(dx, dy)

        if dist > self.radius:
            return StrategyScore(
                strategy_name=self.name,
                score=0.0,
                detail=f"distance {dist:.1f} exceeds radius {self.radius}",
            )

        score = max(0.0, 1.0 - (dist / (self.radius * 1.1)))
        return StrategyScore(
            strategy_name=self.name,
            score=score,
            detail=f"distance {dist:.1f}/{self.radius} units",
        )


class TemporalStrategy(CorrelationStrategy):
    """Co-movement detection from target position history."""

    def __init__(
        self,
        history: TargetHistory,
        *,
        min_samples: int = 3,
        heading_tolerance: float = 45.0,
        speed_ratio_max: float = 3.0,
    ) -> None:
        self.history = history
        self.min_samples = min_samples
        self.heading_tolerance = heading_tolerance
        self.speed_ratio_max = speed_ratio_max

    @property
    def name(self) -> str:
        return "temporal"

    def evaluate(
        self,
        target_a: TrackedTarget,
        target_b: TrackedTarget,
    ) -> StrategyScore:
        trail_a = self.history.get_trail(target_a.target_id, max_points=20)
        trail_b = self.history.get_trail(target_b.target_id, max_points=20)

        if len(trail_a) < self.min_samples or len(trail_b) < self.min_samples:
            return StrategyScore(
                strategy_name=self.name,
                score=0.0,
                detail=f"insufficient history ({len(trail_a)}/{len(trail_b)} samples)",
            )

        heading_a = self._compute_heading(trail_a)
        heading_b = self._compute_heading(trail_b)
        speed_a = self._compute_speed(trail_a)
        speed_b = self._compute_speed(trail_b)

        heading_diff = abs(heading_a - heading_b)
        if heading_diff > 180.0:
            heading_diff = 360.0 - heading_diff

        if speed_a < 0.01 and speed_b < 0.01:
            return StrategyScore(
                strategy_name=self.name,
                score=0.0,
                detail="both targets stationary",
            )

        if heading_diff > self.heading_tolerance:
            heading_score = 0.0
        else:
            heading_score = 1.0 - (heading_diff / self.heading_tolerance)

        max_speed = max(speed_a, speed_b)
        min_speed = min(speed_a, speed_b)
        if min_speed < 0.01:
            speed_score = 0.1
        else:
            ratio = max_speed / min_speed
            if ratio > self.speed_ratio_max:
                speed_score = 0.0
            else:
                speed_score = 1.0 - ((ratio - 1.0) / (self.speed_ratio_max - 1.0))

        score = 0.6 * heading_score + 0.4 * speed_score
        return StrategyScore(
            strategy_name=self.name,
            score=min(1.0, max(0.0, score)),
            detail=(
                f"heading diff {heading_diff:.0f}deg, "
                f"speed {speed_a:.2f}/{speed_b:.2f} u/s"
            ),
        )

    @staticmethod
    def _compute_heading(trail: list[tuple[float, float, float]]) -> float:
        if len(trail) < 2:
            return 0.0
        dx = trail[-1][0] - trail[0][0]
        dy = trail[-1][1] - trail[0][1]
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return 0.0
        return math.degrees(math.atan2(dx, dy)) % 360

    @staticmethod
    def _compute_speed(trail: list[tuple[float, float, float]]) -> float:
        if len(trail) < 2:
            return 0.0
        total_dist = 0.0
        for i in range(1, len(trail)):
            dx = trail[i][0] - trail[i - 1][0]
            dy = trail[i][1] - trail[i - 1][1]
            total_dist += math.hypot(dx, dy)
        dt = trail[-1][2] - trail[0][2]
        if dt <= 0:
            return 0.0
        return total_dist / dt


class SignalPatternStrategy(CorrelationStrategy):
    """Appearance/disappearance timing correlation."""

    def __init__(self, *, appearance_window: float = 10.0) -> None:
        self.appearance_window = appearance_window

    @property
    def name(self) -> str:
        return "signal_pattern"

    def evaluate(
        self,
        target_a: TrackedTarget,
        target_b: TrackedTarget,
    ) -> StrategyScore:
        if target_a.source == target_b.source:
            return StrategyScore(
                strategy_name=self.name,
                score=0.0,
                detail="same source type, signal pattern N/A",
            )

        time_diff = abs(target_a.last_seen - target_b.last_seen)

        if time_diff > self.appearance_window:
            return StrategyScore(
                strategy_name=self.name,
                score=0.0,
                detail=f"last_seen diff {time_diff:.1f}s exceeds window",
            )

        score = 1.0 - (time_diff / self.appearance_window)

        source_pair = frozenset((target_a.source, target_b.source))
        if source_pair == frozenset(("ble", "yolo")):
            score = min(1.0, score * 1.2)

        return StrategyScore(
            strategy_name=self.name,
            score=min(1.0, max(0.0, score)),
            detail=f"last_seen diff {time_diff:.1f}s, sources {target_a.source}+{target_b.source}",
        )


class WiFiProbeStrategy(CorrelationStrategy):
    """WiFi probe request correlation with BLE detections."""

    def __init__(self, *, max_window: float = 10.0) -> None:
        self.max_window = max_window

    @property
    def name(self) -> str:
        return "wifi_probe"

    def evaluate(
        self,
        target_a: TrackedTarget,
        target_b: TrackedTarget,
    ) -> StrategyScore:
        sources = frozenset((target_a.source, target_b.source))
        if sources != frozenset(("ble", "wifi_probe")):
            return StrategyScore(
                strategy_name=self.name,
                score=0.0,
                detail="not a BLE+wifi_probe pair",
            )

        time_diff = abs(target_a.last_seen - target_b.last_seen)
        if time_diff > self.max_window:
            return StrategyScore(
                strategy_name=self.name,
                score=0.0,
                detail=f"time diff {time_diff:.1f}s exceeds window {self.max_window}s",
            )

        score = 1.0 - (time_diff / self.max_window)

        observer_a = getattr(target_a, "observer_id", "")
        observer_b = getattr(target_b, "observer_id", "")
        same_observer = bool(observer_a and observer_a == observer_b)
        if same_observer:
            score = min(1.0, score * 1.3)

        rssi_a = getattr(target_a, "rssi", None)
        rssi_b = getattr(target_b, "rssi", None)
        if rssi_a is not None and rssi_b is not None:
            rssi_diff = abs(float(rssi_a) - float(rssi_b))
            if rssi_diff < 15:
                score = min(1.0, score * 1.1)

        detail = (
            f"BLE+wifi_probe dt={time_diff:.1f}s"
            f"{' same_observer' if same_observer else ''}"
        )

        return StrategyScore(
            strategy_name=self.name,
            score=min(1.0, max(0.0, score)),
            detail=detail,
        )


class DossierStrategy(CorrelationStrategy):
    """Check DossierStore for known prior associations."""

    def __init__(self, dossier_store) -> None:
        self._store = dossier_store

    @property
    def name(self) -> str:
        return "dossier"

    def evaluate(
        self,
        target_a: TrackedTarget,
        target_b: TrackedTarget,
    ) -> StrategyScore:
        dossier = self._store.find_association(
            target_a.target_id, target_b.target_id
        )
        if dossier is not None:
            score = min(1.0, 0.7 + 0.1 * dossier.correlation_count)
            return StrategyScore(
                strategy_name=self.name,
                score=score,
                detail=f"known dossier {dossier.uuid[:8]}, {dossier.correlation_count} prior correlations",
            )

        d_a = self._store.find_by_signal(target_a.target_id)
        d_b = self._store.find_by_signal(target_b.target_id)

        if d_a is not None and d_b is not None and d_a.uuid != d_b.uuid:
            return StrategyScore(
                strategy_name=self.name,
                score=0.0,
                detail="targets belong to different known dossiers",
            )

        return StrategyScore(
            strategy_name=self.name,
            score=0.0,
            detail="no prior association found",
        )
