# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""MovementPatternAnalyzer — detect regular routes, loitering, and anomalies.

Consumes position history from TargetHistory and classifies movement
patterns for each tracked target:

  - **Regular route**: target follows a similar path at similar times
  - **Loitering**: target stays in a small area for an extended period
  - **Unusual deviation**: target departs from its established pattern
  - **Patrol**: target moves along a repeated circuit
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Thresholds
LOITER_RADIUS = 5.0
LOITER_MIN_DURATION = 300.0
SPEED_THRESHOLD = 0.3
DEVIATION_SIGMA = 2.0


@dataclass
class MovementPattern:
    """A detected movement pattern."""

    pattern_type: str
    target_id: str
    timestamp: float
    duration_s: float = 0.0
    center: tuple[float, float] = (0.0, 0.0)
    radius: float = 0.0
    confidence: float = 0.0
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "pattern_type": self.pattern_type,
            "target_id": self.target_id,
            "timestamp": self.timestamp,
            "duration_s": self.duration_s,
            "center": {"x": self.center[0], "y": self.center[1]},
            "radius": self.radius,
            "confidence": self.confidence,
            "details": self.details,
        }


class MovementPatternAnalyzer:
    """Analyzes target position history to detect movement patterns.

    Thread-safe. Call ``analyze()`` for a single target or
    ``analyze_all()`` for all tracked targets.
    """

    def __init__(
        self,
        history=None,
        loiter_radius: float = LOITER_RADIUS,
        loiter_min_duration: float = LOITER_MIN_DURATION,
    ) -> None:
        self._history = history
        self._loiter_radius = loiter_radius
        self._loiter_min_duration = loiter_min_duration
        self._lock = threading.Lock()
        self._patterns: dict[str, list[MovementPattern]] = {}

    def set_history(self, history) -> None:
        """Set or replace the TargetHistory source."""
        with self._lock:
            self._history = history

    def analyze(self, target_id: str, max_points: int = 500) -> list[dict]:
        """Analyze movement patterns for a single target."""
        if self._history is None:
            return []

        trail = self._history.get_trail(target_id, max_points=max_points)
        if len(trail) < 3:
            return []

        patterns: list[MovementPattern] = []

        loiter = self._detect_loitering(target_id, trail)
        patterns.extend(loiter)

        stationary = self._detect_stationary(target_id, trail)
        patterns.extend(stationary)

        routes = self._detect_regular_routes(target_id, trail)
        patterns.extend(routes)

        deviations = self._detect_deviations(target_id, trail)
        patterns.extend(deviations)

        patterns.sort(key=lambda p: p.timestamp)

        with self._lock:
            self._patterns[target_id] = patterns

        return [p.to_dict() for p in patterns]

    def analyze_all(self, target_ids: list[str]) -> dict[str, list[dict]]:
        """Analyze patterns for multiple targets."""
        results = {}
        for tid in target_ids:
            results[tid] = self.analyze(tid)
        return results

    def get_cached_patterns(self, target_id: str) -> list[dict]:
        """Return previously computed patterns without re-analyzing."""
        with self._lock:
            patterns = self._patterns.get(target_id, [])
        return [p.to_dict() for p in patterns]

    def get_summary(self, target_ids: list[str] | None = None) -> dict:
        """Return aggregate summary of all detected patterns."""
        with self._lock:
            if target_ids is not None:
                all_patterns = []
                for tid in target_ids:
                    all_patterns.extend(self._patterns.get(tid, []))
            else:
                all_patterns = []
                for pats in self._patterns.values():
                    all_patterns.extend(pats)

        counts: dict[str, int] = {}
        loitering_targets: list[str] = []

        for p in all_patterns:
            counts[p.pattern_type] = counts.get(p.pattern_type, 0) + 1
            if p.pattern_type == "loitering":
                if p.target_id not in loitering_targets:
                    loitering_targets.append(p.target_id)

        return {
            "total_patterns": len(all_patterns),
            "counts": counts,
            "loitering_targets": loitering_targets,
            "targets_analyzed": len(self._patterns),
        }

    def _detect_loitering(
        self, target_id: str, trail: list[tuple[float, float, float]]
    ) -> list[MovementPattern]:
        """Detect periods where target stays within a small radius."""
        patterns: list[MovementPattern] = []
        if len(trail) < 3:
            return patterns

        i = 0
        while i < len(trail):
            anchor_x, anchor_y, start_t = trail[i]
            j = i + 1

            while j < len(trail):
                dx = trail[j][0] - anchor_x
                dy = trail[j][1] - anchor_y
                dist = math.hypot(dx, dy)
                if dist > self._loiter_radius:
                    break
                j += 1

            end_t = trail[j - 1][2]
            duration = end_t - start_t

            if duration >= self._loiter_min_duration and (j - i) >= 3:
                xs = [trail[k][0] for k in range(i, j)]
                ys = [trail[k][1] for k in range(i, j)]
                cx = sum(xs) / len(xs)
                cy = sum(ys) / len(ys)

                max_r = max(
                    math.hypot(trail[k][0] - cx, trail[k][1] - cy)
                    for k in range(i, j)
                )

                patterns.append(MovementPattern(
                    pattern_type="loitering",
                    target_id=target_id,
                    timestamp=start_t,
                    duration_s=duration,
                    center=(cx, cy),
                    radius=max_r,
                    confidence=min(1.0, duration / (self._loiter_min_duration * 3)),
                    details={"point_count": j - i},
                ))
                i = j
            else:
                i += 1

        return patterns

    def _detect_stationary(
        self, target_id: str, trail: list[tuple[float, float, float]]
    ) -> list[MovementPattern]:
        """Detect periods of zero or near-zero movement."""
        patterns: list[MovementPattern] = []
        if len(trail) < 2:
            return patterns

        i = 0
        while i < len(trail) - 1:
            dx = trail[i + 1][0] - trail[i][0]
            dy = trail[i + 1][1] - trail[i][1]
            dt = trail[i + 1][2] - trail[i][2]
            if dt <= 0:
                i += 1
                continue

            speed = math.hypot(dx, dy) / dt
            if speed < SPEED_THRESHOLD:
                start = i
                j = i + 1
                while j < len(trail) - 1:
                    dx2 = trail[j + 1][0] - trail[j][0]
                    dy2 = trail[j + 1][1] - trail[j][1]
                    dt2 = trail[j + 1][2] - trail[j][2]
                    if dt2 <= 0:
                        j += 1
                        continue
                    speed2 = math.hypot(dx2, dy2) / dt2
                    if speed2 >= SPEED_THRESHOLD:
                        break
                    j += 1

                duration = trail[j][2] - trail[start][2]
                if duration >= 30.0 and (j - start) >= 2:
                    patterns.append(MovementPattern(
                        pattern_type="stationary",
                        target_id=target_id,
                        timestamp=trail[start][2],
                        duration_s=duration,
                        center=(trail[start][0], trail[start][1]),
                        confidence=min(1.0, duration / 120.0),
                    ))
                i = j
            else:
                i += 1

        return patterns

    def _detect_regular_routes(
        self, target_id: str, trail: list[tuple[float, float, float]]
    ) -> list[MovementPattern]:
        """Detect repeated path segments."""
        patterns: list[MovementPattern] = []
        if len(trail) < 20:
            return patterns

        segment_duration = 300.0
        segments: list[list[tuple[float, float, float]]] = []
        current_seg: list[tuple[float, float, float]] = [trail[0]]

        for pt in trail[1:]:
            if pt[2] - current_seg[0][2] > segment_duration:
                if len(current_seg) >= 5:
                    segments.append(current_seg)
                current_seg = [pt]
            else:
                current_seg.append(pt)
        if len(current_seg) >= 5:
            segments.append(current_seg)

        if len(segments) < 2:
            return patterns

        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                similarity = self._segment_similarity(segments[i], segments[j])
                if similarity > 0.7:
                    all_pts = segments[i] + segments[j]
                    cx = sum(p[0] for p in all_pts) / len(all_pts)
                    cy = sum(p[1] for p in all_pts) / len(all_pts)

                    patterns.append(MovementPattern(
                        pattern_type="regular_route",
                        target_id=target_id,
                        timestamp=segments[j][0][2],
                        center=(cx, cy),
                        confidence=similarity,
                        details={
                            "segment_a_time": segments[i][0][2],
                            "segment_b_time": segments[j][0][2],
                            "similarity": round(similarity, 3),
                        },
                    ))

        return patterns

    def _detect_deviations(
        self, target_id: str, trail: list[tuple[float, float, float]]
    ) -> list[MovementPattern]:
        """Detect points that deviate significantly from the mean path."""
        patterns: list[MovementPattern] = []
        if len(trail) < 10:
            return patterns

        xs = [p[0] for p in trail]
        ys = [p[1] for p in trail]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)

        dists = [math.hypot(p[0] - mean_x, p[1] - mean_y) for p in trail]
        mean_dist = sum(dists) / len(dists)
        if mean_dist < 1e-6:
            return patterns

        variance = sum((d - mean_dist) ** 2 for d in dists) / len(dists)
        std_dist = math.sqrt(variance) if variance > 0 else 0.0
        if std_dist < 1e-9:
            return patterns  # all points equidistant from mean — no deviations

        for idx, (x, y, t) in enumerate(trail):
            d = math.hypot(x - mean_x, y - mean_y)
            if d > mean_dist + DEVIATION_SIGMA * std_dist:
                patterns.append(MovementPattern(
                    pattern_type="deviation",
                    target_id=target_id,
                    timestamp=t,
                    center=(x, y),
                    radius=d,
                    confidence=min(1.0, (d - mean_dist) / (std_dist * 3)),
                    details={
                        "distance_from_mean": round(d, 2),
                        "sigma": round((d - mean_dist) / std_dist, 2),
                    },
                ))

        return patterns

    @staticmethod
    def _segment_similarity(
        seg_a: list[tuple[float, float, float]],
        seg_b: list[tuple[float, float, float]],
    ) -> float:
        """Compute a normalised similarity score between two path segments."""
        n = min(len(seg_a), len(seg_b), 20)
        if n < 2:
            return 0.0

        def resample(seg, count):
            indices = [int(i * (len(seg) - 1) / (count - 1)) for i in range(count)]
            return [(seg[i][0], seg[i][1]) for i in indices]

        ra = resample(seg_a, n)
        rb = resample(seg_b, n)

        total_dist = sum(math.hypot(ra[i][0] - rb[i][0], ra[i][1] - rb[i][1]) for i in range(n))
        mean_dist = total_dist / n

        all_pts = ra + rb
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        extent = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)

        similarity = max(0.0, 1.0 - (mean_dist / extent))
        return similarity
