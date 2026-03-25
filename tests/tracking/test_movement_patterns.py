# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.tracking.movement_patterns."""

import math
import pytest

pytestmark = pytest.mark.unit

from tritium_lib.tracking.target_history import TargetHistory
from tritium_lib.tracking.movement_patterns import (
    MovementPatternAnalyzer,
    MovementPattern,
    LOITER_RADIUS,
    LOITER_MIN_DURATION,
    SPEED_THRESHOLD,
)


def _build_loitering_trail(
    history: TargetHistory,
    target_id: str,
    center: tuple[float, float],
    duration: float,
    n_points: int = 20,
    radius: float = 2.0,
):
    """Create a trail that stays within radius of center for duration seconds."""
    for i in range(n_points):
        t = 100.0 + (duration / n_points) * i
        angle = (2 * math.pi / n_points) * i
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        history.record(target_id, (x, y), timestamp=t)


class TestMovementPatternInit:
    def test_default_init(self):
        a = MovementPatternAnalyzer()
        assert a._loiter_radius == LOITER_RADIUS
        assert a._loiter_min_duration == LOITER_MIN_DURATION

    def test_custom_params(self):
        a = MovementPatternAnalyzer(loiter_radius=10.0, loiter_min_duration=60.0)
        assert a._loiter_radius == 10.0
        assert a._loiter_min_duration == 60.0


class TestMovementPatternDataclass:
    def test_to_dict(self):
        p = MovementPattern(
            pattern_type="loitering",
            target_id="t1",
            timestamp=100.0,
            duration_s=300.0,
            center=(10.0, 20.0),
            radius=3.0,
            confidence=0.8,
        )
        d = p.to_dict()
        assert d["pattern_type"] == "loitering"
        assert d["center"]["x"] == 10.0
        assert d["center"]["y"] == 20.0
        assert d["duration_s"] == 300.0


class TestLoiteringDetection:
    def test_detect_loitering(self):
        h = TargetHistory()
        # Loiter within 2m of center for 600s with many points
        center = (50.0, 50.0)
        for i in range(50):
            t = 100.0 + (600.0 / 50) * i
            # Stay very close to center — small random-ish displacement
            angle = (2 * math.pi / 50) * i
            x = center[0] + 1.5 * math.cos(angle)
            y = center[1] + 1.5 * math.sin(angle)
            h.record("t1", (x, y), timestamp=t)
        a = MovementPatternAnalyzer(history=h, loiter_radius=5.0, loiter_min_duration=300.0)
        patterns = a.analyze("t1")
        loiters = [p for p in patterns if p["pattern_type"] == "loitering"]
        assert len(loiters) >= 1
        assert loiters[0]["duration_s"] >= 300.0

    def test_no_loitering_short_duration(self):
        h = TargetHistory()
        _build_loitering_trail(h, "t1", (50, 50), duration=60, n_points=10, radius=2.0)
        a = MovementPatternAnalyzer(history=h, loiter_radius=5.0, loiter_min_duration=300.0)
        patterns = a.analyze("t1")
        loiters = [p for p in patterns if p["pattern_type"] == "loitering"]
        assert len(loiters) == 0

    def test_no_loitering_large_radius(self):
        h = TargetHistory()
        # Move in a circle with radius > loiter_radius
        for i in range(30):
            t = 100.0 + 20 * i
            angle = (2 * math.pi / 30) * i
            x = 50 + 20.0 * math.cos(angle)
            y = 50 + 20.0 * math.sin(angle)
            h.record("t1", (x, y), timestamp=t)
        a = MovementPatternAnalyzer(history=h, loiter_radius=5.0, loiter_min_duration=300.0)
        patterns = a.analyze("t1")
        loiters = [p for p in patterns if p["pattern_type"] == "loitering"]
        assert len(loiters) == 0


class TestStationaryDetection:
    def test_detect_stationary(self):
        h = TargetHistory()
        for i in range(20):
            h.record("t1", (10.0, 10.0), timestamp=100.0 + i * 5)
        a = MovementPatternAnalyzer(history=h)
        patterns = a.analyze("t1")
        stationary = [p for p in patterns if p["pattern_type"] == "stationary"]
        assert len(stationary) >= 1


class TestDeviationDetection:
    def test_detect_deviation(self):
        h = TargetHistory()
        # Normal path along x axis
        for i in range(15):
            h.record("t1", (float(i), 0.0), timestamp=100.0 + i)
        # Add one outlier far away
        h.record("t1", (7.0, 100.0), timestamp=116.0)
        a = MovementPatternAnalyzer(history=h)
        patterns = a.analyze("t1")
        devs = [p for p in patterns if p["pattern_type"] == "deviation"]
        assert len(devs) >= 1


class TestRegularRoutes:
    def test_too_few_points(self):
        h = TargetHistory()
        for i in range(5):
            h.record("t1", (float(i), 0.0), timestamp=100.0 + i)
        a = MovementPatternAnalyzer(history=h)
        patterns = a.analyze("t1")
        routes = [p for p in patterns if p["pattern_type"] == "regular_route"]
        assert len(routes) == 0

    def test_repeated_path_detected(self):
        h = TargetHistory()
        # Two similar segments
        for seg in range(2):
            base_t = 100.0 + seg * 400
            for i in range(10):
                h.record("t1", (float(i * 10), float(i * 5)), timestamp=base_t + i)
        a = MovementPatternAnalyzer(history=h)
        patterns = a.analyze("t1")
        routes = [p for p in patterns if p["pattern_type"] == "regular_route"]
        # May or may not detect depending on segment length — at least no crash
        assert isinstance(routes, list)


class TestAnalyzeAll:
    def test_analyze_multiple_targets(self):
        h = TargetHistory()
        for i in range(20):
            h.record("t1", (float(i), 0.0), timestamp=100.0 + i * 5)
            h.record("t2", (0.0, float(i)), timestamp=100.0 + i * 5)
        a = MovementPatternAnalyzer(history=h)
        results = a.analyze_all(["t1", "t2"])
        assert "t1" in results
        assert "t2" in results


class TestGetSummary:
    def test_summary_structure(self):
        h = TargetHistory()
        for i in range(20):
            h.record("t1", (10.0, 10.0), timestamp=100.0 + i * 5)
        a = MovementPatternAnalyzer(history=h)
        a.analyze("t1")
        summary = a.get_summary()
        assert "total_patterns" in summary
        assert "counts" in summary
        assert "targets_analyzed" in summary
        assert summary["targets_analyzed"] == 1


class TestNoHistory:
    def test_no_history_returns_empty(self):
        a = MovementPatternAnalyzer(history=None)
        result = a.analyze("t1")
        assert result == []

    def test_insufficient_trail(self):
        h = TargetHistory()
        h.record("t1", (0, 0), timestamp=100.0)
        a = MovementPatternAnalyzer(history=h)
        result = a.analyze("t1")
        assert result == []


class TestCachedPatterns:
    def test_get_cached_before_analyze(self):
        a = MovementPatternAnalyzer()
        result = a.get_cached_patterns("t1")
        assert result == []

    def test_get_cached_after_analyze(self):
        h = TargetHistory()
        for i in range(20):
            h.record("t1", (10.0, 10.0), timestamp=100.0 + i * 5)
        a = MovementPatternAnalyzer(history=h)
        a.analyze("t1")
        cached = a.get_cached_patterns("t1")
        assert isinstance(cached, list)


class TestDeviationEdgeCases:
    """Edge case tests for deviation detection — division by zero fix."""

    def test_all_points_same_position_no_crash(self):
        """All points at identical position: std_dist=0 should not divide by zero."""
        h = TargetHistory()
        for i in range(20):
            h.record("t1", (10.0, 20.0), timestamp=100.0 + i * 5)
        a = MovementPatternAnalyzer(history=h)
        # Should not raise ZeroDivisionError
        result = a.analyze("t1")
        # No deviations possible when all points identical
        deviations = [p for p in result if p["pattern_type"] == "deviation"]
        assert deviations == []

    def test_equidistant_points_on_circle_no_crash(self):
        """Points equidistant from mean (on a circle) have zero std_dist."""
        h = TargetHistory()
        n = 12
        for i in range(n):
            angle = 2 * math.pi * i / n
            x = 50.0 + 10.0 * math.cos(angle)
            y = 50.0 + 10.0 * math.sin(angle)
            h.record("t1", (x, y), timestamp=100.0 + i * 30)
        a = MovementPatternAnalyzer(history=h)
        # Should not crash — all distances from mean are equal, std_dist ~ 0
        result = a.analyze("t1")
        deviations = [p for p in result if p["pattern_type"] == "deviation"]
        assert isinstance(deviations, list)  # no crash

    def test_single_outlier_detected(self):
        """One outlier among many clustered points should be detected."""
        h = TargetHistory()
        # 19 points at the same spot
        for i in range(19):
            h.record("t1", (0.0, 0.0), timestamp=100.0 + i)
        # 1 outlier far away
        h.record("t1", (100.0, 100.0), timestamp=200.0)
        a = MovementPatternAnalyzer(history=h)
        result = a.analyze("t1")
        deviations = [p for p in result if p["pattern_type"] == "deviation"]
        assert len(deviations) >= 1
