# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.analytics — real-time analytics engine."""

from __future__ import annotations

import threading
import time

import pytest

from tritium_lib.analytics import (
    AnalyticsEngine,
    Counter,
    Histogram,
    TimeWindow,
    TopN,
    TrendDetector,
    TrendResult,
)


# ===================================================================
# TimeWindow tests
# ===================================================================

class TestTimeWindow:
    """Tests for the O(1) sliding time window."""

    def test_empty_window(self) -> None:
        w = TimeWindow(window_seconds=60.0)
        assert w.count == 0.0
        assert w.rate_per_second == 0.0
        assert w.rate_per_minute == 0.0

    def test_single_add(self) -> None:
        w = TimeWindow(window_seconds=60.0)
        w.add(1.0)
        assert w.count >= 1.0

    def test_multiple_adds(self) -> None:
        w = TimeWindow(window_seconds=60.0)
        now = time.time()
        for i in range(10):
            w.add(1.0, timestamp=now + i * 0.1)
        assert w.count == pytest.approx(10.0, abs=0.01)

    def test_rate_calculation(self) -> None:
        w = TimeWindow(window_seconds=60.0)
        now = time.time()
        for i in range(60):
            w.add(1.0, timestamp=now + i)
        # 60 events in 60 seconds = 1/sec = 60/min
        assert w.rate_per_second == pytest.approx(1.0, abs=0.1)
        assert w.rate_per_minute == pytest.approx(60.0, abs=6.0)

    def test_expiry(self) -> None:
        w = TimeWindow(window_seconds=10.0)
        now = time.time()
        # Add old events that should be expired
        w.add(5.0, timestamp=now - 20.0)
        w.add(3.0, timestamp=now - 15.0)
        # Add a recent event
        w.add(1.0, timestamp=now)
        assert w.count == pytest.approx(1.0, abs=0.01)

    def test_clear(self) -> None:
        w = TimeWindow(window_seconds=60.0)
        w.add(5.0)
        w.clear()
        assert w.count == 0.0

    def test_export(self) -> None:
        w = TimeWindow(window_seconds=60.0)
        w.add(1.0)
        exported = w.export()
        assert "count" in exported
        assert "rate_per_second" in exported
        assert "rate_per_minute" in exported
        assert "window_seconds" in exported
        assert exported["window_seconds"] == 60.0

    def test_custom_bucket_size(self) -> None:
        w = TimeWindow(window_seconds=60.0, bucket_seconds=5.0)
        now = time.time()
        for i in range(10):
            w.add(1.0, timestamp=now + i)
        assert w.count == pytest.approx(10.0, abs=0.01)

    def test_fractional_values(self) -> None:
        w = TimeWindow(window_seconds=60.0)
        now = time.time()
        w.add(0.5, timestamp=now)
        w.add(0.3, timestamp=now + 1)
        assert w.count == pytest.approx(0.8, abs=0.01)


# ===================================================================
# Counter tests
# ===================================================================

class TestCounter:
    """Tests for the multi-horizon event counter."""

    def test_empty_counter(self) -> None:
        c = Counter(name="test")
        assert c.total == 0.0
        assert c.count == 0
        assert c.rate("1min") == 0.0
        assert c.name == "test"

    def test_increment(self) -> None:
        c = Counter(name="events")
        c.increment()
        c.increment()
        c.increment()
        assert c.count == 3
        assert c.total == 3.0

    def test_increment_with_value(self) -> None:
        c = Counter(name="bytes")
        c.increment(100.0)
        c.increment(200.0)
        assert c.total == 300.0
        assert c.count == 2

    def test_rates_all_horizons(self) -> None:
        c = Counter(name="test")
        now = time.time()
        for i in range(10):
            c.increment(timestamp=now + i * 0.1)
        rates = c.rates()
        assert "1min" in rates
        assert "5min" in rates
        assert "1hr" in rates
        assert "24hr" in rates
        # All rates should be positive since events are recent
        assert rates["1min"] > 0

    def test_window_count(self) -> None:
        c = Counter(name="test")
        now = time.time()
        for i in range(5):
            c.increment(timestamp=now + i)
        assert c.window_count("1min") >= 5.0
        assert c.window_count("5min") >= 5.0

    def test_window_count_unknown_horizon(self) -> None:
        c = Counter()
        assert c.window_count("unknown") == 0.0
        assert c.rate("unknown") == 0.0

    def test_clear(self) -> None:
        c = Counter(name="test")
        c.increment()
        c.increment()
        c.clear()
        assert c.total == 0.0
        assert c.count == 0

    def test_export(self) -> None:
        c = Counter(name="detections")
        c.increment()
        exported = c.export()
        assert exported["name"] == "detections"
        assert exported["lifetime_count"] == 1
        assert exported["lifetime_total"] == 1.0
        assert "rates_per_minute" in exported
        assert "window_counts" in exported


# ===================================================================
# Histogram tests
# ===================================================================

class TestHistogram:
    """Tests for the categorical distribution tracker."""

    def test_empty_histogram(self) -> None:
        h = Histogram(name="test")
        assert h.categories == []
        assert h.distribution() == {}
        assert h.percentages() == {}

    def test_single_category(self) -> None:
        h = Histogram(name="sources", window_seconds=60.0)
        now = time.time()
        h.record("ble", timestamp=now)
        h.record("ble", timestamp=now + 1)
        assert h.count("ble") == pytest.approx(2.0, abs=0.01)
        assert "ble" in h.categories

    def test_multiple_categories(self) -> None:
        h = Histogram(name="sources", window_seconds=60.0)
        now = time.time()
        h.record("ble", timestamp=now)
        h.record("ble", timestamp=now + 0.1)
        h.record("wifi", timestamp=now + 0.2)
        h.record("yolo", timestamp=now + 0.3)
        dist = h.distribution()
        assert dist["ble"] == pytest.approx(2.0, abs=0.01)
        assert dist["wifi"] == pytest.approx(1.0, abs=0.01)
        assert dist["yolo"] == pytest.approx(1.0, abs=0.01)

    def test_percentages(self) -> None:
        h = Histogram(name="types", window_seconds=60.0)
        now = time.time()
        for i in range(3):
            h.record("person", timestamp=now + i * 0.1)
        h.record("vehicle", timestamp=now + 0.4)
        pcts = h.percentages()
        assert pcts["person"] == pytest.approx(75.0, abs=1.0)
        assert pcts["vehicle"] == pytest.approx(25.0, abs=1.0)

    def test_lifetime_counts(self) -> None:
        h = Histogram(name="test")
        h.record("a")
        h.record("a")
        h.record("b")
        lc = h.lifetime_counts()
        assert lc["a"] == 2
        assert lc["b"] == 1

    def test_expiry(self) -> None:
        h = Histogram(name="test", window_seconds=10.0)
        now = time.time()
        h.record("old", timestamp=now - 20.0)
        h.record("new", timestamp=now)
        dist = h.distribution()
        assert "old" not in dist or dist.get("old", 0) == 0
        assert dist.get("new", 0) == pytest.approx(1.0, abs=0.01)

    def test_clear(self) -> None:
        h = Histogram(name="test")
        h.record("a")
        h.record("b")
        h.clear()
        assert h.categories == []
        assert h.lifetime_counts() == {}

    def test_export(self) -> None:
        h = Histogram(name="zones", window_seconds=60.0)
        h.record("lobby")
        exported = h.export()
        assert exported["name"] == "zones"
        assert "distribution" in exported
        assert "percentages" in exported
        assert "lifetime_counts" in exported
        assert exported["window_seconds"] == 60.0


# ===================================================================
# TrendDetector tests
# ===================================================================

class TestTrendDetector:
    """Tests for trend detection via linear regression."""

    def test_empty_trend(self) -> None:
        td = TrendDetector(name="test")
        result = td.analyze()
        assert isinstance(result, TrendResult)
        assert result.direction == "stable"
        assert result.slope == 0.0
        assert result.samples == 0

    def test_stable_trend(self) -> None:
        td = TrendDetector(
            name="test",
            window_seconds=300.0,
            bucket_seconds=10.0,
            slope_threshold=0.1,
        )
        now = time.time()
        # Record same amount in each bucket
        for i in range(10):
            for _ in range(5):
                td.record(timestamp=now + i * 10 + 1)
        result = td.analyze()
        assert result.direction == "stable"

    def test_increasing_trend(self) -> None:
        td = TrendDetector(
            name="test",
            window_seconds=300.0,
            bucket_seconds=10.0,
            slope_threshold=0.001,
        )
        now = time.time()
        # Record increasing events in each bucket
        for i in range(10):
            for _ in range(i + 1):
                td.record(timestamp=now + i * 10 + 1)
        result = td.analyze()
        assert result.direction == "increasing"
        assert result.slope > 0
        assert result.samples >= 2

    def test_decreasing_trend(self) -> None:
        td = TrendDetector(
            name="test",
            window_seconds=300.0,
            bucket_seconds=10.0,
            slope_threshold=0.001,
        )
        now = time.time()
        # Record decreasing events in each bucket
        for i in range(10):
            for _ in range(10 - i):
                td.record(timestamp=now + i * 10 + 1)
        result = td.analyze()
        assert result.direction == "decreasing"
        assert result.slope < 0

    def test_confidence_range(self) -> None:
        td = TrendDetector(name="test", bucket_seconds=10.0)
        now = time.time()
        for i in range(5):
            td.record(timestamp=now + i * 10 + 1)
        result = td.analyze()
        assert 0.0 <= result.confidence <= 1.0

    def test_clear(self) -> None:
        td = TrendDetector(name="test")
        td.record()
        td.clear()
        result = td.analyze()
        assert result.samples == 0

    def test_export(self) -> None:
        td = TrendDetector(name="detection_trend")
        td.record()
        exported = td.export()
        assert exported["name"] == "detection_trend"
        assert "direction" in exported
        assert "slope" in exported
        assert "confidence" in exported
        assert "current_rate" in exported
        assert "samples" in exported

    def test_single_bucket(self) -> None:
        """With only one bucket, trend should be stable."""
        td = TrendDetector(name="test", bucket_seconds=10.0)
        now = time.time()
        td.record(timestamp=now)
        td.record(timestamp=now + 1)
        result = td.analyze()
        assert result.direction == "stable"
        assert result.samples == 1


# ===================================================================
# TopN tests
# ===================================================================

class TestTopN:
    """Tests for top-N activity tracking."""

    def test_empty_top(self) -> None:
        tn = TopN(n=5)
        assert tn.top() == []

    def test_single_item(self) -> None:
        tn = TopN(n=5, window_seconds=60.0)
        now = time.time()
        tn.record("target_1", timestamp=now)
        result = tn.top()
        assert len(result) == 1
        assert result[0][0] == "target_1"
        assert result[0][1] >= 1.0

    def test_ordering(self) -> None:
        tn = TopN(n=3, window_seconds=60.0)
        now = time.time()
        for i in range(5):
            tn.record("low", timestamp=now + i * 0.1)
        for i in range(10):
            tn.record("high", timestamp=now + i * 0.1)
        for i in range(3):
            tn.record("medium", timestamp=now + i * 0.1)
        result = tn.top()
        assert len(result) == 3
        assert result[0][0] == "high"
        assert result[1][0] == "low"
        assert result[2][0] == "medium"

    def test_n_limit(self) -> None:
        tn = TopN(n=2, window_seconds=60.0)
        now = time.time()
        for name in ["a", "b", "c", "d"]:
            tn.record(name, timestamp=now)
        result = tn.top()
        assert len(result) == 2

    def test_custom_n(self) -> None:
        tn = TopN(n=10, window_seconds=60.0)
        now = time.time()
        for name in ["a", "b", "c"]:
            tn.record(name, timestamp=now)
        result = tn.top(n=1)
        assert len(result) == 1

    def test_count(self) -> None:
        tn = TopN(n=5, window_seconds=60.0)
        now = time.time()
        tn.record("x", timestamp=now)
        tn.record("x", timestamp=now + 0.1)
        tn.record("x", timestamp=now + 0.2)
        assert tn.count("x") == pytest.approx(3.0, abs=0.01)
        assert tn.count("nonexistent") == 0.0

    def test_lifetime_count(self) -> None:
        tn = TopN(n=5)
        tn.record("a")
        tn.record("a")
        tn.record("b")
        assert tn.lifetime_count("a") == 2
        assert tn.lifetime_count("b") == 1
        assert tn.lifetime_count("c") == 0

    def test_clear(self) -> None:
        tn = TopN(n=5)
        tn.record("a")
        tn.clear()
        assert tn.top() == []
        assert tn.lifetime_count("a") == 0

    def test_export(self) -> None:
        tn = TopN(n=5, window_seconds=60.0)
        tn.record("target_1")
        exported = tn.export()
        assert "top" in exported
        assert "tracked_items" in exported
        assert "window_seconds" in exported
        assert exported["tracked_items"] == 1


# ===================================================================
# AnalyticsEngine tests
# ===================================================================

class TestAnalyticsEngine:
    """Tests for the unified AnalyticsEngine."""

    def test_empty_engine(self) -> None:
        engine = AnalyticsEngine()
        assert engine.detection_rate == 0.0
        assert engine.alert_rate == 0.0
        assert engine.correlation_success_rate == 0.0
        assert engine.zone_activity() == {}
        assert engine.sensor_utilization() == {}
        assert engine.top_targets() == []

    def test_record_detection(self) -> None:
        engine = AnalyticsEngine()
        now = time.time()
        engine.record_detection(
            "ble_aabb",
            source="ble",
            zone="lobby",
            target_type="phone",
            timestamp=now,
        )
        assert engine.detection_rate > 0
        su = engine.sensor_utilization()
        assert "ble" in su
        za = engine.zone_activity()
        assert "lobby" in za

    def test_multiple_detections(self) -> None:
        engine = AnalyticsEngine()
        now = time.time()
        for i in range(20):
            engine.record_detection(
                f"target_{i % 5}",
                source="ble" if i % 2 == 0 else "wifi",
                zone="zone_a" if i < 10 else "zone_b",
                timestamp=now + i * 0.1,
            )
        su = engine.sensor_utilization()
        assert "ble" in su
        assert "wifi" in su
        za = engine.zone_activity()
        assert "zone_a" in za
        assert "zone_b" in za

    def test_record_alert(self) -> None:
        engine = AnalyticsEngine()
        now = time.time()
        engine.record_alert("geofence_entry", severity="warning", timestamp=now)
        engine.record_alert("threat_level", severity="critical", timestamp=now + 1)
        assert engine.alert_rate > 0

    def test_correlation_success_rate(self) -> None:
        engine = AnalyticsEngine()
        now = time.time()
        for i in range(10):
            engine.record_correlation(
                "a", "b",
                success=(i < 7),
                timestamp=now + i * 0.1,
            )
        rate = engine.correlation_success_rate
        assert rate == pytest.approx(0.7, abs=0.05)

    def test_correlation_no_attempts(self) -> None:
        engine = AnalyticsEngine()
        assert engine.correlation_success_rate == 0.0

    def test_target_type_distribution(self) -> None:
        engine = AnalyticsEngine()
        now = time.time()
        for i in range(6):
            engine.record_detection(
                f"t_{i}",
                target_type="person",
                timestamp=now + i * 0.1,
            )
        for i in range(4):
            engine.record_detection(
                f"v_{i}",
                target_type="vehicle",
                timestamp=now + 0.6 + i * 0.1,
            )
        pcts = engine.target_type_distribution()
        assert pcts.get("person", 0) == pytest.approx(60.0, abs=5.0)
        assert pcts.get("vehicle", 0) == pytest.approx(40.0, abs=5.0)

    def test_detection_trend(self) -> None:
        engine = AnalyticsEngine(trend_window=100.0, trend_bucket=10.0)
        now = time.time()
        # Increasing pattern: 1, 2, 3, ... events per bucket
        for bucket_idx in range(5):
            for _ in range(bucket_idx + 1):
                engine.record_detection(
                    "t",
                    timestamp=now + bucket_idx * 10 + 1,
                )
        trend = engine.detection_trend()
        assert isinstance(trend, TrendResult)

    def test_alert_trend(self) -> None:
        engine = AnalyticsEngine(trend_window=100.0, trend_bucket=10.0)
        now = time.time()
        for i in range(5):
            engine.record_alert("test", timestamp=now + i * 10 + 1)
        trend = engine.alert_trend()
        assert isinstance(trend, TrendResult)

    def test_top_targets(self) -> None:
        engine = AnalyticsEngine(top_n=3)
        now = time.time()
        for i in range(10):
            engine.record_detection("frequent", timestamp=now + i * 0.1)
        for i in range(3):
            engine.record_detection("rare", timestamp=now + i * 0.1)
        top = engine.top_targets()
        assert len(top) <= 3
        assert top[0][0] == "frequent"

    def test_top_zones(self) -> None:
        engine = AnalyticsEngine(top_n=2)
        now = time.time()
        for i in range(5):
            engine.record_detection("t", zone="hot_zone", timestamp=now + i * 0.1)
        engine.record_detection("t", zone="cold_zone", timestamp=now + 0.6)
        top = engine.top_zones()
        assert top[0][0] == "hot_zone"

    def test_top_sources(self) -> None:
        engine = AnalyticsEngine(top_n=2)
        now = time.time()
        for i in range(8):
            engine.record_detection("t", source="ble", timestamp=now + i * 0.1)
        for i in range(2):
            engine.record_detection("t", source="wifi", timestamp=now + i * 0.1)
        top = engine.top_sources()
        assert top[0][0] == "ble"

    def test_snapshot(self) -> None:
        engine = AnalyticsEngine()
        now = time.time()
        engine.record_detection("t1", source="ble", zone="z1", timestamp=now)
        engine.record_alert("test", severity="info", timestamp=now)
        engine.record_correlation("a", "b", success=True, timestamp=now)
        snap = engine.snapshot()
        assert "detection_rate" in snap
        assert "alert_rate" in snap
        assert "correlation" in snap
        assert "zone_activity" in snap
        assert "sensor_utilization" in snap
        assert "target_types" in snap
        assert "trends" in snap
        assert "top_targets" in snap
        assert "top_zones" in snap
        assert "top_sources" in snap
        assert "timestamp" in snap

    def test_snapshot_serializable(self) -> None:
        """Snapshot must be JSON-serializable (all standard types)."""
        import json
        engine = AnalyticsEngine()
        now = time.time()
        engine.record_detection("t1", source="ble", zone="z1", target_type="phone", timestamp=now)
        engine.record_alert("test", severity="warning", timestamp=now)
        engine.record_correlation("a", "b", success=True, timestamp=now)
        snap = engine.snapshot()
        # Should not raise
        json_str = json.dumps(snap)
        assert len(json_str) > 10

    def test_clear(self) -> None:
        engine = AnalyticsEngine()
        now = time.time()
        engine.record_detection("t1", source="ble", zone="z1", timestamp=now)
        engine.record_alert("test", timestamp=now)
        engine.record_correlation("a", "b", timestamp=now)
        engine.clear()
        assert engine.detection_rate == 0.0
        assert engine.alert_rate == 0.0
        assert engine.zone_activity() == {}
        assert engine.top_targets() == []

    def test_thread_safety(self) -> None:
        """Concurrent access must not raise."""
        engine = AnalyticsEngine()
        errors: list[Exception] = []

        def writer() -> None:
            try:
                for i in range(100):
                    engine.record_detection(
                        f"target_{i}",
                        source="ble",
                        zone="z1",
                    )
                    engine.record_alert("test", severity="info")
                    engine.record_correlation("a", "b", success=True)
            except Exception as exc:
                errors.append(exc)

        def reader() -> None:
            try:
                for _ in range(50):
                    engine.snapshot()
                    engine.detection_rate
                    engine.zone_activity()
                    engine.top_targets()
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert errors == [], f"Thread errors: {errors}"

    def test_detection_without_optional_fields(self) -> None:
        """record_detection with only target_id should work."""
        engine = AnalyticsEngine()
        engine.record_detection("simple_target")
        assert engine.detection_rate > 0

    def test_high_volume(self) -> None:
        """Engine should handle thousands of events efficiently."""
        engine = AnalyticsEngine()
        now = time.time()
        for i in range(1000):
            engine.record_detection(
                f"target_{i % 50}",
                source=["ble", "wifi", "yolo", "mesh"][i % 4],
                zone=f"zone_{i % 10}",
                target_type=["person", "vehicle", "phone"][i % 3],
                timestamp=now + i * 0.001,
            )
        snap = engine.snapshot()
        assert snap["detection_rate"]["lifetime_count"] == 1000
        assert len(snap["top_targets"]["top"]) <= 10


# ===================================================================
# Integration tests
# ===================================================================

class TestAnalyticsIntegration:
    """Integration tests combining multiple components."""

    def test_full_workflow(self) -> None:
        """Simulate a realistic tracking session."""
        engine = AnalyticsEngine(
            trend_window=60.0,
            trend_bucket=5.0,
            histogram_window=60.0,
            top_n=5,
        )
        now = time.time()

        # Phase 1: BLE detections
        for i in range(20):
            engine.record_detection(
                f"ble_{i:04x}",
                source="ble",
                zone="entrance",
                target_type="phone",
                timestamp=now + i,
            )

        # Phase 2: YOLO detections
        for i in range(15):
            engine.record_detection(
                f"det_person_{i}",
                source="yolo",
                zone="parking",
                target_type="person",
                timestamp=now + 20 + i,
            )

        # Phase 3: Correlations
        for i in range(10):
            engine.record_correlation(
                f"ble_{i:04x}",
                f"det_person_{i}",
                success=(i < 8),
                timestamp=now + 35 + i,
            )

        # Phase 4: Alerts
        for i in range(5):
            engine.record_alert(
                "geofence_entry",
                severity="warning",
                timestamp=now + 45 + i,
            )

        # Verify
        snap = engine.snapshot()
        assert snap["detection_rate"]["lifetime_count"] == 35
        assert snap["correlation"]["attempts"]["lifetime_count"] == 10
        assert snap["correlation"]["successes"]["lifetime_count"] == 8
        assert snap["correlation"]["success_rate_5min"] == pytest.approx(0.8, abs=0.05)

        su = snap["sensor_utilization"]
        assert "ble" in su["distribution"]
        assert "yolo" in su["distribution"]

        za = snap["zone_activity"]
        assert "entrance" in za["distribution"]
        assert "parking" in za["distribution"]

    def test_import_from_package(self) -> None:
        """All public symbols should be importable from the package."""
        from tritium_lib.analytics import (
            AnalyticsEngine,
            Counter,
            Histogram,
            TimeWindow,
            TopN,
            TrendDetector,
            TrendResult,
        )
        # Just verify they exist and are the right types
        assert callable(AnalyticsEngine)
        assert callable(Counter)
        assert callable(Histogram)
        assert callable(TimeWindow)
        assert callable(TopN)
        assert callable(TrendDetector)
        # TrendResult is a dataclass — check fields exist
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(TrendResult)}
        assert "direction" in field_names
