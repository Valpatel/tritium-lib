# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.monitoring — system health monitoring module."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest

from tritium_lib.monitoring import (
    ComponentHealth,
    ComponentStatus,
    HealthCheck,
    HealthMonitor,
    MetricSample,
    MetricsCollector,
    MetricWindow,
    SystemStatus,
)
from tritium_lib.monitoring.health import (
    make_event_bus_check,
    make_fusion_check,
    make_store_check,
    make_tracker_check,
)


# =========================================================================
# MetricSample
# =========================================================================

class TestMetricSample:
    def test_frozen_dataclass(self):
        s = MetricSample(value=1.5, timestamp=100.0)
        assert s.value == 1.5
        assert s.timestamp == 100.0
        with pytest.raises(AttributeError):
            s.value = 2.0  # type: ignore[misc]


# =========================================================================
# MetricWindow
# =========================================================================

class TestMetricWindow:
    def test_add_and_count(self):
        w = MetricWindow(window_seconds=300)
        w.add(1.0)
        w.add(2.0)
        w.add(3.0)
        assert w.count == 3

    def test_total(self):
        w = MetricWindow(window_seconds=300)
        w.add(10.0)
        w.add(20.0)
        assert abs(w.total - 30.0) < 1e-6

    def test_empty_stats(self):
        w = MetricWindow()
        stats = w.get_stats()
        assert stats["count"] == 0
        assert stats["mean"] == 0.0
        assert stats["p95"] == 0.0

    def test_stats_basic(self):
        w = MetricWindow(window_seconds=300)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            w.add(v)
        stats = w.get_stats()
        assert stats["count"] == 5
        assert stats["mean"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["p50"] == 3.0

    def test_window_expiry(self):
        """Samples older than the window are pruned."""
        w = MetricWindow(window_seconds=1.0)
        old_time = time.time() - 5.0
        w.add(100.0, timestamp=old_time)
        w.add(1.0)  # current
        stats = w.get_stats()
        assert stats["count"] == 1
        assert stats["mean"] == 1.0

    def test_max_samples_cap(self):
        w = MetricWindow(window_seconds=9999, max_samples=10)
        for i in range(20):
            w.add(float(i))
        assert w.count <= 10

    def test_clear(self):
        w = MetricWindow()
        w.add(5.0)
        w.add(10.0)
        w.clear()
        assert w.count == 0
        assert w.total == 0.0

    def test_percentile_single_value(self):
        w = MetricWindow()
        w.add(42.0)
        stats = w.get_stats()
        assert stats["p50"] == 42.0
        assert stats["p99"] == 42.0

    def test_get_values(self):
        w = MetricWindow(window_seconds=300)
        w.add(1.0)
        w.add(2.0)
        vals = w.get_values()
        assert vals == [1.0, 2.0]


# =========================================================================
# MetricsCollector
# =========================================================================

class TestMetricsCollector:
    def test_record_and_get_latency(self):
        mc = MetricsCollector()
        mc.record_latency("pipeline", 0.01)
        mc.record_latency("pipeline", 0.02)
        mc.record_latency("pipeline", 0.03)
        stats = mc.get_stats("pipeline")
        assert stats["count"] == 3
        assert abs(stats["mean"] - 0.02) < 1e-6

    def test_get_stats_unknown_metric(self):
        mc = MetricsCollector()
        stats = mc.get_stats("nonexistent")
        assert stats["count"] == 0

    def test_counter_increment(self):
        mc = MetricsCollector()
        assert mc.get_counter("events") == 0.0
        mc.increment("events")
        assert mc.get_counter("events") == 1.0
        mc.increment("events", 5)
        assert mc.get_counter("events") == 6.0

    def test_counter_reset(self):
        mc = MetricsCollector()
        mc.increment("x", 10)
        mc.reset_counter("x")
        assert mc.get_counter("x") == 0.0

    def test_gauge_set_and_get(self):
        mc = MetricsCollector()
        assert mc.get_gauge("targets") == 0.0
        mc.set_gauge("targets", 42.0)
        assert mc.get_gauge("targets") == 42.0

    def test_gauge_with_timestamp(self):
        mc = MetricsCollector()
        mc.set_gauge("q", 7.0)
        val, ts = mc.get_gauge_with_timestamp("q")
        assert val == 7.0
        assert ts > 0

    def test_gauge_unset_returns_zero(self):
        mc = MetricsCollector()
        val, ts = mc.get_gauge_with_timestamp("missing")
        assert val == 0.0
        assert ts == 0.0

    def test_get_all_counters(self):
        mc = MetricsCollector()
        mc.increment("a")
        mc.increment("b", 5)
        all_c = mc.get_all_counters()
        assert all_c["a"] == 1.0
        assert all_c["b"] == 5.0

    def test_get_all_gauges(self):
        mc = MetricsCollector()
        mc.set_gauge("x", 1.0)
        mc.set_gauge("y", 2.0)
        all_g = mc.get_all_gauges()
        assert all_g == {"x": 1.0, "y": 2.0}

    def test_get_all_latency_names(self):
        mc = MetricsCollector()
        mc.record_latency("a.b", 1.0)
        mc.record_latency("c.d", 2.0)
        names = mc.get_all_latency_names()
        assert set(names) == {"a.b", "c.d"}

    def test_export(self):
        mc = MetricsCollector()
        mc.increment("events", 3)
        mc.set_gauge("depth", 10.0)
        mc.record_latency("query", 0.05)
        export = mc.export()
        assert export["counters"]["events"] == 3.0
        assert export["gauges"]["depth"] == 10.0
        assert export["latencies"]["query"]["count"] == 1
        assert "timestamp" in export

    def test_clear(self):
        mc = MetricsCollector()
        mc.increment("a")
        mc.set_gauge("b", 1.0)
        mc.record_latency("c", 0.1)
        mc.clear()
        assert mc.get_counter("a") == 0.0
        assert mc.get_gauge("b") == 0.0
        assert mc.get_stats("c")["count"] == 0

    def test_thread_safety(self):
        mc = MetricsCollector()
        errors = []

        def writer():
            try:
                for i in range(100):
                    mc.record_latency("shared", float(i) * 0.001)
                    mc.increment("shared_counter")
                    mc.set_gauge("shared_gauge", float(i))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert mc.get_counter("shared_counter") == 400.0


# =========================================================================
# ComponentHealth
# =========================================================================

class TestComponentHealth:
    def test_defaults(self):
        ch = ComponentHealth(name="test")
        assert ch.name == "test"
        assert ch.status == ComponentStatus.UNKNOWN
        assert ch.message == ""
        assert ch.details == {}
        assert ch.error == ""

    def test_is_healthy(self):
        ch = ComponentHealth(name="t", status=ComponentStatus.UP)
        assert ch.is_healthy
        assert not ch.is_degraded
        assert not ch.is_down

    def test_is_degraded(self):
        ch = ComponentHealth(name="t", status=ComponentStatus.DEGRADED)
        assert not ch.is_healthy
        assert ch.is_degraded
        assert not ch.is_down

    def test_is_down(self):
        ch = ComponentHealth(name="t", status=ComponentStatus.DOWN)
        assert not ch.is_healthy
        assert not ch.is_degraded
        assert ch.is_down

    def test_to_dict(self):
        ch = ComponentHealth(
            name="tracker",
            status=ComponentStatus.UP,
            message="OK",
            details={"targets": 10},
        )
        d = ch.to_dict()
        assert d["name"] == "tracker"
        assert d["status"] == "up"
        assert d["message"] == "OK"
        assert d["details"]["targets"] == 10


# =========================================================================
# SystemStatus
# =========================================================================

class TestSystemStatus:
    def test_empty_system(self):
        ss = SystemStatus()
        assert ss.component_count == 0
        assert ss.healthy_count == 0

    def test_counts(self):
        ss = SystemStatus(
            components={
                "a": ComponentHealth(name="a", status=ComponentStatus.UP),
                "b": ComponentHealth(name="b", status=ComponentStatus.DEGRADED),
                "c": ComponentHealth(name="c", status=ComponentStatus.DOWN),
            }
        )
        assert ss.component_count == 3
        assert ss.healthy_count == 1
        assert ss.degraded_count == 1
        assert ss.down_count == 1

    def test_to_dict(self):
        ss = SystemStatus(
            overall=ComponentStatus.UP,
            components={
                "x": ComponentHealth(name="x", status=ComponentStatus.UP),
            },
        )
        d = ss.to_dict()
        assert d["overall"] == "up"
        assert d["component_count"] == 1
        assert "x" in d["components"]


# =========================================================================
# HealthMonitor
# =========================================================================

class TestHealthMonitor:
    def test_register_and_check(self):
        monitor = HealthMonitor()
        monitor.register("comp", lambda: ComponentHealth(
            name="comp", status=ComponentStatus.UP, message="all good",
        ))
        result = monitor.check("comp")
        assert result.status == ComponentStatus.UP

    def test_check_unregistered(self):
        monitor = HealthMonitor()
        result = monitor.check("ghost")
        assert result.status == ComponentStatus.UNKNOWN

    def test_unregister(self):
        monitor = HealthMonitor()
        monitor.register("temp", lambda: ComponentHealth(
            name="temp", status=ComponentStatus.UP,
        ))
        assert monitor.unregister("temp")
        assert not monitor.unregister("temp")  # already gone
        assert "temp" not in monitor.registered_components

    def test_registered_components(self):
        monitor = HealthMonitor()
        monitor.register("a", lambda: ComponentHealth(name="a", status=ComponentStatus.UP))
        monitor.register("b", lambda: ComponentHealth(name="b", status=ComponentStatus.UP))
        assert set(monitor.registered_components) == {"a", "b"}

    def test_check_all_all_up(self):
        monitor = HealthMonitor()
        monitor.register("a", lambda: ComponentHealth(name="a", status=ComponentStatus.UP))
        monitor.register("b", lambda: ComponentHealth(name="b", status=ComponentStatus.UP))
        status = monitor.check_all()
        assert status.overall == ComponentStatus.UP
        assert status.healthy_count == 2

    def test_check_all_one_degraded(self):
        monitor = HealthMonitor()
        monitor.register("a", lambda: ComponentHealth(name="a", status=ComponentStatus.UP))
        monitor.register("b", lambda: ComponentHealth(name="b", status=ComponentStatus.DEGRADED))
        status = monitor.check_all()
        assert status.overall == ComponentStatus.DEGRADED

    def test_check_all_one_down(self):
        monitor = HealthMonitor()
        monitor.register("a", lambda: ComponentHealth(name="a", status=ComponentStatus.UP))
        monitor.register("b", lambda: ComponentHealth(name="b", status=ComponentStatus.DOWN))
        status = monitor.check_all()
        assert status.overall == ComponentStatus.DOWN

    def test_check_all_empty(self):
        monitor = HealthMonitor()
        status = monitor.check_all()
        assert status.overall == ComponentStatus.UNKNOWN

    def test_exception_in_check(self):
        def bad_check():
            raise RuntimeError("Database on fire")

        monitor = HealthMonitor()
        monitor.register("broken", bad_check)
        result = monitor.check("broken")
        assert result.status == ComponentStatus.DOWN
        assert "Database on fire" in result.error

    def test_metrics_integration(self):
        mc = MetricsCollector()
        monitor = HealthMonitor(metrics=mc)
        monitor.register("comp", lambda: ComponentHealth(
            name="comp", status=ComponentStatus.UP,
        ))
        monitor.check("comp")
        # Should have recorded check latency and status gauge
        assert mc.get_stats("health.comp.check_time")["count"] == 1
        assert mc.get_gauge("health.comp.status") == 1.0

    def test_metrics_records_down_status(self):
        mc = MetricsCollector()
        monitor = HealthMonitor(metrics=mc)
        monitor.register("bad", lambda: ComponentHealth(
            name="bad", status=ComponentStatus.DOWN,
        ))
        monitor.check("bad")
        assert mc.get_gauge("health.bad.status") == 0.0

    def test_get_last_result(self):
        monitor = HealthMonitor()
        assert monitor.get_last_result("x") is None
        monitor.register("x", lambda: ComponentHealth(
            name="x", status=ComponentStatus.UP, message="fresh",
        ))
        monitor.check("x")
        cached = monitor.get_last_result("x")
        assert cached is not None
        assert cached.message == "fresh"

    def test_get_last_status_never_checked(self):
        monitor = HealthMonitor()
        monitor.register("x", lambda: ComponentHealth(
            name="x", status=ComponentStatus.UP,
        ))
        status = monitor.get_last_status()
        # Never checked => UNKNOWN
        assert status.components["x"].status == ComponentStatus.UNKNOWN

    def test_check_all_records_overall_gauge(self):
        mc = MetricsCollector()
        monitor = HealthMonitor(metrics=mc)
        monitor.register("a", lambda: ComponentHealth(
            name="a", status=ComponentStatus.UP,
        ))
        monitor.check_all()
        assert mc.get_gauge("health.system.overall") == 1.0


# =========================================================================
# Built-in health check factories
# =========================================================================

class TestMakeTrackerCheck:
    def _make_tracker(self, targets):
        """Create a mock tracker with get_all() returning given targets."""
        @dataclass
        class FakeTarget:
            target_id: str = "t1"
            last_seen: float = field(default_factory=time.monotonic)

        tracker = MagicMock()
        tracker.get_all.return_value = [
            FakeTarget(**t) if isinstance(t, dict) else t for t in targets
        ]
        return tracker

    def test_healthy_tracker(self):
        @dataclass
        class T:
            target_id: str = "t1"
            last_seen: float = field(default_factory=time.monotonic)

        tracker = MagicMock()
        tracker.get_all.return_value = [T(), T()]
        check = make_tracker_check(tracker)
        result = check()
        assert result.status == ComponentStatus.UP
        assert result.details["target_count"] == 2

    def test_empty_tracker(self):
        tracker = MagicMock()
        tracker.get_all.return_value = []
        check = make_tracker_check(tracker)
        result = check()
        assert result.status == ComponentStatus.UP
        assert result.details["target_count"] == 0

    def test_stale_targets_degrade(self):
        @dataclass
        class T:
            target_id: str = "t1"
            last_seen: float = 0.0  # very old monotonic time

        tracker = MagicMock()
        tracker.get_all.return_value = [T(), T(), T()]
        check = make_tracker_check(tracker, stale_seconds=10, max_stale_ratio=0.3)
        result = check()
        assert result.status == ComponentStatus.DEGRADED

    def test_tracker_exception(self):
        tracker = MagicMock()
        tracker.get_all.side_effect = RuntimeError("boom")
        check = make_tracker_check(tracker)
        result = check()
        assert result.status == ComponentStatus.DOWN
        assert "boom" in result.error


class TestMakeFusionCheck:
    def test_healthy_fusion(self):
        engine = MagicMock()
        engine.get_fused_targets.return_value = [1, 2, 3]
        engine.metrics = None
        check = make_fusion_check(engine)
        result = check()
        assert result.status == ComponentStatus.UP
        assert result.details["fused_target_count"] == 3

    def test_fusion_exception(self):
        engine = MagicMock()
        engine.get_fused_targets.side_effect = RuntimeError("fusion error")
        check = make_fusion_check(engine)
        result = check()
        assert result.status == ComponentStatus.DOWN

    def test_low_fusion_rate_degrades(self):
        engine = MagicMock()
        engine.get_fused_targets.return_value = []
        metrics = MagicMock()
        metrics.get_status.return_value = {
            "total_fusions": 5,
            "hourly_rate": 0.5,
            "total_pending": 2,
        }
        engine.metrics = metrics
        check = make_fusion_check(engine, min_fusion_rate=10.0)
        result = check()
        assert result.status == ComponentStatus.DEGRADED
        assert "Low fusion rate" in result.message


class TestMakeEventBusCheck:
    def test_healthy_bus(self):
        bus = MagicMock()
        bus._subscribers = {
            "topic.a": [1, 2],
            "topic.b": [3],
        }
        bus._queue = None
        check = make_event_bus_check(bus)
        result = check()
        assert result.status == ComponentStatus.UP
        assert result.details["subscriber_count"] == 3
        assert result.details["topic_count"] == 2

    def test_queue_overflow_degrades(self):
        import queue

        bus = MagicMock()
        bus._subscribers = {"t": [1]}
        q = queue.Queue()
        for i in range(50):
            q.put(i)
        bus._queue = q
        check = make_event_bus_check(bus, max_queue_depth=10)
        result = check()
        assert result.status == ComponentStatus.DEGRADED


class TestMakeStoreCheck:
    def test_healthy_store(self):
        store = MagicMock()
        conn = MagicMock()
        store._conn = conn
        store._db_path = ":memory:"

        # Mock cursor for SELECT 1
        cursor1 = MagicMock()
        cursor1.fetchone.return_value = (1,)

        # Mock cursor for table list
        cursor_tables = MagicMock()
        cursor_tables.fetchall.return_value = [("targets",), ("events",)]

        # Mock cursor for count queries
        cursor_count = MagicMock()
        cursor_count.fetchone.return_value = (42,)

        conn.execute.side_effect = [cursor1, cursor_tables, cursor_count, cursor_count]

        check = make_store_check(store, name="test_store")
        result = check()
        assert result.status == ComponentStatus.UP

    def test_store_no_connection(self):
        store = MagicMock()
        store._conn = None
        check = make_store_check(store, name="dead_store")
        result = check()
        assert result.status == ComponentStatus.DOWN
        assert "No database connection" in result.message

    def test_store_query_failure(self):
        store = MagicMock()
        conn = MagicMock()
        store._conn = conn
        conn.execute.side_effect = RuntimeError("disk full")
        check = make_store_check(store, name="bad_store")
        result = check()
        assert result.status == ComponentStatus.DOWN
        assert "disk full" in result.error
