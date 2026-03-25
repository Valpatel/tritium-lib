# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.intelligence.anomaly_engine module."""
import time

import pytest

from tritium_lib.intelligence.anomaly_engine import (
    AnomalyAlert,
    AnomalyEngine,
    KnownPattern,
    ZoneBaseline,
    _mean_std,
    _score_from_sigma,
    _severity_from_sigma,
)
from tritium_lib.events.bus import EventBus


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _feed_normal_observations(
    engine: AnomalyEngine,
    zone_id: str = "zone_a",
    count: int = 50,
    speed_base: float = 1.5,
    dwell_base: float = 120.0,
    entity_count_base: float = 10.0,
    hour: int = 14,
):
    """Feed normal observations into a zone to build a realistic baseline."""
    for i in range(count):
        # Add slight variation to simulate real-world data
        speed = speed_base + (i % 5) * 0.1 - 0.2
        dwell = dwell_base + (i % 7) * 5.0 - 15.0
        ent_count = entity_count_base + (i % 3) - 1.0
        engine.observe(
            zone_id,
            target_id=f"target_{i % 20}",
            speed=speed,
            dwell_seconds=dwell,
            entity_count=ent_count,
            hour_of_day=hour,
        )


class TestHelperFunctions:
    """Test the module-level utility functions."""

    def test_mean_std_empty(self):
        mean, std = _mean_std([])
        assert mean == 0.0
        assert std == 0.0

    def test_mean_std_single(self):
        mean, std = _mean_std([5.0])
        assert mean == 5.0
        assert std == 0.0

    def test_mean_std_normal(self):
        mean, std = _mean_std([10.0, 10.0, 10.0, 10.0])
        assert mean == 10.0
        assert std == 0.0

    def test_mean_std_with_variance(self):
        mean, std = _mean_std([8.0, 12.0])
        assert mean == 10.0
        assert std == 2.0

    def test_severity_from_sigma(self):
        assert _severity_from_sigma(1.0) == "low"
        assert _severity_from_sigma(3.0) == "medium"
        assert _severity_from_sigma(4.0) == "high"
        assert _severity_from_sigma(5.0) == "critical"
        assert _severity_from_sigma(10.0) == "critical"

    def test_score_from_sigma(self):
        assert _score_from_sigma(0.0) == 0.0
        assert _score_from_sigma(2.5) == 0.5
        assert _score_from_sigma(5.0) == 1.0
        assert _score_from_sigma(10.0) == 1.0  # capped at 1.0


class TestAnomalyAlert:
    """Test the AnomalyAlert dataclass."""

    def test_default_values(self):
        alert = AnomalyAlert()
        assert alert.alert_type == ""
        assert alert.severity == "low"
        assert alert.score == 0.0
        assert alert.suppressed is False

    def test_to_dict(self):
        alert = AnomalyAlert(
            alert_type="speed",
            zone_id="zone_a",
            target_id="ble_abc",
            severity="high",
            score=0.85,
            observed_value=15.0,
            baseline_mean=1.5,
            baseline_std=0.3,
            deviation_sigma=45.0,
            detail="Speed anomaly",
        )
        d = alert.to_dict()
        assert d["alert_type"] == "speed"
        assert d["zone_id"] == "zone_a"
        assert d["severity"] == "high"
        assert d["score"] == 0.85
        assert d["suppressed"] is False

    def test_to_dict_rounding(self):
        alert = AnomalyAlert(
            score=0.123456789,
            observed_value=1.23456789,
            baseline_mean=2.34567891,
            baseline_std=0.11111111,
            deviation_sigma=3.14159,
        )
        d = alert.to_dict()
        assert d["score"] == 0.1235
        assert d["deviation_sigma"] == 3.14


class TestBaselineLearning:
    """Test that the engine correctly learns baselines from observations."""

    def test_observe_creates_baseline(self):
        engine = AnomalyEngine()
        engine.observe("zone_lobby", speed=1.5, dwell_seconds=60.0)
        bl = engine.get_baseline("zone_lobby")
        assert bl is not None
        assert bl.zone_id == "zone_lobby"
        assert bl.observation_count == 1

    def test_observe_accumulates(self):
        engine = AnomalyEngine()
        for i in range(20):
            engine.observe("zone_a", speed=float(i), hour_of_day=10)
        bl = engine.get_baseline("zone_a")
        assert bl.observation_count == 20
        assert len(bl.speed_all) == 20
        assert len(bl.speed_by_hour[10]) == 20

    def test_observe_tracks_targets(self):
        engine = AnomalyEngine()
        engine.observe("zone_a", target_id="t1")
        engine.observe("zone_a", target_id="t1")
        engine.observe("zone_a", target_id="t2")
        bl = engine.get_baseline("zone_a")
        assert bl.target_visits["t1"] == 2
        assert bl.target_visits["t2"] == 1

    def test_observe_batch(self):
        engine = AnomalyEngine()
        batch = [
            {"zone_id": "z1", "speed": 1.0, "target_id": "t1"},
            {"zone_id": "z1", "speed": 2.0, "target_id": "t2"},
            {"zone_id": "z2", "speed": 3.0, "target_id": "t3"},
            {"zone_id": "", "speed": 4.0},  # skipped — no zone_id
        ]
        count = engine.observe_batch(batch)
        assert count == 3
        assert engine.get_baseline("z1").observation_count == 2
        assert engine.get_baseline("z2").observation_count == 1
        assert engine.get_baseline("") is None

    def test_observe_hourly_segmentation(self):
        engine = AnomalyEngine()
        engine.observe("z1", speed=1.0, hour_of_day=8)
        engine.observe("z1", speed=10.0, hour_of_day=20)
        bl = engine.get_baseline("z1")
        assert len(bl.speed_by_hour[8]) == 1
        assert len(bl.speed_by_hour[20]) == 1
        assert bl.speed_by_hour[8][0] == 1.0
        assert bl.speed_by_hour[20][0] == 10.0

    def test_baseline_returns_none_for_unknown_zone(self):
        engine = AnomalyEngine()
        assert engine.get_baseline("nonexistent") is None

    def test_get_zone_stats(self):
        engine = AnomalyEngine(min_baseline_samples=5)
        _feed_normal_observations(engine, "z1", count=30, hour=10)
        stats = engine.get_zone_stats("z1", hour=10)
        assert stats["zone_id"] == "z1"
        assert stats["speed"]["samples"] == 30
        assert stats["dwell"]["samples"] == 30
        assert stats["speed"]["mean"] > 0

    def test_get_zone_stats_global(self):
        engine = AnomalyEngine()
        _feed_normal_observations(engine, "z1", count=30, hour=10)
        stats = engine.get_zone_stats("z1")
        assert stats["hour"] is None
        assert stats["speed"]["samples"] == 30

    def test_get_zone_stats_unknown_zone(self):
        engine = AnomalyEngine()
        stats = engine.get_zone_stats("nonexistent")
        assert stats["status"] == "no_baseline"


class TestSpeedAnomaly:
    """Test speed anomaly detection."""

    def test_no_anomaly_for_normal_speed(self):
        engine = AnomalyEngine(min_baseline_samples=10, speed_threshold_sigma=2.5)
        _feed_normal_observations(engine, speed_base=1.5, count=50)
        alerts = engine.check_target("zone_a", speed=1.5, hour_of_day=14)
        assert len(alerts) == 0

    def test_speed_anomaly_above(self):
        engine = AnomalyEngine(min_baseline_samples=10, speed_threshold_sigma=2.5)
        _feed_normal_observations(engine, speed_base=1.5, count=50)
        alerts = engine.check_target("zone_a", speed=50.0, hour_of_day=14)
        speed_alerts = [a for a in alerts if a.alert_type == "speed"]
        assert len(speed_alerts) == 1
        assert speed_alerts[0].observed_value == 50.0
        assert speed_alerts[0].deviation_sigma > 2.5

    def test_speed_anomaly_below(self):
        engine = AnomalyEngine(min_baseline_samples=10, speed_threshold_sigma=2.5)
        # Build baseline with high speeds
        for i in range(50):
            engine.observe("z1", speed=100.0 + (i % 5), hour_of_day=14)
        alerts = engine.check_target("z1", speed=0.1, hour_of_day=14)
        speed_alerts = [a for a in alerts if a.alert_type == "speed"]
        assert len(speed_alerts) == 1
        assert "below" in speed_alerts[0].detail

    def test_insufficient_baseline_no_alert(self):
        engine = AnomalyEngine(min_baseline_samples=10)
        for i in range(5):  # Only 5 samples, need 10
            engine.observe("z1", speed=1.0, hour_of_day=14)
        alerts = engine.check_target("z1", speed=100.0, hour_of_day=14)
        assert len(alerts) == 0


class TestDwellAnomaly:
    """Test dwell time anomaly detection."""

    def test_no_anomaly_for_normal_dwell(self):
        engine = AnomalyEngine(min_baseline_samples=10, dwell_threshold_sigma=2.5)
        _feed_normal_observations(engine, dwell_base=120.0, count=50)
        alerts = engine.check_target("zone_a", dwell_seconds=120.0, hour_of_day=14)
        assert all(a.alert_type != "dwell" for a in alerts)

    def test_dwell_anomaly_long_stay(self):
        engine = AnomalyEngine(min_baseline_samples=10, dwell_threshold_sigma=2.5)
        _feed_normal_observations(engine, dwell_base=120.0, count=50)
        alerts = engine.check_target("zone_a", dwell_seconds=5000.0, hour_of_day=14)
        dwell_alerts = [a for a in alerts if a.alert_type == "dwell"]
        assert len(dwell_alerts) == 1
        assert dwell_alerts[0].observed_value == 5000.0
        assert dwell_alerts[0].severity in ("medium", "high", "critical")


class TestCountAnomaly:
    """Test zone entity count anomaly detection."""

    def test_no_anomaly_normal_count(self):
        engine = AnomalyEngine(min_baseline_samples=10, count_threshold_sigma=2.5)
        _feed_normal_observations(engine, entity_count_base=10.0, count=50)
        alert = engine.check_zone_count("zone_a", 10.0, hour_of_day=14)
        assert alert is None

    def test_count_anomaly_surge(self):
        engine = AnomalyEngine(min_baseline_samples=10, count_threshold_sigma=2.5)
        _feed_normal_observations(engine, entity_count_base=10.0, count=50)
        alert = engine.check_zone_count("zone_a", 200.0, hour_of_day=14)
        assert alert is not None
        assert alert.alert_type == "count"
        assert alert.observed_value == 200.0

    def test_count_anomaly_deserted(self):
        engine = AnomalyEngine(min_baseline_samples=10, count_threshold_sigma=2.5)
        # Baseline with normally busy zone
        for i in range(50):
            engine.observe("z1", entity_count=50.0 + (i % 5), hour_of_day=14)
        alert = engine.check_zone_count("z1", 0.0, hour_of_day=14)
        assert alert is not None
        assert alert.alert_type == "count"

    def test_count_no_baseline(self):
        engine = AnomalyEngine()
        alert = engine.check_zone_count("nonexistent", 100.0)
        assert alert is None


class TestRouteAnomaly:
    """Test route (never-seen-in-zone) anomaly detection."""

    def test_no_route_anomaly_for_known_target(self):
        engine = AnomalyEngine(min_baseline_samples=5, route_min_visits=10)
        # Build zone with many distinct targets
        for i in range(30):
            engine.observe("z1", target_id=f"t_{i}", speed=1.0)
        # t_0 has been here before
        alerts = engine.check_target("z1", target_id="t_0", speed=1.0)
        route_alerts = [a for a in alerts if a.alert_type == "route"]
        assert len(route_alerts) == 0

    def test_route_anomaly_for_new_target(self):
        engine = AnomalyEngine(
            min_baseline_samples=5,
            route_min_visits=10,
            cooldown_seconds=0,
        )
        # Build zone with 25 distinct targets
        for i in range(25):
            engine.observe("z1", target_id=f"known_{i}", speed=1.0)
        # Totally new target appears
        alerts = engine.check_target("z1", target_id="intruder", speed=1.0)
        route_alerts = [a for a in alerts if a.alert_type == "route"]
        assert len(route_alerts) == 1
        assert route_alerts[0].target_id == "intruder"

    def test_route_anomaly_needs_min_diversity(self):
        engine = AnomalyEngine(route_min_visits=20, cooldown_seconds=0)
        # Only 5 unique targets — not enough to judge
        for i in range(5):
            engine.observe("z1", target_id=f"t_{i}")
        alerts = engine.check_target("z1", target_id="new_target")
        route_alerts = [a for a in alerts if a.alert_type == "route"]
        assert len(route_alerts) == 0


class TestReappearanceAnomaly:
    """Test target reappearance anomaly detection."""

    def test_no_anomaly_short_absence(self):
        engine = AnomalyEngine()
        alert = engine.check_reappearance("z1", "t1", absence_seconds=3600)
        assert alert is None  # 1 hour is not anomalous

    def test_reappearance_anomaly_long_absence(self):
        engine = AnomalyEngine()
        # 5 days absent
        alert = engine.check_reappearance("z1", "t1", absence_seconds=5 * 86400)
        assert alert is not None
        assert alert.alert_type == "reappearance"
        assert alert.severity in ("medium", "high", "critical")

    def test_reappearance_just_below_threshold(self):
        engine = AnomalyEngine()
        # 1.5 days — ratio=1.5, sigma=1.5 < 2.0 threshold
        alert = engine.check_reappearance("z1", "t1", absence_seconds=1.5 * 86400)
        assert alert is None


class TestSeverityCalculation:
    """Test that severity levels are assigned correctly based on deviation."""

    def test_extreme_deviation_is_critical(self):
        engine = AnomalyEngine(min_baseline_samples=10, speed_threshold_sigma=2.0)
        # Build very tight baseline
        for i in range(50):
            engine.observe("z1", speed=10.0, hour_of_day=12)
        # Huge deviation
        alerts = engine.check_target("z1", speed=1000.0, hour_of_day=12)
        speed_alerts = [a for a in alerts if a.alert_type == "speed"]
        assert len(speed_alerts) == 1
        assert speed_alerts[0].severity == "critical" or speed_alerts[0].severity == "high"
        assert speed_alerts[0].score >= 0.8


class TestKnownPatternSuppression:
    """Test false-positive handling via known patterns."""

    def test_suppressed_alert_is_marked(self):
        engine = AnomalyEngine(min_baseline_samples=10, speed_threshold_sigma=2.0, cooldown_seconds=0)
        _feed_normal_observations(engine, speed_base=1.5, count=50)

        engine.add_known_pattern(KnownPattern(
            pattern_id="delivery_truck",
            zone_id="zone_a",
            alert_type="speed",
            max_value=60.0,
            reason="Delivery trucks routinely drive fast through zone_a",
        ))

        alerts = engine.check_target("zone_a", speed=50.0, hour_of_day=14)
        speed_alerts = [a for a in alerts if a.alert_type == "speed"]
        assert len(speed_alerts) == 1
        assert speed_alerts[0].suppressed is True
        assert "delivery_truck" in speed_alerts[0].detail

    def test_unsuppressed_when_value_exceeds_pattern(self):
        engine = AnomalyEngine(min_baseline_samples=10, speed_threshold_sigma=2.0, cooldown_seconds=0)
        _feed_normal_observations(engine, speed_base=1.5, count=50)

        engine.add_known_pattern(KnownPattern(
            pattern_id="delivery_truck",
            zone_id="zone_a",
            alert_type="speed",
            max_value=30.0,
            reason="Delivery trucks cap at 30",
        ))

        # Speed exceeds the pattern max_value
        alerts = engine.check_target("zone_a", speed=50.0, hour_of_day=14)
        speed_alerts = [a for a in alerts if a.alert_type == "speed"]
        assert len(speed_alerts) == 1
        assert speed_alerts[0].suppressed is False

    def test_remove_known_pattern(self):
        engine = AnomalyEngine()
        engine.add_known_pattern(KnownPattern(pattern_id="p1"))
        engine.add_known_pattern(KnownPattern(pattern_id="p2"))
        assert len(engine.get_known_patterns()) == 2

        removed = engine.remove_known_pattern("p1")
        assert removed is True
        assert len(engine.get_known_patterns()) == 1

        removed = engine.remove_known_pattern("nonexistent")
        assert removed is False

    def test_suppressed_alerts_excluded_from_history_by_default(self):
        engine = AnomalyEngine(min_baseline_samples=10, speed_threshold_sigma=2.0, cooldown_seconds=0)
        _feed_normal_observations(engine, speed_base=1.5, count=50)

        engine.add_known_pattern(KnownPattern(
            pattern_id="suppress_all",
            alert_type="speed",
            max_value=1000.0,
        ))

        engine.check_target("zone_a", speed=50.0, hour_of_day=14)

        # Default excludes suppressed
        history = engine.get_alert_history()
        assert len(history) == 0

        # Include suppressed
        history_all = engine.get_alert_history(include_suppressed=True)
        assert len(history_all) == 1
        assert history_all[0].suppressed is True


class TestCooldown:
    """Test alert cooldown to prevent alert flooding."""

    def test_cooldown_suppresses_rapid_alerts(self):
        engine = AnomalyEngine(
            min_baseline_samples=10,
            speed_threshold_sigma=2.0,
            cooldown_seconds=60.0,
        )
        _feed_normal_observations(engine, speed_base=1.5, count=50)

        ts = time.time()
        alerts1 = engine.check_target(
            "zone_a", target_id="t1", speed=50.0,
            hour_of_day=14, timestamp=ts,
        )
        alerts2 = engine.check_target(
            "zone_a", target_id="t1", speed=50.0,
            hour_of_day=14, timestamp=ts + 10,
        )

        speed1 = [a for a in alerts1 if a.alert_type == "speed"]
        speed2 = [a for a in alerts2 if a.alert_type == "speed"]

        assert len(speed1) == 1  # First fires
        assert len(speed2) == 0  # Second suppressed by cooldown

    def test_cooldown_allows_after_expiry(self):
        engine = AnomalyEngine(
            min_baseline_samples=10,
            speed_threshold_sigma=2.0,
            cooldown_seconds=60.0,
        )
        _feed_normal_observations(engine, speed_base=1.5, count=50)

        ts = time.time()
        alerts1 = engine.check_target(
            "zone_a", target_id="t1", speed=50.0,
            hour_of_day=14, timestamp=ts,
        )
        alerts2 = engine.check_target(
            "zone_a", target_id="t1", speed=50.0,
            hour_of_day=14, timestamp=ts + 120,  # after cooldown
        )

        speed1 = [a for a in alerts1 if a.alert_type == "speed"]
        speed2 = [a for a in alerts2 if a.alert_type == "speed"]

        assert len(speed1) == 1
        assert len(speed2) == 1


class TestEventBusIntegration:
    """Test EventBus integration for real-time alert publishing."""

    def test_alerts_published_to_event_bus(self):
        bus = EventBus()
        received = []
        bus.subscribe("anomaly.alert", lambda e: received.append(e))

        engine = AnomalyEngine(
            event_bus=bus,
            min_baseline_samples=10,
            speed_threshold_sigma=2.0,
            cooldown_seconds=0,
        )
        _feed_normal_observations(engine, speed_base=1.5, count=50)

        engine.check_target("zone_a", speed=50.0, hour_of_day=14)
        assert len(received) >= 1
        assert received[0].data["alert_type"] == "speed"

    def test_typed_topic_published(self):
        bus = EventBus()
        speed_alerts = []
        bus.subscribe("anomaly.alert.speed", lambda e: speed_alerts.append(e))

        engine = AnomalyEngine(
            event_bus=bus,
            min_baseline_samples=10,
            speed_threshold_sigma=2.0,
            cooldown_seconds=0,
        )
        _feed_normal_observations(engine, speed_base=1.5, count=50)

        engine.check_target("zone_a", speed=50.0, hour_of_day=14)
        assert len(speed_alerts) == 1
        assert speed_alerts[0].source == "anomaly_engine"

    def test_suppressed_alerts_not_published(self):
        bus = EventBus()
        received = []
        bus.subscribe("anomaly.alert", lambda e: received.append(e))

        engine = AnomalyEngine(
            event_bus=bus,
            min_baseline_samples=10,
            speed_threshold_sigma=2.0,
            cooldown_seconds=0,
        )
        _feed_normal_observations(engine, speed_base=1.5, count=50)
        engine.add_known_pattern(KnownPattern(
            pattern_id="suppress",
            alert_type="speed",
            max_value=1000.0,
        ))

        engine.check_target("zone_a", speed=50.0, hour_of_day=14)
        assert len(received) == 0  # Suppressed, not published

    def test_no_crash_without_event_bus(self):
        engine = AnomalyEngine(
            min_baseline_samples=10,
            speed_threshold_sigma=2.0,
            cooldown_seconds=0,
        )
        _feed_normal_observations(engine, speed_base=1.5, count=50)
        # Should not raise
        alerts = engine.check_target("zone_a", speed=50.0, hour_of_day=14)
        assert len(alerts) >= 1


class TestFusionEngineIntegration:
    """Test integration with FusionEngine-style target objects."""

    def test_ingest_from_fusion_with_duck_typed_targets(self):
        engine = AnomalyEngine(
            min_baseline_samples=5,
            speed_threshold_sigma=2.0,
            cooldown_seconds=0,
        )
        # Build a baseline first
        for i in range(20):
            engine.observe("z1", target_id=f"t_{i}", speed=1.5, entity_count=10.0)

        class FakeTarget:
            def __init__(self, tid, spd, zones):
                self.target_id = tid
                self.speed = spd
                self.zones = zones
                self.position = (0.0, 0.0)

        targets = [
            FakeTarget("t_normal", 1.5, {"z1"}),
            FakeTarget("t_fast", 100.0, {"z1"}),
        ]

        alerts = engine.ingest_from_fusion(targets)
        # t_fast should trigger a speed anomaly
        speed_alerts = [a for a in alerts if a.alert_type == "speed"]
        assert len(speed_alerts) >= 1

    def test_ingest_with_zone_assignments(self):
        engine = AnomalyEngine(
            min_baseline_samples=5,
            speed_threshold_sigma=2.0,
            cooldown_seconds=0,
        )
        for i in range(20):
            engine.observe("z1", target_id=f"t_{i}", speed=1.5, entity_count=5.0)

        class SimpleTarget:
            def __init__(self, tid, spd):
                self.target_id = tid
                self.speed = spd
                self.position = (0.0, 0.0)

        targets = [SimpleTarget("t1", 1.5)]
        zone_map = {"z1": {"t1"}}

        # Should not crash, and zone count observation occurs
        alerts = engine.ingest_from_fusion(targets, zone_assignments=zone_map)
        assert isinstance(alerts, list)


class TestAlertHistory:
    """Test alert history retrieval and filtering."""

    def test_alert_history_ordering(self):
        engine = AnomalyEngine(min_baseline_samples=10, speed_threshold_sigma=2.0, cooldown_seconds=0)
        _feed_normal_observations(engine, speed_base=1.5, count=50)

        ts = time.time()
        engine.check_target("zone_a", target_id="t1", speed=50.0, hour_of_day=14, timestamp=ts)
        engine.check_target("zone_a", target_id="t2", speed=60.0, hour_of_day=14, timestamp=ts + 1)

        history = engine.get_alert_history()
        assert len(history) >= 2
        # Most recent first
        assert history[0].timestamp >= history[1].timestamp

    def test_alert_history_filter_by_zone(self):
        engine = AnomalyEngine(min_baseline_samples=10, speed_threshold_sigma=2.0, cooldown_seconds=0)
        _feed_normal_observations(engine, "z1", speed_base=1.5, count=50)
        _feed_normal_observations(engine, "z2", speed_base=1.5, count=50)

        engine.check_target("z1", speed=50.0, hour_of_day=14)
        engine.check_target("z2", speed=50.0, hour_of_day=14)

        z1_history = engine.get_alert_history(zone_id="z1")
        z2_history = engine.get_alert_history(zone_id="z2")

        assert all(a.zone_id == "z1" for a in z1_history)
        assert all(a.zone_id == "z2" for a in z2_history)

    def test_alert_history_filter_by_type(self):
        engine = AnomalyEngine(min_baseline_samples=10, speed_threshold_sigma=2.0, cooldown_seconds=0)
        _feed_normal_observations(engine, speed_base=1.5, count=50)

        engine.check_target("zone_a", speed=50.0, dwell_seconds=50000.0, hour_of_day=14)

        speed_history = engine.get_alert_history(alert_type="speed")
        assert all(a.alert_type == "speed" for a in speed_history)

    def test_alert_history_limit(self):
        engine = AnomalyEngine(min_baseline_samples=10, speed_threshold_sigma=2.0, cooldown_seconds=0)
        _feed_normal_observations(engine, speed_base=1.5, count=50)

        ts = time.time()
        for i in range(10):
            engine.check_target(
                "zone_a", target_id=f"t_{i}", speed=50.0 + i,
                hour_of_day=14, timestamp=ts + i,
            )

        history = engine.get_alert_history(limit=3)
        assert len(history) <= 3


class TestEngineStats:
    """Test engine statistics and reset."""

    def test_stats_reflect_activity(self):
        engine = AnomalyEngine(min_baseline_samples=10, cooldown_seconds=0)
        _feed_normal_observations(engine, count=50)

        stats = engine.get_stats()
        assert stats["total_observations"] == 50
        assert stats["zone_count"] == 1
        assert stats["total_alerts"] == 0

    def test_stats_after_alerts(self):
        engine = AnomalyEngine(
            min_baseline_samples=10,
            speed_threshold_sigma=2.0,
            cooldown_seconds=0,
        )
        _feed_normal_observations(engine, speed_base=1.5, count=50)
        engine.check_target("zone_a", speed=50.0, hour_of_day=14)

        stats = engine.get_stats()
        assert stats["total_alerts"] >= 1

    def test_reset_specific_zone(self):
        engine = AnomalyEngine()
        engine.observe("z1", speed=1.0)
        engine.observe("z2", speed=2.0)
        engine.reset(zone_id="z1")
        assert engine.get_baseline("z1") is None
        assert engine.get_baseline("z2") is not None

    def test_reset_all(self):
        engine = AnomalyEngine()
        engine.observe("z1", speed=1.0)
        engine.observe("z2", speed=2.0)
        engine.reset()
        stats = engine.get_stats()
        assert stats["total_observations"] == 0
        assert stats["zone_count"] == 0


class TestHourlyFallback:
    """Test that the engine falls back to global data when hourly data is sparse."""

    def test_uses_hourly_when_available(self):
        engine = AnomalyEngine(min_baseline_samples=10, speed_threshold_sigma=2.0, cooldown_seconds=0)
        # Feed many observations at hour=10
        for i in range(50):
            engine.observe("z1", speed=1.5 + (i % 3) * 0.1, hour_of_day=10)

        # Also feed different data at another hour
        for i in range(50):
            engine.observe("z1", speed=20.0 + (i % 3) * 0.1, hour_of_day=22)

        # Check at hour=10 with a value that's anomalous for hour=10 but normal for hour=22
        alerts = engine.check_target("z1", speed=20.0, hour_of_day=10)
        speed_alerts = [a for a in alerts if a.alert_type == "speed"]
        assert len(speed_alerts) == 1  # Should compare against hour=10 baseline

    def test_falls_back_to_global_when_hourly_sparse(self):
        engine = AnomalyEngine(min_baseline_samples=10, speed_threshold_sigma=2.0, cooldown_seconds=0)
        # Only 3 observations at hour=10 (below min_baseline_samples)
        for i in range(3):
            engine.observe("z1", speed=1.5, hour_of_day=10)
        # But 50 total across other hours
        for i in range(50):
            engine.observe("z1", speed=1.5 + (i % 3) * 0.1, hour_of_day=14)

        # Should fall back to global data
        alerts = engine.check_target("z1", speed=50.0, hour_of_day=10)
        speed_alerts = [a for a in alerts if a.alert_type == "speed"]
        assert len(speed_alerts) == 1


class TestZeroVarianceBaseline:
    """Test handling of zero-variance baselines (all identical values)."""

    def test_zero_variance_speed_flags_deviation(self):
        engine = AnomalyEngine(min_baseline_samples=10, speed_threshold_sigma=2.0, cooldown_seconds=0)
        # All same speed
        for i in range(50):
            engine.observe("z1", speed=5.0, hour_of_day=12)
        alerts = engine.check_target("z1", speed=10.0, hour_of_day=12)
        speed_alerts = [a for a in alerts if a.alert_type == "speed"]
        assert len(speed_alerts) == 1
        assert speed_alerts[0].severity == "critical" or speed_alerts[0].severity == "high"

    def test_zero_variance_no_alert_for_same_value(self):
        engine = AnomalyEngine(min_baseline_samples=10, speed_threshold_sigma=2.0, cooldown_seconds=0)
        for i in range(50):
            engine.observe("z1", speed=5.0, hour_of_day=12)
        alerts = engine.check_target("z1", speed=5.0, hour_of_day=12)
        speed_alerts = [a for a in alerts if a.alert_type == "speed"]
        assert len(speed_alerts) == 0
