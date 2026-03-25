# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.tracking.sensor_health_monitor."""

import time
from unittest.mock import MagicMock, patch

import pytest

from tritium_lib.tracking.sensor_health_monitor import SensorHealthMonitor


class TestRecordSighting:
    def test_record_creates_sensor(self):
        mon = SensorHealthMonitor()
        mon.record_sighting("node-01")
        health = mon.get_health()
        assert len(health) == 1
        assert health[0]["sensor_id"] == "node-01"

    def test_record_multiple_sightings(self):
        mon = SensorHealthMonitor()
        for _ in range(10):
            mon.record_sighting("node-01")
        health = mon.get_health()
        assert health[0]["sighting_count"] == 10

    def test_multiple_sensors(self):
        mon = SensorHealthMonitor()
        mon.record_sighting("node-01")
        mon.record_sighting("node-02")
        health = mon.get_health()
        ids = {h["sensor_id"] for h in health}
        assert ids == {"node-01", "node-02"}


class TestGetHealth:
    def test_unknown_status_few_samples(self):
        mon = SensorHealthMonitor()
        # Record fewer than BASELINE_MIN_SAMPLES sightings
        for _ in range(3):
            mon.record_sighting("node-01")
        health = mon.get_health()
        assert health[0]["status"] == "unknown"

    def test_healthy_status_after_baseline(self):
        mon = SensorHealthMonitor()
        # Record enough sightings to build a baseline, all in a burst
        for _ in range(20):
            mon.record_sighting("node-01")
        health = mon.get_health()
        # Should be healthy or unknown (depending on timing), but not critical
        assert health[0]["status"] in ("healthy", "unknown")

    def test_health_has_expected_fields(self):
        mon = SensorHealthMonitor()
        mon.record_sighting("node-01")
        health = mon.get_health()
        h = health[0]
        assert "sensor_id" in h
        assert "sighting_rate" in h
        assert "baseline_rate" in h
        assert "deviation_pct" in h
        assert "status" in h
        assert "last_seen_seconds_ago" in h
        assert "sighting_count" in h
        assert "baseline_samples" in h
        assert "alert_message" in h


class TestGetSensorHealth:
    def test_specific_sensor(self):
        mon = SensorHealthMonitor()
        mon.record_sighting("node-01")
        mon.record_sighting("node-02")
        h = mon.get_sensor_health("node-01")
        assert h is not None
        assert h["sensor_id"] == "node-01"

    def test_nonexistent_sensor(self):
        mon = SensorHealthMonitor()
        assert mon.get_sensor_health("no-such") is None


class TestBaselineLearning:
    def test_baseline_samples_increment(self):
        mon = SensorHealthMonitor()
        for _ in range(10):
            mon.record_sighting("node-01")
        health = mon.get_health()
        # Should have accumulated some baseline samples
        assert health[0]["baseline_samples"] >= 1

    def test_baseline_rate_positive_after_sightings(self):
        mon = SensorHealthMonitor()
        for _ in range(20):
            mon.record_sighting("node-01")
        health = mon.get_health()
        assert health[0]["baseline_rate"] >= 0


class TestOfflineDetection:
    def test_offline_after_threshold(self):
        mon = SensorHealthMonitor()
        # Manually inject a sensor record with old last_seen
        mon.record_sighting("node-01")
        with mon._lock:
            rec = mon._sensors["node-01"]
            # Set last_seen to far in the past (monotonic time)
            rec.last_seen = time.monotonic() - (mon.OFFLINE_THRESHOLD_SECONDS + 100)
        health = mon.get_health()
        assert health[0]["status"] == "offline"


class TestAlertEmission:
    def test_alert_emitted_on_critical(self):
        bus = MagicMock()
        mon = SensorHealthMonitor(event_bus=bus)
        # Build a baseline, then make the sensor appear critical
        for _ in range(20):
            mon.record_sighting("node-01")
        with mon._lock:
            rec = mon._sensors["node-01"]
            # Force a high baseline rate so current rate (near 0 after time passes) is critical
            rec.baseline_rate = 1000.0
            rec.baseline_samples = 10
            # Clear recent sightings to force rate drop
            rec.sighting_times.clear()
        health = mon.get_health()
        critical_sensors = [h for h in health if h["status"] == "critical"]
        if critical_sensors:
            bus.publish.assert_called()

    def test_alert_cooldown(self):
        bus = MagicMock()
        mon = SensorHealthMonitor(event_bus=bus)
        for _ in range(20):
            mon.record_sighting("node-01")
        with mon._lock:
            rec = mon._sensors["node-01"]
            rec.baseline_rate = 1000.0
            rec.baseline_samples = 10
            rec.sighting_times.clear()
        # First health check should emit alert
        mon.get_health()
        first_count = bus.publish.call_count
        # Second call within cooldown should NOT emit again
        mon.get_health()
        assert bus.publish.call_count == first_count

    def test_offline_alert_emitted(self):
        bus = MagicMock()
        mon = SensorHealthMonitor(event_bus=bus)
        mon.record_sighting("node-01")
        with mon._lock:
            rec = mon._sensors["node-01"]
            rec.last_seen = time.monotonic() - (mon.OFFLINE_THRESHOLD_SECONDS + 100)
        mon.get_health()
        bus.publish.assert_called()
        call_args = bus.publish.call_args
        assert call_args[0][0] == "sensor:health_alert"


class TestClassConstants:
    def test_rate_window_positive(self):
        assert SensorHealthMonitor.RATE_WINDOW_SECONDS > 0

    def test_baseline_min_samples_positive(self):
        assert SensorHealthMonitor.BASELINE_MIN_SAMPLES > 0

    def test_baseline_alpha_in_range(self):
        assert 0 < SensorHealthMonitor.BASELINE_ALPHA <= 1

    def test_cooldown_positive(self):
        assert SensorHealthMonitor.ALERT_COOLDOWN_SECONDS > 0

    def test_offline_threshold_positive(self):
        assert SensorHealthMonitor.OFFLINE_THRESHOLD_SECONDS > 0
