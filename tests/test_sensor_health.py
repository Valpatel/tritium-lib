# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for sensor health monitoring models."""

import pytest
from datetime import datetime, timezone

from tritium_lib.models.sensor_health import (
    SensorAlert,
    SensorArrayHealth,
    SensorBaseline,
    SensorHealthMetrics,
    SensorHealthStatus,
    classify_sensor_health,
)


class TestSensorHealthStatus:
    def test_enum_values(self):
        assert SensorHealthStatus.HEALTHY == "healthy"
        assert SensorHealthStatus.DEGRADED == "degraded"
        assert SensorHealthStatus.CRITICAL == "critical"
        assert SensorHealthStatus.OFFLINE == "offline"
        assert SensorHealthStatus.UNKNOWN == "unknown"


class TestSensorHealthMetrics:
    def test_defaults(self):
        m = SensorHealthMetrics(sensor_id="node-01")
        assert m.sensor_id == "node-01"
        assert m.sighting_rate == 0.0
        assert m.baseline_rate == 0.0
        assert m.deviation_pct == 0.0
        assert m.status == SensorHealthStatus.UNKNOWN
        assert m.last_seen is None
        assert m.window_seconds == 300.0

    def test_is_healthy(self):
        m = SensorHealthMetrics(sensor_id="a", status=SensorHealthStatus.HEALTHY)
        assert m.is_healthy()
        m2 = SensorHealthMetrics(sensor_id="b", status=SensorHealthStatus.DEGRADED)
        assert not m2.is_healthy()

    def test_to_alert_dict(self):
        m = SensorHealthMetrics(
            sensor_id="node-02",
            status=SensorHealthStatus.CRITICAL,
            deviation_pct=-65.3,
            sighting_rate=1.2,
            baseline_rate=3.5,
            alert_message="Sighting rate dropped >50%",
        )
        d = m.to_alert_dict()
        assert d["sensor_id"] == "node-02"
        assert d["status"] == "critical"
        assert d["deviation_pct"] == -65.3
        assert d["alert_message"] == "Sighting rate dropped >50%"

    def test_roundtrip(self):
        m = SensorHealthMetrics(
            sensor_id="x",
            sighting_rate=5.0,
            baseline_rate=10.0,
            deviation_pct=-50.0,
            status=SensorHealthStatus.CRITICAL,
            last_seen=datetime(2026, 1, 1, tzinfo=timezone.utc),
            sighting_count=25,
        )
        d = m.model_dump()
        m2 = SensorHealthMetrics(**d)
        assert m2.sensor_id == m.sensor_id
        assert m2.sighting_rate == m.sighting_rate
        assert m2.status == SensorHealthStatus.CRITICAL


class TestClassifySensorHealth:
    def test_healthy(self):
        result = classify_sensor_health(9.0, 10.0)
        assert result == SensorHealthStatus.HEALTHY

    def test_degraded(self):
        # 40% below baseline
        result = classify_sensor_health(6.0, 10.0)
        assert result == SensorHealthStatus.DEGRADED

    def test_critical(self):
        # 60% below baseline
        result = classify_sensor_health(4.0, 10.0)
        assert result == SensorHealthStatus.CRITICAL

    def test_offline(self):
        result = classify_sensor_health(0.0, 10.0, seconds_since_last=600.0)
        assert result == SensorHealthStatus.OFFLINE

    def test_unknown_no_baseline(self):
        result = classify_sensor_health(5.0, 0.0)
        assert result == SensorHealthStatus.UNKNOWN

    def test_above_baseline_healthy(self):
        result = classify_sensor_health(15.0, 10.0)
        assert result == SensorHealthStatus.HEALTHY

    def test_exact_25pct_drop_still_healthy(self):
        result = classify_sensor_health(7.5, 10.0)
        assert result == SensorHealthStatus.HEALTHY

    def test_just_below_25pct_is_degraded(self):
        result = classify_sensor_health(7.4, 10.0)
        assert result == SensorHealthStatus.DEGRADED

    def test_exactly_50pct_drop_is_degraded(self):
        result = classify_sensor_health(5.0, 10.0)
        assert result == SensorHealthStatus.DEGRADED

    def test_just_below_50pct_is_critical(self):
        result = classify_sensor_health(4.9, 10.0)
        assert result == SensorHealthStatus.CRITICAL


class TestSensorArrayHealth:
    def test_compute_overall_all_healthy(self):
        arr = SensorArrayHealth(sensors=[
            SensorHealthMetrics(sensor_id="a", status=SensorHealthStatus.HEALTHY),
            SensorHealthMetrics(sensor_id="b", status=SensorHealthStatus.HEALTHY),
        ])
        arr.compute_overall()
        assert arr.overall_status == SensorHealthStatus.HEALTHY
        assert arr.healthy_count == 2
        assert arr.degraded_count == 0

    def test_compute_overall_with_degraded(self):
        arr = SensorArrayHealth(sensors=[
            SensorHealthMetrics(sensor_id="a", status=SensorHealthStatus.HEALTHY),
            SensorHealthMetrics(sensor_id="b", status=SensorHealthStatus.DEGRADED),
        ])
        arr.compute_overall()
        assert arr.overall_status == SensorHealthStatus.DEGRADED

    def test_compute_overall_with_critical(self):
        arr = SensorArrayHealth(sensors=[
            SensorHealthMetrics(sensor_id="a", status=SensorHealthStatus.HEALTHY),
            SensorHealthMetrics(sensor_id="b", status=SensorHealthStatus.CRITICAL),
        ])
        arr.compute_overall()
        assert arr.overall_status == SensorHealthStatus.CRITICAL

    def test_compute_overall_with_offline(self):
        arr = SensorArrayHealth(sensors=[
            SensorHealthMetrics(sensor_id="a", status=SensorHealthStatus.HEALTHY),
            SensorHealthMetrics(sensor_id="b", status=SensorHealthStatus.OFFLINE),
        ])
        arr.compute_overall()
        assert arr.overall_status == SensorHealthStatus.CRITICAL
        assert arr.offline_count == 1

    def test_compute_overall_empty(self):
        arr = SensorArrayHealth()
        arr.compute_overall()
        assert arr.overall_status == SensorHealthStatus.UNKNOWN


class TestSensorBaseline:
    def test_defaults(self):
        b = SensorBaseline(sensor_id="node-01")
        assert b.sensor_id == "node-01"
        assert b.sighting_rate_mean == 0.0
        assert b.sighting_rate_stddev == 0.0
        assert b.training_window_hours == 24.0
        assert b.sample_count == 0
        assert not b.is_valid
        assert b.created_at is None

    def test_deviation_from_valid(self):
        b = SensorBaseline(
            sensor_id="x",
            sighting_rate_mean=10.0,
            sighting_rate_stddev=2.0,
            is_valid=True,
        )
        # Exactly at mean
        assert b.deviation_from(10.0) == 0.0
        # 1 sigma above
        assert b.deviation_from(12.0) == pytest.approx(1.0)
        # 2 sigma below
        assert b.deviation_from(6.0) == pytest.approx(-2.0)

    def test_deviation_from_invalid(self):
        b = SensorBaseline(sensor_id="y", is_valid=False)
        assert b.deviation_from(5.0) == 0.0

    def test_deviation_from_zero_stddev(self):
        b = SensorBaseline(
            sensor_id="z",
            sighting_rate_mean=10.0,
            sighting_rate_stddev=0.0,
            is_valid=True,
        )
        assert b.deviation_from(5.0) == 0.0

    def test_roundtrip(self):
        ts = datetime(2026, 3, 14, tzinfo=timezone.utc)
        b = SensorBaseline(
            sensor_id="node-02",
            sighting_rate_mean=8.5,
            sighting_rate_stddev=1.5,
            min_sighting_rate=3.0,
            max_sighting_rate=15.0,
            training_window_hours=48.0,
            sample_count=288,
            created_at=ts,
            updated_at=ts,
            is_valid=True,
        )
        d = b.model_dump()
        b2 = SensorBaseline(**d)
        assert b2.sensor_id == b.sensor_id
        assert b2.sighting_rate_mean == b.sighting_rate_mean
        assert b2.is_valid is True
        assert b2.sample_count == 288


class TestSensorAlert:
    def test_defaults(self):
        a = SensorAlert(sensor_id="node-01")
        assert a.sensor_id == "node-01"
        assert a.alert_type == "deviation"
        assert a.severity == SensorHealthStatus.UNKNOWN
        assert a.message == ""
        assert not a.acknowledged

    def test_to_notification_dict(self):
        ts = datetime(2026, 3, 14, 12, 0, 0, tzinfo=timezone.utc)
        a = SensorAlert(
            sensor_id="node-03",
            alert_type="tamper",
            severity=SensorHealthStatus.CRITICAL,
            message="Zero sightings for 10 minutes",
            sighting_rate=0.0,
            baseline_rate=12.0,
            deviation_pct=-100.0,
            deviation_sigma=-4.5,
            timestamp=ts,
            recommended_action="Inspect sensor for physical obstruction",
        )
        d = a.to_notification_dict()
        assert d["sensor_id"] == "node-03"
        assert d["alert_type"] == "tamper"
        assert d["severity"] == "critical"
        assert d["deviation_pct"] == -100.0
        assert d["deviation_sigma"] == -4.5
        assert d["recommended_action"] == "Inspect sensor for physical obstruction"
        assert "2026-03-14" in d["timestamp"]

    def test_roundtrip(self):
        a = SensorAlert(
            sensor_id="x",
            alert_type="offline",
            severity=SensorHealthStatus.OFFLINE,
            message="Sensor went offline",
            deviation_pct=-100.0,
        )
        d = a.model_dump()
        a2 = SensorAlert(**d)
        assert a2.sensor_id == a.sensor_id
        assert a2.alert_type == "offline"
        assert a2.severity == SensorHealthStatus.OFFLINE
