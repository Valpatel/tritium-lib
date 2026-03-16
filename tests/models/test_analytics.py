# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for DailyAnalytics model."""

import pytest
from datetime import datetime, timezone

from tritium_lib.models.analytics import DailyAnalytics, DeviceActivity


class TestDeviceActivity:
    def test_defaults(self):
        d = DeviceActivity()
        assert d.device_id == ""
        assert d.sighting_count == 0

    def test_to_dict(self):
        d = DeviceActivity(device_id="node-01", sighting_count=42, target_count=5, last_seen=1000.0)
        out = d.to_dict()
        assert out["device_id"] == "node-01"
        assert out["sighting_count"] == 42

    def test_from_dict(self):
        data = {"device_id": "x", "sighting_count": 10, "target_count": 2, "last_seen": 500.0}
        d = DeviceActivity.from_dict(data)
        assert d.device_id == "x"
        assert d.sighting_count == 10

    def test_roundtrip(self):
        original = DeviceActivity(device_id="abc", sighting_count=99, target_count=7, last_seen=12345.0)
        restored = DeviceActivity.from_dict(original.to_dict())
        assert restored.device_id == original.device_id
        assert restored.sighting_count == original.sighting_count
        assert restored.last_seen == original.last_seen


class TestDailyAnalytics:
    def test_defaults(self):
        a = DailyAnalytics()
        assert a.new_targets == 0
        assert a.correlations == 0
        assert a.threats == 0
        assert a.threat_level == "GREEN"
        assert a.report_date is not None
        assert a.generated_at is not None

    def test_to_dict(self):
        a = DailyAnalytics(
            new_targets=5,
            correlations=3,
            threats=2,
            zone_events=1,
            total_sightings=100,
            sightings_by_source={"ble": 60, "wifi": 30, "yolo": 10},
            threat_level="YELLOW",
        )
        out = a.to_dict()
        assert out["new_targets"] == 5
        assert out["correlations"] == 3
        assert out["sightings_by_source"]["ble"] == 60
        assert out["threat_level"] == "YELLOW"

    def test_from_dict(self):
        data = {
            "report_date": "2026-03-14",
            "new_targets": 12,
            "correlations": 5,
            "threats": 1,
            "zone_events": 4,
            "total_sightings": 200,
            "sightings_by_source": {"ble": 150, "wifi": 50},
            "top_devices": [
                {"device_id": "node-01", "sighting_count": 100, "target_count": 10, "last_seen": 99.0}
            ],
            "threat_level": "YELLOW",
            "uptime_percent": 99.5,
        }
        a = DailyAnalytics.from_dict(data)
        assert a.report_date == "2026-03-14"
        assert a.new_targets == 12
        assert len(a.top_devices) == 1
        assert a.top_devices[0].device_id == "node-01"
        assert a.uptime_percent == 99.5

    def test_roundtrip(self):
        original = DailyAnalytics(
            new_targets=8,
            correlations=2,
            threats=0,
            zone_events=3,
            investigations_opened=1,
            total_sightings=50,
            sightings_by_source={"mesh": 20, "ble": 30},
            top_devices=[
                DeviceActivity(device_id="d1", sighting_count=30),
                DeviceActivity(device_id="d2", sighting_count=20),
            ],
            threat_level="GREEN",
            uptime_percent=100.0,
            extra={"custom": "data"},
        )
        restored = DailyAnalytics.from_dict(original.to_dict())
        assert restored.new_targets == original.new_targets
        assert restored.correlations == original.correlations
        assert len(restored.top_devices) == 2
        assert restored.top_devices[0].device_id == "d1"
        assert restored.extra == {"custom": "data"}

    def test_generated_at_preserved(self):
        ts = datetime(2026, 3, 14, 12, 0, 0, tzinfo=timezone.utc)
        a = DailyAnalytics(generated_at=ts)
        d = a.to_dict()
        restored = DailyAnalytics.from_dict(d)
        assert restored.generated_at == ts
