# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for SystemSummary model."""

from datetime import datetime, timezone

from tritium_lib.models.summary import (
    FleetSummary,
    SystemSummary,
    TargetCounts,
)


class TestTargetCounts:
    def test_defaults(self):
        tc = TargetCounts()
        assert tc.total == 0
        assert tc.friendly == 0
        assert tc.hostile == 0
        assert tc.unknown == 0
        assert tc.ble == 0

    def test_to_dict(self):
        tc = TargetCounts(total=10, friendly=3, hostile=2, unknown=5, ble=4, yolo=3, mesh=2, rf_motion=1)
        d = tc.to_dict()
        assert d["total"] == 10
        assert d["by_alliance"]["friendly"] == 3
        assert d["by_alliance"]["hostile"] == 2
        assert d["by_source"]["ble"] == 4
        assert d["by_source"]["yolo"] == 3
        assert d["by_source"]["rf_motion"] == 1

    def test_roundtrip(self):
        tc = TargetCounts(total=5, friendly=1, hostile=2, unknown=2, ble=3, wifi=2)
        d = tc.to_dict()
        tc2 = TargetCounts.from_dict(d)
        assert tc2.total == 5
        assert tc2.friendly == 1
        assert tc2.hostile == 2
        assert tc2.ble == 3
        assert tc2.wifi == 2

    def test_from_dict_missing_fields(self):
        tc = TargetCounts.from_dict({})
        assert tc.total == 0
        assert tc.friendly == 0


class TestFleetSummary:
    def test_defaults(self):
        fs = FleetSummary()
        assert fs.total_devices == 0
        assert fs.online == 0

    def test_roundtrip(self):
        fs = FleetSummary(total_devices=5, online=3, offline=2, low_battery=1)
        d = fs.to_dict()
        fs2 = FleetSummary.from_dict(d)
        assert fs2.total_devices == 5
        assert fs2.online == 3
        assert fs2.offline == 2
        assert fs2.low_battery == 1


class TestSystemSummary:
    def test_defaults(self):
        ss = SystemSummary()
        assert ss.targets.total == 0
        assert ss.dossier_count == 0
        assert ss.plugin_count == 0
        assert ss.active_plugins == []
        assert ss.demo_active is False
        assert ss.timestamp is not None

    def test_auto_timestamp(self):
        ss = SystemSummary()
        assert ss.timestamp is not None
        assert ss.timestamp.tzinfo is not None

    def test_to_dict(self):
        ss = SystemSummary(
            targets=TargetCounts(total=10, hostile=3),
            dossier_count=5,
            active_plugins=["edge_tracker", "meshtastic"],
            plugin_count=2,
            active_alerts=1,
            active_investigations=0,
            fleet=FleetSummary(total_devices=3, online=2, offline=1),
            demo_active=True,
            uptime_seconds=3600.0,
            mqtt_connected=True,
            version="0.2.0",
            extra={"custom": "value"},
        )
        d = ss.to_dict()
        assert d["targets"]["total"] == 10
        assert d["targets"]["by_alliance"]["hostile"] == 3
        assert d["dossier_count"] == 5
        assert d["active_plugins"] == ["edge_tracker", "meshtastic"]
        assert d["fleet"]["online"] == 2
        assert d["demo_active"] is True
        assert d["mqtt_connected"] is True
        assert d["version"] == "0.2.0"
        assert d["extra"]["custom"] == "value"

    def test_roundtrip(self):
        ts = datetime(2026, 3, 14, 12, 0, 0, tzinfo=timezone.utc)
        ss = SystemSummary(
            timestamp=ts,
            targets=TargetCounts(total=7, ble=3, yolo=4),
            dossier_count=12,
            active_plugins=["gis_layers"],
            plugin_count=1,
            active_alerts=2,
            uptime_seconds=7200.0,
        )
        d = ss.to_dict()
        ss2 = SystemSummary.from_dict(d)
        assert ss2.targets.total == 7
        assert ss2.targets.ble == 3
        assert ss2.targets.yolo == 4
        assert ss2.dossier_count == 12
        assert ss2.active_plugins == ["gis_layers"]
        assert ss2.active_alerts == 2
        assert ss2.uptime_seconds == 7200.0
        assert ss2.timestamp == ts

    def test_from_dict_empty(self):
        ss = SystemSummary.from_dict({})
        assert ss.targets.total == 0
        assert ss.plugin_count == 0
        assert ss.version == "0.1.0"

    def test_importable_from_models(self):
        from tritium_lib.models import SystemSummary, TargetCounts, FleetSummary
        assert SystemSummary is not None
        assert TargetCounts is not None
        assert FleetSummary is not None
