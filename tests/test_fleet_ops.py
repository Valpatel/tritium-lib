# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for fleet operations models."""

import pytest

from tritium_lib.models.fleet_ops import (
    BUILTIN_TEMPLATES,
    ConfigTemplate,
    ConfigTemplateName,
    CoveragePoint,
    DeviceUptimeRecord,
    FleetAnalyticsSnapshot,
    FleetCommand,
    FleetCommandStatus,
    FleetCommandType,
    SightingRateRecord,
)


class TestFleetCommandType:
    def test_all_command_types_exist(self):
        assert FleetCommandType.REBOOT == "reboot"
        assert FleetCommandType.SCAN_BURST == "scan_burst"
        assert FleetCommandType.INCREASE_RATE == "increase_rate"
        assert FleetCommandType.DECREASE_RATE == "decrease_rate"
        assert FleetCommandType.OTA_UPDATE == "ota_update"
        assert FleetCommandType.APPLY_TEMPLATE == "apply_template"
        assert FleetCommandType.SET_GROUP == "set_group"
        assert FleetCommandType.IDENTIFY == "identify"
        assert FleetCommandType.SLEEP == "sleep"


class TestFleetCommand:
    def test_create_minimal(self):
        cmd = FleetCommand(id="cmd-001", command_type=FleetCommandType.REBOOT, target_group="perimeter")
        assert cmd.id == "cmd-001"
        assert cmd.command_type == FleetCommandType.REBOOT
        assert cmd.target_group == "perimeter"
        assert cmd.status == FleetCommandStatus.PENDING
        assert cmd.expected_targets == 0
        assert cmd.acked_targets == 0
        assert cmd.payload == {}

    def test_create_with_payload(self):
        cmd = FleetCommand(
            id="cmd-002",
            command_type=FleetCommandType.OTA_UPDATE,
            target_group="all",
            payload={"firmware_url": "http://example.com/fw.bin"},
            expected_targets=5,
        )
        assert cmd.payload["firmware_url"] == "http://example.com/fw.bin"
        assert cmd.expected_targets == 5

    def test_serialization_roundtrip(self):
        cmd = FleetCommand(
            id="cmd-003",
            command_type=FleetCommandType.SCAN_BURST,
            target_group="interior",
        )
        data = cmd.model_dump()
        restored = FleetCommand(**data)
        assert restored.id == cmd.id
        assert restored.command_type == cmd.command_type


class TestConfigTemplate:
    def test_create_custom(self):
        tpl = ConfigTemplate(
            id="custom-1",
            name="Custom Fast Scan",
            ble_scan_interval_ms=2000,
            power_mode="high_performance",
        )
        assert tpl.id == "custom-1"
        assert tpl.ble_scan_interval_ms == 2000
        assert tpl.template_type == ConfigTemplateName.CUSTOM

    def test_defaults(self):
        tpl = ConfigTemplate(id="default", name="Default")
        assert tpl.ble_scan_interval_ms == 10000
        assert tpl.wifi_scan_interval_ms == 30000
        assert tpl.heartbeat_interval_ms == 30000
        assert tpl.sighting_interval_ms == 15000
        assert tpl.power_mode == "normal"


class TestBuiltinTemplates:
    def test_three_builtin_templates(self):
        assert len(BUILTIN_TEMPLATES) == 3

    def test_perimeter_high_security(self):
        tpl = BUILTIN_TEMPLATES["perimeter_high_security"]
        assert tpl.ble_scan_interval_ms == 5000
        assert tpl.power_mode == "high_performance"

    def test_indoor_normal(self):
        tpl = BUILTIN_TEMPLATES["indoor_normal"]
        assert tpl.ble_scan_interval_ms == 10000
        assert tpl.power_mode == "normal"

    def test_power_saver_mobile(self):
        tpl = BUILTIN_TEMPLATES["power_saver_mobile"]
        assert tpl.ble_scan_interval_ms == 30000
        assert tpl.power_mode == "low_power"


class TestFleetAnalyticsSnapshot:
    def test_create_minimal(self):
        snap = FleetAnalyticsSnapshot(timestamp=1000.0)
        assert snap.total_devices == 0
        assert snap.online_devices == 0
        assert snap.groups == {}

    def test_create_full(self):
        snap = FleetAnalyticsSnapshot(
            timestamp=1000.0,
            total_devices=10,
            online_devices=8,
            offline_devices=2,
            avg_uptime_s=3600.0,
            avg_battery_pct=78.5,
            total_ble_sightings=500,
            total_wifi_sightings=200,
            uptime_records=[
                DeviceUptimeRecord(device_id="dev-1", timestamp=1000.0, uptime_s=3600),
            ],
            sighting_rates=[
                SightingRateRecord(device_id="dev-1", timestamp=1000.0, ble_rate=5.0),
            ],
            coverage_points=[
                CoveragePoint(lat=33.0, lng=-97.0, sensor_count=2),
            ],
            groups={"perimeter": 4, "interior": 6},
        )
        assert snap.total_devices == 10
        assert len(snap.uptime_records) == 1
        assert len(snap.coverage_points) == 1
        assert snap.groups["perimeter"] == 4


class TestImportsFromInit:
    def test_imports_from_models_init(self):
        from tritium_lib.models import (
            ConfigTemplate,
            FleetCommand,
            FleetCommandType,
            FleetAnalyticsSnapshot,
            FLEET_BUILTIN_TEMPLATES,
        )
        assert len(FLEET_BUILTIN_TEMPLATES) == 3
        assert FleetCommandType.REBOOT == "reboot"
