# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for config sync models."""

import pytest
from tritium_lib.models.config import (
    ConfigDrift,
    ConfigDriftSeverity,
    DeviceConfig,
    FleetConfigStatus,
    compute_config_drift,
    compute_fleet_config_status,
    classify_drift_severity,
)


class TestClassifyDriftSeverity:
    def test_critical_keys(self):
        assert classify_drift_severity("server_url") == ConfigDriftSeverity.CRITICAL
        assert classify_drift_severity("mqtt_broker") == ConfigDriftSeverity.CRITICAL
        assert classify_drift_severity("ca_pem") == ConfigDriftSeverity.CRITICAL

    def test_moderate_keys(self):
        assert classify_drift_severity("heartbeat_interval_s") == ConfigDriftSeverity.MODERATE
        assert classify_drift_severity("wifi_ssid") == ConfigDriftSeverity.MODERATE
        assert classify_drift_severity("ble_enabled") == ConfigDriftSeverity.MODERATE

    def test_minor_keys(self):
        assert classify_drift_severity("display_brightness") == ConfigDriftSeverity.MINOR
        assert classify_drift_severity("device_name") == ConfigDriftSeverity.MINOR
        assert classify_drift_severity("custom_key") == ConfigDriftSeverity.MINOR


class TestComputeConfigDrift:
    def test_identical_configs(self):
        desired = {"heartbeat_interval_s": 60, "ble_enabled": True}
        reported = {"heartbeat_interval_s": 60, "ble_enabled": True}
        assert compute_config_drift(desired, reported) == []

    def test_empty_configs(self):
        assert compute_config_drift({}, {}) == []

    def test_value_differs(self):
        desired = {"heartbeat_interval_s": 30}
        reported = {"heartbeat_interval_s": 60}
        drifts = compute_config_drift(desired, reported)
        assert len(drifts) == 1
        assert drifts[0].key == "heartbeat_interval_s"
        assert drifts[0].desired_value == 30
        assert drifts[0].reported_value == 60
        assert drifts[0].severity == ConfigDriftSeverity.MODERATE

    def test_missing_in_reported(self):
        desired = {"ble_enabled": True}
        reported = {}
        drifts = compute_config_drift(desired, reported)
        assert len(drifts) == 1
        assert drifts[0].is_missing
        assert not drifts[0].is_extra

    def test_extra_in_reported(self):
        desired = {}
        reported = {"custom_key": "value"}
        drifts = compute_config_drift(desired, reported)
        assert len(drifts) == 1
        assert drifts[0].is_extra
        assert not drifts[0].is_missing

    def test_multiple_drifts(self):
        desired = {"server_url": "http://a", "heartbeat_interval_s": 30, "name": "x"}
        reported = {"server_url": "http://b", "heartbeat_interval_s": 60, "name": "x"}
        drifts = compute_config_drift(desired, reported)
        assert len(drifts) == 2  # server_url and heartbeat differ, name matches
        keys = {d.key for d in drifts}
        assert "server_url" in keys
        assert "heartbeat_interval_s" in keys
        assert "name" not in keys

    def test_critical_drift_detected(self):
        desired = {"server_url": "http://good"}
        reported = {"server_url": "http://bad"}
        drifts = compute_config_drift(desired, reported)
        assert drifts[0].severity == ConfigDriftSeverity.CRITICAL


class TestDeviceConfig:
    def test_synced(self):
        dc = DeviceConfig(device_id="node-1", drifts=[])
        assert dc.is_synced
        assert dc.drift_count == 0
        assert dc.max_severity == ConfigDriftSeverity.NONE

    def test_drifted(self):
        dc = DeviceConfig(
            device_id="node-2",
            drifts=[
                ConfigDrift("key1", "a", "b", ConfigDriftSeverity.MINOR),
                ConfigDrift("key2", "c", "d", ConfigDriftSeverity.MODERATE),
            ],
        )
        assert not dc.is_synced
        assert dc.drift_count == 2
        assert dc.max_severity == ConfigDriftSeverity.MODERATE

    def test_critical_severity_wins(self):
        dc = DeviceConfig(
            device_id="node-3",
            drifts=[
                ConfigDrift("k1", 1, 2, ConfigDriftSeverity.MINOR),
                ConfigDrift("k2", 3, 4, ConfigDriftSeverity.CRITICAL),
            ],
        )
        assert dc.max_severity == ConfigDriftSeverity.CRITICAL


class TestFleetConfigStatus:
    def test_empty_fleet(self):
        status = compute_fleet_config_status([])
        assert status.total_devices == 0
        assert status.synced_count == 0
        assert status.sync_ratio == 1.0

    def test_all_synced(self):
        devices = [
            {"device_id": "n1", "desired_config": {"k": 1}, "reported_config": {"k": 1}},
            {"device_id": "n2", "desired_config": {"k": 2}, "reported_config": {"k": 2}},
        ]
        status = compute_fleet_config_status(devices)
        assert status.total_devices == 2
        assert status.synced_count == 2
        assert status.drifted_count == 0
        assert status.sync_ratio == 1.0

    def test_some_drifted(self):
        devices = [
            {"device_id": "n1", "desired_config": {"k": 1}, "reported_config": {"k": 1}},
            {"device_id": "n2", "desired_config": {"k": 1}, "reported_config": {"k": 2}},
            {"device_id": "n3", "desired_config": {"server_url": "a"}, "reported_config": {"server_url": "b"}},
        ]
        status = compute_fleet_config_status(devices)
        assert status.total_devices == 3
        assert status.synced_count == 1
        assert status.drifted_count == 2
        assert status.critical_drift_count == 1
        assert status.sync_ratio == pytest.approx(1 / 3)

    def test_no_config_keys(self):
        devices = [
            {"device_id": "n1", "desired_config": {}, "reported_config": {}},
        ]
        status = compute_fleet_config_status(devices)
        assert status.synced_count == 1

    def test_missing_config_fields(self):
        """Devices without config fields should be treated as synced (empty == empty)."""
        devices = [{"device_id": "n1"}]
        status = compute_fleet_config_status(devices)
        assert status.synced_count == 1
