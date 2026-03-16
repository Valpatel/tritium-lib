# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for radio scheduler models."""

import pytest
from tritium_lib.models.radio import (
    CameraMqttConfig,
    CameraMqttStats,
    RadioMode,
    RadioSchedulerConfig,
    RadioSchedulerStatus,
)


class TestRadioMode:
    def test_enum_values(self):
        assert RadioMode.IDLE == "idle"
        assert RadioMode.WIFI_ACTIVE == "wifi"
        assert RadioMode.BLE_SCANNING == "ble"
        assert RadioMode.TRANSITIONING == "transition"

    def test_from_string(self):
        assert RadioMode("wifi") == RadioMode.WIFI_ACTIVE
        assert RadioMode("ble") == RadioMode.BLE_SCANNING


class TestRadioSchedulerConfig:
    def test_defaults(self):
        cfg = RadioSchedulerConfig()
        assert cfg.wifi_slot_ms == 25000
        assert cfg.ble_slot_ms == 10000
        assert cfg.transition_ms == 2000
        assert cfg.enable_ble is True
        assert cfg.enable_wifi is True
        assert cfg.wifi_first is True

    def test_custom_values(self):
        cfg = RadioSchedulerConfig(wifi_slot_ms=30000, ble_slot_ms=5000)
        assert cfg.wifi_slot_ms == 30000
        assert cfg.ble_slot_ms == 5000

    def test_validation_min(self):
        with pytest.raises(Exception):
            RadioSchedulerConfig(wifi_slot_ms=500)  # Below min 1000

    def test_serialization(self):
        cfg = RadioSchedulerConfig()
        d = cfg.model_dump()
        assert d["wifi_slot_ms"] == 25000
        cfg2 = RadioSchedulerConfig.model_validate(d)
        assert cfg2 == cfg


class TestRadioSchedulerStatus:
    def test_defaults(self):
        status = RadioSchedulerStatus()
        assert status.mode == RadioMode.IDLE
        assert status.wifi_cycles == 0
        assert status.ble_cycles == 0

    def test_with_config(self):
        status = RadioSchedulerStatus(
            mode=RadioMode.WIFI_ACTIVE,
            wifi_cycles=10,
            ble_cycles=8,
            slot_remaining_ms=15000,
            config=RadioSchedulerConfig(),
        )
        assert status.mode == RadioMode.WIFI_ACTIVE
        assert status.config is not None
        assert status.config.wifi_slot_ms == 25000

    def test_json_roundtrip(self):
        status = RadioSchedulerStatus(mode=RadioMode.BLE_SCANNING, ble_cycles=5)
        j = status.model_dump_json()
        status2 = RadioSchedulerStatus.model_validate_json(j)
        assert status2.mode == RadioMode.BLE_SCANNING
        assert status2.ble_cycles == 5


class TestCameraMqttConfig:
    def test_defaults(self):
        cfg = CameraMqttConfig()
        assert cfg.target_fps == 2.0
        assert cfg.jpeg_quality == 15
        assert cfg.auto_start is True

    def test_fps_bounds(self):
        with pytest.raises(Exception):
            CameraMqttConfig(target_fps=0.05)  # Below 0.1
        with pytest.raises(Exception):
            CameraMqttConfig(target_fps=15.0)  # Above 10.0


class TestCameraMqttStats:
    def test_defaults(self):
        stats = CameraMqttStats()
        assert stats.active is False
        assert stats.frames_published == 0

    def test_with_data(self):
        stats = CameraMqttStats(
            active=True,
            frames_published=100,
            frames_failed=2,
            avg_latency_ms=45,
            max_frame_bytes=32000,
            target_fps=2.0,
            actual_fps=1.8,
        )
        assert stats.frames_published == 100
        assert stats.actual_fps == 1.8
