# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for capability advertisement models."""

import pytest
from tritium_lib.models.capability import (
    CapabilityAdvertisement,
    CapabilityType,
    DeviceCapability,
)


class TestDeviceCapability:
    def test_create_basic(self):
        cap = DeviceCapability(cap_type=CapabilityType.BLE_SCANNER)
        assert cap.cap_type == CapabilityType.BLE_SCANNER
        assert cap.version == "1.0"
        assert cap.enabled is True
        assert cap.config == {}

    def test_create_with_config(self):
        cap = DeviceCapability(
            cap_type=CapabilityType.CAMERA,
            version="2.1",
            config={"resolution": "640x480", "fps": 15},
        )
        assert cap.version == "2.1"
        assert cap.config["resolution"] == "640x480"

    def test_to_summary(self):
        cap = DeviceCapability(cap_type=CapabilityType.WIFI_SCANNER, version="1.2")
        assert "wifi_scanner" in cap.to_summary()
        assert "v1.2" in cap.to_summary()
        assert "[ON]" in cap.to_summary()

    def test_disabled_summary(self):
        cap = DeviceCapability(cap_type=CapabilityType.GPS, enabled=False)
        assert "[OFF]" in cap.to_summary()


class TestCapabilityAdvertisement:
    def _make_advert(self):
        return CapabilityAdvertisement(
            device_id="esp32-test-001",
            board="touch-lcd-43c-box",
            firmware_version="0.5.0",
            capabilities=[
                DeviceCapability(cap_type=CapabilityType.BLE_SCANNER),
                DeviceCapability(cap_type=CapabilityType.WIFI_SCANNER),
                DeviceCapability(cap_type=CapabilityType.CAMERA, enabled=False),
                DeviceCapability(cap_type=CapabilityType.DISPLAY),
                DeviceCapability(cap_type=CapabilityType.HEARTBEAT),
            ],
        )

    def test_has_capability(self):
        advert = self._make_advert()
        assert advert.has_capability(CapabilityType.BLE_SCANNER) is True
        assert advert.has_capability(CapabilityType.WIFI_SCANNER) is True
        # Camera is disabled
        assert advert.has_capability(CapabilityType.CAMERA) is False
        # GPS not present
        assert advert.has_capability(CapabilityType.GPS) is False

    def test_get_capability(self):
        advert = self._make_advert()
        ble = advert.get_capability(CapabilityType.BLE_SCANNER)
        assert ble is not None
        assert ble.cap_type == CapabilityType.BLE_SCANNER

        camera = advert.get_capability(CapabilityType.CAMERA)
        assert camera is None  # disabled

    def test_capability_types(self):
        advert = self._make_advert()
        types = advert.capability_types()
        assert "ble_scanner" in types
        assert "wifi_scanner" in types
        assert "display" in types
        assert "heartbeat" in types
        # Camera disabled, should not be listed
        assert "camera" not in types

    def test_to_heartbeat_list(self):
        advert = self._make_advert()
        hb_list = advert.to_heartbeat_list()
        assert "ble" in hb_list  # ble_scanner maps to "ble"
        assert "wifi" in hb_list  # wifi_scanner maps to "wifi"
        assert "display" in hb_list
        assert "camera" not in hb_list  # disabled

    def test_empty_capabilities(self):
        advert = CapabilityAdvertisement(device_id="test")
        assert advert.has_capability(CapabilityType.BLE_SCANNER) is False
        assert advert.capability_types() == []

    def test_serialization_roundtrip(self):
        advert = self._make_advert()
        data = advert.model_dump()
        restored = CapabilityAdvertisement(**data)
        assert restored.device_id == advert.device_id
        assert len(restored.capabilities) == len(advert.capabilities)
        assert restored.has_capability(CapabilityType.BLE_SCANNER) is True

    def test_json_roundtrip(self):
        advert = self._make_advert()
        json_str = advert.model_dump_json()
        restored = CapabilityAdvertisement.model_validate_json(json_str)
        assert restored.device_id == "esp32-test-001"
        assert restored.board == "touch-lcd-43c-box"


class TestCapabilityType:
    def test_all_types_are_strings(self):
        for ct in CapabilityType:
            assert isinstance(ct.value, str)

    def test_known_types(self):
        assert CapabilityType.BLE_SCANNER == "ble_scanner"
        assert CapabilityType.CAMERA == "camera"
        assert CapabilityType.HEARTBEAT == "heartbeat"
