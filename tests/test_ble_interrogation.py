# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for BLE GATT interrogation models."""

import pytest
from datetime import datetime, timezone

from tritium_lib.models.ble_interrogation import (
    BleDeviceProfile,
    BleGATTCharacteristic,
    BleGATTService,
    BleInterrogationResult,
    BleInterrogationQueue,
    STANDARD_SERVICE_UUIDS,
    classify_device_from_profile,
    lookup_service_name,
)


class TestLookupServiceName:
    """Test service UUID name lookup."""

    def test_known_service(self):
        assert lookup_service_name(0x180A) == "Device Information"

    def test_battery_service(self):
        assert lookup_service_name(0x180F) == "Battery Service"

    def test_gap(self):
        assert lookup_service_name(0x1800) == "Generic Access (GAP)"

    def test_heart_rate(self):
        assert lookup_service_name(0x180D) == "Heart Rate"

    def test_hid(self):
        assert lookup_service_name(0x1812) == "Human Interface Device (HID)"

    def test_unknown_uuid(self):
        result = lookup_service_name(0xFFFF)
        assert "Unknown" in result
        assert "0xFFFF" in result

    def test_google_fast_pair(self):
        assert lookup_service_name(0xFE2C) == "Google Fast Pair"

    def test_tile_tracker(self):
        assert lookup_service_name(0xFEED) == "Tile Tracker"


class TestBleGATTService:
    """Test BleGATTService model."""

    def test_create_standard_service(self):
        svc = BleGATTService(
            uuid="0x180A",
            uuid16=0x180A,
            name="Device Information",
            is_standard=True,
        )
        assert svc.uuid == "0x180A"
        assert svc.uuid16 == 0x180A
        assert svc.is_standard is True

    def test_create_custom_service(self):
        svc = BleGATTService(
            uuid="12345678-1234-1234-1234-123456789abc",
            name="Custom Service",
        )
        assert svc.uuid16 is None
        assert svc.is_standard is False

    def test_with_characteristics(self):
        char1 = BleGATTCharacteristic(
            uuid="0x2A29",
            name="Manufacturer Name",
            value="Apple Inc.",
            properties=["read"],
        )
        svc = BleGATTService(
            uuid="0x180A",
            uuid16=0x180A,
            name="Device Information",
            is_standard=True,
            characteristics=[char1],
        )
        assert len(svc.characteristics) == 1
        assert svc.characteristics[0].value == "Apple Inc."


class TestBleDeviceProfile:
    """Test BleDeviceProfile model."""

    def _make_profile(self, **kwargs):
        defaults = {
            "mac": "AA:BB:CC:DD:EE:FF",
            "manufacturer": "Apple Inc.",
            "model": "iPhone 15",
            "firmware_rev": "17.3",
            "device_name": "Matt's iPhone",
            "battery_level": 85,
            "services": [
                BleGATTService(uuid="0x1800", uuid16=0x1800, name="Generic Access (GAP)", is_standard=True),
                BleGATTService(uuid="0x180A", uuid16=0x180A, name="Device Information", is_standard=True),
                BleGATTService(uuid="0x180F", uuid16=0x180F, name="Battery Service", is_standard=True),
            ],
            "connection_duration_ms": 450,
        }
        defaults.update(kwargs)
        return BleDeviceProfile(**defaults)

    def test_basic_creation(self):
        p = self._make_profile()
        assert p.mac == "AA:BB:CC:DD:EE:FF"
        assert p.manufacturer == "Apple Inc."
        assert p.model == "iPhone 15"
        assert p.battery_level == 85

    def test_has_device_info(self):
        p = self._make_profile()
        assert p.has_device_info is True

    def test_no_device_info(self):
        p = self._make_profile(manufacturer="", model="", firmware_rev="",
                                hardware_rev="", software_rev="", serial_number="")
        assert p.has_device_info is False

    def test_service_uuids_16bit(self):
        p = self._make_profile()
        assert 0x1800 in p.service_uuids_16bit
        assert 0x180A in p.service_uuids_16bit
        assert 0x180F in p.service_uuids_16bit

    def test_service_names(self):
        p = self._make_profile()
        names = p.service_names
        assert "Generic Access (GAP)" in names
        assert "Device Information" in names

    def test_has_service(self):
        p = self._make_profile()
        assert p.has_service(0x180A) is True
        assert p.has_service(0x1812) is False

    def test_to_enrichment_dict(self):
        p = self._make_profile()
        d = p.to_enrichment_dict()
        assert d["mac"] == "AA:BB:CC:DD:EE:FF"
        assert d["source"] == "gatt_interrogation"
        assert d["manufacturer"] == "Apple Inc."
        assert d["model"] == "iPhone 15"
        assert d["battery_level"] == 85
        assert len(d["services"]) == 3

    def test_enrichment_dict_omits_empty(self):
        p = self._make_profile(manufacturer="", model="", firmware_rev="",
                                hardware_rev="", software_rev="", serial_number="",
                                device_name="", appearance=None, battery_level=None)
        d = p.to_enrichment_dict()
        assert "manufacturer" not in d
        assert "model" not in d
        assert "battery_level" not in d

    def test_timestamp_default(self):
        p = self._make_profile()
        assert isinstance(p.interrogated_at, datetime)
        assert p.interrogated_at.tzinfo is not None

    def test_empty_services(self):
        p = BleDeviceProfile(mac="AA:BB:CC:DD:EE:FF")
        assert p.services == []
        assert p.service_uuids_16bit == []
        assert p.has_device_info is False


class TestBleInterrogationResult:
    """Test BleInterrogationResult model."""

    def test_success_result(self):
        profile = BleDeviceProfile(mac="AA:BB:CC:DD:EE:FF", manufacturer="Apple")
        result = BleInterrogationResult(
            mac="AA:BB:CC:DD:EE:FF",
            success=True,
            profile=profile,
            duration_ms=350,
            node_id="node-01",
        )
        assert result.success is True
        assert result.profile is not None
        assert result.profile.manufacturer == "Apple"
        assert result.node_id == "node-01"

    def test_failure_result(self):
        result = BleInterrogationResult(
            mac="AA:BB:CC:DD:EE:FF",
            success=False,
            error="Connection timeout",
            duration_ms=2000,
        )
        assert result.success is False
        assert result.profile is None
        assert result.error == "Connection timeout"

    def test_serialization(self):
        result = BleInterrogationResult(
            mac="AA:BB:CC:DD:EE:FF",
            success=True,
            duration_ms=500,
        )
        d = result.model_dump()
        assert d["mac"] == "AA:BB:CC:DD:EE:FF"
        assert d["success"] is True


class TestBleInterrogationQueue:
    """Test BleInterrogationQueue model."""

    def test_empty_queue(self):
        q = BleInterrogationQueue()
        assert q.pending == []
        assert q.completed == 0
        assert q.failed == 0
        assert q.active is False

    def test_active_queue(self):
        q = BleInterrogationQueue(
            pending=["AA:BB:CC:DD:EE:FF", "11:22:33:44:55:66"],
            completed=5,
            failed=1,
            on_cooldown=3,
            active=True,
        )
        assert len(q.pending) == 2
        assert q.completed == 5
        assert q.active is True


class TestClassifyDeviceFromProfile:
    """Test GATT-based device classification."""

    def _make_profile(self, services=None, manufacturer="", model="", device_name=""):
        svcs = []
        for uuid16 in (services or []):
            svcs.append(BleGATTService(
                uuid=f"0x{uuid16:04X}",
                uuid16=uuid16,
                name=lookup_service_name(uuid16),
                is_standard=True,
            ))
        return BleDeviceProfile(
            mac="AA:BB:CC:DD:EE:FF",
            services=svcs,
            manufacturer=manufacturer,
            model=model,
            device_name=device_name,
        )

    def test_fitness_tracker(self):
        p = self._make_profile(services=[0x1800, 0x180D, 0x1814])
        assert classify_device_from_profile(p) == "fitness_tracker"

    def test_watch_heart_rate_only(self):
        p = self._make_profile(services=[0x1800, 0x180D])
        assert classify_device_from_profile(p) == "watch"

    def test_hid_keyboard(self):
        p = self._make_profile(services=[0x1800, 0x1812], device_name="Logitech Keyboard K380")
        assert classify_device_from_profile(p) == "keyboard"

    def test_hid_mouse(self):
        p = self._make_profile(services=[0x1800, 0x1812], device_name="MX Mouse")
        assert classify_device_from_profile(p) == "mouse"

    def test_hid_generic(self):
        p = self._make_profile(services=[0x1800, 0x1812], device_name="BT Device")
        assert classify_device_from_profile(p) == "peripheral"

    def test_medical_device(self):
        p = self._make_profile(services=[0x1800, 0x1810])
        assert classify_device_from_profile(p) == "medical_device"

    def test_scale(self):
        p = self._make_profile(services=[0x1800, 0x181D])
        assert classify_device_from_profile(p) == "scale"

    def test_environmental_sensor(self):
        p = self._make_profile(services=[0x1800, 0x181A])
        assert classify_device_from_profile(p) == "environmental_sensor"

    def test_mesh_device(self):
        p = self._make_profile(services=[0x1800, 0x1827])
        assert classify_device_from_profile(p) == "mesh_device"

    def test_apple_watch(self):
        p = self._make_profile(manufacturer="Apple Inc.", model="Apple Watch Series 9")
        assert classify_device_from_profile(p) == "watch"

    def test_apple_iphone(self):
        p = self._make_profile(manufacturer="Apple Inc.", model="iPhone 15 Pro")
        assert classify_device_from_profile(p) == "phone"

    def test_apple_ipad(self):
        p = self._make_profile(manufacturer="Apple Inc.", model="iPad Air")
        assert classify_device_from_profile(p) == "tablet"

    def test_apple_macbook(self):
        p = self._make_profile(manufacturer="Apple Inc.", model="MacBook Pro")
        assert classify_device_from_profile(p) == "laptop"

    def test_apple_airpods(self):
        p = self._make_profile(manufacturer="Apple Inc.", model="AirPods Pro")
        assert classify_device_from_profile(p) == "earbuds"

    def test_samsung_phone(self):
        p = self._make_profile(manufacturer="Samsung", model="Galaxy S24")
        assert classify_device_from_profile(p) == "phone"

    def test_samsung_watch(self):
        p = self._make_profile(manufacturer="Samsung", model="Galaxy Watch 6")
        assert classify_device_from_profile(p) == "watch"

    def test_samsung_buds(self):
        p = self._make_profile(manufacturer="Samsung", model="Galaxy Buds2 Pro")
        assert classify_device_from_profile(p) == "earbuds"

    def test_fitbit(self):
        p = self._make_profile(manufacturer="Fitbit", device_name="Charge 6")
        assert classify_device_from_profile(p) == "fitness_tracker"

    def test_garmin(self):
        p = self._make_profile(manufacturer="Garmin", device_name="Fenix 7")
        assert classify_device_from_profile(p) == "fitness_tracker"

    def test_tile_tracker(self):
        p = self._make_profile(manufacturer="Tile", device_name="Tile Pro")
        assert classify_device_from_profile(p) == "tracker"

    def test_bose_headphones(self):
        p = self._make_profile(manufacturer="Bose", device_name="QC45")
        assert classify_device_from_profile(p) == "headphones"

    def test_unknown_device(self):
        p = self._make_profile(services=[0x1800, 0x1801])
        assert classify_device_from_profile(p) == "unknown"

    def test_simple_peripheral(self):
        p = self._make_profile(services=[0x1800, 0x1801, 0x180F])
        assert classify_device_from_profile(p) == "simple_peripheral"

    def test_fitness_machine(self):
        p = self._make_profile(services=[0x1800, 0x1826])
        assert classify_device_from_profile(p) == "fitness_machine"

    def test_gps_device(self):
        p = self._make_profile(services=[0x1800, 0x1819])
        assert classify_device_from_profile(p) == "gps_device"


class TestStandardServiceUuids:
    """Test the UUID lookup table completeness."""

    def test_has_core_services(self):
        assert 0x1800 in STANDARD_SERVICE_UUIDS
        assert 0x1801 in STANDARD_SERVICE_UUIDS
        assert 0x180A in STANDARD_SERVICE_UUIDS
        assert 0x180F in STANDARD_SERVICE_UUIDS

    def test_has_health_services(self):
        assert 0x180D in STANDARD_SERVICE_UUIDS  # Heart Rate
        assert 0x1810 in STANDARD_SERVICE_UUIDS  # Blood Pressure
        assert 0x1809 in STANDARD_SERVICE_UUIDS  # Health Thermometer

    def test_has_audio_services(self):
        assert 0x1843 in STANDARD_SERVICE_UUIDS  # Audio Input Control
        assert 0x1844 in STANDARD_SERVICE_UUIDS  # Volume Control

    def test_has_mesh_services(self):
        assert 0x1827 in STANDARD_SERVICE_UUIDS  # Mesh Provisioning
        assert 0x1828 in STANDARD_SERVICE_UUIDS  # Mesh Proxy

    def test_table_size(self):
        # Should have a reasonable number of entries
        assert len(STANDARD_SERVICE_UUIDS) >= 50
