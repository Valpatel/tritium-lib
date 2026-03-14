# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.classifier.device_classifier — DeviceClassifier."""

import pytest

from tritium_lib.classifier import DeviceClassifier, DeviceClassification


@pytest.fixture
def dc() -> DeviceClassifier:
    return DeviceClassifier()


class TestDeviceClassifierInit:
    """Test initialization and fingerprint loading."""

    def test_loaded(self, dc: DeviceClassifier):
        assert dc.loaded is True

    def test_bad_path_still_works(self):
        dc = DeviceClassifier(fingerprints_path="/nonexistent/path.json")
        assert dc.loaded is False
        result = dc.classify_ble(name="iPhone 15")
        # Name pattern still works without fingerprint data
        assert result.device_type == "phone"


class TestClassifyBLE:
    """Test BLE device classification with various signals."""

    def test_iphone_by_name(self, dc: DeviceClassifier):
        r = dc.classify_ble(name="iPhone 15 Pro")
        assert r.device_type == "phone"
        assert r.confidence >= 0.9

    def test_airpods_by_name(self, dc: DeviceClassifier):
        r = dc.classify_ble(name="AirPods Pro")
        assert r.device_type == "earbuds"
        assert r.confidence >= 0.9

    def test_apple_watch_by_name(self, dc: DeviceClassifier):
        r = dc.classify_ble(name="Apple Watch")
        assert r.device_type == "watch"
        assert r.confidence >= 0.8

    def test_esp32_by_name(self, dc: DeviceClassifier):
        r = dc.classify_ble(name="ESP32-Sensor")
        assert r.device_type == "microcontroller"
        assert r.confidence >= 0.9

    def test_tesla_by_name(self, dc: DeviceClassifier):
        r = dc.classify_ble(name="Tesla Model 3")
        assert r.device_type == "vehicle"

    def test_oui_apple(self, dc: DeviceClassifier):
        r = dc.classify_ble(mac="AC:BC:32:AA:BB:CC")
        assert r.manufacturer == "Apple"

    def test_oui_espressif(self, dc: DeviceClassifier):
        r = dc.classify_ble(mac="24:0A:C4:12:34:56")
        assert r.manufacturer == "Espressif"
        assert r.device_type == "microcontroller"

    def test_gap_appearance_phone(self, dc: DeviceClassifier):
        r = dc.classify_ble(appearance=0x0040)
        assert r.device_type == "phone"
        assert r.confidence >= 0.9

    def test_gap_appearance_watch(self, dc: DeviceClassifier):
        r = dc.classify_ble(appearance=0x00C0)
        assert r.device_type == "watch"
        assert r.confidence >= 0.9

    def test_gap_appearance_computer(self, dc: DeviceClassifier):
        r = dc.classify_ble(appearance=0x0080)
        assert r.device_type == "computer"

    def test_service_uuid_heart_rate(self, dc: DeviceClassifier):
        r = dc.classify_ble(service_uuids=["0x180D"])
        assert r.device_type == "fitness"

    def test_service_uuid_phone_alert(self, dc: DeviceClassifier):
        r = dc.classify_ble(service_uuids=["0x180E"])
        assert r.device_type == "phone"

    def test_company_id_apple(self, dc: DeviceClassifier):
        r = dc.classify_ble(company_id=76)
        assert r.manufacturer == "Apple"
        assert r.device_type in ("phone", "watch", "computer", "audio", "tag")

    def test_company_id_samsung(self, dc: DeviceClassifier):
        r = dc.classify_ble(company_id=117)
        assert r.manufacturer == "Samsung Electronics"

    def test_company_id_tesla(self, dc: DeviceClassifier):
        r = dc.classify_ble(company_id=555)
        assert r.device_type == "vehicle"

    def test_fast_pair_sony_xm4(self, dc: DeviceClassifier):
        r = dc.classify_ble(fast_pair_model_id="0x01EEB4")
        assert r.device_type == "headphones"
        assert "Sony" in r.device_name

    def test_fast_pair_galaxy_phone(self, dc: DeviceClassifier):
        r = dc.classify_ble(fast_pair_model_id="0x0577B1")
        assert r.device_type == "phone"

    def test_apple_device_class_iphone(self, dc: DeviceClassifier):
        r = dc.classify_ble(apple_device_class="0x02")
        assert r.device_type == "phone"
        assert r.confidence >= 0.9

    def test_apple_device_class_watch(self, dc: DeviceClassifier):
        r = dc.classify_ble(apple_device_class="0x0E")
        assert r.device_type == "watch"

    def test_apple_device_class_mac(self, dc: DeviceClassifier):
        r = dc.classify_ble(apple_device_class="0x0A")
        assert r.device_type == "computer"

    def test_multi_signal_highest_confidence_wins(self, dc: DeviceClassifier):
        """GAP appearance (0.9) should beat name pattern when both match."""
        r = dc.classify_ble(
            name="AirPods Pro",
            appearance=0x0040,  # phone
        )
        # Appearance says phone (0.9), name says earbuds (0.95)
        # Earbuds wins since 0.95 > 0.9
        assert r.device_type == "earbuds"

    def test_unknown_device(self, dc: DeviceClassifier):
        r = dc.classify_ble(mac="FF:FF:FF:00:00:00")
        assert r.device_type == "unknown"

    def test_no_signals(self, dc: DeviceClassifier):
        r = dc.classify_ble()
        assert r.device_type == "unknown"

    def test_signals_populated(self, dc: DeviceClassifier):
        r = dc.classify_ble(
            mac="AC:BC:32:AA:BB:CC",
            name="iPhone 15",
            appearance=0x0040,
        )
        assert len(r.signals) >= 2  # OUI + name + appearance

    def test_to_dict(self, dc: DeviceClassifier):
        r = dc.classify_ble(name="iPhone 15")
        d = r.to_dict()
        assert d["device_type"] == "phone"
        assert d["confidence"] >= 0.9
        assert isinstance(d["signals"], list)


class TestClassifyWiFi:
    """Test WiFi device classification."""

    def test_iphone_hotspot(self, dc: DeviceClassifier):
        r = dc.classify_wifi(ssid="iPhone-Matt")
        assert r.device_type == "phone"

    def test_printer_direct(self, dc: DeviceClassifier):
        r = dc.classify_wifi(ssid="DIRECT-HP-Printer")
        assert r.device_type == "printer"

    def test_laptop_ssid(self, dc: DeviceClassifier):
        r = dc.classify_wifi(ssid="LAPTOP-ABC123")
        assert r.device_type == "computer"

    def test_bssid_oui(self, dc: DeviceClassifier):
        r = dc.classify_wifi(bssid="DC:A6:32:AA:BB:CC")
        assert r.manufacturer == "Raspberry Pi"

    def test_probed_ssids(self, dc: DeviceClassifier):
        r = dc.classify_wifi(probed_ssids=["iPhone-Matt", "HomeNetwork"])
        assert r.device_type == "phone"

    def test_unknown_ssid(self, dc: DeviceClassifier):
        r = dc.classify_wifi(ssid="MyHomeNetwork")
        assert r.device_type == "unknown"

    def test_no_signals(self, dc: DeviceClassifier):
        r = dc.classify_wifi()
        assert r.device_type == "unknown"


class TestVendorUUIDPatterns:
    """Test vendor UUID pattern matching for 128-bit UUIDs."""

    def test_fitbit_uuid(self, dc: DeviceClassifier):
        r = dc.classify_ble(
            service_uuids=["adab1234-6e7d-4601-bda2-bffaa68956ba"]
        )
        assert r.device_type == "fitness"

    def test_kontakt_beacon(self, dc: DeviceClassifier):
        r = dc.classify_ble(
            service_uuids=["f7826da6-4fa2-4e98-8024-bc5b71e0893e"]
        )
        assert r.device_type == "beacon"
