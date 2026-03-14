# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for DeviceClassifier — BLE and WiFi device identification."""

import pytest

from tritium_lib.classifier import DeviceClassification, DeviceClassifier


@pytest.fixture
def classifier():
    """Create a classifier instance (loads data files once)."""
    c = DeviceClassifier()
    return c


# ── MAC utilities ──────────────────────────────────────────────────────

class TestMACUtilities:
    def test_normalize_mac_colon(self):
        assert DeviceClassifier.normalize_mac("aa:bb:cc:dd:ee:ff") == "AA:BB:CC:DD:EE:FF"

    def test_normalize_mac_dash(self):
        assert DeviceClassifier.normalize_mac("aa-bb-cc-dd-ee-ff") == "AA:BB:CC:DD:EE:FF"

    def test_normalize_mac_no_separator(self):
        assert DeviceClassifier.normalize_mac("aabbccddeeff") == "AA:BB:CC:DD:EE:FF"

    def test_is_randomized_true(self):
        # Second hex digit is 2 -> locally administered
        assert DeviceClassifier.is_randomized_mac("02:11:22:33:44:55") is True
        assert DeviceClassifier.is_randomized_mac("06:11:22:33:44:55") is True
        assert DeviceClassifier.is_randomized_mac("0A:11:22:33:44:55") is True
        assert DeviceClassifier.is_randomized_mac("0E:11:22:33:44:55") is True
        assert DeviceClassifier.is_randomized_mac("12:34:56:78:9A:BC") is True
        assert DeviceClassifier.is_randomized_mac("1A:34:56:78:9A:BC") is True

    def test_is_randomized_false(self):
        # Second hex digit is 0 -> globally unique (real OUI)
        assert DeviceClassifier.is_randomized_mac("00:11:22:33:44:55") is False
        assert DeviceClassifier.is_randomized_mac("04:11:22:33:44:55") is False
        assert DeviceClassifier.is_randomized_mac("08:11:22:33:44:55") is False
        assert DeviceClassifier.is_randomized_mac("10:11:22:33:44:55") is False

    def test_get_oui_prefix(self):
        assert DeviceClassifier.get_oui_prefix("AA:BB:CC:DD:EE:FF") == "AA:BB:CC"


# ── BLE classification ────────────────────────────────────────────────

class TestBLEClassification:
    def test_classify_by_appearance_phone(self, classifier):
        result = classifier.classify_ble(appearance=64)
        assert result.device_type == "phone"
        assert result.confidence >= 0.90
        assert "ble_appearance" in result.sources

    def test_classify_by_appearance_watch(self, classifier):
        result = classifier.classify_ble(appearance=192)
        assert result.device_type == "watch"

    def test_classify_by_appearance_laptop(self, classifier):
        result = classifier.classify_ble(appearance=131)
        assert result.device_type == "laptop"

    def test_classify_by_appearance_keyboard(self, classifier):
        result = classifier.classify_ble(appearance=961)
        assert result.device_type == "keyboard"

    def test_classify_by_appearance_mouse(self, classifier):
        result = classifier.classify_ble(appearance=962)
        assert result.device_type == "mouse"

    def test_classify_by_appearance_heart_rate(self, classifier):
        result = classifier.classify_ble(appearance=832)
        assert result.device_type == "health_monitor"

    def test_classify_by_appearance_speaker(self, classifier):
        result = classifier.classify_ble(appearance=2113)
        assert result.device_type == "speaker"

    def test_classify_by_appearance_sensor(self, classifier):
        result = classifier.classify_ble(appearance=1344)
        assert result.device_type == "sensor"

    def test_classify_by_appearance_smart_light(self, classifier):
        result = classifier.classify_ble(appearance=1408)
        assert result.device_type == "smart_light"

    def test_classify_by_appearance_thermostat(self, classifier):
        result = classifier.classify_ble(appearance=1537)
        assert result.device_type == "thermostat"

    def test_classify_by_appearance_smart_lock(self, classifier):
        result = classifier.classify_ble(appearance=1800)
        assert result.device_type == "smart_lock"

    def test_classify_by_name_iphone(self, classifier):
        result = classifier.classify_ble(name="iPhone 15 Pro")
        assert result.device_type == "phone"
        assert result.manufacturer == "Apple"
        assert result.confidence >= 0.90
        assert "ble_name_pattern" in result.sources

    def test_classify_by_name_airpods(self, classifier):
        result = classifier.classify_ble(name="AirPods Pro")
        assert result.device_type == "earbud"
        assert result.manufacturer == "Apple"

    def test_classify_by_name_galaxy_watch(self, classifier):
        result = classifier.classify_ble(name="Galaxy Watch6")
        assert result.device_type == "watch"
        assert result.manufacturer == "Samsung"

    def test_classify_by_name_galaxy_buds(self, classifier):
        result = classifier.classify_ble(name="Galaxy Buds2 Pro")
        assert result.device_type == "earbud"
        assert result.manufacturer == "Samsung"

    def test_classify_by_name_sony_headphones(self, classifier):
        result = classifier.classify_ble(name="WH-1000XM5")
        assert result.device_type == "headphones"
        assert result.manufacturer == "Sony"

    def test_classify_by_name_pixel_buds(self, classifier):
        result = classifier.classify_ble(name="Pixel Buds Pro")
        assert result.device_type == "earbud"
        assert result.manufacturer == "Google"

    def test_classify_by_name_fitbit(self, classifier):
        result = classifier.classify_ble(name="Fitbit Charge 5")
        assert result.device_type == "fitness_tracker"
        assert result.manufacturer == "Fitbit"

    def test_classify_by_name_garmin(self, classifier):
        result = classifier.classify_ble(name="Forerunner 965")
        assert result.device_type == "watch"
        assert result.manufacturer == "Garmin"

    def test_classify_by_name_jbl_speaker(self, classifier):
        result = classifier.classify_ble(name="JBL Flip 6")
        assert result.device_type == "speaker"
        assert result.manufacturer == "JBL"

    def test_classify_by_name_bose(self, classifier):
        result = classifier.classify_ble(name="Bose QuietComfort 45")
        assert result.device_type == "headphones"
        assert result.manufacturer == "Bose"

    def test_classify_by_name_tile(self, classifier):
        result = classifier.classify_ble(name="Tile Mate")
        assert result.device_type == "tracker"
        assert result.manufacturer == "Tile"

    def test_classify_by_name_meshtastic(self, classifier):
        result = classifier.classify_ble(name="Meshtastic Node")
        assert result.device_type == "mesh_radio"
        assert result.manufacturer == "Meshtastic"

    def test_classify_by_name_esp32(self, classifier):
        result = classifier.classify_ble(name="ESP32-DevKit")
        assert result.device_type == "iot"
        assert result.manufacturer == "Espressif"

    def test_classify_by_name_macbook(self, classifier):
        result = classifier.classify_ble(name="MacBook Pro")
        assert result.device_type == "laptop"
        assert result.manufacturer == "Apple"

    def test_classify_by_name_homepod(self, classifier):
        result = classifier.classify_ble(name="HomePod mini")
        assert result.device_type == "smart_speaker"
        assert result.manufacturer == "Apple"

    def test_classify_by_name_nothing_ear(self, classifier):
        result = classifier.classify_ble(name="Nothing Ear (2)")
        assert result.device_type == "earbud"
        assert result.manufacturer == "Nothing"

    def test_classify_by_name_dji_drone(self, classifier):
        result = classifier.classify_ble(name="DJI Mini 3")
        assert result.device_type == "drone"
        assert result.manufacturer == "DJI"

    def test_classify_by_name_xbox_controller(self, classifier):
        result = classifier.classify_ble(name="Xbox Controller")
        assert result.device_type == "controller"
        assert result.manufacturer == "Microsoft"

    def test_classify_by_name_switchbot(self, classifier):
        result = classifier.classify_ble(name="SwitchBot Curtain")
        assert result.device_type == "iot"
        assert result.manufacturer == "SwitchBot"

    def test_classify_by_company_id_apple(self, classifier):
        result = classifier.classify_ble(company_id=76)
        assert result.manufacturer == "Apple, Inc."
        assert result.device_type == "phone"
        assert "ble_company_id" in result.sources

    def test_classify_by_company_id_samsung(self, classifier):
        result = classifier.classify_ble(company_id=117)
        assert result.manufacturer == "Samsung Electronics Co. Ltd."

    def test_classify_by_company_id_google(self, classifier):
        result = classifier.classify_ble(company_id=224)
        assert result.manufacturer == "Google Inc."

    def test_classify_by_company_id_bose(self, classifier):
        result = classifier.classify_ble(company_id=157)
        assert result.manufacturer == "Bose Corporation"
        assert result.device_type == "headphones"

    def test_classify_by_oui_apple(self, classifier):
        result = classifier.classify_ble(mac="04:0C:CE:AA:BB:CC")
        assert result.manufacturer == "Apple, Inc."
        assert "oui_lookup" in result.sources

    def test_classify_by_oui_samsung(self, classifier):
        result = classifier.classify_ble(mac="00:07:AB:11:22:33")
        assert result.manufacturer == "Samsung Electronics Co., Ltd."

    def test_classify_by_oui_espressif(self, classifier):
        result = classifier.classify_ble(mac="30:AE:A4:11:22:33")
        assert result.manufacturer == "Espressif Inc."
        assert result.device_type == "iot"

    def test_classify_by_oui_raspberry_pi(self, classifier):
        result = classifier.classify_ble(mac="B8:27:EB:11:22:33")
        assert result.manufacturer == "Raspberry Pi Foundation"
        assert result.device_type == "single_board_computer"

    def test_classify_randomized_mac_no_oui(self, classifier):
        # Randomized MAC — OUI lookup should not contribute
        result = classifier.classify_ble(mac="02:11:22:33:44:55")
        assert result.is_randomized_mac is True
        assert "oui_lookup" not in result.sources

    def test_classify_by_service_uuid_heart_rate(self, classifier):
        result = classifier.classify_ble(service_uuids=["0x180D"])
        assert result.device_type == "fitness_tracker"
        assert "ble_service_uuid" in result.sources

    def test_classify_by_service_uuid_hid(self, classifier):
        result = classifier.classify_ble(service_uuids=["0x1812"])
        assert result.device_type == "hid"

    def test_classify_by_service_uuid_fitness(self, classifier):
        result = classifier.classify_ble(service_uuids=["0x1826"])
        assert result.device_type == "fitness_equipment"

    def test_classify_apple_continuity_nearby_info(self, classifier):
        # Simulate raw advertising with Apple company ID + Nearby Info
        raw = bytes([0x4C, 0x00, 0x10, 0x05, 0x01])
        result = classifier.classify_ble(raw_adv=raw)
        assert result.manufacturer == "Apple"
        assert "apple_continuity_nearby_info" in result.sources

    def test_classify_apple_continuity_findmy(self, classifier):
        raw = bytes([0x4C, 0x00, 0x12, 0x19, 0x00])
        result = classifier.classify_ble(raw_adv=raw)
        assert result.device_type == "tracker"
        assert result.manufacturer == "Apple"

    def test_classify_apple_proximity_pairing(self, classifier):
        # AirPods Pro model ID 0x0E20
        raw = bytes([0x4C, 0x00, 0x05, 0x0E, 0x20])
        result = classifier.classify_ble(raw_adv=raw)
        assert result.device_type == "earbud"
        assert result.manufacturer == "Apple"
        assert "AirPods Pro" in (result.model_hint or "")

    def test_classify_combined_signals(self, classifier):
        """Test that multiple signal sources combine correctly."""
        result = classifier.classify_ble(
            mac="04:0C:CE:AA:BB:CC",
            name="iPhone 15",
            company_id=76,
            appearance=64,
        )
        assert result.device_type == "phone"
        assert result.manufacturer is not None
        assert result.confidence >= 0.90
        assert len(result.sources) >= 2

    def test_classify_unknown_device(self, classifier):
        result = classifier.classify_ble(mac="02:FF:FF:FF:FF:FF")
        assert result.device_type == "unknown"
        assert result.confidence == 0.0
        assert result.is_randomized_mac is True


# ── WiFi classification ───────────────────────────────────────────────

class TestWiFiClassification:
    def test_classify_by_ssid_iphone_hotspot(self, classifier):
        result = classifier.classify_wifi(ssid="iPhone 15 Pro")
        assert result.device_type == "phone"
        assert result.manufacturer == "Apple"
        assert "wifi_ssid_pattern" in result.sources

    def test_classify_by_ssid_android_hotspot(self, classifier):
        result = classifier.classify_wifi(ssid="AndroidAP1234")
        assert result.device_type == "phone"
        assert result.confidence >= 0.85

    def test_classify_by_ssid_ring(self, classifier):
        result = classifier.classify_wifi(ssid="Ring-ABC123")
        assert result.device_type == "doorbell"
        assert result.manufacturer == "Amazon (Ring)"

    def test_classify_by_ssid_nest(self, classifier):
        result = classifier.classify_wifi(ssid="Nest-Thermostat")
        assert result.manufacturer == "Google (Nest)"

    def test_classify_by_ssid_eero(self, classifier):
        result = classifier.classify_wifi(ssid="eero-mesh")
        assert result.device_type == "router"
        assert result.manufacturer == "Amazon (eero)"

    def test_classify_by_ssid_eduroam(self, classifier):
        result = classifier.classify_wifi(ssid="eduroam")
        assert "wifi_ssid_pattern" in result.sources
        assert result.confidence >= 0.90

    def test_classify_by_ssid_hp_printer(self, classifier):
        result = classifier.classify_wifi(ssid="HP-Print-A1-LaserJet")
        assert result.device_type == "printer"
        assert result.manufacturer == "HP"

    def test_classify_by_ssid_gopro(self, classifier):
        result = classifier.classify_wifi(ssid="GoPro-HERO12")
        assert result.device_type == "camera"
        assert result.manufacturer == "GoPro"

    def test_classify_by_ssid_dji(self, classifier):
        result = classifier.classify_wifi(ssid="DJI-Mini3Pro")
        assert result.device_type == "drone"
        assert result.manufacturer == "DJI"

    def test_classify_by_dhcp_vendor_android(self, classifier):
        result = classifier.classify_wifi(vendor_class="android-dhcp-14")
        assert result.device_type == "phone"
        assert result.os_hint == "Android"
        assert "dhcp_vendor_class" in result.sources

    def test_classify_by_dhcp_vendor_windows(self, classifier):
        result = classifier.classify_wifi(vendor_class="MSFT 5.0")
        assert result.os_hint is not None
        assert "Windows" in result.os_hint

    def test_classify_by_dhcp_vendor_cisco(self, classifier):
        result = classifier.classify_wifi(vendor_class="Cisco Systems 2960")
        assert result.device_type == "router"

    def test_classify_by_hostname_iphone(self, classifier):
        result = classifier.classify_wifi(hostname="iPhone-15-Pro")
        assert result.device_type == "phone"
        assert result.os_hint == "iOS"
        assert "dhcp_hostname" in result.sources

    def test_classify_by_hostname_windows_desktop(self, classifier):
        result = classifier.classify_wifi(hostname="DESKTOP-ABC1234")
        assert result.device_type == "desktop"
        assert result.os_hint == "Windows"

    def test_classify_by_hostname_windows_laptop(self, classifier):
        result = classifier.classify_wifi(hostname="LAPTOP-XYZ5678")
        assert result.device_type == "laptop"
        assert result.os_hint == "Windows"

    def test_classify_by_hostname_raspberrypi(self, classifier):
        result = classifier.classify_wifi(hostname="raspberrypi")
        assert result.device_type == "single_board_computer"
        assert result.os_hint == "Linux"

    def test_classify_by_hostname_xbox(self, classifier):
        result = classifier.classify_wifi(hostname="XBOX-One")
        assert result.device_type == "gaming"

    def test_classify_by_hostname_chromecast(self, classifier):
        result = classifier.classify_wifi(hostname="Chromecast-Ultra")
        assert result.device_type == "streaming_device"

    def test_classify_by_mdns_airplay(self, classifier):
        result = classifier.classify_wifi(mdns_services=["_airplay._tcp"])
        assert result.device_type == "streaming_device"
        assert result.manufacturer == "Apple"
        assert "mdns_service" in result.sources

    def test_classify_by_mdns_googlecast(self, classifier):
        result = classifier.classify_wifi(mdns_services=["_googlecast._tcp"])
        assert result.device_type == "streaming_device"
        assert result.manufacturer == "Google"

    def test_classify_by_mdns_printer(self, classifier):
        result = classifier.classify_wifi(mdns_services=["_ipp._tcp"])
        assert result.device_type == "printer"

    def test_classify_by_mdns_homekit(self, classifier):
        result = classifier.classify_wifi(mdns_services=["_hap._tcp"])
        assert result.device_type == "smart_home"
        assert result.manufacturer == "Apple"

    def test_classify_by_mdns_spotify(self, classifier):
        result = classifier.classify_wifi(mdns_services=["_spotify-connect._tcp"])
        assert result.device_type == "speaker"

    def test_classify_by_oui_wifi(self, classifier):
        result = classifier.classify_wifi(mac="00:18:4D:11:22:33")
        assert result.manufacturer == "Ubiquiti Inc."
        assert "oui_lookup" in result.sources

    def test_classify_combined_wifi(self, classifier):
        result = classifier.classify_wifi(
            mac="00:17:88:11:22:33",
            hostname="Philips-Hue-Bridge",
            mdns_services=["_hap._tcp"],
        )
        assert result.manufacturer is not None
        assert len(result.sources) >= 1


# ── Lookup utilities ──────────────────────────────────────────────────

class TestLookupUtilities:
    def test_lookup_oui(self, classifier):
        entry = classifier.lookup_oui("30:AE:A4:11:22:33")
        assert entry is not None
        assert entry["manufacturer"] == "Espressif Inc."

    def test_lookup_oui_not_found(self, classifier):
        entry = classifier.lookup_oui("FF:FF:FF:11:22:33")
        assert entry is None

    def test_lookup_company_id(self, classifier):
        entry = classifier.lookup_company_id(76)
        assert entry is not None
        assert "Apple" in entry["name"]

    def test_lookup_company_id_not_found(self, classifier):
        entry = classifier.lookup_company_id(99999)
        assert entry is None

    def test_lookup_appearance(self, classifier):
        entry = classifier.lookup_appearance(64)
        assert entry is not None
        assert entry["category"] == "Phone"

    def test_lookup_service_uuid(self, classifier):
        entry = classifier.lookup_service_uuid("180D")
        assert entry is not None
        assert entry["name"] == "Heart Rate"


# ── DeviceClassification merging ──────────────────────────────────────

class TestClassificationMerge:
    def test_merge_higher_confidence_wins(self):
        a = DeviceClassification(
            device_type="unknown",
            confidence=0.3,
            sources=["oui_lookup"],
        )
        b = DeviceClassification(
            device_type="phone",
            manufacturer="Apple",
            confidence=0.9,
            sources=["ble_name_pattern"],
        )
        merged = a.merge(b)
        assert merged.device_type == "phone"
        assert merged.manufacturer == "Apple"
        assert merged.confidence == 0.9
        assert "oui_lookup" in merged.sources
        assert "ble_name_pattern" in merged.sources

    def test_merge_fills_missing_fields(self):
        a = DeviceClassification(
            device_type="phone",
            manufacturer="Apple",
            confidence=0.9,
            sources=["ble_name_pattern"],
        )
        b = DeviceClassification(
            device_type="unknown",
            os_hint="iOS",
            confidence=0.5,
            sources=["dhcp_vendor_class"],
        )
        merged = a.merge(b)
        assert merged.device_type == "phone"
        assert merged.os_hint == "iOS"
        assert merged.manufacturer == "Apple"

    def test_merge_preserves_randomized_mac(self):
        a = DeviceClassification(is_randomized_mac=True, confidence=0.0)
        b = DeviceClassification(is_randomized_mac=False, confidence=0.5)
        merged = a.merge(b)
        assert merged.is_randomized_mac is True


# ── Edge cases ────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_inputs(self, classifier):
        result = classifier.classify_ble()
        assert result.device_type == "unknown"
        assert result.confidence == 0.0

    def test_empty_wifi_inputs(self, classifier):
        result = classifier.classify_wifi()
        assert result.device_type == "unknown"

    def test_unknown_appearance(self, classifier):
        result = classifier.classify_ble(appearance=99999)
        assert result.device_type == "unknown"

    def test_case_insensitive_name(self, classifier):
        result = classifier.classify_ble(name="iphone 15")
        assert result.device_type == "phone"
        assert result.manufacturer == "Apple"

    def test_case_insensitive_ssid(self, classifier):
        result = classifier.classify_wifi(ssid="ANDROIDAP")
        assert result.device_type == "phone"
