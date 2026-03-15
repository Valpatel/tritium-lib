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
        assert "Apple" in r.manufacturer
        assert r.device_type in ("phone", "watch", "computer", "audio", "tag")

    def test_company_id_samsung(self, dc: DeviceClassifier):
        r = dc.classify_ble(company_id=117)
        assert "Samsung Electronics" in r.manufacturer

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


class TestExpandedCompanyIDs:
    """Test expanded BLE company ID database (500+ entries)."""

    def test_company_id_count(self, dc: DeviceClassifier):
        """Verify we have 500+ company IDs in the fingerprint database."""
        company_ids = dc._data.get("company_ids", {})
        assert len(company_ids) >= 500, f"Expected 500+ company IDs, got {len(company_ids)}"

    def test_company_id_google(self, dc: DeviceClassifier):
        r = dc.classify_ble(company_id=224)
        assert "Google" in r.manufacturer

    def test_company_id_amazon(self, dc: DeviceClassifier):
        r = dc.classify_ble(company_id=321)
        assert "Amazon" in r.manufacturer

    def test_company_id_xiaomi(self, dc: DeviceClassifier):
        r = dc.classify_ble(company_id=911)
        assert "Xiaomi" in r.manufacturer

    def test_company_id_huawei(self, dc: DeviceClassifier):
        r = dc.classify_ble(company_id=361)
        assert "Huawei" in r.manufacturer

    def test_company_id_sony(self, dc: DeviceClassifier):
        r = dc.classify_ble(company_id=301)
        assert "Sony" in r.manufacturer

    def test_company_id_lg(self, dc: DeviceClassifier):
        r = dc.classify_ble(company_id=196)
        assert "LG" in r.manufacturer

    def test_company_id_bose(self, dc: DeviceClassifier):
        r = dc.classify_ble(company_id=158)
        assert "Bose" in r.manufacturer
        assert r.device_type == "audio"

    def test_company_id_garmin(self, dc: DeviceClassifier):
        r = dc.classify_ble(company_id=135)
        assert "Garmin" in r.manufacturer
        assert r.device_type in ("fitness", "watch", "gps")

    def test_company_id_fitbit(self, dc: DeviceClassifier):
        r = dc.classify_ble(company_id=449)
        assert "Fitbit" in r.manufacturer

    def test_company_id_tile(self, dc: DeviceClassifier):
        r = dc.classify_ble(company_id=450)
        assert "Tile" in r.manufacturer
        assert r.device_type == "tag"

    def test_company_id_ring(self, dc: DeviceClassifier):
        r = dc.classify_ble(company_id=405)
        assert "Ring" in r.manufacturer

    def test_company_id_philips_signify(self, dc: DeviceClassifier):
        r = dc.classify_ble(company_id=477)
        assert "Philips" in r.manufacturer or "Signify" in r.manufacturer
        assert r.device_type == "smart_home"

    def test_company_id_ikea(self, dc: DeviceClassifier):
        r = dc.classify_ble(company_id=478)
        assert "IKEA" in r.manufacturer
        assert r.device_type == "smart_home"

    def test_company_id_bmw(self, dc: DeviceClassifier):
        r = dc.classify_ble(company_id=1515)
        assert "BMW" in r.manufacturer
        assert r.device_type == "vehicle"

    def test_company_id_mercedes(self, dc: DeviceClassifier):
        r = dc.classify_ble(company_id=380)
        assert "Mercedes" in r.manufacturer
        assert r.device_type == "vehicle"

    def test_company_id_ford(self, dc: DeviceClassifier):
        r = dc.classify_ble(company_id=1827)
        assert "Ford" in r.manufacturer
        assert r.device_type == "vehicle"

    def test_company_id_dexcom(self, dc: DeviceClassifier):
        r = dc.classify_ble(company_id=1202)
        assert "Dexcom" in r.manufacturer
        assert r.device_type == "medical"

    def test_company_id_medtronic(self, dc: DeviceClassifier):
        r = dc.classify_ble(company_id=1201)
        assert "Medtronic" in r.manufacturer
        assert r.device_type == "medical"

    def test_company_id_omron(self, dc: DeviceClassifier):
        r = dc.classify_ble(company_id=1200)
        assert "Omron" in r.manufacturer
        assert r.device_type == "medical"

    def test_company_id_whoop(self, dc: DeviceClassifier):
        r = dc.classify_ble(company_id=414)
        assert "Whoop" in r.manufacturer
        assert r.device_type == "fitness"

    def test_company_id_oura(self, dc: DeviceClassifier):
        r = dc.classify_ble(company_id=423)
        assert "Oura" in r.manufacturer
        assert r.device_type == "fitness"

    def test_company_id_jabra(self, dc: DeviceClassifier):
        r = dc.classify_ble(company_id=397)
        assert "Jabra" in r.manufacturer
        assert r.device_type == "audio"

    def test_company_id_sennheiser(self, dc: DeviceClassifier):
        r = dc.classify_ble(company_id=1172)
        assert "Sennheiser" in r.manufacturer
        assert r.device_type == "audio"

    def test_company_id_bang_olufsen(self, dc: DeviceClassifier):
        r = dc.classify_ble(company_id=259)
        assert "Bang" in r.manufacturer
        assert r.device_type == "audio"

    def test_company_id_meta(self, dc: DeviceClassifier):
        r = dc.classify_ble(company_id=427)
        assert "Meta" in r.manufacturer
        assert r.device_type == "vr_headset"

    def test_company_id_nintendo(self, dc: DeviceClassifier):
        r = dc.classify_ble(company_id=1363)
        assert "Nintendo" in r.manufacturer
        assert r.device_type == "gamepad"

    def test_company_id_logitech(self, dc: DeviceClassifier):
        r = dc.classify_ble(company_id=474)
        assert "Logitech" in r.manufacturer

    def test_company_id_razer(self, dc: DeviceClassifier):
        r = dc.classify_ble(company_id=1678)
        assert "Razer" in r.manufacturer

    def test_company_id_gopro(self, dc: DeviceClassifier):
        r = dc.classify_ble(company_id=754)
        assert "GoPro" in r.manufacturer
        assert r.device_type == "camera"

    def test_company_id_espressif(self, dc: DeviceClassifier):
        r = dc.classify_ble(company_id=1100)
        assert "Espressif" in r.manufacturer
        assert r.device_type == "iot_device"

    def test_company_id_onplus(self, dc: DeviceClassifier):
        r = dc.classify_ble(company_id=1839)
        assert "OnePlus" in r.manufacturer
        assert r.device_type == "phone"

    def test_company_id_dell(self, dc: DeviceClassifier):
        r = dc.classify_ble(company_id=1054)
        assert "Dell" in r.manufacturer
        assert r.device_type == "computer"

    def test_company_id_lenovo(self, dc: DeviceClassifier):
        r = dc.classify_ble(company_id=709)
        assert "Lenovo" in r.manufacturer
        assert r.device_type == "computer"

    def test_company_id_htc_vr(self, dc: DeviceClassifier):
        r = dc.classify_ble(company_id=749)
        assert "HTC" in r.manufacturer

    def test_company_id_sonos(self, dc: DeviceClassifier):
        r = dc.classify_ble(company_id=1447)
        assert "Sonos" in r.manufacturer
        assert r.device_type == "audio"

    def test_company_id_withings(self, dc: DeviceClassifier):
        r = dc.classify_ble(company_id=1023)
        assert "Withings" in r.manufacturer

    def test_all_company_ids_have_name(self, dc: DeviceClassifier):
        """Every company ID entry must have a non-empty name."""
        company_ids = dc._data.get("company_ids", {})
        for cid, entry in company_ids.items():
            assert entry.get("name"), f"Company ID {cid} missing name"

    def test_all_company_ids_have_types(self, dc: DeviceClassifier):
        """Every company ID entry must have at least one type."""
        company_ids = dc._data.get("company_ids", {})
        for cid, entry in company_ids.items():
            types = entry.get("types", [])
            assert len(types) > 0, f"Company ID {cid} ({entry.get('name')}) has no types"


class TestBleCompanyIdsJson:
    """Test the standalone ble_company_ids.json database."""

    def test_standalone_db_loads(self):
        import json
        import os
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "src", "tritium_lib", "data"
        )
        path = os.path.join(data_dir, "ble_company_ids.json")
        with open(path) as f:
            data = json.load(f)
        companies = data.get("companies", {})
        assert len(companies) >= 500, f"Expected 500+ companies, got {len(companies)}"

    def test_standalone_db_has_major_brands(self):
        import json
        import os
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "src", "tritium_lib", "data"
        )
        path = os.path.join(data_dir, "ble_company_ids.json")
        with open(path) as f:
            data = json.load(f)
        companies = data["companies"]
        # Check key company IDs exist
        assert "76" in companies, "Apple (76) missing"
        assert "117" in companies, "Samsung (117) missing"
        assert "224" in companies, "Google (224) missing"
        assert "6" in companies, "Microsoft (6) missing"
        assert "135" in companies, "Garmin (135) missing"
        assert "301" in companies, "Sony (301) missing"

    def test_standalone_db_entries_valid(self):
        import json
        import os
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "src", "tritium_lib", "data"
        )
        path = os.path.join(data_dir, "ble_company_ids.json")
        with open(path) as f:
            data = json.load(f)
        for cid, entry in data["companies"].items():
            assert "name" in entry, f"Company {cid} missing name"
            assert "device_types" in entry, f"Company {cid} missing device_types"
            assert len(entry["device_types"]) > 0, f"Company {cid} has empty device_types"


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
