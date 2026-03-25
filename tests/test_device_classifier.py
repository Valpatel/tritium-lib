# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.classifier.device_classifier — DeviceClassifier."""

import pytest

from tritium_lib.classifier import DeviceClassifier, DeviceClassification, is_mac_randomized


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


# ======================================================================
# MAC Randomization Detection
# ======================================================================


class TestMACRandomization:
    """Test MAC address randomization detection."""

    def test_locally_administered_second_char_2(self):
        assert is_mac_randomized("02:AA:BB:CC:DD:EE") is True

    def test_locally_administered_second_char_6(self):
        assert is_mac_randomized("06:AA:BB:CC:DD:EE") is True

    def test_locally_administered_second_char_A(self):
        assert is_mac_randomized("0A:AA:BB:CC:DD:EE") is True

    def test_locally_administered_second_char_E(self):
        assert is_mac_randomized("0E:AA:BB:CC:DD:EE") is True

    def test_locally_administered_second_char_3(self):
        assert is_mac_randomized("F3:AA:BB:CC:DD:EE") is True

    def test_locally_administered_second_char_7(self):
        assert is_mac_randomized("D7:AA:BB:CC:DD:EE") is True

    def test_locally_administered_second_char_B(self):
        assert is_mac_randomized("4B:AA:BB:CC:DD:EE") is True

    def test_locally_administered_second_char_F(self):
        assert is_mac_randomized("5F:AA:BB:CC:DD:EE") is True

    def test_globally_unique_apple(self):
        assert is_mac_randomized("AC:BC:32:AA:BB:CC") is False

    def test_globally_unique_espressif(self):
        assert is_mac_randomized("24:0A:C4:12:34:56") is False

    def test_globally_unique_samsung(self):
        assert is_mac_randomized("40:B4:CD:12:34:56") is False

    def test_empty_mac(self):
        assert is_mac_randomized("") is False

    def test_short_mac(self):
        assert is_mac_randomized("AA") is False

    def test_dash_separator(self):
        assert is_mac_randomized("02-AA-BB-CC-DD-EE") is True

    def test_dot_separator(self):
        assert is_mac_randomized("02.AA.BB.CC.DD.EE") is True

    def test_lowercase_randomized(self):
        assert is_mac_randomized("4a:bb:cc:dd:ee:ff") is True

    def test_lowercase_not_randomized(self):
        assert is_mac_randomized("40:b4:cd:12:34:56") is False

    def test_ble_skips_oui_for_randomized_mac(self, dc: DeviceClassifier):
        """Randomized MACs should not use OUI lookup — it's unreliable."""
        r = dc.classify_ble(mac="02:BC:32:AA:BB:CC")
        assert r.mac_randomized is True
        # Should not have any OUI signals since MAC is randomized
        oui_signals = [s for s in r.signals if s["signal"] == "oui"]
        assert len(oui_signals) == 0

    def test_ble_randomized_still_classifies_by_name(self, dc: DeviceClassifier):
        """Even with randomized MAC, name classification should work."""
        r = dc.classify_ble(mac="02:BC:32:AA:BB:CC", name="iPhone 15")
        assert r.mac_randomized is True
        assert r.device_type == "phone"
        assert r.confidence >= 0.9

    def test_wifi_randomized_bssid(self, dc: DeviceClassifier):
        """Randomized BSSID should be detected in WiFi classification."""
        r = dc.classify_wifi(bssid="02:AA:BB:CC:DD:EE", ssid="iPhone-Matt")
        assert r.mac_randomized is True
        assert r.device_type == "phone"

    def test_to_dict_includes_mac_randomized(self, dc: DeviceClassifier):
        r = dc.classify_ble(mac="02:BC:32:AA:BB:CC")
        d = r.to_dict()
        assert "mac_randomized" in d
        assert d["mac_randomized"] is True


# ======================================================================
# Supplementary OUI Database (461 prefixes)
# ======================================================================


class TestSupplementaryOUI:
    """Test OUI lookup using the supplementary oui_device_types.json database."""

    def test_oui_db_loaded(self, dc: DeviceClassifier):
        stats = dc.database_stats
        assert stats["oui_prefixes"] >= 400, f"Expected 400+ OUI prefixes, got {stats['oui_prefixes']}"

    def test_oui_apple_prefix_not_in_hardcoded(self, dc: DeviceClassifier):
        """An Apple OUI prefix that's in the DB but not hardcoded."""
        # 00:03:93 is Apple in the DB but not in the hardcoded _OUI_MANUFACTURERS
        r = dc.classify_ble(mac="00:03:93:AA:BB:CC")
        assert r.manufacturer == "Apple, Inc."
        assert r.device_type in ("phone", "tablet", "laptop", "watch", "earbud")

    def test_oui_samsung_from_db(self, dc: DeviceClassifier):
        """Samsung prefix from supplementary DB."""
        r = dc.classify_ble(mac="00:07:AB:AA:BB:CC")
        assert "Samsung" in r.manufacturer

    def test_oui_device_type_candidates(self, dc: DeviceClassifier):
        """DB entries should provide device_type_candidates list."""
        r = dc.classify_ble(mac="00:03:93:AA:BB:CC")
        oui_signals = [s for s in r.signals if "oui" in s["signal"]]
        assert len(oui_signals) > 0
        oui_sig = oui_signals[0]
        assert "device_type_candidates" in oui_sig
        assert len(oui_sig["device_type_candidates"]) > 1

    def test_hardcoded_oui_takes_priority(self, dc: DeviceClassifier):
        """Hardcoded OUI entries should still match before the DB."""
        r = dc.classify_ble(mac="AC:BC:32:AA:BB:CC")
        assert r.manufacturer == "Apple"  # Hardcoded says "Apple", not "Apple, Inc."

    def test_unknown_oui_returns_none(self, dc: DeviceClassifier):
        """Completely unknown OUI returns no classification."""
        r = dc.classify_ble(mac="FF:FF:FF:00:00:00")
        assert r.device_type == "unknown"
        assert r.manufacturer == ""


# ======================================================================
# Supplementary BLE Name Patterns (217 patterns)
# ======================================================================


class TestSupplementaryBLENamePatterns:
    """Test BLE name classification using the supplementary ble_name_patterns.json."""

    def test_name_db_loaded(self, dc: DeviceClassifier):
        stats = dc.database_stats
        assert stats["ble_name_patterns"] >= 200, f"Expected 200+ name patterns, got {stats['ble_name_patterns']}"

    def test_hardcoded_pattern_still_works(self, dc: DeviceClassifier):
        """Hardcoded patterns should still match first."""
        r = dc.classify_ble(name="iPhone 15 Pro")
        assert r.device_type == "phone"
        name_sigs = [s for s in r.signals if s["signal"] == "name_pattern"]
        assert len(name_sigs) > 0  # Matched by hardcoded, not DB

    def test_db_pattern_for_apple_tv(self, dc: DeviceClassifier):
        """Apple TV might be in DB but not hardcoded list."""
        r = dc.classify_ble(name="Apple TV")
        # Should match either hardcoded or DB
        assert r.device_type != "unknown"

    def test_name_db_provides_manufacturer(self, dc: DeviceClassifier):
        """DB name patterns include manufacturer metadata."""
        # iPad is in the hardcoded list so check it matches
        r = dc.classify_ble(name="iPad Pro 12.9")
        assert r.device_type == "tablet"

    def test_unknown_name_no_match(self, dc: DeviceClassifier):
        """Completely unknown name returns no name signal."""
        r = dc.classify_ble(name="xyzzy_random_device_12345")
        # Should not match any pattern
        name_sigs = [s for s in r.signals if "name" in s.get("signal", "")]
        assert len(name_sigs) == 0


# ======================================================================
# Supplementary WiFi SSID Patterns (72 patterns)
# ======================================================================


class TestSupplementaryWiFiSSID:
    """Test WiFi SSID classification using supplementary wifi_ssid_patterns.json."""

    def test_ssid_db_loaded(self, dc: DeviceClassifier):
        stats = dc.database_stats
        assert stats["wifi_ssid_patterns"] >= 60, f"Expected 60+ WiFi SSID patterns, got {stats['wifi_ssid_patterns']}"

    def test_iphone_hotspot_from_db(self, dc: DeviceClassifier):
        """iPhone hotspot matches DB pattern with manufacturer metadata."""
        r = dc.classify_wifi(ssid="iPhone-14-Pro")
        assert r.device_type == "phone"
        # DB pattern should provide manufacturer
        assert r.manufacturer == "Apple"

    def test_macbook_hotspot(self, dc: DeviceClassifier):
        r = dc.classify_wifi(ssid="MacBook Pro")
        assert r.device_type in ("computer", "laptop")

    def test_android_hotspot(self, dc: DeviceClassifier):
        r = dc.classify_wifi(ssid="Android-ABCD1234")
        assert r.device_type == "phone"

    def test_chromecast_ssid(self, dc: DeviceClassifier):
        r = dc.classify_wifi(ssid="ChromeCast-Ultra")
        assert r.device_type in ("media_player", "streaming_device")

    def test_roku_ssid(self, dc: DeviceClassifier):
        r = dc.classify_wifi(ssid="Roku-Ultra-4800")
        assert r.device_type in ("media_player", "streaming_device")

    def test_db_null_device_type_falls_through(self, dc: DeviceClassifier):
        """DB entries with null device_type should not block hardcoded patterns."""
        # DIRECT- is in the DB with null device_type, but hardcoded has "printer"
        r = dc.classify_wifi(ssid="DIRECT-HP-OfficeJet")
        assert r.device_type == "printer"

    def test_tesla_ssid(self, dc: DeviceClassifier):
        r = dc.classify_wifi(ssid="TeslaModel3-ABCDEF")
        assert r.device_type in ("vehicle", "automotive")

    def test_xfinity_hotspot(self, dc: DeviceClassifier):
        r = dc.classify_wifi(ssid="xfinitywifi")
        assert r.device_type in ("hotspot", "router")

    def test_multiple_probed_ssids(self, dc: DeviceClassifier):
        """Multiple probed SSIDs — best match wins."""
        r = dc.classify_wifi(probed_ssids=["HomeWiFi", "iPhone-Matt", "CoffeeShop"])
        assert r.device_type == "phone"


# ======================================================================
# Supplementary Appearance Database (217 codes)
# ======================================================================


class TestSupplementaryAppearance:
    """Test GAP appearance classification using supplementary ble_appearance_values.json."""

    def test_appearance_db_loaded(self, dc: DeviceClassifier):
        stats = dc.database_stats
        assert stats["appearance_codes"] >= 200, f"Expected 200+ appearance codes, got {stats['appearance_codes']}"

    def test_phone_appearance(self, dc: DeviceClassifier):
        """Phone appearance (0x0040 = 64) from primary or supplementary DB."""
        r = dc.classify_ble(appearance=0x0040)
        assert r.device_type == "phone"
        assert r.confidence >= 0.9

    def test_watch_appearance(self, dc: DeviceClassifier):
        r = dc.classify_ble(appearance=0x00C0)
        assert r.device_type == "watch"

    def test_computer_appearance(self, dc: DeviceClassifier):
        r = dc.classify_ble(appearance=0x0080)
        assert r.device_type == "computer"

    def test_keyboard_appearance(self, dc: DeviceClassifier):
        """Keyboard appearance (961 = 0x03C1)."""
        r = dc.classify_ble(appearance=961)
        assert r.device_type in ("keyboard", "hid", "computer", "peripheral")

    def test_mouse_appearance(self, dc: DeviceClassifier):
        """Mouse appearance (962 = 0x03C2)."""
        r = dc.classify_ble(appearance=962)
        assert r.device_type in ("mouse", "hid", "computer", "peripheral")

    def test_unknown_appearance_returns_none(self, dc: DeviceClassifier):
        """Appearance 0 (unknown) should not produce a signal."""
        r = dc.classify_ble(appearance=0)
        app_sigs = [s for s in r.signals if "appearance" in s.get("signal", "")]
        assert len(app_sigs) == 0

    def test_category_fallback(self, dc: DeviceClassifier):
        """Specific appearance falls back to category (upper byte)."""
        # 0x0041 is a specific phone sub-type; should fall back to 0x0040 category
        r = dc.classify_ble(appearance=0x0041)
        assert r.device_type == "phone"


# ======================================================================
# Supplementary Service UUID Database (77 services)
# ======================================================================


class TestSupplementaryServiceUUIDs:
    """Test service UUID classification using supplementary ble_service_uuids.json."""

    def test_service_uuid_db_loaded(self, dc: DeviceClassifier):
        stats = dc.database_stats
        assert stats["service_uuids"] >= 70, f"Expected 70+ service UUIDs, got {stats['service_uuids']}"

    def test_heart_rate_uuid(self, dc: DeviceClassifier):
        """Heart Rate Service (0x180D) should classify as fitness."""
        r = dc.classify_ble(service_uuids=["0x180D"])
        assert r.device_type == "fitness"

    def test_phone_alert_uuid(self, dc: DeviceClassifier):
        """Phone Alert Status (0x180E) should classify as phone."""
        r = dc.classify_ble(service_uuids=["0x180E"])
        assert r.device_type == "phone"

    def test_blood_pressure_uuid(self, dc: DeviceClassifier):
        """Blood Pressure (0x1810) should classify as medical/health."""
        r = dc.classify_ble(service_uuids=["0x1810"])
        assert r.device_type in ("medical", "health", "health_monitor")

    def test_multiple_service_uuids(self, dc: DeviceClassifier):
        """Multiple UUIDs — highest confidence wins."""
        r = dc.classify_ble(service_uuids=["0x1800", "0x180D"])
        # 0x1800 (Generic Access) has no device_hint; 0x180D (Heart Rate) does
        assert r.device_type == "fitness"

    def test_immediate_alert_tracker(self, dc: DeviceClassifier):
        """Immediate Alert (0x1802) should hint at tracker."""
        r = dc.classify_ble(service_uuids=["0x1802"])
        assert r.device_type in ("tracker", "tag", "beacon")


# ======================================================================
# Supplementary Company ID Database (654 companies)
# ======================================================================


class TestSupplementaryCompanyIDs:
    """Test company ID classification using supplementary ble_company_ids.json."""

    def test_company_id_db_loaded(self, dc: DeviceClassifier):
        stats = dc.database_stats
        assert stats["company_ids_standalone"] >= 600, f"Expected 600+ standalone company IDs, got {stats['company_ids_standalone']}"

    def test_primary_db_takes_priority(self, dc: DeviceClassifier):
        """Primary fingerprint DB entries should match before standalone DB."""
        r = dc.classify_ble(company_id=76)  # Apple
        assert "Apple" in r.manufacturer
        # Should come from primary DB
        cid_sigs = [s for s in r.signals if s["signal"] == "company_id"]
        assert len(cid_sigs) > 0


# ======================================================================
# DHCP Classification
# ======================================================================


class TestClassifyDHCP:
    """Test DHCP-based device classification."""

    def test_dhcp_vendor_patterns_loaded(self, dc: DeviceClassifier):
        stats = dc.database_stats
        assert stats["dhcp_vendor_patterns"] >= 20, f"Expected 20+ DHCP vendor patterns, got {stats['dhcp_vendor_patterns']}"

    def test_dhcp_hostname_patterns_loaded(self, dc: DeviceClassifier):
        stats = dc.database_stats
        assert stats["dhcp_hostname_patterns"] >= 20, f"Expected 20+ DHCP hostname patterns, got {stats['dhcp_hostname_patterns']}"

    def test_android_dhcp_vendor_class(self, dc: DeviceClassifier):
        r = dc.classify_dhcp(vendor_class="android-dhcp-14")
        assert r.device_type == "phone"
        assert r.os_hint == "Android"

    def test_xbox_dhcp_vendor_class(self, dc: DeviceClassifier):
        r = dc.classify_dhcp(vendor_class="MSFT XBOX")
        assert r.device_type == "gaming"

    def test_cisco_vendor_class(self, dc: DeviceClassifier):
        r = dc.classify_dhcp(vendor_class="Cisco Systems Inc.")
        assert r.device_type == "router"

    def test_hp_printer_vendor_class(self, dc: DeviceClassifier):
        r = dc.classify_dhcp(vendor_class="HP Ethernet Multi-Function")
        assert r.device_type == "printer"

    def test_roku_vendor_class(self, dc: DeviceClassifier):
        r = dc.classify_dhcp(vendor_class="Roku/8.0")
        assert r.device_type == "streaming_device"

    def test_iphone_dhcp_hostname(self, dc: DeviceClassifier):
        r = dc.classify_dhcp(hostname="iPhone-Matt")
        assert r.device_type == "phone"
        assert r.os_hint == "iOS"

    def test_ipad_dhcp_hostname(self, dc: DeviceClassifier):
        r = dc.classify_dhcp(hostname="iPad-Pro")
        assert r.device_type == "tablet"
        assert r.os_hint == "iPadOS"

    def test_macbook_dhcp_hostname(self, dc: DeviceClassifier):
        r = dc.classify_dhcp(hostname="MacBook-Pro")
        assert r.device_type == "laptop"

    def test_galaxy_dhcp_hostname(self, dc: DeviceClassifier):
        r = dc.classify_dhcp(hostname="Galaxy-S24")
        assert r.device_type == "phone"

    def test_desktop_hostname(self, dc: DeviceClassifier):
        r = dc.classify_dhcp(hostname="DESKTOP-ABC123")
        assert r.device_type == "desktop"
        assert r.os_hint == "Windows"

    def test_laptop_hostname(self, dc: DeviceClassifier):
        r = dc.classify_dhcp(hostname="LAPTOP-XYZ789")
        assert r.device_type == "laptop"

    def test_raspberrypi_hostname(self, dc: DeviceClassifier):
        r = dc.classify_dhcp(hostname="raspberrypi")
        assert r.device_type == "single_board_computer"

    def test_esp32_hostname(self, dc: DeviceClassifier):
        r = dc.classify_dhcp(hostname="esp32-sensor")
        assert r.device_type == "iot"

    def test_playstation_hostname(self, dc: DeviceClassifier):
        r = dc.classify_dhcp(hostname="PlayStation5")
        assert r.device_type == "gaming"

    def test_roomba_hostname(self, dc: DeviceClassifier):
        r = dc.classify_dhcp(hostname="Roomba-i7")
        assert r.device_type == "robot_vacuum"

    def test_chromecast_hostname(self, dc: DeviceClassifier):
        r = dc.classify_dhcp(hostname="Chromecast-Ultra")
        assert r.device_type == "streaming_device"

    def test_both_vendor_and_hostname(self, dc: DeviceClassifier):
        """When both vendor class and hostname match, best confidence wins."""
        r = dc.classify_dhcp(
            vendor_class="android-dhcp-14",
            hostname="Galaxy-S24",
        )
        assert r.device_type == "phone"
        assert len(r.signals) == 2

    def test_vendor_class_without_device_type(self, dc: DeviceClassifier):
        """MSFT vendor class has no device_type — should still capture OS."""
        r = dc.classify_dhcp(vendor_class="MSFT 5.0")
        assert r.os_hint in ("Windows", "Windows 2000/XP")

    def test_unknown_vendor_class(self, dc: DeviceClassifier):
        r = dc.classify_dhcp(vendor_class="totally-unknown-vendor-12345")
        assert r.device_type == "unknown"

    def test_unknown_hostname(self, dc: DeviceClassifier):
        r = dc.classify_dhcp(hostname="xyzzy-random-host")
        assert r.device_type == "unknown"

    def test_empty_dhcp(self, dc: DeviceClassifier):
        r = dc.classify_dhcp()
        assert r.device_type == "unknown"

    def test_os_hint_in_to_dict(self, dc: DeviceClassifier):
        r = dc.classify_dhcp(hostname="iPhone-Matt")
        d = r.to_dict()
        assert "os_hint" in d
        assert d["os_hint"] == "iOS"


# ======================================================================
# mDNS Classification
# ======================================================================


class TestClassifyMDNS:
    """Test mDNS/Bonjour service-based classification."""

    def test_mdns_db_loaded(self, dc: DeviceClassifier):
        stats = dc.database_stats
        assert stats["mdns_services"] >= 40, f"Expected 40+ mDNS services, got {stats['mdns_services']}"

    def test_googlecast(self, dc: DeviceClassifier):
        r = dc.classify_mdns(services=["_googlecast._tcp"])
        assert r.device_type == "streaming_device"
        assert r.manufacturer == "Google"

    def test_airplay(self, dc: DeviceClassifier):
        r = dc.classify_mdns(services=["_airplay._tcp"])
        assert r.device_type == "streaming_device"
        assert r.manufacturer == "Apple"

    def test_printer_ipp(self, dc: DeviceClassifier):
        r = dc.classify_mdns(services=["_ipp._tcp"])
        assert r.device_type == "printer"

    def test_printer_lpr(self, dc: DeviceClassifier):
        r = dc.classify_mdns(services=["_printer._tcp"])
        assert r.device_type == "printer"

    def test_homekit(self, dc: DeviceClassifier):
        r = dc.classify_mdns(services=["_hap._tcp"])
        assert r.device_type == "smart_home"
        assert r.manufacturer == "Apple"

    def test_spotify_connect(self, dc: DeviceClassifier):
        r = dc.classify_mdns(services=["_spotify-connect._tcp"])
        assert r.device_type == "speaker"

    def test_ssh_computer(self, dc: DeviceClassifier):
        r = dc.classify_mdns(services=["_ssh._tcp"])
        assert r.device_type == "computer"

    def test_amazon_devices(self, dc: DeviceClassifier):
        r = dc.classify_mdns(services=["_amzn-wplay._tcp"])
        assert r.device_type == "smart_speaker"
        assert r.manufacturer == "Amazon"

    def test_nvidia_gamestream(self, dc: DeviceClassifier):
        r = dc.classify_mdns(services=["_nvstream._tcp"])
        assert r.device_type == "gaming"
        assert r.manufacturer == "NVIDIA"

    def test_multiple_mdns_services(self, dc: DeviceClassifier):
        """Multiple services — pick best device_type match."""
        r = dc.classify_mdns(
            services=["_http._tcp", "_googlecast._tcp", "_spotify-connect._tcp"]
        )
        # _http has no device_hint; _googlecast and _spotify both have hints
        assert r.device_type in ("streaming_device", "speaker")

    def test_scanner_mdns(self, dc: DeviceClassifier):
        r = dc.classify_mdns(services=["_scanner._tcp"])
        assert r.device_type == "scanner"

    def test_time_machine(self, dc: DeviceClassifier):
        r = dc.classify_mdns(services=["_adisk._tcp"])
        assert r.device_type == "nas"

    def test_unknown_service(self, dc: DeviceClassifier):
        r = dc.classify_mdns(services=["_totally-unknown._tcp"])
        assert r.device_type == "unknown"

    def test_empty_services(self, dc: DeviceClassifier):
        r = dc.classify_mdns(services=[])
        assert r.device_type == "unknown"

    def test_none_services(self, dc: DeviceClassifier):
        r = dc.classify_mdns(services=None)
        assert r.device_type == "unknown"


# ======================================================================
# Multi-Protocol Classification (classify_multi)
# ======================================================================


class TestClassifyMulti:
    """Test combined multi-protocol classification."""

    def test_ble_only(self, dc: DeviceClassifier):
        r = dc.classify_multi(mac="AC:BC:32:AA:BB:CC", ble_name="iPhone 15")
        assert r.device_type == "phone"
        assert r.manufacturer == "Apple"

    def test_wifi_only(self, dc: DeviceClassifier):
        r = dc.classify_multi(ssid="DIRECT-HP-Printer")
        assert r.device_type == "printer"

    def test_dhcp_only(self, dc: DeviceClassifier):
        r = dc.classify_multi(dhcp_vendor_class="android-dhcp-14")
        assert r.device_type == "phone"
        assert r.os_hint == "Android"

    def test_mdns_only(self, dc: DeviceClassifier):
        r = dc.classify_multi(mdns_services=["_googlecast._tcp"])
        assert r.device_type == "streaming_device"

    def test_ble_plus_wifi(self, dc: DeviceClassifier):
        """BLE + WiFi combined — highest confidence wins."""
        r = dc.classify_multi(
            ble_name="iPhone 15",
            ssid="iPhone-Matt",
        )
        assert r.device_type == "phone"
        assert len(r.signals) >= 2

    def test_ble_plus_dhcp(self, dc: DeviceClassifier):
        """BLE name + DHCP hostname — both contribute signals."""
        r = dc.classify_multi(
            ble_name="Galaxy Buds Pro",
            dhcp_hostname="Galaxy-S24",
        )
        assert len(r.signals) >= 2

    def test_all_protocols(self, dc: DeviceClassifier):
        """All four protocols contributing signals."""
        r = dc.classify_multi(
            ble_name="iPhone 15",
            ssid="iPhone-Matt",
            dhcp_hostname="iPhone-Matt",
            mdns_services=["_airplay._tcp"],
        )
        assert r.device_type == "phone"
        assert len(r.signals) >= 3
        assert r.manufacturer == "Apple"

    def test_empty_multi(self, dc: DeviceClassifier):
        r = dc.classify_multi()
        assert r.device_type == "unknown"
        assert len(r.signals) == 0

    def test_conflicting_signals_highest_confidence_wins(self, dc: DeviceClassifier):
        """When BLE says phone and mDNS says printer, highest confidence wins."""
        r = dc.classify_multi(
            appearance=0x0040,  # phone, confidence 0.9
            mdns_services=["_ipp._tcp"],  # printer, confidence 0.7
        )
        assert r.device_type == "phone"

    def test_mac_randomized_propagates(self, dc: DeviceClassifier):
        """MAC randomization flag should propagate through classify_multi."""
        r = dc.classify_multi(mac="02:AA:BB:CC:DD:EE", ble_name="iPhone 15")
        assert r.mac_randomized is True

    def test_os_hint_propagates(self, dc: DeviceClassifier):
        """OS hint from DHCP should propagate through classify_multi."""
        r = dc.classify_multi(
            ble_name="iPhone 15",
            dhcp_hostname="iPhone-Matt",
        )
        assert r.os_hint == "iOS"


# ======================================================================
# Database Stats
# ======================================================================


class TestDatabaseStats:
    """Test the database_stats property."""

    def test_stats_has_all_keys(self, dc: DeviceClassifier):
        stats = dc.database_stats
        expected_keys = [
            "ble_fingerprints",
            "oui_prefixes",
            "ble_name_patterns",
            "wifi_ssid_patterns",
            "appearance_codes",
            "service_uuids",
            "company_ids_standalone",
            "company_ids_fingerprints",
            "dhcp_vendor_patterns",
            "dhcp_hostname_patterns",
            "mdns_services",
        ]
        for key in expected_keys:
            assert key in stats, f"Missing key: {key}"

    def test_stats_all_loaded(self, dc: DeviceClassifier):
        """All databases should have non-zero counts."""
        stats = dc.database_stats
        for key, count in stats.items():
            assert count > 0, f"Database {key} has 0 entries"

    def test_fingerprints_loaded(self, dc: DeviceClassifier):
        stats = dc.database_stats
        assert stats["ble_fingerprints"] == 1
        assert stats["company_ids_fingerprints"] >= 900


# ======================================================================
# Real-World BLE Advertisement Scenarios
# ======================================================================


class TestRealWorldBLEScenarios:
    """Test with realistic BLE advertisement data combinations."""

    def test_iphone_full_advertisement(self, dc: DeviceClassifier):
        """Full iPhone BLE advertisement: Apple OUI + name + appearance + Apple class."""
        r = dc.classify_ble(
            mac="AC:BC:32:11:22:33",
            name="iPhone 15 Pro",
            appearance=0x0040,
            company_id=76,
            apple_device_class="0x02",
        )
        assert r.device_type == "phone"
        assert r.manufacturer == "Apple"
        assert r.confidence >= 0.9
        assert len(r.signals) >= 3

    def test_galaxy_watch_advertisement(self, dc: DeviceClassifier):
        """Samsung Galaxy Watch: name + company ID."""
        r = dc.classify_ble(
            name="Galaxy Watch 6",
            company_id=117,
            appearance=0x00C0,
        )
        assert r.device_type == "watch"
        assert r.confidence >= 0.9

    def test_airpods_advertisement(self, dc: DeviceClassifier):
        """AirPods: name + Apple company + Apple device class."""
        r = dc.classify_ble(
            name="AirPods Pro",
            company_id=76,
            apple_device_class="0x02",
        )
        assert r.device_type in ("earbuds", "phone")
        assert r.confidence >= 0.9

    def test_fitbit_advertisement(self, dc: DeviceClassifier):
        """Fitbit: name + company ID + heart rate service UUID."""
        r = dc.classify_ble(
            name="Fitbit Charge 5",
            company_id=449,
            service_uuids=["0x180D"],
        )
        assert r.device_type == "fitness"

    def test_tile_tracker_advertisement(self, dc: DeviceClassifier):
        """Tile tracker: company ID + name."""
        r = dc.classify_ble(
            name="Tile Pro",
            company_id=450,
        )
        assert r.device_type == "tag"

    def test_esp32_iot_sensor(self, dc: DeviceClassifier):
        """ESP32 IoT sensor: Espressif OUI + name."""
        r = dc.classify_ble(
            mac="24:0A:C4:12:34:56",
            name="ESP32-Temperature",
        )
        assert r.device_type == "microcontroller"
        assert r.manufacturer == "Espressif"

    def test_randomized_mac_phone(self, dc: DeviceClassifier):
        """Modern phone with randomized MAC — only name and appearance help."""
        r = dc.classify_ble(
            mac="4A:BB:CC:DD:EE:FF",
            name="Pixel 8",
            appearance=0x0040,
        )
        assert r.mac_randomized is True
        assert r.device_type == "phone"
        assert r.confidence >= 0.85
        # No OUI signal should be present
        oui_sigs = [s for s in r.signals if s["signal"] == "oui"]
        assert len(oui_sigs) == 0

    def test_anonymous_ble_beacon(self, dc: DeviceClassifier):
        """Anonymous BLE beacon: randomized MAC, no name, just advertising."""
        r = dc.classify_ble(mac="02:11:22:33:44:55")
        assert r.mac_randomized is True
        assert r.device_type == "unknown"

    def test_sony_headphones_fast_pair(self, dc: DeviceClassifier):
        """Sony WH-1000XM4: Fast Pair model ID + name."""
        r = dc.classify_ble(
            name="Sony WH-1000XM4",
            fast_pair_model_id="0x01EEB4",
        )
        assert r.device_type == "headphones"
        assert r.confidence >= 0.85

    def test_tesla_key_fob(self, dc: DeviceClassifier):
        """Tesla BLE key: company ID + name."""
        r = dc.classify_ble(
            name="Tesla Model 3",
            company_id=555,
        )
        assert r.device_type == "vehicle"

    def test_medical_device_dexcom(self, dc: DeviceClassifier):
        """Dexcom CGM: company ID."""
        r = dc.classify_ble(company_id=1202)
        assert r.device_type == "medical"
        assert "Dexcom" in r.manufacturer

    def test_smart_home_govee(self, dc: DeviceClassifier):
        """Govee smart light: name pattern."""
        r = dc.classify_ble(name="Govee_H6071")
        assert r.device_type == "smart_home"

    def test_gamepad_dualsense(self, dc: DeviceClassifier):
        """DualSense controller: name pattern."""
        r = dc.classify_ble(name="DualSense Wireless Controller")
        assert r.device_type == "gamepad"

    def test_vr_headset_meta_quest(self, dc: DeviceClassifier):
        """Meta Quest VR headset: name + company ID."""
        r = dc.classify_ble(
            name="Meta Quest 3",
            company_id=427,
        )
        assert r.device_type == "vr_headset"


# ======================================================================
# Real-World WiFi Scenarios
# ======================================================================


class TestRealWorldWiFiScenarios:
    """Test with realistic WiFi probe and association data."""

    def test_android_probe_request(self, dc: DeviceClassifier):
        """Android phone probing for known networks."""
        r = dc.classify_wifi(
            probed_ssids=["Android-ABCD", "HomeNetwork", "Office-5G"],
        )
        assert r.device_type == "phone"

    def test_laptop_ssid_with_bssid(self, dc: DeviceClassifier):
        """Windows laptop visible as SSID with known BSSID OUI."""
        r = dc.classify_wifi(ssid="LAPTOP-MATT123")
        assert r.device_type == "computer"

    def test_desktop_ssid(self, dc: DeviceClassifier):
        r = dc.classify_wifi(ssid="DESKTOP-WORK42")
        assert r.device_type == "computer"

    def test_fire_tv_ssid(self, dc: DeviceClassifier):
        r = dc.classify_wifi(ssid="FireTV-Stick-4K")
        assert r.device_type in ("media_player", "streaming_device")


# ======================================================================
# Edge Cases and Error Handling
# ======================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_string_inputs(self, dc: DeviceClassifier):
        r = dc.classify_ble(mac="", name="", service_uuids=[])
        assert r.device_type == "unknown"

    def test_none_service_uuids(self, dc: DeviceClassifier):
        r = dc.classify_ble(service_uuids=None)
        assert r.device_type == "unknown"

    def test_malformed_mac_short(self, dc: DeviceClassifier):
        r = dc.classify_ble(mac="AA:BB")
        assert r.device_type == "unknown"

    def test_mac_with_dashes(self, dc: DeviceClassifier):
        """MAC with dash separators should be normalized."""
        r = dc.classify_ble(mac="24-0A-C4-12-34-56")
        assert r.manufacturer == "Espressif"

    def test_mac_lowercase(self, dc: DeviceClassifier):
        """Lowercase MAC should be normalized."""
        r = dc.classify_ble(mac="24:0a:c4:12:34:56")
        assert r.manufacturer == "Espressif"

    def test_service_uuid_normalization(self, dc: DeviceClassifier):
        """UUIDs with various formats should be normalized."""
        r1 = dc.classify_ble(service_uuids=["0x180D"])
        r2 = dc.classify_ble(service_uuids=["180D"])
        r3 = dc.classify_ble(service_uuids=["0X180D"])
        assert r1.device_type == r2.device_type == r3.device_type

    def test_company_id_zero(self, dc: DeviceClassifier):
        """Company ID 0 — Ericsson Technology Licensing."""
        r = dc.classify_ble(company_id=0)
        # Should not crash; may or may not classify
        assert isinstance(r.device_type, str)

    def test_very_large_company_id(self, dc: DeviceClassifier):
        """Very large company ID that doesn't exist."""
        r = dc.classify_ble(company_id=99999)
        assert r.device_type == "unknown"

    def test_appearance_max_value(self, dc: DeviceClassifier):
        """Max 16-bit appearance value."""
        r = dc.classify_ble(appearance=0xFFFF)
        assert isinstance(r.device_type, str)

    def test_classification_to_dict_roundtrip(self, dc: DeviceClassifier):
        """to_dict should produce a JSON-serializable dictionary."""
        import json as json_mod
        r = dc.classify_ble(
            mac="AC:BC:32:AA:BB:CC",
            name="iPhone 15",
            appearance=0x0040,
        )
        d = r.to_dict()
        serialized = json_mod.dumps(d)
        deserialized = json_mod.loads(serialized)
        assert deserialized["device_type"] == r.device_type
        assert deserialized["confidence"] == r.confidence
        assert deserialized["mac_randomized"] == r.mac_randomized

    def test_bad_data_dir_graceful_degradation(self):
        """Bad data directory should degrade gracefully — hardcoded patterns still work."""
        dc = DeviceClassifier(
            fingerprints_path="/nonexistent/fp.json",
            data_dir="/nonexistent/data/",
        )
        assert dc.loaded is False
        # Hardcoded patterns still work
        r = dc.classify_ble(name="iPhone 15")
        assert r.device_type == "phone"
        # Hardcoded OUI still works
        r2 = dc.classify_ble(mac="24:0A:C4:12:34:56")
        assert r2.manufacturer == "Espressif"

    def test_dhcp_empty_patterns_no_crash(self):
        """DHCP classification with no patterns loaded should not crash."""
        dc = DeviceClassifier(
            fingerprints_path="/nonexistent/fp.json",
            data_dir="/nonexistent/data/",
        )
        r = dc.classify_dhcp(vendor_class="unknown", hostname="unknown")
        assert r.device_type == "unknown"

    def test_mdns_empty_db_no_crash(self):
        """mDNS classification with no DB loaded should not crash."""
        dc = DeviceClassifier(
            fingerprints_path="/nonexistent/fp.json",
            data_dir="/nonexistent/data/",
        )
        r = dc.classify_mdns(services=["_airplay._tcp"])
        assert r.device_type == "unknown"

    def test_classify_multi_empty_no_crash(self):
        """classify_multi with no data should not crash."""
        dc = DeviceClassifier(
            fingerprints_path="/nonexistent/fp.json",
            data_dir="/nonexistent/data/",
        )
        r = dc.classify_multi(
            ble_name="iPhone",
            ssid="iPhone-Matt",
            dhcp_hostname="iPhone-Matt",
            mdns_services=["_airplay._tcp"],
        )
        assert r.device_type == "phone"  # Hardcoded name pattern still works
