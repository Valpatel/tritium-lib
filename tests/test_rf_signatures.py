# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for RF signature simulation module."""

import random
import re
import time

import pytest

from tritium_lib.sim_engine.ai.rf_signatures import (
    APPLE_COMPANY_ID,
    GOOGLE_COMPANY_ID,
    MAC_ROTATION_INTERVAL_S,
    SAMSUNG_COMPANY_ID,
    BuildingRFProfile,
    PersonRFProfile,
    RFSignatureGenerator,
    VehicleRFProfile,
    _random_mac,
    _random_plate,
    _random_tpms_id,
)


# ---------------------------------------------------------------------------
# MAC address helpers
# ---------------------------------------------------------------------------

_MAC_RE = re.compile(r"^[0-9A-F]{2}(:[0-9A-F]{2}){5}$")


def _is_valid_mac(mac: str) -> bool:
    return bool(_MAC_RE.match(mac))


def _is_locally_administered(mac: str) -> bool:
    """Check that bit 1 of first octet is set (locally administered)."""
    first = int(mac.split(":")[0], 16)
    return bool(first & 0x02)


def _is_unicast(mac: str) -> bool:
    """Check that bit 0 of first octet is clear (unicast)."""
    first = int(mac.split(":")[0], 16)
    return not (first & 0x01)


class TestRandomMac:
    def test_format(self):
        mac = _random_mac()
        assert _is_valid_mac(mac), f"Invalid MAC format: {mac}"

    def test_locally_administered(self):
        for _ in range(50):
            mac = _random_mac()
            assert _is_locally_administered(mac)

    def test_unicast(self):
        for _ in range(50):
            mac = _random_mac()
            assert _is_unicast(mac)

    def test_uniqueness(self):
        macs = {_random_mac() for _ in range(100)}
        assert len(macs) == 100, "Generated duplicate MACs"

    def test_seeded_reproducibility(self):
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        assert _random_mac(rng1) == _random_mac(rng2)


class TestRandomTpmsId:
    def test_format(self):
        tid = _random_tpms_id()
        assert re.match(r"^[0-9A-F]{8}$", tid), f"Invalid TPMS ID: {tid}"

    def test_uniqueness(self):
        ids = {_random_tpms_id() for _ in range(100)}
        assert len(ids) == 100


class TestRandomPlate:
    def test_ca_format(self):
        plate = _random_plate("CA")
        # CA format: digit + 3 letters + 3 digits = 7 chars
        assert len(plate) == 7
        assert plate[0].isdigit()
        assert plate[1:4].isalpha()
        assert plate[4:7].isdigit()

    def test_tx_format(self):
        plate = _random_plate("TX")
        # TX format: LLL-DDDD = 8 chars
        assert len(plate) == 8
        assert plate[3] == "-"

    def test_default_state(self):
        plate = _random_plate()
        assert len(plate) == 7  # CA default


# ---------------------------------------------------------------------------
# PersonRFProfile
# ---------------------------------------------------------------------------

class TestPersonRFProfile:
    def test_emit_ble_phone_only(self):
        person = PersonRFProfile(
            has_phone=True,
            phone_mac="AA:BB:CC:DD:EE:FF",
            phone_model="iPhone 15",
            phone_ble_company_id=APPLE_COMPANY_ID,
            phone_ecosystem="apple",
        )
        ads = person.emit_ble_advertisements((100.0, 200.0))
        assert len(ads) == 1
        ad = ads[0]
        assert ad["mac"] == "AA:BB:CC:DD:EE:FF"
        assert ad["source"] == "ble"
        assert ad["device_type"] == "phone"
        assert ad["company_id"] == APPLE_COMPANY_ID
        assert ad["position_x"] == 100.0
        assert ad["position_y"] == 200.0
        assert ad["simulated"] is True
        assert "rssi" in ad
        assert isinstance(ad["timestamp"], float)

    def test_emit_ble_full_loadout(self):
        person = PersonRFProfile(
            has_phone=True,
            phone_mac="AA:BB:CC:DD:EE:01",
            phone_model="Galaxy S24",
            phone_ble_company_id=SAMSUNG_COMPANY_ID,
            has_smartwatch=True,
            watch_mac="AA:BB:CC:DD:EE:02",
            watch_model="Galaxy Watch6",
            watch_ble_company_id=SAMSUNG_COMPANY_ID,
            has_earbuds=True,
            earbuds_mac="AA:BB:CC:DD:EE:03",
            earbuds_model="Galaxy Buds2 Pro",
            earbuds_ble_company_id=SAMSUNG_COMPANY_ID,
        )
        ads = person.emit_ble_advertisements((50.0, 75.0))
        assert len(ads) == 3
        types = {a["device_type"] for a in ads}
        assert types == {"phone", "smartwatch", "earbuds"}

    def test_emit_ble_no_devices(self):
        person = PersonRFProfile(has_phone=False, has_smartwatch=False, has_earbuds=False)
        ads = person.emit_ble_advertisements((0.0, 0.0))
        assert len(ads) == 0

    def test_emit_wifi_probes(self):
        person = PersonRFProfile(
            has_phone=True,
            phone_mac="AA:BB:CC:DD:EE:FF",
            phone_model="Pixel 8",
            phone_wifi_probes=["xfinitywifi", "HOME-1234", "MyNetwork"],
        )
        probes = person.emit_wifi_probes((10.0, 20.0))
        assert 1 <= len(probes) <= 3
        for p in probes:
            assert p["mac"] == "AA:BB:CC:DD:EE:FF"
            assert p["source"] == "wifi_probe"
            assert p["ssid"] in ["xfinitywifi", "HOME-1234", "MyNetwork"]
            assert p["simulated"] is True
            assert -75 <= p["rssi"] <= -45

    def test_emit_wifi_probes_no_phone(self):
        person = PersonRFProfile(has_phone=False)
        assert person.emit_wifi_probes((0.0, 0.0)) == []

    def test_rssi_varies(self):
        person = PersonRFProfile(
            has_phone=True,
            phone_mac="AA:BB:CC:DD:EE:FF",
            phone_model="iPhone 15",
            phone_ble_company_id=APPLE_COMPANY_ID,
        )
        rssi_values = set()
        for _ in range(50):
            ads = person.emit_ble_advertisements((0.0, 0.0), rssi_at_1m=-59)
            rssi_values.add(ads[0]["rssi"])
        # Should have multiple distinct RSSI values (random jitter)
        assert len(rssi_values) > 1

    def test_mac_rotation(self):
        person = PersonRFProfile(
            has_phone=True,
            phone_mac="AA:BB:CC:DD:EE:FF",
            phone_model="iPhone 15",
            phone_ble_company_id=APPLE_COMPANY_ID,
            has_smartwatch=True,
            watch_mac="11:22:33:44:55:66",
            watch_model="Apple Watch Series 9",
            watch_ble_company_id=APPLE_COMPANY_ID,
        )
        old_phone_mac = person.phone_mac
        old_watch_mac = person.watch_mac
        old_company_id = person.phone_ble_company_id

        person.rotate_mac()

        # MACs changed
        assert person.phone_mac != old_phone_mac
        assert person.watch_mac != old_watch_mac
        # Company ID preserved
        assert person.phone_ble_company_id == old_company_id
        # New MACs are valid
        assert _is_valid_mac(person.phone_mac)
        assert _is_valid_mac(person.watch_mac)

    def test_should_rotate_mac_timing(self):
        person = PersonRFProfile(has_phone=True)
        person._mac_last_rotated = time.time()
        assert not person.should_rotate_mac()

        person._mac_last_rotated = time.time() - MAC_ROTATION_INTERVAL_S - 1
        assert person.should_rotate_mac()


# ---------------------------------------------------------------------------
# VehicleRFProfile
# ---------------------------------------------------------------------------

class TestVehicleRFProfile:
    def test_emit_tpms(self):
        vehicle = VehicleRFProfile(
            tpms_ids=["AABB0001", "AABB0002", "AABB0003", "AABB0004"],
            tpms_frequency=315.0,
            license_plate="7ABC123",
            make_model="2020 Toyota Camry",
        )
        readings = vehicle.emit_tpms((300.0, 400.0))
        assert len(readings) == 4

        tire_positions = set()
        for r in readings:
            assert r["source"] == "sdr_ism"
            assert r["classification"] == "ism_device"
            assert r["position_x"] == 300.0
            assert r["position_y"] == 400.0
            assert r["simulated"] is True
            meta = r["metadata"]
            assert meta["device_type"] == "TPMS"
            assert meta["frequency_mhz"] == 315.0
            assert 25.0 <= meta["pressure_psi"] <= 40.0
            assert 15.0 <= meta["temperature_c"] <= 50.0
            assert meta["vehicle_plate"] == "7ABC123"
            tire_positions.add(meta["tire_position"])

        assert tire_positions == {"FL", "FR", "RL", "RR"}

    def test_tpms_target_id_format(self):
        vehicle = VehicleRFProfile(
            tpms_ids=["DEADBEEF", "CAFEBABE", "12345678", "ABCD0000"],
        )
        readings = vehicle.emit_tpms((0.0, 0.0))
        for r in readings:
            assert r["target_id"].startswith("ism_tpms_")

    def test_emit_keyfob_ble(self):
        vehicle = VehicleRFProfile(
            has_keyfob=True,
            keyfob_mac="FF:EE:DD:CC:BB:AA",
            make_model="2023 BMW 3 Series",
        )
        ads = vehicle.emit_keyfob_ble((10.0, 20.0))
        assert len(ads) == 1
        assert ads[0]["mac"] == "FF:EE:DD:CC:BB:AA"
        assert ads[0]["source"] == "ble"
        assert ads[0]["device_type"] == "keyfob"

    def test_emit_keyfob_disabled(self):
        vehicle = VehicleRFProfile(has_keyfob=False)
        assert vehicle.emit_keyfob_ble((0.0, 0.0)) == []

    def test_emit_dashcam_wifi(self):
        vehicle = VehicleRFProfile(
            has_dashcam_wifi=True,
            dashcam_ssid="VIOFO-A229-1234",
            dashcam_mac="AA:BB:CC:DD:EE:FF",
        )
        beacons = vehicle.emit_dashcam_wifi((50.0, 60.0))
        assert len(beacons) == 1
        assert beacons[0]["ssid"] == "VIOFO-A229-1234"
        assert beacons[0]["source"] == "wifi_beacon"

    def test_emit_dashcam_disabled(self):
        vehicle = VehicleRFProfile(has_dashcam_wifi=False)
        assert vehicle.emit_dashcam_wifi((0.0, 0.0)) == []


# ---------------------------------------------------------------------------
# BuildingRFProfile
# ---------------------------------------------------------------------------

class TestBuildingRFProfile:
    def test_emit_wifi_beacons(self):
        building = BuildingRFProfile(
            building_type="residential",
            wifi_aps=[
                {"ssid": "HOME-ABCD", "bssid": "AA:BB:CC:00:11:22", "channel": 6, "signal_strength": -45},
                {"ssid": "HOME-ABCD_5G", "bssid": "AA:BB:CC:00:11:23", "channel": 36, "signal_strength": -50},
            ],
        )
        beacons = building.emit_wifi_beacons((25.0, 30.0))
        assert len(beacons) == 2
        for b in beacons:
            assert b["source"] == "wifi_beacon"
            assert b["device_type"] == "access_point"
            assert b["simulated"] is True
            assert b["position_x"] == 25.0

    def test_emit_iot_signals(self):
        building = BuildingRFProfile(
            iot_devices=[
                {"type": "doorbell", "mac": "AA:BB:CC:DD:EE:01", "protocol": "ble", "name": "Doorbell"},
                {"type": "thermostat", "mac": "AA:BB:CC:DD:EE:02", "protocol": "wifi", "name": "Thermostat"},
            ],
        )
        signals = building.emit_iot_signals((50.0, 50.0))
        assert len(signals) == 2
        types = {s["device_type"] for s in signals}
        assert types == {"doorbell", "thermostat"}
        for s in signals:
            assert s["simulated"] is True
            assert "mac" in s


# ---------------------------------------------------------------------------
# RFSignatureGenerator — random profile factories
# ---------------------------------------------------------------------------

class TestRFSignatureGenerator:
    def test_random_person_has_valid_mac(self):
        rng = random.Random(42)
        person = RFSignatureGenerator.random_person(rng=rng)
        if person.has_phone:
            assert _is_valid_mac(person.phone_mac)
        if person.has_smartwatch:
            assert _is_valid_mac(person.watch_mac)
        if person.has_earbuds:
            assert _is_valid_mac(person.earbuds_mac)

    def test_random_person_distribution(self):
        """Test that device ownership follows expected distributions."""
        rng = random.Random(123)
        n = 500
        phone_count = 0
        watch_count = 0
        earbuds_count = 0
        apple_count = 0
        android_count = 0

        for _ in range(n):
            p = RFSignatureGenerator.random_person(rng=rng)
            if p.has_phone:
                phone_count += 1
            if p.has_smartwatch:
                watch_count += 1
            if p.has_earbuds:
                earbuds_count += 1
            if p.phone_ecosystem == "apple":
                apple_count += 1
            elif p.phone_ecosystem == "android":
                android_count += 1

        # Allow wide margins for statistical tests
        assert phone_count > n * 0.70, f"Phone rate too low: {phone_count}/{n}"
        assert watch_count > n * 0.15, f"Watch rate too low: {watch_count}/{n}"
        assert earbuds_count > n * 0.20, f"Earbuds rate too low: {earbuds_count}/{n}"
        assert apple_count > n * 0.30, f"Apple rate too low: {apple_count}/{n}"
        assert android_count > n * 0.30, f"Android rate too low: {android_count}/{n}"

    def test_random_person_has_wifi_probes(self):
        rng = random.Random(99)
        person = RFSignatureGenerator.random_person(rng=rng)
        if person.has_phone:
            assert len(person.phone_wifi_probes) >= 2

    def test_random_person_company_id_matches_ecosystem(self):
        rng = random.Random(77)
        for _ in range(100):
            p = RFSignatureGenerator.random_person(rng=rng)
            if p.phone_ecosystem == "apple":
                assert p.phone_ble_company_id == APPLE_COMPANY_ID

    def test_random_vehicle_has_4_tpms(self):
        rng = random.Random(42)
        v = RFSignatureGenerator.random_vehicle(rng=rng)
        assert len(v.tpms_ids) == 4
        for tid in v.tpms_ids:
            assert re.match(r"^[0-9A-F]{8}$", tid)

    def test_random_vehicle_has_plate(self):
        rng = random.Random(42)
        v = RFSignatureGenerator.random_vehicle(rng=rng)
        assert len(v.license_plate) > 0
        assert len(v.make_model) > 0
        assert len(v.color) > 0

    def test_random_vehicle_tpms_frequency(self):
        rng = random.Random(42)
        freqs = set()
        for _ in range(100):
            v = RFSignatureGenerator.random_vehicle(rng=rng)
            freqs.add(v.tpms_frequency)
        assert 315.0 in freqs
        # 433.92 may or may not appear depending on seed, but 315 always does

    def test_random_building_residential(self):
        rng = random.Random(42)
        b = RFSignatureGenerator.random_building("residential", rng=rng)
        assert b.building_type == "residential"
        assert 1 <= len(b.wifi_aps) <= 2
        assert 1 <= len(b.iot_devices) <= 5
        for ap in b.wifi_aps:
            assert "ssid" in ap
            assert "bssid" in ap
            assert _is_valid_mac(ap["bssid"])

    def test_random_building_commercial(self):
        rng = random.Random(42)
        b = RFSignatureGenerator.random_building("commercial", rng=rng)
        assert b.building_type == "commercial"
        assert 2 <= len(b.wifi_aps) <= 4
        assert 3 <= len(b.iot_devices) <= 8

    def test_random_building_iot_devices_have_macs(self):
        rng = random.Random(42)
        b = RFSignatureGenerator.random_building("residential", rng=rng)
        for dev in b.iot_devices:
            assert _is_valid_mac(dev["mac"])
            assert dev["protocol"] in ("ble", "wifi")

    def test_varied_profiles(self):
        """Ensure random profiles are actually varied."""
        rng = random.Random(42)
        phone_models = set()
        ecosystems = set()
        for _ in range(50):
            p = RFSignatureGenerator.random_person(rng=rng)
            if p.has_phone:
                phone_models.add(p.phone_model)
            ecosystems.add(p.phone_ecosystem)
        assert len(phone_models) > 5, "Not enough model variety"
        assert len(ecosystems) >= 2, "Not enough ecosystem variety"

    def test_emitted_data_matches_sensor_format(self):
        """Verify emitted data has the fields real sensors produce."""
        person = RFSignatureGenerator.random_person(rng=random.Random(42))
        if person.has_phone:
            ads = person.emit_ble_advertisements((100.0, 200.0))
            ad = ads[0]
            # These fields are what edge_tracker expects
            required_ble_fields = {"mac", "rssi", "name", "source", "timestamp"}
            assert required_ble_fields.issubset(ad.keys())

            probes = person.emit_wifi_probes((100.0, 200.0))
            if probes:
                probe = probes[0]
                required_wifi_fields = {"mac", "ssid", "rssi", "source", "timestamp"}
                assert required_wifi_fields.issubset(probe.keys())

    def test_vehicle_tpms_matches_ism_format(self):
        """Verify TPMS output matches ISMDevice.to_target_dict() format."""
        vehicle = RFSignatureGenerator.random_vehicle(rng=random.Random(42))
        readings = vehicle.emit_tpms((0.0, 0.0))
        for r in readings:
            # ISMDevice.to_target_dict() produces these exact keys
            assert "target_id" in r
            assert "source" in r
            assert r["source"] == "sdr_ism"
            assert "classification" in r
            assert r["classification"] == "ism_device"
            assert "metadata" in r
            meta = r["metadata"]
            assert "device_type" in meta
            assert "frequency_mhz" in meta

    def test_seeded_reproducibility(self):
        """Same seed produces identical profiles."""
        p1 = RFSignatureGenerator.random_person(rng=random.Random(999))
        p2 = RFSignatureGenerator.random_person(rng=random.Random(999))
        assert p1.phone_mac == p2.phone_mac
        assert p1.phone_model == p2.phone_model
        assert p1.phone_ecosystem == p2.phone_ecosystem
        assert p1.has_smartwatch == p2.has_smartwatch

    def test_static_helpers(self):
        assert _is_valid_mac(RFSignatureGenerator.random_mac())
        assert re.match(r"^[0-9A-F]{8}$", RFSignatureGenerator.random_tpms_id())
        assert len(RFSignatureGenerator.random_plate()) > 0
