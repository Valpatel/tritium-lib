# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for WiFi probe request and fingerprinting models."""

from datetime import datetime, timezone

import pytest

from tritium_lib.models.wifi import (
    WiFiFingerprint,
    WiFiNetwork,
    WiFiNetworkType,
    WiFiProbeRequest,
)
from tritium_lib.mqtt.topics import TritiumTopics


# ── WiFiNetworkType ────────────────────────────────────────────────

class TestWiFiNetworkType:
    def test_all_values(self):
        expected = {
            "corporate", "home", "hotspot", "iot",
            "mesh", "guest", "public", "unknown",
        }
        assert {t.value for t in WiFiNetworkType} == expected

    def test_str_enum(self):
        assert WiFiNetworkType.CORPORATE == "corporate"
        assert isinstance(WiFiNetworkType.HOME, str)


# ── WiFiProbeRequest ───────────────────────────────────────────────

class TestWiFiProbeRequest:
    def test_minimal(self):
        probe = WiFiProbeRequest(mac="aa:bb:cc:dd:ee:ff")
        assert probe.mac == "AA:BB:CC:DD:EE:FF"  # normalized to uppercase
        assert probe.ssid_probed == ""
        assert probe.rssi == -100
        assert probe.channel == 0
        assert probe.observer_id == ""
        assert isinstance(probe.timestamp, datetime)

    def test_full(self):
        ts = datetime(2026, 3, 13, 12, 0, 0, tzinfo=timezone.utc)
        probe = WiFiProbeRequest(
            mac="aa:bb:cc:dd:ee:ff",
            ssid_probed="CORP-5G",
            rssi=-45,
            timestamp=ts,
            channel=6,
            observer_id="edge-001",
        )
        assert probe.ssid_probed == "CORP-5G"
        assert probe.rssi == -45
        assert probe.channel == 6
        assert probe.observer_id == "edge-001"
        assert probe.timestamp == ts

    def test_serialization_roundtrip(self):
        probe = WiFiProbeRequest(
            mac="11:22:33:44:55:66",
            ssid_probed="HomeNet",
            rssi=-60,
            channel=11,
            observer_id="edge-002",
        )
        data = probe.model_dump()
        restored = WiFiProbeRequest(**data)
        assert restored.mac == probe.mac
        assert restored.ssid_probed == probe.ssid_probed
        assert restored.rssi == probe.rssi

    def test_json_roundtrip(self):
        probe = WiFiProbeRequest(mac="de:ad:be:ef:00:01", ssid_probed="IoT_Net")
        json_str = probe.model_dump_json()
        restored = WiFiProbeRequest.model_validate_json(json_str)
        assert restored.mac == probe.mac


# ── WiFiNetwork ────────────────────────────────────────────────────

class TestWiFiNetwork:
    def test_minimal(self):
        net = WiFiNetwork(bssid="00:11:22:33:44:55")
        assert net.bssid == "00:11:22:33:44:55"  # already uppercase hex digits
        assert net.ssid == ""
        assert net.auth_type == "open"
        assert net.network_type == WiFiNetworkType.UNKNOWN

    def test_full(self):
        net = WiFiNetwork(
            bssid="00:11:22:33:44:55",
            ssid="CorpNet-5G",
            rssi=-30,
            channel=36,
            auth_type="wpa2-enterprise",
            network_type=WiFiNetworkType.CORPORATE,
            observer_id="edge-003",
        )
        assert net.ssid == "CorpNet-5G"
        assert net.channel == 36
        assert net.auth_type == "wpa2-enterprise"
        assert net.network_type == WiFiNetworkType.CORPORATE

    def test_network_type_from_string(self):
        net = WiFiNetwork(
            bssid="aa:bb:cc:dd:ee:ff",
            network_type="hotspot",
        )
        assert net.bssid == "AA:BB:CC:DD:EE:FF"  # normalized
        assert net.network_type == WiFiNetworkType.HOTSPOT

    def test_serialization_roundtrip(self):
        net = WiFiNetwork(
            bssid="ff:ee:dd:cc:bb:aa",
            ssid="GuestWiFi",
            network_type=WiFiNetworkType.GUEST,
        )
        data = net.model_dump()
        restored = WiFiNetwork(**data)
        assert restored.ssid == net.ssid
        assert restored.network_type == WiFiNetworkType.GUEST


# ── WiFiFingerprint ────────────────────────────────────────────────

class TestWiFiFingerprint:
    def test_minimal(self):
        fp = WiFiFingerprint(mac="aa:bb:cc:dd:ee:ff")
        assert fp.mac == "AA:BB:CC:DD:EE:FF"  # normalized to uppercase
        assert fp.probed_ssids == []
        assert fp.network_associations == []
        assert fp.device_type_hint == "unknown"
        assert fp.probe_count == 0

    def test_full(self):
        fp = WiFiFingerprint(
            mac="aa:bb:cc:dd:ee:ff",
            probed_ssids=["CORP-5G", "eduroam", "iPhone-Hotspot"],
            network_associations=["CORP-5G"],
            device_type_hint="laptop",
            observer_id="edge-001",
            probe_count=42,
        )
        assert len(fp.probed_ssids) == 3
        assert "eduroam" in fp.probed_ssids
        assert fp.device_type_hint == "laptop"
        assert fp.probe_count == 42

    def test_timestamps(self):
        ts1 = datetime(2026, 3, 1, tzinfo=timezone.utc)
        ts2 = datetime(2026, 3, 13, tzinfo=timezone.utc)
        fp = WiFiFingerprint(
            mac="11:22:33:44:55:66",
            first_seen=ts1,
            last_seen=ts2,
        )
        assert fp.first_seen == ts1
        assert fp.last_seen == ts2
        assert fp.last_seen > fp.first_seen

    def test_json_roundtrip(self):
        fp = WiFiFingerprint(
            mac="de:ad:be:ef:00:01",
            probed_ssids=["SmartHome_2G", "Nest_Setup"],
            device_type_hint="iot",
            probe_count=7,
        )
        json_str = fp.model_dump_json()
        restored = WiFiFingerprint.model_validate_json(json_str)
        assert restored.probed_ssids == fp.probed_ssids
        assert restored.device_type_hint == "iot"
        assert restored.probe_count == 7


# ── MQTT Topics ────────────────────────────────────────────────────

class TestWiFiMqttTopics:
    def setup_method(self):
        self.topics = TritiumTopics(site_id="lab")

    def test_wifi_probe_topic(self):
        topic = self.topics.wifi_probe("edge-001")
        assert topic == "tritium/lab/edge/edge-001/wifi_probe"

    def test_wifi_scan_topic(self):
        topic = self.topics.wifi_scan("edge-002")
        assert topic == "tritium/lab/edge/edge-002/wifi_scan"

    def test_default_site(self):
        topics = TritiumTopics()
        assert topics.wifi_probe("n1") == "tritium/home/edge/n1/wifi_probe"
        assert topics.wifi_scan("n1") == "tritium/home/edge/n1/wifi_scan"


# ── Import from models package ─────────────────────────────────────

class TestWiFiExports:
    def test_importable_from_models(self):
        from tritium_lib.models import (
            WiFiFingerprint,
            WiFiNetwork,
            WiFiNetworkType,
            WiFiProbeRequest,
        )
        assert WiFiProbeRequest is not None
        assert WiFiNetwork is not None
        assert WiFiFingerprint is not None
        assert WiFiNetworkType is not None
