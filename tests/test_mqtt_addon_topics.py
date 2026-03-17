# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for SDR, Meshtastic extended, and generic addon device MQTT topics."""

import pytest

from tritium_lib.mqtt.topics import TritiumTopics, parse_site_topic


@pytest.fixture
def topics() -> TritiumTopics:
    return TritiumTopics(site_id="home")


# --- SDR topics ---

class TestSDRTopics:
    def test_sdr_spectrum(self, topics: TritiumTopics) -> None:
        assert topics.sdr_spectrum("hackrf-01") == "tritium/home/sdr/hackrf-01/spectrum"

    def test_sdr_status(self, topics: TritiumTopics) -> None:
        assert topics.sdr_status("hackrf-01") == "tritium/home/sdr/hackrf-01/status"

    def test_sdr_command(self, topics: TritiumTopics) -> None:
        assert topics.sdr_command("hackrf-01") == "tritium/home/sdr/hackrf-01/command"

    def test_all_sdr(self, topics: TritiumTopics) -> None:
        assert topics.all_sdr() == "tritium/home/sdr/+/#"

    def test_sdr_different_site(self) -> None:
        t = TritiumTopics(site_id="alpha")
        assert t.sdr_spectrum("rtl-001") == "tritium/alpha/sdr/rtl-001/spectrum"


# --- Meshtastic extended topics ---

class TestMeshtasticExtendedTopics:
    def test_meshtastic_status(self, topics: TritiumTopics) -> None:
        assert topics.meshtastic_status("lora-01") == "tritium/home/meshtastic/lora-01/status"

    def test_meshtastic_position(self, topics: TritiumTopics) -> None:
        assert topics.meshtastic_position("lora-01") == "tritium/home/meshtastic/lora-01/position"


# --- Generic addon device topics ---

class TestAddonDeviceTopics:
    def test_addon_device_custom(self, topics: TritiumTopics) -> None:
        assert topics.addon_device("adsb", "receiver-01", "aircraft") == \
            "tritium/home/adsb/receiver-01/aircraft"

    def test_addon_device_status(self, topics: TritiumTopics) -> None:
        assert topics.addon_device_status("adsb", "receiver-01") == \
            "tritium/home/adsb/receiver-01/status"

    def test_addon_device_command(self, topics: TritiumTopics) -> None:
        assert topics.addon_device_command("adsb", "receiver-01") == \
            "tritium/home/adsb/receiver-01/command"

    def test_addon_device_arbitrary_domain(self, topics: TritiumTopics) -> None:
        """Verify addon_device works for any domain string."""
        assert topics.addon_device("lidar", "velodyne-1", "pointcloud") == \
            "tritium/home/lidar/velodyne-1/pointcloud"

    def test_addon_device_status_delegates(self, topics: TritiumTopics) -> None:
        """addon_device_status should produce the same result as addon_device with 'status'."""
        assert topics.addon_device_status("radar", "unit-5") == \
            topics.addon_device("radar", "unit-5", "status")

    def test_addon_device_command_delegates(self, topics: TritiumTopics) -> None:
        """addon_device_command should produce the same result as addon_device with 'command'."""
        assert topics.addon_device_command("radar", "unit-5") == \
            topics.addon_device("radar", "unit-5", "command")

    def test_all_addon_domain(self, topics: TritiumTopics) -> None:
        assert topics.all_addon_domain("adsb") == "tritium/home/adsb/+/#"

    def test_all_addon_domain_arbitrary(self, topics: TritiumTopics) -> None:
        assert topics.all_addon_domain("sonar") == "tritium/home/sonar/+/#"


# --- ParsedTopic for site-scoped topics ---

class TestParseSiteTopic:
    def test_parse_sdr_spectrum(self) -> None:
        parsed = parse_site_topic("tritium/home/sdr/hackrf-01/spectrum")
        assert parsed is not None
        assert parsed.device_id == "hackrf-01"
        assert parsed.domain == "sdr"
        assert parsed.data_type == "spectrum"
        assert parsed.site == "home"
        assert parsed.message_type == "sdr/spectrum"

    def test_parse_sdr_status(self) -> None:
        parsed = parse_site_topic("tritium/home/sdr/rtl-001/status")
        assert parsed is not None
        assert parsed.device_id == "rtl-001"
        assert parsed.domain == "sdr"
        assert parsed.data_type == "status"

    def test_parse_addon_device(self) -> None:
        parsed = parse_site_topic("tritium/alpha/adsb/receiver-01/aircraft")
        assert parsed is not None
        assert parsed.site == "alpha"
        assert parsed.domain == "adsb"
        assert parsed.device_id == "receiver-01"
        assert parsed.data_type == "aircraft"

    def test_parse_meshtastic_position(self) -> None:
        parsed = parse_site_topic("tritium/home/meshtastic/lora-01/position")
        assert parsed is not None
        assert parsed.domain == "meshtastic"
        assert parsed.data_type == "position"

    def test_parse_invalid_topic(self) -> None:
        assert parse_site_topic("invalid/topic") is None

    def test_parse_too_short(self) -> None:
        assert parse_site_topic("tritium/home/sdr") is None

    def test_parse_nested_data_type(self) -> None:
        """Data types with slashes should be captured fully."""
        parsed = parse_site_topic("tritium/home/edge/esp32-001/ota/status")
        assert parsed is not None
        assert parsed.data_type == "ota/status"
        assert parsed.domain == "edge"
