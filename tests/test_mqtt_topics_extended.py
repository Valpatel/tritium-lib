# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Extended MQTT topic builder and parser tests."""

import pytest

from tritium_lib.mqtt.topics import (
    ParsedTopic,
    TritiumTopics,
    device_commands,
    device_heartbeat,
    device_ota_status,
    device_sensors,
    fleet_broadcast,
    parse_site_topic,
    parse_topic,
)


# ── Device-centric topic functions ──────────────────────────────────

class TestDeviceTopicFunctions:
    """Tests for the simple device topic builder functions."""

    def test_device_heartbeat(self):
        assert device_heartbeat("esp32-001") == "tritium/devices/esp32-001/heartbeat"

    def test_device_sensors(self):
        assert device_sensors("esp32-001", "temperature") == \
            "tritium/devices/esp32-001/sensors/temperature"

    def test_device_commands(self):
        assert device_commands("esp32-001") == "tritium/devices/esp32-001/commands"

    def test_device_ota_status(self):
        assert device_ota_status("esp32-001") == "tritium/devices/esp32-001/ota/status"

    def test_fleet_broadcast(self):
        assert fleet_broadcast() == "tritium/fleet/broadcast"

    def test_special_chars_in_device_id(self):
        topic = device_heartbeat("node-with-dashes")
        assert "node-with-dashes" in topic


# ── Topic parser ────────────────────────────────────────────────────

class TestParseTopic:
    """Tests for parse_topic()."""

    def test_parse_heartbeat(self):
        result = parse_topic("tritium/devices/esp32-001/heartbeat")
        assert result is not None
        assert result.device_id == "esp32-001"
        assert result.message_type == "heartbeat"
        assert result.sensor_type is None

    def test_parse_sensor_topic(self):
        result = parse_topic("tritium/devices/node-5/sensors/temperature")
        assert result is not None
        assert result.device_id == "node-5"
        assert result.message_type == "sensors/temperature"
        assert result.sensor_type == "temperature"

    def test_parse_ota_status(self):
        result = parse_topic("tritium/devices/abc/ota/status")
        assert result is not None
        assert result.device_id == "abc"
        assert result.message_type == "ota/status"

    def test_parse_commands(self):
        result = parse_topic("tritium/devices/esp32-001/commands")
        assert result is not None
        assert result.device_id == "esp32-001"
        assert result.message_type == "commands"

    def test_non_matching_topic_returns_none(self):
        assert parse_topic("other/topic/path") is None

    def test_empty_string_returns_none(self):
        assert parse_topic("") is None

    def test_fleet_broadcast_not_device_topic(self):
        assert parse_topic("tritium/fleet/broadcast") is None


# ── Site-scoped topic parser ────────────────────────────────────────

class TestParseSiteTopic:
    """Tests for parse_site_topic()."""

    def test_parse_sdr_spectrum(self):
        result = parse_site_topic("tritium/home/sdr/hackrf-01/spectrum")
        assert result is not None
        assert result.device_id == "hackrf-01"
        assert result.site == "home"
        assert result.domain == "sdr"
        assert result.data_type == "spectrum"
        assert result.message_type == "sdr/spectrum"

    def test_parse_camera_frame(self):
        result = parse_site_topic("tritium/hq/cameras/cam-01/frame")
        assert result is not None
        assert result.site == "hq"
        assert result.domain == "cameras"
        assert result.device_id == "cam-01"
        assert result.data_type == "frame"

    def test_parse_edge_heartbeat(self):
        result = parse_site_topic("tritium/home/edge/esp32-001/heartbeat")
        assert result is not None
        assert result.site == "home"
        assert result.domain == "edge"

    def test_non_matching_returns_none(self):
        assert parse_site_topic("not/a/tritium/topic") is None

    def test_too_few_segments_returns_none(self):
        assert parse_site_topic("tritium/home/edge") is None


# ── ParsedTopic dataclass ──────────────────────────────────────────

class TestParsedTopic:
    """Tests for the ParsedTopic dataclass."""

    def test_defaults(self):
        pt = ParsedTopic(device_id="d1", message_type="heartbeat")
        assert pt.sensor_type is None
        assert pt.site is None
        assert pt.domain is None
        assert pt.data_type is None


# ── TritiumTopics builder ──────────────────────────────────────────

class TestTritiumTopics:
    """Tests for the TritiumTopics site-scoped builder."""

    def setup_method(self):
        self.topics = TritiumTopics(site_id="hq")

    # Edge topics
    def test_edge_heartbeat(self):
        assert self.topics.edge_heartbeat("node-1") == "tritium/hq/edge/node-1/heartbeat"

    def test_edge_telemetry(self):
        assert self.topics.edge_telemetry("node-1") == "tritium/hq/edge/node-1/telemetry"

    def test_edge_command(self):
        assert self.topics.edge_command("node-1") == "tritium/hq/edge/node-1/command"

    def test_edge_ota_status(self):
        assert self.topics.edge_ota_status("node-1") == "tritium/hq/edge/node-1/ota"

    def test_edge_capabilities(self):
        assert self.topics.edge_capabilities("node-1") == "tritium/hq/edge/node-1/capabilities"

    # Sensor topics
    def test_sensor(self):
        assert self.topics.sensor("s1", "temperature") == "tritium/hq/sensors/s1/temperature"

    def test_sensor_wildcard_default(self):
        assert self.topics.sensor_wildcard() == "tritium/hq/sensors/+/#"

    def test_sensor_wildcard_specific(self):
        assert self.topics.sensor_wildcard("s1") == "tritium/hq/sensors/s1/#"

    # Camera topics
    def test_camera_frame(self):
        assert self.topics.camera_frame("cam-1") == "tritium/hq/cameras/cam-1/frame"

    def test_camera_detections(self):
        assert self.topics.camera_detections("cam-1") == "tritium/hq/cameras/cam-1/detections"

    def test_camera_command(self):
        assert self.topics.camera_command("cam-1") == "tritium/hq/cameras/cam-1/command"

    def test_camera_feed(self):
        assert self.topics.camera_feed("cam-1") == "tritium/hq/cameras/cam-1/feed"

    def test_camera_snapshot(self):
        assert self.topics.camera_snapshot("cam-1") == "tritium/hq/cameras/cam-1/snapshot"

    # Audio topics
    def test_audio_stream(self):
        assert self.topics.audio_stream("mic-1") == "tritium/hq/audio/mic-1/stream"

    def test_audio_vad(self):
        assert self.topics.audio_vad("mic-1") == "tritium/hq/audio/mic-1/vad"

    # Mesh topics
    def test_mesh_peers(self):
        assert self.topics.mesh_peers("node-1") == "tritium/hq/mesh/node-1/peers"

    # Meshtastic topics
    def test_meshtastic_nodes(self):
        assert self.topics.meshtastic_nodes("bridge-1") == \
            "tritium/hq/meshtastic/bridge-1/nodes"

    def test_meshtastic_message(self):
        assert self.topics.meshtastic_message("bridge-1") == \
            "tritium/hq/meshtastic/bridge-1/message"

    def test_meshtastic_command(self):
        assert self.topics.meshtastic_command("bridge-1") == \
            "tritium/hq/meshtastic/bridge-1/command"

    def test_meshtastic_status(self):
        assert self.topics.meshtastic_status("bridge-1") == \
            "tritium/hq/meshtastic/bridge-1/status"

    def test_meshtastic_position(self):
        assert self.topics.meshtastic_position("bridge-1") == \
            "tritium/hq/meshtastic/bridge-1/position"

    # WiFi topics
    def test_wifi_probe(self):
        assert self.topics.wifi_probe("node-1") == "tritium/hq/edge/node-1/wifi_probe"

    def test_wifi_scan(self):
        assert self.topics.wifi_scan("node-1") == "tritium/hq/edge/node-1/wifi_scan"

    # Robot topics
    def test_robot_telemetry(self):
        assert self.topics.robot_telemetry("robot-1") == "tritium/hq/robots/robot-1/telemetry"

    def test_robot_command(self):
        assert self.topics.robot_command("robot-1") == "tritium/hq/robots/robot-1/command"

    def test_robot_thoughts(self):
        assert self.topics.robot_thoughts("robot-1") == "tritium/hq/robots/robot-1/thoughts"

    # SDR topics
    def test_sdr_spectrum(self):
        assert self.topics.sdr_spectrum("hackrf-01") == "tritium/hq/sdr/hackrf-01/spectrum"

    def test_sdr_status(self):
        assert self.topics.sdr_status("hackrf-01") == "tritium/hq/sdr/hackrf-01/status"

    def test_sdr_command(self):
        assert self.topics.sdr_command("hackrf-01") == "tritium/hq/sdr/hackrf-01/command"

    # System topics
    def test_alerts(self):
        assert self.topics.alerts() == "tritium/hq/amy/alerts"

    def test_escalation(self):
        assert self.topics.escalation() == "tritium/hq/escalation/change"

    # Wildcards
    def test_all_edge(self):
        assert self.topics.all_edge() == "tritium/hq/edge/+/#"

    def test_all_sensors(self):
        assert self.topics.all_sensors() == "tritium/hq/sensors/+/#"

    def test_all_cameras(self):
        assert self.topics.all_cameras() == "tritium/hq/cameras/+/#"

    def test_all_meshtastic(self):
        assert self.topics.all_meshtastic() == "tritium/hq/meshtastic/+/#"

    def test_all_sdr(self):
        assert self.topics.all_sdr() == "tritium/hq/sdr/+/#"

    # Generic addon topics
    def test_addon_device(self):
        assert self.topics.addon_device("adsb", "receiver-1", "aircraft") == \
            "tritium/hq/adsb/receiver-1/aircraft"

    def test_addon_device_status(self):
        assert self.topics.addon_device_status("adsb", "r1") == \
            "tritium/hq/adsb/r1/status"

    def test_addon_device_command(self):
        assert self.topics.addon_device_command("adsb", "r1") == \
            "tritium/hq/adsb/r1/command"

    def test_all_addon_domain(self):
        assert self.topics.all_addon_domain("adsb") == "tritium/hq/adsb/+/#"


class TestTritiumTopicsSiteId:
    """Test site_id variations."""

    def test_default_site_is_home(self):
        topics = TritiumTopics()
        assert topics.site == "home"
        assert topics.edge_heartbeat("n1").startswith("tritium/home/")

    def test_custom_site(self):
        topics = TritiumTopics(site_id="warehouse-3")
        assert topics.site == "warehouse-3"
        assert topics.edge_heartbeat("n1") == "tritium/warehouse-3/edge/n1/heartbeat"
