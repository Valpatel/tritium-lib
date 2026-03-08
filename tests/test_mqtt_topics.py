# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for MQTT topic generation patterns."""

from tritium_lib.mqtt import TritiumTopics


class TestTritiumTopics:
    def setup_method(self):
        self.topics = TritiumTopics(site_id="home")

    def test_prefix(self):
        assert self.topics.prefix == "tritium/home"

    def test_custom_site(self):
        t = TritiumTopics(site_id="warehouse-3")
        assert t.prefix == "tritium/warehouse-3"

    # --- Edge topics ---

    def test_edge_heartbeat(self):
        assert self.topics.edge_heartbeat("dev-001") == "tritium/home/edge/dev-001/heartbeat"

    def test_edge_telemetry(self):
        assert self.topics.edge_telemetry("dev-001") == "tritium/home/edge/dev-001/telemetry"

    def test_edge_command(self):
        assert self.topics.edge_command("dev-001") == "tritium/home/edge/dev-001/command"

    def test_edge_ota_status(self):
        assert self.topics.edge_ota_status("dev-001") == "tritium/home/edge/dev-001/ota"

    # --- Sensor topics ---

    def test_sensor(self):
        assert self.topics.sensor("dev-001", "temperature") == "tritium/home/sensors/dev-001/temperature"

    def test_sensor_wildcard_specific(self):
        assert self.topics.sensor_wildcard("dev-001") == "tritium/home/sensors/dev-001/#"

    def test_sensor_wildcard_all(self):
        assert self.topics.sensor_wildcard() == "tritium/home/sensors/+/#"

    # --- Camera topics ---

    def test_camera_frame(self):
        assert self.topics.camera_frame("cam-01") == "tritium/home/cameras/cam-01/frame"

    def test_camera_detections(self):
        assert self.topics.camera_detections("cam-01") == "tritium/home/cameras/cam-01/detections"

    def test_camera_command(self):
        assert self.topics.camera_command("cam-01") == "tritium/home/cameras/cam-01/command"

    # --- Audio topics ---

    def test_audio_stream(self):
        assert self.topics.audio_stream("mic-01") == "tritium/home/audio/mic-01/stream"

    def test_audio_vad(self):
        assert self.topics.audio_vad("mic-01") == "tritium/home/audio/mic-01/vad"

    # --- Mesh topics ---

    def test_mesh_peers(self):
        assert self.topics.mesh_peers("node-01") == "tritium/home/mesh/node-01/peers"

    # --- Robot topics ---

    def test_robot_telemetry(self):
        assert self.topics.robot_telemetry("bot-01") == "tritium/home/robots/bot-01/telemetry"

    def test_robot_command(self):
        assert self.topics.robot_command("bot-01") == "tritium/home/robots/bot-01/command"

    def test_robot_thoughts(self):
        assert self.topics.robot_thoughts("bot-01") == "tritium/home/robots/bot-01/thoughts"

    # --- System topics ---

    def test_alerts(self):
        assert self.topics.alerts() == "tritium/home/amy/alerts"

    def test_escalation(self):
        assert self.topics.escalation() == "tritium/home/escalation/change"

    # --- Wildcards ---

    def test_all_edge(self):
        assert self.topics.all_edge() == "tritium/home/edge/+/#"

    def test_all_sensors(self):
        assert self.topics.all_sensors() == "tritium/home/sensors/+/#"

    def test_all_cameras(self):
        assert self.topics.all_cameras() == "tritium/home/cameras/+/#"
