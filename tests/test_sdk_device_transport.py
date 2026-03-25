# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for DeviceTransport — local and MQTT transport layers."""

import asyncio

import pytest

from tritium_lib.sdk.device_transport import (
    DeviceTransport,
    LocalTransport,
    MQTTTransport,
)


# ── LocalTransport ──────────────────────────────────────────────────

class TestLocalTransport:
    """Tests for the local (USB/serial) transport."""

    def test_transport_type(self):
        t = LocalTransport("/dev/ttyUSB0", "hackrf")
        assert t.transport_type == "local"

    def test_device_path(self):
        t = LocalTransport("/dev/ttyACM0")
        assert t.device_path == "/dev/ttyACM0"

    def test_device_type(self):
        t = LocalTransport("/dev/x", "meshtastic")
        assert t.device_type == "meshtastic"

    def test_default_device_type(self):
        t = LocalTransport("/dev/x")
        assert t.device_type == "unknown"

    def test_initial_not_connected(self):
        t = LocalTransport("/dev/x")
        assert t.is_connected is False

    def test_connect(self):
        t = LocalTransport("/dev/x")
        result = asyncio.run(t.connect())
        assert result is True
        assert t.is_connected is True

    def test_disconnect(self):
        t = LocalTransport("/dev/x")
        asyncio.run(t.connect())
        asyncio.run(t.disconnect())
        assert t.is_connected is False

    def test_send_command(self):
        t = LocalTransport("/dev/x")
        result = asyncio.run(t.send_command("restart", {"delay": 5}))
        assert result == {"status": "ok"}

    def test_on_data_callback(self):
        t = LocalTransport("/dev/x")
        received = []
        t.on_data(lambda d: received.append(d))
        t._emit_data({"sensor": "temp", "value": 23.5})
        assert len(received) == 1
        assert received[0]["sensor"] == "temp"

    def test_multiple_data_callbacks(self):
        t = LocalTransport("/dev/x")
        r1, r2 = [], []
        t.on_data(lambda d: r1.append(d))
        t.on_data(lambda d: r2.append(d))
        t._emit_data({"value": 1})
        assert len(r1) == 1
        assert len(r2) == 1


# ── MQTTTransport ──────────────────────────────────────────────────

class TestMQTTTransport:
    """Tests for the MQTT-based transport."""

    def test_transport_type(self):
        t = MQTTTransport("hackrf-01", "sdr")
        assert t.transport_type == "mqtt"

    def test_initial_not_connected(self):
        t = MQTTTransport("hackrf-01", "sdr")
        assert t.is_connected is False

    def test_command_topic(self):
        t = MQTTTransport("hackrf-01", "sdr", site_id="home")
        assert t.command_topic == "tritium/home/sdr/hackrf-01/command"

    def test_status_topic(self):
        t = MQTTTransport("cam-01", "cameras", site_id="hq")
        assert t.status_topic == "tritium/hq/cameras/cam-01/status"

    def test_data_topic(self):
        t = MQTTTransport("dev1", "sensors", site_id="warehouse")
        assert t.data_topic == "tritium/warehouse/sensors/dev1/data"

    def test_connect_without_client(self):
        t = MQTTTransport("dev1", "sdr")
        result = asyncio.run(t.connect())
        assert result is True
        assert t.is_connected is True

    def test_disconnect_without_client(self):
        t = MQTTTransport("dev1", "sdr")
        asyncio.run(t.connect())
        asyncio.run(t.disconnect())
        assert t.is_connected is False

    def test_send_command_without_client(self):
        t = MQTTTransport("dev1", "sdr")
        result = asyncio.run(t.send_command("scan", {"freq": 433}))
        assert result["status"] == "queued"

    def test_on_mqtt_message_data_topic(self):
        t = MQTTTransport("dev1", "sdr", site_id="home")
        received = []
        t.on_data(lambda d: received.append(d))
        t.on_mqtt_message(t.data_topic, {"spectrum": [1, 2, 3]})
        assert len(received) == 1
        assert received[0]["spectrum"] == [1, 2, 3]

    def test_on_mqtt_message_status_topic(self):
        t = MQTTTransport("dev1", "sdr", site_id="home")
        received = []
        t.on_data(lambda d: received.append(d))
        t.on_mqtt_message(t.status_topic, {"status": "online"})
        assert len(received) == 1

    def test_on_mqtt_message_unrelated_topic_ignored(self):
        t = MQTTTransport("dev1", "sdr", site_id="home")
        received = []
        t.on_data(lambda d: received.append(d))
        t.on_mqtt_message("unrelated/topic", {"data": 1})
        assert len(received) == 0

    def test_connect_with_mock_client(self):
        class MockMQTT:
            def __init__(self):
                self.subscribed = []
            def subscribe(self, topic):
                self.subscribed.append(topic)

        mock = MockMQTT()
        t = MQTTTransport("dev1", "sdr", mqtt_client=mock)
        asyncio.run(t.connect())
        assert t.is_connected is True
        assert len(mock.subscribed) == 2  # status + data topics

    def test_send_command_with_mock_client(self):
        class MockMQTT:
            def __init__(self):
                self.published = []
            def publish(self, topic, payload):
                self.published.append((topic, payload))
            def subscribe(self, topic):
                pass

        mock = MockMQTT()
        t = MQTTTransport("dev1", "sdr", mqtt_client=mock)
        result = asyncio.run(t.send_command("scan", {"freq": 433}))
        assert result["status"] == "sent"
        assert len(mock.published) == 1
        assert mock.published[0][0] == t.command_topic
