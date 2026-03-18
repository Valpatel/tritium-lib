# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under Apache-2.0 — see LICENSE for details.
"""Tests for DeviceTransport abstraction."""

import asyncio
from unittest.mock import MagicMock

import pytest

from tritium_lib.sdk.device_transport import (
    DeviceTransport,
    LocalTransport,
    MQTTTransport,
)


def _run(coro):
    """Helper to run an async coroutine in tests."""
    return asyncio.run(coro)


# --- Abstract base class ---


def test_device_transport_cannot_instantiate():
    """DeviceTransport is abstract and cannot be instantiated directly."""
    with pytest.raises(TypeError):
        DeviceTransport()


# --- LocalTransport ---


def test_local_transport_connect_disconnect():
    t = LocalTransport("/dev/ttyUSB0", device_type="esp32")
    assert not t.is_connected
    result = _run(t.connect())
    assert result is True
    assert t.is_connected
    _run(t.disconnect())
    assert not t.is_connected


def test_local_transport_properties():
    t = LocalTransport("/dev/ttyACM0", device_type="meshtastic")
    assert t.transport_type == "local"
    assert t.device_path == "/dev/ttyACM0"
    assert t.device_type == "meshtastic"


def test_local_transport_default_device_type():
    t = LocalTransport("/dev/ttyUSB0")
    assert t.device_type == "unknown"


def test_local_transport_send_command():
    t = LocalTransport("/dev/ttyUSB0")
    _run(t.connect())
    result = _run(t.send_command("reboot", {"delay": 5}))
    assert result == {"status": "ok"}
    assert t._last_command == ("reboot", {"delay": 5})


def test_local_transport_send_command_no_payload():
    t = LocalTransport("/dev/ttyUSB0")
    result = _run(t.send_command("ping"))
    assert result == {"status": "ok"}
    assert t._last_command == ("ping", None)


def test_local_transport_data_callback():
    t = LocalTransport("/dev/ttyUSB0")
    received = []
    t.on_data(lambda d: received.append(d))
    t._emit_data({"temperature": 42})
    assert received == [{"temperature": 42}]


def test_local_transport_multiple_callbacks():
    t = LocalTransport("/dev/ttyUSB0")
    a, b = [], []
    t.on_data(lambda d: a.append(d))
    t.on_data(lambda d: b.append(d))
    t._emit_data({"x": 1})
    assert a == [{"x": 1}]
    assert b == [{"x": 1}]


# --- MQTTTransport ---


def test_mqtt_transport_topic_formatting():
    t = MQTTTransport("node-01", "sensors", site_id="hq")
    assert t.command_topic == "tritium/hq/sensors/node-01/command"
    assert t.status_topic == "tritium/hq/sensors/node-01/status"
    assert t.data_topic == "tritium/hq/sensors/node-01/data"


def test_mqtt_transport_default_site():
    t = MQTTTransport("cam-1", "cameras")
    assert t.command_topic == "tritium/home/cameras/cam-1/command"


def test_mqtt_transport_type():
    t = MQTTTransport("x", "y")
    assert t.transport_type == "mqtt"


def test_mqtt_transport_connect_disconnect_no_client():
    t = MQTTTransport("n1", "domain")
    assert not t.is_connected
    result = _run(t.connect())
    assert result is True
    assert t.is_connected
    _run(t.disconnect())
    assert not t.is_connected


def test_mqtt_transport_connect_with_client():
    client = MagicMock()
    t = MQTTTransport("n1", "sensors", mqtt_client=client)
    _run(t.connect())
    assert t.is_connected
    client.subscribe.assert_any_call(t.status_topic)
    client.subscribe.assert_any_call(t.data_topic)


def test_mqtt_transport_disconnect_with_client():
    client = MagicMock()
    t = MQTTTransport("n1", "sensors", mqtt_client=client)
    _run(t.connect())
    _run(t.disconnect())
    assert not t.is_connected
    client.unsubscribe.assert_any_call(t.status_topic)
    client.unsubscribe.assert_any_call(t.data_topic)


def test_mqtt_transport_send_command_no_client():
    t = MQTTTransport("n1", "sensors")
    result = _run(t.send_command("scan", {"duration": 10}))
    assert result == {"status": "queued"}
    assert t._last_command == ("scan", {"duration": 10})


def test_mqtt_transport_send_command_with_client():
    client = MagicMock()
    t = MQTTTransport("n1", "sensors", mqtt_client=client)
    result = _run(t.send_command("reboot"))
    assert result == {"status": "sent"}
    client.publish.assert_called_once()
    topic_arg = client.publish.call_args[0][0]
    assert topic_arg == t.command_topic


def test_mqtt_transport_on_mqtt_message_routes_data():
    t = MQTTTransport("n1", "sensors")
    received = []
    t.on_data(lambda d: received.append(d))
    t.on_mqtt_message(t.data_topic, {"rssi": -55})
    assert received == [{"rssi": -55}]


def test_mqtt_transport_on_mqtt_message_routes_status():
    t = MQTTTransport("n1", "sensors")
    received = []
    t.on_data(lambda d: received.append(d))
    t.on_mqtt_message(t.status_topic, {"online": True})
    assert received == [{"online": True}]


def test_mqtt_transport_on_mqtt_message_ignores_unrelated():
    t = MQTTTransport("n1", "sensors")
    received = []
    t.on_data(lambda d: received.append(d))
    t.on_mqtt_message("tritium/other/topic", {"x": 1})
    assert received == []


def test_mqtt_transport_connect_client_error():
    """If mqtt_client.subscribe raises, connect returns False."""
    client = MagicMock()
    client.subscribe.side_effect = Exception("broker down")
    t = MQTTTransport("n1", "sensors", mqtt_client=client)
    result = _run(t.connect())
    assert result is False
    assert not t.is_connected
