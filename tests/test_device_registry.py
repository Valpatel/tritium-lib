# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under Apache-2.0 — see LICENSE for details.
"""Tests for the DeviceRegistry addon SDK component."""

import time

import pytest

from tritium_lib.sdk.device_registry import (
    DeviceRegistry,
    DeviceState,
    RegisteredDevice,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockTransport:
    """Minimal mock that satisfies connect/disconnect interface."""

    def __init__(self, *, fail_connect: bool = False) -> None:
        self._fail = fail_connect
        self.connected = False

    async def connect(self) -> bool:
        if self._fail:
            raise RuntimeError("connection refused")
        self.connected = True
        return True

    async def disconnect(self) -> None:
        self.connected = False


def _make_registry(addon_id: str = "test_addon") -> DeviceRegistry:
    return DeviceRegistry(addon_id)


# ---------------------------------------------------------------------------
# RegisteredDevice
# ---------------------------------------------------------------------------

class TestRegisteredDevice:
    def test_to_dict_excludes_transport(self):
        dev = RegisteredDevice(
            device_id="d1",
            device_type="hackrf",
            transport_type="local",
            transport=object(),
        )
        d = dev.to_dict()
        assert "transport" not in d
        assert d["device_id"] == "d1"
        assert d["state"] == "disconnected"

    def test_is_local(self):
        dev = RegisteredDevice(device_id="d1", device_type="x", transport_type="local")
        assert dev.is_local is True
        assert dev.is_remote is False

    def test_is_remote(self):
        dev = RegisteredDevice(device_id="d1", device_type="x", transport_type="mqtt")
        assert dev.is_local is False
        assert dev.is_remote is True


# ---------------------------------------------------------------------------
# Add / Remove / Get
# ---------------------------------------------------------------------------

class TestAddRemoveGet:
    def test_add_and_get(self):
        reg = _make_registry()
        dev = reg.add_device("hrf-001", "hackrf")
        assert dev.device_id == "hrf-001"
        assert reg.get_device("hrf-001") is dev

    def test_add_with_metadata_and_transport(self):
        reg = _make_registry()
        t = MockTransport()
        dev = reg.add_device(
            "hrf-002", "hackrf",
            transport_type="mqtt",
            metadata={"serial": "ABC"},
            transport=t,
        )
        assert dev.metadata == {"serial": "ABC"}
        assert dev.transport is t

    def test_duplicate_device_id_rejected(self):
        reg = _make_registry()
        reg.add_device("d1", "hackrf")
        with pytest.raises(ValueError, match="already registered"):
            reg.add_device("d1", "hackrf")

    def test_remove_existing(self):
        reg = _make_registry()
        reg.add_device("d1", "hackrf")
        assert reg.remove_device("d1") is True
        assert reg.get_device("d1") is None

    def test_remove_nonexistent_returns_false(self):
        reg = _make_registry()
        assert reg.remove_device("ghost") is False

    def test_get_nonexistent_returns_none(self):
        reg = _make_registry()
        assert reg.get_device("nope") is None


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------

class TestListing:
    def test_list_devices(self):
        reg = _make_registry()
        reg.add_device("a", "hackrf")
        reg.add_device("b", "rtl-sdr")
        assert len(reg.list_devices()) == 2

    def test_list_connected(self):
        reg = _make_registry()
        reg.add_device("a", "hackrf")
        reg.add_device("b", "rtl-sdr")
        reg.set_state("a", DeviceState.CONNECTED)
        connected = reg.list_connected()
        assert len(connected) == 1
        assert connected[0].device_id == "a"

    def test_list_by_type(self):
        reg = _make_registry()
        reg.add_device("a", "hackrf")
        reg.add_device("b", "rtl-sdr")
        reg.add_device("c", "hackrf")
        hackers = reg.list_by_type("hackrf")
        assert len(hackers) == 2
        assert all(d.device_type == "hackrf" for d in hackers)


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------

class TestState:
    def test_set_state(self):
        reg = _make_registry()
        reg.add_device("d1", "hackrf")
        assert reg.set_state("d1", DeviceState.CONNECTING) is True
        assert reg.get_device("d1").state == DeviceState.CONNECTING

    def test_set_state_with_error(self):
        reg = _make_registry()
        reg.add_device("d1", "hackrf")
        reg.set_state("d1", DeviceState.ERROR, error="timeout")
        dev = reg.get_device("d1")
        assert dev.state == DeviceState.ERROR
        assert dev.error == "timeout"

    def test_set_state_nonexistent(self):
        reg = _make_registry()
        assert reg.set_state("ghost", DeviceState.CONNECTED) is False


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_update_metadata(self):
        reg = _make_registry()
        reg.add_device("d1", "hackrf")
        assert reg.update_metadata("d1", firmware="1.2.3", serial="XYZ") is True
        dev = reg.get_device("d1")
        assert dev.metadata["firmware"] == "1.2.3"
        assert dev.metadata["serial"] == "XYZ"

    def test_update_metadata_nonexistent(self):
        reg = _make_registry()
        assert reg.update_metadata("ghost", key="val") is False


# ---------------------------------------------------------------------------
# Touch
# ---------------------------------------------------------------------------

class TestTouch:
    def test_touch_updates_last_seen(self):
        reg = _make_registry()
        reg.add_device("d1", "hackrf")
        before = time.time()
        reg.touch("d1")
        after = time.time()
        ls = reg.get_device("d1").last_seen
        assert before <= ls <= after

    def test_touch_nonexistent(self):
        reg = _make_registry()
        assert reg.touch("ghost") is False


# ---------------------------------------------------------------------------
# connect_all / disconnect_all
# ---------------------------------------------------------------------------

class TestConnectDisconnect:
    @pytest.mark.asyncio
    async def test_connect_all_with_transports(self):
        reg = _make_registry()
        t1 = MockTransport()
        t2 = MockTransport()
        reg.add_device("a", "hackrf", transport=t1)
        reg.add_device("b", "rtl-sdr", transport=t2)
        results = await reg.connect_all()
        assert results == {"a": True, "b": True}
        assert reg.get_device("a").state == DeviceState.CONNECTED
        assert reg.get_device("b").state == DeviceState.CONNECTED
        assert t1.connected
        assert t2.connected

    @pytest.mark.asyncio
    async def test_connect_all_no_transport(self):
        reg = _make_registry()
        reg.add_device("a", "hackrf")  # no transport
        results = await reg.connect_all()
        assert results == {"a": False}
        # State should remain DISCONNECTED (no transport to attempt)
        assert reg.get_device("a").state == DeviceState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_connect_all_with_failure(self):
        reg = _make_registry()
        t1 = MockTransport(fail_connect=True)
        reg.add_device("a", "hackrf", transport=t1)
        results = await reg.connect_all()
        assert results == {"a": False}
        dev = reg.get_device("a")
        assert dev.state == DeviceState.ERROR
        assert "connection refused" in dev.error

    @pytest.mark.asyncio
    async def test_disconnect_all(self):
        reg = _make_registry()
        t1 = MockTransport()
        reg.add_device("a", "hackrf", transport=t1)
        await reg.connect_all()
        assert t1.connected
        await reg.disconnect_all()
        assert not t1.connected
        assert reg.get_device("a").state == DeviceState.DISCONNECTED


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_to_dict(self):
        reg = _make_registry("my_addon")
        reg.add_device("d1", "hackrf", metadata={"sn": "123"})
        reg.set_state("d1", DeviceState.CONNECTED)
        d = reg.to_dict()
        assert d["addon_id"] == "my_addon"
        assert d["device_count"] == 1
        assert d["connected_count"] == 1
        assert "d1" in d["devices"]
        assert d["devices"]["d1"]["state"] == "connected"
        assert d["devices"]["d1"]["metadata"] == {"sn": "123"}


# ---------------------------------------------------------------------------
# Dunder methods
# ---------------------------------------------------------------------------

class TestDunder:
    def test_contains(self):
        reg = _make_registry()
        reg.add_device("d1", "hackrf")
        assert "d1" in reg
        assert "d2" not in reg

    def test_len(self):
        reg = _make_registry()
        assert len(reg) == 0
        reg.add_device("a", "hackrf")
        reg.add_device("b", "rtl-sdr")
        assert len(reg) == 2


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestProperties:
    def test_device_count(self):
        reg = _make_registry()
        assert reg.device_count == 0
        reg.add_device("a", "hackrf")
        assert reg.device_count == 1

    def test_connected_count(self):
        reg = _make_registry()
        reg.add_device("a", "hackrf")
        reg.add_device("b", "rtl-sdr")
        assert reg.connected_count == 0
        reg.set_state("a", DeviceState.CONNECTED)
        assert reg.connected_count == 1
        reg.set_state("b", DeviceState.CONNECTED)
        assert reg.connected_count == 2
