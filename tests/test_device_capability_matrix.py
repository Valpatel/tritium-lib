# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for CapabilityMatrix model."""

from tritium_lib.models.device_capability_matrix import (
    CapabilityMatrix,
    DeviceCapabilityEntry,
)


def _make_device(did: str, name: str = "", online: bool = True) -> DeviceCapabilityEntry:
    return DeviceCapabilityEntry(device_id=did, device_name=name or did, online=online)


def test_empty_matrix():
    m = CapabilityMatrix()
    assert m.device_count == 0
    assert m.capability_count == 0
    assert m.coverage_gaps() == []


def test_add_capability_then_device():
    m = CapabilityMatrix()
    m.add_capability("ble")
    m.add_capability("wifi")
    m.add_capability("camera")
    m.add_device(_make_device("d1"), [True, True, False])
    m.add_device(_make_device("d2"), [False, True, True])
    assert m.device_count == 2
    assert m.capability_count == 3


def test_get_device_capabilities():
    m = CapabilityMatrix()
    m.add_capability("ble")
    m.add_capability("wifi")
    m.add_capability("gps")
    m.add_device(_make_device("d1"), [True, False, True])
    caps = m.get_device_capabilities("d1")
    assert caps == ["ble", "gps"]


def test_get_devices_with_capability():
    m = CapabilityMatrix()
    m.add_capability("ble")
    m.add_capability("wifi")
    m.add_device(_make_device("d1"), [True, True])
    m.add_device(_make_device("d2"), [False, True])
    m.add_device(_make_device("d3"), [True, False])
    assert sorted(m.get_devices_with_capability("ble")) == ["d1", "d3"]
    assert sorted(m.get_devices_with_capability("wifi")) == ["d1", "d2"]
    assert m.get_devices_with_capability("nonexistent") == []


def test_coverage_gaps():
    m = CapabilityMatrix()
    m.add_capability("ble")
    m.add_capability("acoustic")
    m.add_capability("camera")
    # d1 is online, has ble
    m.add_device(_make_device("d1", online=True), [True, False, False])
    # d2 is offline, has acoustic + camera
    m.add_device(_make_device("d2", online=False), [False, True, True])
    # acoustic and camera have no ONLINE device
    gaps = m.coverage_gaps()
    assert "acoustic" in gaps
    assert "camera" in gaps
    assert "ble" not in gaps


def test_add_device_pads_caps():
    m = CapabilityMatrix()
    m.add_capability("ble")
    m.add_capability("wifi")
    m.add_capability("gps")
    # Provide fewer caps than capabilities
    m.add_device(_make_device("d1"), [True])
    caps = m.get_device_capabilities("d1")
    assert caps == ["ble"]


def test_roundtrip():
    m = CapabilityMatrix()
    m.add_capability("ble")
    m.add_capability("wifi")
    m.add_device(_make_device("d1"), [True, False])
    m.add_device(_make_device("d2", online=False), [False, True])
    d = m.to_dict()
    m2 = CapabilityMatrix.from_dict(d)
    assert m2.device_count == 2
    assert m2.capability_count == 2
    assert m2.get_device_capabilities("d1") == ["ble"]
    assert m2.devices[1].online is False


def test_device_entry_roundtrip():
    e = DeviceCapabilityEntry(device_id="abc", device_name="Node A", device_type="43c", online=True)
    d = e.to_dict()
    e2 = DeviceCapabilityEntry.from_dict(d)
    assert e2.device_id == "abc"
    assert e2.device_type == "43c"
    assert e2.online is True
