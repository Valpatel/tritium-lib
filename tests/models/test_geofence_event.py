# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for GeofenceEvent model."""

import time

from tritium_lib.models.geofence_event import GeofenceEvent


def test_basic_creation():
    ev = GeofenceEvent(
        target_id="ble_AA:BB:CC:DD:EE:FF",
        zone_id="zone_abc123",
        direction="enter",
        target_alliance="hostile",
        zone_type="restricted",
        zone_name="Perimeter North",
        position=(33.749, -84.388),
    )
    assert ev.target_id == "ble_AA:BB:CC:DD:EE:FF"
    assert ev.direction == "enter"
    assert ev.target_alliance == "hostile"
    assert ev.zone_type == "restricted"
    assert ev.zone_name == "Perimeter North"
    assert ev.position == (33.749, -84.388)


def test_defaults():
    ev = GeofenceEvent(
        target_id="det_person_1",
        zone_id="z1",
        direction="exit",
    )
    assert ev.target_alliance == "unknown"
    assert ev.zone_type == "monitored"
    assert ev.zone_name == ""
    assert ev.position is None
    assert ev.timestamp > 0


def test_to_dict():
    ev = GeofenceEvent(
        target_id="mesh_node_7",
        zone_id="z2",
        direction="enter",
        timestamp=1700000000.0,
        target_alliance="friendly",
        zone_type="safe",
        zone_name="Base Camp",
        position=(34.0, -118.0),
    )
    d = ev.to_dict()
    assert d["target_id"] == "mesh_node_7"
    assert d["zone_id"] == "z2"
    assert d["direction"] == "enter"
    assert d["timestamp"] == 1700000000.0
    assert d["target_alliance"] == "friendly"
    assert d["zone_type"] == "safe"
    assert d["zone_name"] == "Base Camp"
    assert d["position"] == [34.0, -118.0]


def test_to_dict_no_optionals():
    ev = GeofenceEvent(
        target_id="t1",
        zone_id="z1",
        direction="exit",
    )
    d = ev.to_dict()
    assert "zone_name" not in d  # empty string omitted
    assert "position" not in d


def test_from_dict():
    d = {
        "target_id": "wifi_bssid_123",
        "zone_id": "zone_9",
        "direction": "exit",
        "timestamp": 1700000000.0,
        "target_alliance": "hostile",
        "zone_type": "restricted",
        "zone_name": "Exclusion Zone",
        "position": [40.7, -74.0],
    }
    ev = GeofenceEvent.from_dict(d)
    assert ev.target_id == "wifi_bssid_123"
    assert ev.direction == "exit"
    assert ev.position == (40.7, -74.0)
    assert ev.zone_name == "Exclusion Zone"


def test_roundtrip():
    original = GeofenceEvent(
        target_id="det_vehicle_5",
        zone_id="z_parking",
        direction="enter",
        timestamp=1700000001.5,
        target_alliance="unknown",
        zone_type="monitored",
        zone_name="Parking Lot A",
        position=(33.0, -84.0),
    )
    restored = GeofenceEvent.from_dict(original.to_dict())
    assert restored.target_id == original.target_id
    assert restored.zone_id == original.zone_id
    assert restored.direction == original.direction
    assert restored.timestamp == original.timestamp
    assert restored.target_alliance == original.target_alliance
    assert restored.zone_type == original.zone_type
    assert restored.zone_name == original.zone_name
    assert restored.position == original.position


def test_import_from_package():
    from tritium_lib.models import GeofenceEvent as GE
    assert GE is GeofenceEvent
