# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for TacticalEvent model."""

from datetime import datetime, timedelta, timezone

from tritium_lib.models.tactical_event import (
    EventPosition,
    TacticalEvent,
    TacticalEventType,
    TacticalSeverity,
    filter_events,
)


def test_event_type_enum():
    assert TacticalEventType.DETECTION.value == "detection"
    assert TacticalEventType.GEOFENCE.value == "geofence"


def test_severity_enum():
    assert TacticalSeverity.CRITICAL.value == "critical"


def test_event_position_roundtrip():
    pos = EventPosition(lat=40.7, lng=-74.0, alt_m=30.0, accuracy_m=5.0)
    d = pos.to_dict()
    pos2 = EventPosition.from_dict(d)
    assert pos2.lat == 40.7
    assert pos2.alt_m == 30.0


def test_event_defaults():
    evt = TacticalEvent()
    assert evt.event_id  # uuid generated
    assert evt.timestamp is not None
    assert evt.is_active
    assert not evt.is_spatial
    assert not evt.resolved


def test_acknowledge():
    evt = TacticalEvent(severity=TacticalSeverity.HIGH)
    assert evt.is_active
    evt.acknowledge("operator1")
    assert evt.acknowledged
    assert evt.acknowledged_by == "operator1"
    assert not evt.is_active


def test_resolve():
    evt = TacticalEvent()
    evt.resolve()
    assert evt.resolved
    assert not evt.is_active


def test_is_spatial():
    evt = TacticalEvent(position=EventPosition(lat=1.0, lng=2.0))
    assert evt.is_spatial


def test_involves_entity():
    evt = TacticalEvent(entities=["ble_aa:bb:cc", "det_person_1"])
    assert evt.involves_entity("ble_aa:bb:cc")
    assert not evt.involves_entity("mesh_node3")


def test_is_expired():
    evt = TacticalEvent(
        ttl_sec=60,
        timestamp=datetime.now(timezone.utc) - timedelta(seconds=120),
    )
    # Override __post_init__ by setting timestamp directly
    assert evt.is_expired

    evt2 = TacticalEvent(ttl_sec=0)
    assert not evt2.is_expired  # permanent

    evt3 = TacticalEvent(ttl_sec=9999)
    assert not evt3.is_expired  # not yet


def test_roundtrip():
    evt = TacticalEvent(
        event_type=TacticalEventType.ALERT,
        severity=TacticalSeverity.HIGH,
        position=EventPosition(lat=35.0, lng=-120.0),
        description="Hostile target detected",
        source="edge_tracker",
        entities=["ble_aa:bb:cc"],
        site_id="alpha",
        tags=["hostile", "ble"],
        metadata={"rssi": -45},
    )
    d = evt.to_dict()
    evt2 = TacticalEvent.from_dict(d)
    assert evt2.event_type == TacticalEventType.ALERT
    assert evt2.severity == TacticalSeverity.HIGH
    assert evt2.position is not None
    assert evt2.position.lat == 35.0
    assert evt2.description == "Hostile target detected"
    assert evt2.source == "edge_tracker"
    assert "ble_aa:bb:cc" in evt2.entities
    assert evt2.metadata["rssi"] == -45


def test_filter_by_type():
    events = [
        TacticalEvent(event_type=TacticalEventType.DETECTION),
        TacticalEvent(event_type=TacticalEventType.ALERT),
        TacticalEvent(event_type=TacticalEventType.DETECTION),
    ]
    filtered = filter_events(events, event_type=TacticalEventType.DETECTION)
    assert len(filtered) == 2


def test_filter_by_severity():
    events = [
        TacticalEvent(severity=TacticalSeverity.LOW),
        TacticalEvent(severity=TacticalSeverity.CRITICAL),
        TacticalEvent(severity=TacticalSeverity.LOW),
    ]
    filtered = filter_events(events, severity=TacticalSeverity.CRITICAL)
    assert len(filtered) == 1


def test_filter_active_only():
    evt1 = TacticalEvent()
    evt2 = TacticalEvent()
    evt2.acknowledge("op")
    evt3 = TacticalEvent()
    evt3.resolve()
    events = [evt1, evt2, evt3]
    filtered = filter_events(events, active_only=True)
    assert len(filtered) == 1


def test_filter_spatial_only():
    events = [
        TacticalEvent(position=EventPosition(lat=1.0, lng=2.0)),
        TacticalEvent(),
        TacticalEvent(position=EventPosition(lat=3.0, lng=4.0)),
    ]
    filtered = filter_events(events, spatial_only=True)
    assert len(filtered) == 2


def test_filter_by_entity():
    events = [
        TacticalEvent(entities=["ble_aa"]),
        TacticalEvent(entities=["mesh_1", "ble_aa"]),
        TacticalEvent(entities=["det_person_1"]),
    ]
    filtered = filter_events(events, entity_id="ble_aa")
    assert len(filtered) == 2


def test_filter_by_source():
    events = [
        TacticalEvent(source="edge_tracker"),
        TacticalEvent(source="yolo_detector"),
        TacticalEvent(source="edge_tracker"),
    ]
    filtered = filter_events(events, source="yolo_detector")
    assert len(filtered) == 1
