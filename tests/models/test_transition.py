# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for TransitionEvent and TransitionHistory models."""

import time
import pytest

from tritium_lib.models.transition import (
    TransitionEvent,
    TransitionHistory,
    TransitionType,
)


class TestTransitionEvent:
    """Tests for TransitionEvent dataclass."""

    def test_create_minimal(self):
        ev = TransitionEvent(
            target_id="ble_aabbccddee",
            from_state="outdoor",
            to_state="indoor",
        )
        assert ev.target_id == "ble_aabbccddee"
        assert ev.from_state == "outdoor"
        assert ev.to_state == "indoor"
        assert ev.transition_type == TransitionType.CUSTOM
        assert ev.confidence == 1.0
        assert ev.event_id  # auto-generated

    def test_create_full(self):
        ev = TransitionEvent(
            target_id="ble_112233445566",
            from_state="outdoor",
            to_state="indoor",
            transition_type=TransitionType.INDOOR_OUTDOOR,
            position=(37.7749, -122.4194),
            confidence=0.85,
            source="indoor_outdoor_detector",
            node_id="node-01",
            metadata={"positioning_method": "wifi_fingerprint"},
        )
        assert ev.transition_type == TransitionType.INDOOR_OUTDOOR
        assert ev.position == (37.7749, -122.4194)
        assert ev.confidence == 0.85
        assert ev.metadata["positioning_method"] == "wifi_fingerprint"

    def test_to_dict(self):
        ev = TransitionEvent(
            target_id="t1",
            from_state="a",
            to_state="b",
            transition_type=TransitionType.ZONE_CROSSING,
            position=(1.0, 2.0),
            confidence=0.9,
            source="geofence",
            node_id="n1",
            metadata={"zone_id": "z1"},
        )
        d = ev.to_dict()
        assert d["target_id"] == "t1"
        assert d["from_state"] == "a"
        assert d["to_state"] == "b"
        assert d["transition_type"] == "zone_crossing"
        assert d["position"] == [1.0, 2.0]
        assert d["confidence"] == 0.9
        assert d["node_id"] == "n1"
        assert d["metadata"]["zone_id"] == "z1"

    def test_from_dict(self):
        d = {
            "target_id": "ble_abc",
            "from_state": "outdoor",
            "to_state": "indoor",
            "transition_type": "indoor_outdoor",
            "position": [10.0, 20.0],
            "confidence": 0.7,
            "source": "detector",
            "node_id": "n2",
            "metadata": {"note": "test"},
        }
        ev = TransitionEvent.from_dict(d)
        assert ev.target_id == "ble_abc"
        assert ev.transition_type == TransitionType.INDOOR_OUTDOOR
        assert ev.position == (10.0, 20.0)
        assert ev.confidence == 0.7
        assert ev.metadata["note"] == "test"

    def test_from_dict_unknown_type(self):
        d = {
            "target_id": "t1",
            "from_state": "a",
            "to_state": "b",
            "transition_type": "never_heard_of_this",
        }
        ev = TransitionEvent.from_dict(d)
        assert ev.transition_type == "never_heard_of_this"

    def test_roundtrip(self):
        ev = TransitionEvent(
            target_id="t1",
            from_state="x",
            to_state="y",
            transition_type=TransitionType.SPEED_CHANGE,
            position=(5.0, 6.0),
            confidence=0.55,
            source="speed_monitor",
        )
        d = ev.to_dict()
        ev2 = TransitionEvent.from_dict(d)
        assert ev2.target_id == ev.target_id
        assert ev2.from_state == ev.from_state
        assert ev2.to_state == ev.to_state
        assert ev2.transition_type == ev.transition_type
        assert ev2.confidence == 0.55

    def test_transition_types(self):
        for tt in TransitionType:
            assert isinstance(tt.value, str)
        assert TransitionType.INDOOR_OUTDOOR.value == "indoor_outdoor"
        assert TransitionType.ZONE_CROSSING.value == "zone_crossing"
        assert TransitionType.VISIBILITY.value == "visibility"


class TestTransitionHistory:
    """Tests for TransitionHistory."""

    def test_empty_history(self):
        h = TransitionHistory(target_id="t1")
        assert len(h.transitions) == 0
        assert h.count_by_type(TransitionType.INDOOR_OUTDOOR) == 0
        assert h.last_transition() is None

    def test_add_and_count(self):
        h = TransitionHistory(target_id="t1")
        h.add(TransitionEvent(
            target_id="t1", from_state="outdoor", to_state="indoor",
            transition_type=TransitionType.INDOOR_OUTDOOR,
        ))
        h.add(TransitionEvent(
            target_id="t1", from_state="zone_a", to_state="zone_b",
            transition_type=TransitionType.ZONE_CROSSING,
        ))
        h.add(TransitionEvent(
            target_id="t1", from_state="indoor", to_state="outdoor",
            transition_type=TransitionType.INDOOR_OUTDOOR,
        ))
        assert h.count_by_type(TransitionType.INDOOR_OUTDOOR) == 2
        assert h.count_by_type(TransitionType.ZONE_CROSSING) == 1

    def test_last_transition(self):
        h = TransitionHistory(target_id="t1")
        h.add(TransitionEvent(
            target_id="t1", from_state="a", to_state="b",
            transition_type=TransitionType.SPEED_CHANGE,
        ))
        h.add(TransitionEvent(
            target_id="t1", from_state="b", to_state="c",
            transition_type=TransitionType.INDOOR_OUTDOOR,
        ))
        last = h.last_transition()
        assert last is not None
        assert last.to_state == "c"

        last_io = h.last_transition(TransitionType.INDOOR_OUTDOOR)
        assert last_io is not None
        assert last_io.to_state == "c"

        last_speed = h.last_transition(TransitionType.SPEED_CHANGE)
        assert last_speed is not None
        assert last_speed.to_state == "b"

    def test_max_history(self):
        h = TransitionHistory(target_id="t1", max_history=5)
        for i in range(10):
            h.add(TransitionEvent(
                target_id="t1", from_state=str(i), to_state=str(i + 1),
            ))
        assert len(h.transitions) == 5
        assert h.transitions[0].from_state == "5"

    def test_to_dict(self):
        h = TransitionHistory(target_id="t1")
        h.add(TransitionEvent(target_id="t1", from_state="a", to_state="b"))
        d = h.to_dict()
        assert d["target_id"] == "t1"
        assert d["transition_count"] == 1
        assert len(d["transitions"]) == 1


class TestTransitionImport:
    """Test that TransitionEvent is importable from the models package."""

    def test_import_from_models(self):
        from tritium_lib.models import TransitionEvent, TransitionHistory, TransitionType
        assert TransitionEvent is not None
        assert TransitionHistory is not None
        assert TransitionType is not None
