# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for Meshtastic integration models."""

from datetime import datetime, timedelta

import pytest

from tritium_lib.models.meshtastic import (
    MeshtasticConnectionType,
    MeshtasticMessage,
    MeshtasticNode,
    MeshtasticStatus,
    MeshtasticWaypoint,
)


# ── MeshtasticNode ──────────────────────────────────────────────────

class TestMeshtasticNode:
    def test_minimal(self):
        node = MeshtasticNode(node_id="!aabbccdd")
        assert node.node_id == "!aabbccdd"
        assert node.long_name == ""
        assert node.short_name == ""
        assert node.hw_model == ""
        assert node.lat is None
        assert node.lng is None
        assert node.alt is None
        assert node.battery_level is None
        assert node.snr is None
        assert node.last_heard is None

    def test_full(self):
        now = datetime.now()
        node = MeshtasticNode(
            node_id="!aabbccdd",
            long_name="Base Camp",
            short_name="BC",
            hw_model="TBEAM",
            lat=37.7749,
            lng=-122.4194,
            alt=10.5,
            battery_level=85,
            snr=12.3,
            last_heard=now,
        )
        assert node.long_name == "Base Camp"
        assert node.hw_model == "TBEAM"
        assert node.battery_level == 85
        assert node.snr == 12.3
        assert node.last_heard == now

    def test_has_position_true(self):
        node = MeshtasticNode(node_id="!1", lat=1.0, lng=2.0)
        assert node.has_position is True

    def test_has_position_false_no_coords(self):
        node = MeshtasticNode(node_id="!1")
        assert node.has_position is False

    def test_has_position_false_partial(self):
        node = MeshtasticNode(node_id="!1", lat=1.0)
        assert node.has_position is False

    def test_serialization_roundtrip(self):
        now = datetime.now()
        node = MeshtasticNode(
            node_id="!ff",
            long_name="Test",
            lat=0.0,
            lng=0.0,
            last_heard=now,
        )
        data = node.model_dump()
        restored = MeshtasticNode.model_validate(data)
        assert restored.node_id == node.node_id
        assert restored.lat == 0.0
        assert restored.last_heard == now

    def test_json_roundtrip(self):
        node = MeshtasticNode(node_id="!aa", battery_level=50)
        json_str = node.model_dump_json()
        restored = MeshtasticNode.model_validate_json(json_str)
        assert restored == node

    def test_battery_edge_values(self):
        node_zero = MeshtasticNode(node_id="!1", battery_level=0)
        assert node_zero.battery_level == 0
        node_full = MeshtasticNode(node_id="!1", battery_level=100)
        assert node_full.battery_level == 100


# ── MeshtasticMessage ───────────────────────────────────────────────

class TestMeshtasticMessage:
    def test_defaults(self):
        msg = MeshtasticMessage(from_id="!aa")
        assert msg.to_id == "^all"
        assert msg.text == ""
        assert msg.channel == 0
        assert msg.hop_count == 0
        assert msg.timestamp is None

    def test_broadcast_detection(self):
        msg = MeshtasticMessage(from_id="!aa")
        assert msg.is_broadcast is True

    def test_direct_message(self):
        msg = MeshtasticMessage(from_id="!aa", to_id="!bb", text="hello")
        assert msg.is_broadcast is False
        assert msg.text == "hello"

    def test_serialization_roundtrip(self):
        now = datetime.now()
        msg = MeshtasticMessage(
            from_id="!aa",
            to_id="!bb",
            text="test message",
            channel=3,
            timestamp=now,
            hop_count=2,
        )
        data = msg.model_dump()
        restored = MeshtasticMessage.model_validate(data)
        assert restored == msg

    def test_json_roundtrip(self):
        msg = MeshtasticMessage(from_id="!aa", text="ping")
        restored = MeshtasticMessage.model_validate_json(msg.model_dump_json())
        assert restored == msg


# ── MeshtasticWaypoint ──────────────────────────────────────────────

class TestMeshtasticWaypoint:
    def test_minimal(self):
        wp = MeshtasticWaypoint(lat=37.0, lng=-122.0)
        assert wp.name == ""
        assert wp.description == ""
        assert wp.expire_time is None

    def test_full(self):
        future = datetime.now() + timedelta(hours=1)
        wp = MeshtasticWaypoint(
            lat=37.0,
            lng=-122.0,
            name="Rally Point",
            description="Meet here",
            expire_time=future,
        )
        assert wp.name == "Rally Point"
        assert wp.is_expired is False

    def test_expired(self):
        past = datetime.now() - timedelta(hours=1)
        wp = MeshtasticWaypoint(lat=0, lng=0, expire_time=past)
        assert wp.is_expired is True

    def test_no_expiry_not_expired(self):
        wp = MeshtasticWaypoint(lat=0, lng=0)
        assert wp.is_expired is False

    def test_serialization_roundtrip(self):
        wp = MeshtasticWaypoint(lat=1.5, lng=-3.5, name="Alpha")
        restored = MeshtasticWaypoint.model_validate(wp.model_dump())
        assert restored == wp

    def test_json_roundtrip(self):
        wp = MeshtasticWaypoint(lat=10.0, lng=20.0, name="B")
        restored = MeshtasticWaypoint.model_validate_json(wp.model_dump_json())
        assert restored == wp


# ── MeshtasticStatus ────────────────────────────────────────────────

class TestMeshtasticStatus:
    def test_defaults(self):
        status = MeshtasticStatus()
        assert status.connected is False
        assert status.connection_type is None
        assert status.node_count == 0
        assert status.my_node_id is None

    def test_connected(self):
        status = MeshtasticStatus(
            connected=True,
            connection_type=MeshtasticConnectionType.BLE,
            node_count=5,
            my_node_id="!aabb",
        )
        assert status.connected is True
        assert status.connection_type == MeshtasticConnectionType.BLE
        assert status.node_count == 5

    def test_connection_types(self):
        for ct in MeshtasticConnectionType:
            status = MeshtasticStatus(connection_type=ct)
            assert status.connection_type == ct

    def test_serialization_roundtrip(self):
        status = MeshtasticStatus(
            connected=True,
            connection_type=MeshtasticConnectionType.SERIAL,
            node_count=3,
        )
        restored = MeshtasticStatus.model_validate(status.model_dump())
        assert restored == status

    def test_json_roundtrip(self):
        status = MeshtasticStatus(connected=True, node_count=10)
        restored = MeshtasticStatus.model_validate_json(status.model_dump_json())
        assert restored == status


# ── Import from top-level ───────────────────────────────────────────

class TestTopLevelImport:
    def test_importable_from_models(self):
        from tritium_lib.models import (
            MeshtasticConnectionType,
            MeshtasticMessage,
            MeshtasticNode,
            MeshtasticStatus,
            MeshtasticWaypoint,
        )
        # Verify they are the correct classes
        assert MeshtasticNode(node_id="!1").node_id == "!1"
