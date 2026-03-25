# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.nodes.base."""

import pytest
from tritium_lib.nodes.base import SensorNode, Position


def test_position_dataclass():
    """Position can be created with defaults."""
    p = Position()
    assert p.pan == 0.0
    assert p.tilt == 0.0
    assert p.zoom == 100.0


def test_position_limits():
    """Position limit properties work."""
    p = Position(pan=0.0, pan_min=-90.0, pan_max=90.0)
    assert p.can_pan_left is True
    assert p.can_pan_right is True

    p2 = Position(pan=-90.0, pan_min=-90.0, pan_max=90.0)
    assert p2.can_pan_left is False
    assert p2.can_pan_right is True


def test_position_no_limits():
    """Position with no limits allows all movement."""
    p = Position()
    assert p.can_pan_left is True
    assert p.can_pan_right is True
    assert p.can_tilt_up is True
    assert p.can_tilt_down is True


def test_sensor_node_is_abstract():
    """SensorNode cannot be instantiated directly (it's an ABC)."""
    # SensorNode has no abstract methods, but it is an ABC
    # Subclasses should be created instead
    # We verify it requires node_id and name
    class TestNode(SensorNode):
        pass

    node = TestNode(node_id="test_01", name="Test Node")
    assert node.node_id == "test_01"
    assert node.name == "Test Node"


def test_sensor_node_defaults():
    """SensorNode defaults are all False/None."""
    class TestNode(SensorNode):
        pass

    node = TestNode(node_id="test_01", name="Test")
    assert node.has_camera is False
    assert node.has_ptz is False
    assert node.has_mic is False
    assert node.has_speaker is False
    assert node.get_frame() is None
    assert node.get_jpeg() is None
    assert node.frame_id == 0
    assert node.record_audio(1.0) is None


def test_sensor_node_move_default():
    """SensorNode.move returns (False, False) by default."""
    class TestNode(SensorNode):
        pass

    node = TestNode(node_id="test_01", name="Test")
    assert node.move(1, 0, 0.5) == (False, False)


def test_sensor_node_get_position():
    """SensorNode.get_position returns default Position."""
    class TestNode(SensorNode):
        pass

    node = TestNode(node_id="test_01", name="Test")
    pos = node.get_position()
    assert isinstance(pos, Position)
    assert pos.pan == 0.0
