# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for SensorNode base class and Position dataclass."""

import pytest

from tritium_lib.nodes.base import Position, SensorNode


# ── Position dataclass ──────────────────────────────────────────────

class TestPosition:
    """Tests for PTZ Position."""

    def test_defaults(self):
        pos = Position()
        assert pos.pan == 0.0
        assert pos.tilt == 0.0
        assert pos.zoom == 100.0

    def test_can_pan_left_no_limits(self):
        pos = Position(pan=0.0)
        assert pos.can_pan_left is True

    def test_can_pan_left_at_limit(self):
        pos = Position(pan=-90.0, pan_min=-90.0)
        assert pos.can_pan_left is False

    def test_can_pan_left_within_limit(self):
        pos = Position(pan=0.0, pan_min=-90.0)
        assert pos.can_pan_left is True

    def test_can_pan_right_no_limits(self):
        pos = Position(pan=0.0)
        assert pos.can_pan_right is True

    def test_can_pan_right_at_limit(self):
        pos = Position(pan=90.0, pan_max=90.0)
        assert pos.can_pan_right is False

    def test_can_pan_right_within_limit(self):
        pos = Position(pan=0.0, pan_max=90.0)
        assert pos.can_pan_right is True

    def test_can_tilt_up_no_limits(self):
        pos = Position(tilt=0.0)
        assert pos.can_tilt_up is True

    def test_can_tilt_up_at_limit(self):
        pos = Position(tilt=45.0, tilt_max=45.0)
        assert pos.can_tilt_up is False

    def test_can_tilt_down_no_limits(self):
        pos = Position(tilt=0.0)
        assert pos.can_tilt_down is True

    def test_can_tilt_down_at_limit(self):
        pos = Position(tilt=-45.0, tilt_min=-45.0)
        assert pos.can_tilt_down is False


# ── SensorNode defaults ────────────────────────────────────────────

class ConcreteSensorNode(SensorNode):
    """Minimal concrete implementation for testing defaults."""
    pass


class TestSensorNodeDefaults:
    """Tests for SensorNode default behavior (no-op implementations)."""

    def test_node_id_and_name(self):
        node = ConcreteSensorNode("cam-01", "Front Camera")
        assert node.node_id == "cam-01"
        assert node.name == "Front Camera"

    def test_has_camera_false(self):
        node = ConcreteSensorNode("n1", "N1")
        assert node.has_camera is False

    def test_has_ptz_false(self):
        node = ConcreteSensorNode("n1", "N1")
        assert node.has_ptz is False

    def test_has_mic_false(self):
        node = ConcreteSensorNode("n1", "N1")
        assert node.has_mic is False

    def test_has_speaker_false(self):
        node = ConcreteSensorNode("n1", "N1")
        assert node.has_speaker is False

    def test_get_frame_returns_none(self):
        node = ConcreteSensorNode("n1", "N1")
        assert node.get_frame() is None

    def test_get_jpeg_returns_none(self):
        node = ConcreteSensorNode("n1", "N1")
        assert node.get_jpeg() is None

    def test_frame_id_zero(self):
        node = ConcreteSensorNode("n1", "N1")
        assert node.frame_id == 0

    def test_move_returns_false_tuple(self):
        node = ConcreteSensorNode("n1", "N1")
        assert node.move(1, 0, 0.5) == (False, False)

    def test_get_position_default(self):
        node = ConcreteSensorNode("n1", "N1")
        pos = node.get_position()
        assert isinstance(pos, Position)
        assert pos.pan == 0.0

    def test_reset_position_no_error(self):
        node = ConcreteSensorNode("n1", "N1")
        node.reset_position()  # Should not raise

    def test_record_audio_returns_none(self):
        node = ConcreteSensorNode("n1", "N1")
        assert node.record_audio(1.0) is None

    def test_play_audio_no_error(self):
        node = ConcreteSensorNode("n1", "N1")
        node.play_audio(b"\x00\x00")  # Should not raise

    def test_start_stop_lifecycle(self):
        node = ConcreteSensorNode("n1", "N1")
        node.start()  # Should not raise
        node.stop()   # Should not raise
