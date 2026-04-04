# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for sim_engine.world.comms — inter-unit communication signals."""

import time
import pytest
from unittest.mock import MagicMock

from tritium_lib.sim_engine.world.comms import (
    Signal,
    Message,
    UnitComms,
    SIGNAL_DISTRESS,
    SIGNAL_CONTACT,
    SIGNAL_REGROUP,
)


def _make_mock_target(target_id, alliance, position, speed=0.0, heading=0.0):
    """Create a mock SimulationTarget."""
    t = MagicMock()
    t.target_id = target_id
    t.alliance = alliance
    t.position = position
    t.speed = speed
    t.heading = heading
    t.status = "active"
    return t


class TestSignal:
    """Tests for the Signal dataclass."""

    def test_not_expired_immediately(self):
        sig = Signal(
            signal_type="distress",
            sender_id="unit_1",
            sender_alliance="friendly",
            position=(10.0, 20.0),
        )
        assert not sig.expired

    def test_expired_after_ttl(self):
        sig = Signal(
            signal_type="distress",
            sender_id="unit_1",
            sender_alliance="friendly",
            position=(10.0, 20.0),
            created_at=time.time() - 20.0,  # 20 seconds ago
            ttl=10.0,
        )
        assert sig.expired


class TestMessage:
    """Tests for the Message dataclass."""

    def test_not_expired_immediately(self):
        msg = Message(
            sender_id="unit_1",
            content="hello",
            position=(0.0, 0.0),
        )
        assert not msg.expired

    def test_expired_after_ttl(self):
        msg = Message(
            sender_id="unit_1",
            content="hello",
            position=(0.0, 0.0),
            created_at=time.time() - 20.0,
            ttl=5.0,
        )
        assert msg.expired


class TestUnitComms:
    """Tests for UnitComms signal management."""

    def test_broadcast_creates_signal(self):
        comms = UnitComms()
        sig = comms.broadcast("distress", "u1", "friendly", (10.0, 20.0))
        assert isinstance(sig, Signal)
        assert sig.signal_type == "distress"
        assert sig.sender_id == "u1"

    def test_emit_distress(self):
        comms = UnitComms()
        sig = comms.emit_distress("u1", (10.0, 20.0), "friendly")
        assert sig.signal_type == SIGNAL_DISTRESS

    def test_emit_contact(self):
        comms = UnitComms()
        sig = comms.emit_contact("u1", (10.0, 20.0), "friendly", enemy_pos=(30.0, 40.0))
        assert sig.signal_type == SIGNAL_CONTACT
        assert sig.target_position == (30.0, 40.0)

    def test_emit_retreat(self):
        comms = UnitComms()
        sig = comms.emit_retreat("u1", (10.0, 20.0), "friendly")
        assert sig.signal_type == "retreat"

    def test_get_signals_for_unit_same_alliance(self):
        comms = UnitComms()
        comms.broadcast("distress", "u1", "friendly", (10.0, 10.0))
        unit = _make_mock_target("u2", "friendly", (15.0, 15.0))
        signals = comms.get_signals_for_unit(unit)
        assert len(signals) == 1

    def test_get_signals_for_unit_different_alliance(self):
        comms = UnitComms()
        comms.broadcast("distress", "u1", "hostile", (10.0, 10.0))
        unit = _make_mock_target("u2", "friendly", (15.0, 15.0))
        signals = comms.get_signals_for_unit(unit)
        assert len(signals) == 0

    def test_get_signals_excludes_own(self):
        comms = UnitComms()
        comms.broadcast("distress", "u1", "friendly", (10.0, 10.0))
        unit = _make_mock_target("u1", "friendly", (10.0, 10.0))
        signals = comms.get_signals_for_unit(unit)
        assert len(signals) == 0

    def test_get_signals_out_of_range(self):
        comms = UnitComms()
        comms.broadcast("distress", "u1", "friendly", (10.0, 10.0), signal_range=5.0)
        unit = _make_mock_target("u2", "friendly", (100.0, 100.0))
        signals = comms.get_signals_for_unit(unit)
        assert len(signals) == 0

    def test_send_message(self):
        comms = UnitComms()
        msg = comms.send("u1", "hello world", (5.0, 5.0))
        assert isinstance(msg, Message)
        assert msg.content == "hello world"

    def test_get_messages_for(self):
        comms = UnitComms()
        comms.send("u1", "hello", (5.0, 5.0))
        messages = comms.get_messages_for("u2", (8.0, 8.0), "rover")
        assert len(messages) == 1

    def test_get_messages_excludes_own(self):
        comms = UnitComms()
        comms.send("u1", "hello", (5.0, 5.0))
        messages = comms.get_messages_for("u1", (5.0, 5.0), "rover")
        assert len(messages) == 0

    def test_tick_cleans_expired(self):
        comms = UnitComms()
        comms.broadcast(
            "distress", "u1", "friendly", (10.0, 10.0),
            ttl=0.0,
        )
        # Force the created_at to be in the past
        comms._signals[0].created_at = time.time() - 1.0
        comms.tick(0.1, {})
        assert len(comms.get_all_signals()) == 0

    def test_reset(self):
        comms = UnitComms()
        comms.broadcast("distress", "u1", "friendly", (10.0, 10.0))
        comms.send("u1", "hello", (5.0, 5.0))
        comms.reset()
        assert len(comms.get_all_signals()) == 0
        assert len(comms._messages) == 0

    def test_event_bus_publish(self):
        bus = MagicMock()
        comms = UnitComms(event_bus=bus)
        comms.broadcast("distress", "u1", "friendly", (10.0, 20.0))
        bus.publish.assert_called_once()
        args = bus.publish.call_args
        assert args[0][0] == "unit_signal"

    def test_no_event_bus(self):
        comms = UnitComms(event_bus=None)
        # Should not raise
        comms.broadcast("distress", "u1", "friendly", (10.0, 20.0))

    def test_filter_by_signal_type(self):
        comms = UnitComms()
        comms.broadcast("distress", "u1", "friendly", (10.0, 10.0))
        comms.broadcast("contact", "u1", "friendly", (10.0, 10.0))
        unit = _make_mock_target("u2", "friendly", (15.0, 15.0))
        distress_only = comms.get_signals_for_unit(unit, signal_type="distress")
        assert len(distress_only) == 1
        assert distress_only[0].signal_type == "distress"
