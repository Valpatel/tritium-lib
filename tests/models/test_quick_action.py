# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for QuickAction model."""

import time

import pytest

from tritium_lib.models.quick_action import (
    QuickAction,
    QuickActionLog,
    QuickActionType,
)


class TestQuickAction:
    def test_create_investigate(self):
        action = QuickAction(
            action_type=QuickActionType.INVESTIGATE,
            target_id="ble_aa:bb:cc:dd:ee:ff",
            operator="operator1",
        )
        assert action.action_type == QuickActionType.INVESTIGATE
        assert action.target_id == "ble_aa:bb:cc:dd:ee:ff"
        assert action.operator == "operator1"
        assert action.action_id  # auto-generated UUID
        assert action.timestamp > 0

    def test_create_classify_with_params(self):
        action = QuickAction(
            action_type=QuickActionType.CLASSIFY,
            target_id="det_person_1",
            params={"alliance": "hostile"},
            operator="admin",
        )
        assert action.params["alliance"] == "hostile"

    def test_create_watch(self):
        action = QuickAction(
            action_type=QuickActionType.WATCH,
            target_id="wifi_test_bssid",
        )
        assert action.operator == "system"  # default

    def test_create_track(self):
        action = QuickAction(
            action_type=QuickActionType.TRACK,
            target_id="mesh_node_1",
            params={"prediction_cone": True, "minutes_ahead": 5},
        )
        assert action.params["prediction_cone"] is True

    def test_to_event(self):
        action = QuickAction(
            action_type=QuickActionType.ESCALATE,
            target_id="ble_test",
            operator="op1",
            notes="Suspicious behavior",
        )
        event = action.to_event()
        assert event["event_type"] == "quick_action"
        assert event["action_type"] == "escalate"
        assert event["target_id"] == "ble_test"
        assert event["operator"] == "op1"
        assert event["notes"] == "Suspicious behavior"

    def test_action_types_enum(self):
        types = [t.value for t in QuickActionType]
        assert "investigate" in types
        assert "watch" in types
        assert "classify" in types
        assert "track" in types
        assert "dismiss" in types
        assert "escalate" in types
        assert "annotate" in types


class TestQuickActionLog:
    def test_add_and_recent(self):
        log = QuickActionLog()
        a1 = QuickAction(action_type=QuickActionType.WATCH, target_id="t1")
        a2 = QuickAction(action_type=QuickActionType.TRACK, target_id="t2")
        log.add(a1)
        log.add(a2)
        recent = log.recent(10)
        assert len(recent) == 2
        assert recent[0].target_id == "t2"  # newest first

    def test_for_target(self):
        log = QuickActionLog()
        log.add(QuickAction(action_type=QuickActionType.WATCH, target_id="t1"))
        log.add(QuickAction(action_type=QuickActionType.TRACK, target_id="t2"))
        log.add(QuickAction(action_type=QuickActionType.CLASSIFY, target_id="t1"))
        result = log.for_target("t1")
        assert len(result) == 2

    def test_by_type(self):
        log = QuickActionLog()
        log.add(QuickAction(action_type=QuickActionType.WATCH, target_id="t1"))
        log.add(QuickAction(action_type=QuickActionType.WATCH, target_id="t2"))
        log.add(QuickAction(action_type=QuickActionType.TRACK, target_id="t3"))
        result = log.by_type(QuickActionType.WATCH)
        assert len(result) == 2

    def test_max_size_trim(self):
        log = QuickActionLog(max_size=5)
        for i in range(10):
            log.add(QuickAction(action_type=QuickActionType.WATCH, target_id=f"t{i}"))
        assert len(log.actions) == 5
        assert log.actions[0].target_id == "t5"  # oldest remaining
