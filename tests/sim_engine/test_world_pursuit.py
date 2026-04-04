# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for sim_engine.world.pursuit — intercept waypoint computation."""

import math
import pytest
from unittest.mock import MagicMock

from tritium_lib.sim_engine.world.pursuit import PursuitSystem


def _make_target(tid, alliance, pos, speed=0.0, heading=0.0, status="active"):
    t = MagicMock()
    t.target_id = tid
    t.alliance = alliance
    t.position = pos
    t.speed = speed
    t.heading = heading
    t.status = status
    return t


class TestPursuitSystem:
    """Tests for PursuitSystem intercept waypoints."""

    def test_assign_and_get(self):
        ps = PursuitSystem()
        ps.assign("rover_1", "hostile_1")
        assert ps.get_assignment("rover_1") == "hostile_1"

    def test_no_assignment(self):
        ps = PursuitSystem()
        assert ps.get_assignment("rover_1") is None

    def test_tick_computes_intercept_for_hostile(self):
        ps = PursuitSystem()
        hostile = _make_target("h1", "hostile", (50.0, 0.0), speed=5.0, heading=0.0)
        targets = {"h1": hostile}
        ps.tick(0.1, targets)
        intercept = ps.get_intercept_point("h1")
        assert intercept is not None
        # Heading 0 = north (+y), so predicted y > 0
        assert intercept[1] > 0.0

    def test_stationary_hostile_intercept_at_position(self):
        ps = PursuitSystem()
        hostile = _make_target("h1", "hostile", (30.0, 20.0), speed=0.0)
        targets = {"h1": hostile}
        ps.tick(0.1, targets)
        intercept = ps.get_intercept_point("h1")
        assert intercept == (30.0, 20.0)

    def test_auto_assign_friendly_to_hostile(self):
        ps = PursuitSystem()
        hostile = _make_target("h1", "hostile", (50.0, 0.0), speed=3.0)
        friendly = _make_target("f1", "friendly", (0.0, 0.0), speed=5.0)
        targets = {"h1": hostile, "f1": friendly}
        ps.tick(0.1, targets)
        assert ps.get_pursuit_target("f1") == "h1"

    def test_pursuit_waypoint(self):
        ps = PursuitSystem()
        hostile = _make_target("h1", "hostile", (50.0, 0.0), speed=5.0, heading=90.0)
        friendly = _make_target("f1", "friendly", (0.0, 0.0), speed=10.0)
        targets = {"h1": hostile, "f1": friendly}
        ps.tick(0.1, targets)
        wp = ps.get_pursuit_waypoint("f1")
        assert wp is not None

    def test_remove_unit(self):
        ps = PursuitSystem()
        ps.assign("f1", "h1")
        ps._intercept_points["h1"] = (50.0, 50.0)
        ps.remove_unit("h1")
        assert ps.get_intercept_point("h1") is None
        ps.remove_unit("f1")
        assert ps.get_pursuit_target("f1") is None

    def test_reset(self):
        ps = PursuitSystem()
        ps.assign("f1", "h1")
        ps._intercept_points["h1"] = (50.0, 50.0)
        ps.reset()
        assert ps.get_assignment("f1") is None
        assert ps.get_intercept_point("h1") is None

    def test_reassign_when_target_gone(self):
        ps = PursuitSystem()
        h1 = _make_target("h1", "hostile", (50.0, 0.0), speed=3.0)
        h2 = _make_target("h2", "hostile", (30.0, 0.0), speed=3.0)
        f1 = _make_target("f1", "friendly", (0.0, 0.0), speed=5.0)
        # First tick: assign f1 to nearest hostile
        ps.tick(0.1, {"h1": h1, "h2": h2, "f1": f1})
        first_target = ps.get_pursuit_target("f1")
        assert first_target == "h2"  # h2 is closer at (30,0) vs (50,0)
        # Remove h2 -- f1 should re-assign to h1
        ps.tick(0.1, {"h1": h1, "f1": f1})
        assert ps.get_pursuit_target("f1") == "h1"

    def test_inactive_hostiles_ignored(self):
        ps = PursuitSystem()
        h = _make_target("h1", "hostile", (50.0, 0.0), speed=3.0, status="eliminated")
        targets = {"h1": h}
        ps.tick(0.1, targets)
        assert ps.get_intercept_point("h1") is None
