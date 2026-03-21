# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Unit tests for behavior tree primitives and pre-built trees.

Tests every node type (Sequence, Selector, Parallel, Inverter, Repeater,
Cooldown, Action, Condition) and all pre-built trees (patrol, hostile,
civilian, vehicle).
"""

from __future__ import annotations

import pytest

from tritium_lib.sim_engine.ai.behavior_tree import (
    Status,
    Node,
    Sequence,
    Selector,
    Parallel,
    Inverter,
    Repeater,
    Cooldown,
    Action,
    Condition,
    make_patrol_tree,
    make_hostile_tree,
    make_civilian_tree,
    make_vehicle_tree,
    make_friendly_tree,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _always_success(ctx: dict) -> Status:
    return Status.SUCCESS

def _always_failure(ctx: dict) -> Status:
    return Status.FAILURE

def _always_running(ctx: dict) -> Status:
    return Status.RUNNING

def _return_none(ctx: dict) -> Status | None:
    return None

def _counter_action(ctx: dict) -> Status:
    ctx["count"] = ctx.get("count", 0) + 1
    return Status.SUCCESS

def _true_predicate(ctx: dict) -> bool:
    return True

def _false_predicate(ctx: dict) -> bool:
    return False


# ---------------------------------------------------------------------------
# Action node
# ---------------------------------------------------------------------------

class TestAction:
    def test_success(self) -> None:
        node = Action(_always_success)
        assert node.tick({}) == Status.SUCCESS

    def test_failure(self) -> None:
        node = Action(_always_failure)
        assert node.tick({}) == Status.FAILURE

    def test_running(self) -> None:
        node = Action(_always_running)
        assert node.tick({}) == Status.RUNNING

    def test_none_returns_success(self) -> None:
        node = Action(_return_none)
        assert node.tick({}) == Status.SUCCESS

    def test_repr(self) -> None:
        node = Action(_always_success, name="my_action")
        assert "my_action" in repr(node)


# ---------------------------------------------------------------------------
# Condition node
# ---------------------------------------------------------------------------

class TestCondition:
    def test_true_returns_success(self) -> None:
        node = Condition(_true_predicate)
        assert node.tick({}) == Status.SUCCESS

    def test_false_returns_failure(self) -> None:
        node = Condition(_false_predicate)
        assert node.tick({}) == Status.FAILURE

    def test_repr(self) -> None:
        node = Condition(_true_predicate, name="check_ammo")
        assert "check_ammo" in repr(node)


# ---------------------------------------------------------------------------
# Sequence node
# ---------------------------------------------------------------------------

class TestSequence:
    def test_all_success(self) -> None:
        seq = Sequence([Action(_always_success), Action(_always_success)])
        assert seq.tick({}) == Status.SUCCESS

    def test_first_failure_aborts(self) -> None:
        ctx = {}
        seq = Sequence([Action(_always_failure), Action(_counter_action)])
        assert seq.tick(ctx) == Status.FAILURE
        assert ctx.get("count", 0) == 0  # second never ran

    def test_running_pauses(self) -> None:
        seq = Sequence([Action(_always_running), Action(_always_success)])
        assert seq.tick({}) == Status.RUNNING

    def test_resume_after_running(self) -> None:
        calls = []
        def first(ctx: dict) -> Status:
            calls.append("first")
            return Status.RUNNING if len(calls) < 2 else Status.SUCCESS
        def second(ctx: dict) -> Status:
            calls.append("second")
            return Status.SUCCESS

        seq = Sequence([Action(first), Action(second)])
        assert seq.tick({}) == Status.RUNNING
        # Second tick should resume from the running child
        assert seq.tick({}) == Status.SUCCESS

    def test_reset(self) -> None:
        seq = Sequence([Action(_always_running)])
        seq.tick({})
        assert seq._running_idx == 0  # only one child
        seq.reset()
        assert seq._running_idx == 0

    def test_empty_sequence_succeeds(self) -> None:
        seq = Sequence([])
        assert seq.tick({}) == Status.SUCCESS


# ---------------------------------------------------------------------------
# Selector node
# ---------------------------------------------------------------------------

class TestSelector:
    def test_first_success_returns(self) -> None:
        sel = Selector([Action(_always_success), Action(_always_failure)])
        assert sel.tick({}) == Status.SUCCESS

    def test_all_fail(self) -> None:
        sel = Selector([Action(_always_failure), Action(_always_failure)])
        assert sel.tick({}) == Status.FAILURE

    def test_running_pauses(self) -> None:
        sel = Selector([Action(_always_running), Action(_always_success)])
        assert sel.tick({}) == Status.RUNNING

    def test_skips_failure_tries_next(self) -> None:
        ctx = {}
        sel = Selector([Action(_always_failure), Action(_counter_action)])
        assert sel.tick(ctx) == Status.SUCCESS
        assert ctx["count"] == 1

    def test_empty_selector_fails(self) -> None:
        sel = Selector([])
        assert sel.tick({}) == Status.FAILURE


# ---------------------------------------------------------------------------
# Parallel node
# ---------------------------------------------------------------------------

class TestParallel:
    def test_all_succeed_with_default_threshold(self) -> None:
        par = Parallel([Action(_always_success), Action(_always_success)])
        assert par.tick({}) == Status.SUCCESS

    def test_one_fail_with_threshold_1(self) -> None:
        par = Parallel([Action(_always_success), Action(_always_failure)], threshold=1)
        assert par.tick({}) == Status.SUCCESS

    def test_all_fail(self) -> None:
        par = Parallel([Action(_always_failure), Action(_always_failure)])
        assert par.tick({}) == Status.FAILURE

    def test_running_mixed(self) -> None:
        par = Parallel([Action(_always_running), Action(_always_success)], threshold=2)
        assert par.tick({}) == Status.RUNNING

    def test_failure_impossible_threshold(self) -> None:
        # threshold=2, but one already failed, so threshold can never be met
        par = Parallel([Action(_always_failure), Action(_always_running)], threshold=2)
        assert par.tick({}) == Status.FAILURE


# ---------------------------------------------------------------------------
# Inverter decorator
# ---------------------------------------------------------------------------

class TestInverter:
    def test_invert_success(self) -> None:
        inv = Inverter(Action(_always_success))
        assert inv.tick({}) == Status.FAILURE

    def test_invert_failure(self) -> None:
        inv = Inverter(Action(_always_failure))
        assert inv.tick({}) == Status.SUCCESS

    def test_running_passes_through(self) -> None:
        inv = Inverter(Action(_always_running))
        assert inv.tick({}) == Status.RUNNING


# ---------------------------------------------------------------------------
# Repeater decorator
# ---------------------------------------------------------------------------

class TestRepeater:
    def test_repeats_n_times(self) -> None:
        ctx = {}
        rep = Repeater(Action(_counter_action), count=3)
        # Each tick processes one iteration
        assert rep.tick(ctx) == Status.RUNNING  # 1st
        assert rep.tick(ctx) == Status.RUNNING  # 2nd
        assert rep.tick(ctx) == Status.SUCCESS  # 3rd - done

    def test_infinite_repeater_stays_running(self) -> None:
        rep = Repeater(Action(_always_success), count=0)
        for _ in range(10):
            assert rep.tick({}) == Status.RUNNING

    def test_failure_stops_repeating(self) -> None:
        rep = Repeater(Action(_always_failure), count=5)
        assert rep.tick({}) == Status.FAILURE

    def test_reset_clears_count(self) -> None:
        ctx = {}
        rep = Repeater(Action(_counter_action), count=3)
        rep.tick(ctx)
        rep.reset()
        assert rep._done == 0


# ---------------------------------------------------------------------------
# Cooldown decorator
# ---------------------------------------------------------------------------

class TestCooldown:
    def test_first_tick_runs(self) -> None:
        cd = Cooldown(Action(_always_success), seconds=5.0)
        assert cd.tick({"time": 0.0}) == Status.SUCCESS

    def test_blocked_during_cooldown(self) -> None:
        cd = Cooldown(Action(_always_success), seconds=5.0)
        cd.tick({"time": 0.0})  # triggers cooldown
        assert cd.tick({"time": 2.0}) == Status.FAILURE  # still cooling

    def test_available_after_cooldown(self) -> None:
        cd = Cooldown(Action(_always_success), seconds=5.0)
        cd.tick({"time": 0.0})
        assert cd.tick({"time": 6.0}) == Status.SUCCESS

    def test_reset_clears_cooldown(self) -> None:
        cd = Cooldown(Action(_always_success), seconds=5.0)
        cd.tick({"time": 0.0})
        cd.reset()
        assert cd.tick({"time": 1.0}) == Status.SUCCESS  # no longer blocked


# ---------------------------------------------------------------------------
# Pre-built trees — make_vehicle_tree
# ---------------------------------------------------------------------------

class TestVehicleTree:
    def test_at_destination_parks(self) -> None:
        tree = make_vehicle_tree()
        ctx = {"at_destination": True, "waypoints": True}
        tree.tick(ctx)
        assert ctx["decision"] == "park"

    def test_has_waypoints_drives(self) -> None:
        tree = make_vehicle_tree()
        ctx = {"at_destination": False, "waypoints": True}
        tree.tick(ctx)
        assert ctx["decision"] == "drive"

    def test_no_waypoints_picks_destination(self) -> None:
        tree = make_vehicle_tree()
        ctx = {"at_destination": False, "waypoints": False}
        tree.tick(ctx)
        assert ctx["decision"] == "pick_destination"


# ---------------------------------------------------------------------------
# Pre-built trees — make_civilian_tree edge cases
# ---------------------------------------------------------------------------

class TestCivilianTreeEdgeCases:
    def test_low_health_with_threat_hides(self) -> None:
        tree = make_civilian_tree()
        ctx = {
            "threats": [{"id": "h1"}],
            "health": 0.1,
            "retreat_threshold": 0.3,
            "recently_threatened": True,
        }
        tree.tick(ctx)
        assert ctx["decision"] == "hide"

    def test_healthy_with_threat_flees(self) -> None:
        tree = make_civilian_tree()
        ctx = {
            "threats": [{"id": "h1"}],
            "health": 1.0,
            "retreat_threshold": 0.3,
            "recently_threatened": True,
        }
        tree.tick(ctx)
        assert ctx["decision"] == "flee"

    def test_no_threat_calm_wanders(self) -> None:
        tree = make_civilian_tree()
        ctx = {
            "threats": [],
            "health": 1.0,
            "recently_threatened": False,
        }
        tree.tick(ctx)
        assert ctx["decision"] == "wander"


# ---------------------------------------------------------------------------
# Pre-built trees — make_hostile_tree edge cases
# ---------------------------------------------------------------------------

class TestHostileTreeEdgeCases:
    def test_hurt_retreats(self) -> None:
        tree = make_hostile_tree()
        ctx = {
            "threats": [{"id": "u1"}],
            "threat_in_range": True,
            "health": 0.1,
            "retreat_threshold": 0.3,
            "time": 0.0,
        }
        tree.tick(ctx)
        assert ctx["decision"] == "retreat"

    def test_threat_in_range_seeks_cover_first(self) -> None:
        tree = make_hostile_tree()
        ctx = {
            "threats": [{"id": "u1"}],
            "threat_in_range": True,
            "health": 1.0,
            "retreat_threshold": 0.3,
            "time": 0.0,
        }
        tree.tick(ctx)
        assert ctx["decision"] == "seek_cover"

    def test_threat_in_range_engages_after_cooldown(self) -> None:
        tree = make_hostile_tree()
        ctx = {
            "threats": [{"id": "u1"}],
            "threat_in_range": True,
            "health": 1.0,
            "retreat_threshold": 0.3,
            "time": 0.0,
        }
        tree.tick(ctx)  # seek_cover (cooldown starts)
        ctx["time"] = 1.0  # within cooldown
        tree.tick(ctx)
        assert ctx["decision"] == "engage"

    def test_no_threats_regroups(self) -> None:
        tree = make_hostile_tree()
        ctx = {
            "threats": [],
            "threat_in_range": False,
            "health": 1.0,
            "retreat_threshold": 0.3,
            "time": 0.0,
        }
        tree.tick(ctx)
        assert ctx["decision"] == "regroup"

    def test_threat_detected_but_out_of_range_approaches(self) -> None:
        tree = make_hostile_tree()
        ctx = {
            "threats": [{"id": "u1"}],
            "threat_in_range": False,
            "health": 1.0,
            "retreat_threshold": 0.3,
            "time": 100.0,  # well past any cooldown
        }
        tree.tick(ctx)
        assert ctx["decision"] == "approach"


# ---------------------------------------------------------------------------
# Pre-built trees — make_patrol_tree completeness
# ---------------------------------------------------------------------------

class TestPatrolTreeCompleteness:
    def test_no_threat_no_waypoints_idles(self) -> None:
        tree = make_patrol_tree()
        ctx = {"threats": [], "threat_in_range": False, "waypoints": False}
        tree.tick(ctx)
        assert ctx["decision"] == "idle"

    def test_threat_in_range_engages(self) -> None:
        tree = make_patrol_tree()
        ctx = {"threats": [{"id": "h1"}], "threat_in_range": True, "waypoints": True}
        tree.tick(ctx)
        assert ctx["decision"] == "engage"

    def test_threat_out_of_range_pursues(self) -> None:
        tree = make_patrol_tree()
        ctx = {"threats": [{"id": "h1"}], "threat_in_range": False, "waypoints": True}
        tree.tick(ctx)
        assert ctx["decision"] == "pursue"

    def test_no_threat_has_waypoints_patrols(self) -> None:
        tree = make_patrol_tree()
        ctx = {"threats": [], "threat_in_range": False, "waypoints": True}
        tree.tick(ctx)
        assert ctx["decision"] == "patrol"


# ---------------------------------------------------------------------------
# Pre-built trees — make_friendly_tree (used by UnitAISystem for friendly infantry)
# ---------------------------------------------------------------------------

class TestFriendlyTree:
    def test_hurt_retreats(self) -> None:
        tree = make_friendly_tree()
        ctx = {
            "threats": [{"id": "h1"}],
            "threat_in_range": True,
            "health": 0.1,
            "retreat_threshold": 0.3,
            "time": 0.0,
        }
        tree.tick(ctx)
        assert ctx["decision"] == "retreat"

    def test_threat_in_range_seeks_cover_first(self) -> None:
        tree = make_friendly_tree()
        ctx = {
            "threats": [{"id": "h1"}],
            "threat_in_range": True,
            "health": 1.0,
            "retreat_threshold": 0.3,
            "time": 0.0,
        }
        tree.tick(ctx)
        assert ctx["decision"] == "seek_cover"

    def test_threat_in_range_engages_after_cooldown(self) -> None:
        tree = make_friendly_tree()
        ctx = {
            "threats": [{"id": "h1"}],
            "threat_in_range": True,
            "health": 1.0,
            "retreat_threshold": 0.3,
            "time": 0.0,
        }
        tree.tick(ctx)  # seek_cover, cooldown starts
        ctx["time"] = 1.0  # within 6s cooldown
        tree.tick(ctx)
        assert ctx["decision"] == "engage"

    def test_threat_detected_out_of_range_approaches(self) -> None:
        tree = make_friendly_tree()
        ctx = {
            "threats": [{"id": "h1"}],
            "threat_in_range": False,
            "health": 1.0,
            "retreat_threshold": 0.3,
            "time": 0.0,
            "waypoints": True,
        }
        tree.tick(ctx)
        assert ctx["decision"] == "approach"

    def test_no_threats_with_waypoints_patrols(self) -> None:
        tree = make_friendly_tree()
        ctx = {
            "threats": [],
            "threat_in_range": False,
            "health": 1.0,
            "retreat_threshold": 0.3,
            "time": 0.0,
            "waypoints": True,
        }
        tree.tick(ctx)
        assert ctx["decision"] == "patrol"

    def test_no_threats_no_waypoints_idles(self) -> None:
        tree = make_friendly_tree()
        ctx = {
            "threats": [],
            "threat_in_range": False,
            "health": 1.0,
            "retreat_threshold": 0.3,
            "time": 0.0,
            "waypoints": False,
        }
        tree.tick(ctx)
        assert ctx["decision"] == "idle"
