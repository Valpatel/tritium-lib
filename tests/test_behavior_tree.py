# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the behavior tree module."""

from __future__ import annotations

import pytest

from tritium_lib.movement.behavior_tree import (
    Action,
    Condition,
    Cooldown,
    Inverter,
    Parallel,
    Repeater,
    Selector,
    Sequence,
    Status,
    make_civilian_tree,
    make_hostile_tree,
    make_patrol_tree,
    make_vehicle_tree,
)


# ---------------------------------------------------------------------------
# Composite tests
# ---------------------------------------------------------------------------

class TestSequence:
    def test_all_succeed(self):
        seq = Sequence([
            Action(lambda ctx: Status.SUCCESS),
            Action(lambda ctx: Status.SUCCESS),
            Action(lambda ctx: Status.SUCCESS),
        ])
        assert seq.tick({}) == Status.SUCCESS

    def test_fail_on_first_failure(self):
        calls = []
        def track(n):
            def fn(ctx):
                calls.append(n)
                return Status.SUCCESS if n < 2 else Status.FAILURE
            return fn

        seq = Sequence([Action(track(1)), Action(track(2)), Action(track(3))])
        assert seq.tick({}) == Status.FAILURE
        assert calls == [1, 2], "Should stop after child 2 fails"

    def test_running_pauses(self):
        seq = Sequence([
            Action(lambda ctx: Status.SUCCESS),
            Action(lambda ctx: Status.RUNNING),
            Action(lambda ctx: Status.SUCCESS),
        ])
        assert seq.tick({}) == Status.RUNNING

    def test_resumes_from_running_child(self):
        call_count = [0]
        def counting(ctx):
            call_count[0] += 1
            return Status.SUCCESS
        first = Action(counting)

        attempts = [0]
        def eventually_succeed(ctx):
            attempts[0] += 1
            return Status.SUCCESS if attempts[0] >= 2 else Status.RUNNING
        second = Action(eventually_succeed)

        seq = Sequence([first, second])

        assert seq.tick({}) == Status.RUNNING
        assert call_count[0] == 1
        # Second tick resumes from running child (index 1), skips first
        assert seq.tick({}) == Status.SUCCESS
        assert call_count[0] == 1  # first child not re-ticked


class TestSelector:
    def test_stops_at_first_success(self):
        calls = []
        def track(n):
            def fn(ctx):
                calls.append(n)
                return Status.FAILURE if n == 1 else Status.SUCCESS
            return fn

        sel = Selector([Action(track(1)), Action(track(2)), Action(track(3))])
        assert sel.tick({}) == Status.SUCCESS
        assert calls == [1, 2], "Should stop after child 2 succeeds"

    def test_all_fail(self):
        sel = Selector([
            Action(lambda ctx: Status.FAILURE),
            Action(lambda ctx: Status.FAILURE),
        ])
        assert sel.tick({}) == Status.FAILURE

    def test_running_pauses(self):
        sel = Selector([
            Action(lambda ctx: Status.FAILURE),
            Action(lambda ctx: Status.RUNNING),
        ])
        assert sel.tick({}) == Status.RUNNING


class TestParallel:
    def test_all_succeed(self):
        par = Parallel([
            Action(lambda ctx: Status.SUCCESS),
            Action(lambda ctx: Status.SUCCESS),
        ])
        assert par.tick({}) == Status.SUCCESS

    def test_threshold(self):
        par = Parallel([
            Action(lambda ctx: Status.SUCCESS),
            Action(lambda ctx: Status.FAILURE),
            Action(lambda ctx: Status.SUCCESS),
        ], threshold=2)
        assert par.tick({}) == Status.SUCCESS

    def test_failure_when_threshold_impossible(self):
        par = Parallel([
            Action(lambda ctx: Status.FAILURE),
            Action(lambda ctx: Status.FAILURE),
            Action(lambda ctx: Status.RUNNING),
        ], threshold=2)
        assert par.tick({}) == Status.FAILURE

    def test_running_when_pending(self):
        par = Parallel([
            Action(lambda ctx: Status.SUCCESS),
            Action(lambda ctx: Status.RUNNING),
        ], threshold=2)
        assert par.tick({}) == Status.RUNNING


# ---------------------------------------------------------------------------
# Decorator tests
# ---------------------------------------------------------------------------

class TestInverter:
    def test_flips_success_to_failure(self):
        inv = Inverter(Action(lambda ctx: Status.SUCCESS))
        assert inv.tick({}) == Status.FAILURE

    def test_flips_failure_to_success(self):
        inv = Inverter(Action(lambda ctx: Status.FAILURE))
        assert inv.tick({}) == Status.SUCCESS

    def test_running_passes_through(self):
        inv = Inverter(Action(lambda ctx: Status.RUNNING))
        assert inv.tick({}) == Status.RUNNING


class TestRepeater:
    def test_repeat_n_times(self):
        count = [0]
        def inc(ctx):
            count[0] += 1
            return Status.SUCCESS
        rep = Repeater(Action(inc), count=3)
        # First two ticks return RUNNING
        assert rep.tick({}) == Status.RUNNING
        assert rep.tick({}) == Status.RUNNING
        # Third tick completes
        assert rep.tick({}) == Status.SUCCESS
        assert count[0] == 3

    def test_stops_on_failure(self):
        rep = Repeater(Action(lambda ctx: Status.FAILURE), count=5)
        assert rep.tick({}) == Status.FAILURE


class TestCooldown:
    def test_prevents_rapid_re_execution(self):
        t = [0.0]
        cd = Cooldown(Action(lambda ctx: Status.SUCCESS), seconds=10.0)

        ctx = {"time": t[0]}
        assert cd.tick(ctx) == Status.SUCCESS  # first call goes through

        t[0] = 5.0
        ctx["time"] = t[0]
        assert cd.tick(ctx) == Status.FAILURE  # within cooldown

        t[0] = 11.0
        ctx["time"] = t[0]
        assert cd.tick(ctx) == Status.SUCCESS  # cooldown expired

    def test_running_does_not_trigger_cooldown(self):
        t = [0.0]
        cd = Cooldown(Action(lambda ctx: Status.RUNNING), seconds=10.0)
        ctx = {"time": t[0]}
        assert cd.tick(ctx) == Status.RUNNING
        t[0] = 1.0
        ctx["time"] = t[0]
        # Should still be able to tick since RUNNING doesn't set cooldown
        assert cd.tick(ctx) == Status.RUNNING


# ---------------------------------------------------------------------------
# Leaf node tests
# ---------------------------------------------------------------------------

class TestAction:
    def test_none_becomes_success(self):
        act = Action(lambda ctx: None)
        assert act.tick({}) == Status.SUCCESS

    def test_passes_context(self):
        def check(ctx):
            return Status.SUCCESS if ctx.get("ready") else Status.FAILURE
        act = Action(check)
        assert act.tick({"ready": True}) == Status.SUCCESS
        assert act.tick({"ready": False}) == Status.FAILURE


class TestCondition:
    def test_truthy_is_success(self):
        cond = Condition(lambda ctx: True)
        assert cond.tick({}) == Status.SUCCESS

    def test_falsy_is_failure(self):
        cond = Condition(lambda ctx: False)
        assert cond.tick({}) == Status.FAILURE

    def test_uses_context(self):
        cond = Condition(lambda ctx: ctx.get("armed", False))
        assert cond.tick({"armed": True}) == Status.SUCCESS
        assert cond.tick({}) == Status.FAILURE


# ---------------------------------------------------------------------------
# Pre-built tree tests
# ---------------------------------------------------------------------------

class TestCivilianTree:
    def test_wander_when_no_threat(self):
        tree = make_civilian_tree()
        ctx: dict = {}
        tree.tick(ctx)
        assert ctx["decision"] == "wander"

    def test_flee_when_threat(self):
        tree = make_civilian_tree()
        ctx: dict = {"threats": ["enemy_1"], "health": 1.0}
        tree.tick(ctx)
        assert ctx["decision"] == "flee"

    def test_hide_when_hurt_and_threatened(self):
        tree = make_civilian_tree()
        ctx: dict = {"threats": ["enemy_1"], "health": 0.1, "retreat_threshold": 0.3}
        tree.tick(ctx)
        assert ctx["decision"] == "hide"

    def test_resume_wander_after_calm(self):
        tree = make_civilian_tree()
        # First: flee
        ctx: dict = {"threats": ["enemy_1"], "health": 1.0}
        tree.tick(ctx)
        assert ctx["decision"] == "flee"
        # Then: calm down
        ctx2: dict = {"threats": [], "recently_threatened": False}
        tree.tick(ctx2)
        assert ctx2["decision"] == "wander"


class TestPatrolTree:
    def test_idle_by_default(self):
        tree = make_patrol_tree()
        ctx: dict = {}
        tree.tick(ctx)
        assert ctx["decision"] == "idle"

    def test_patrol_with_waypoints(self):
        tree = make_patrol_tree()
        ctx: dict = {"waypoints": [(0, 0), (1, 1)]}
        tree.tick(ctx)
        assert ctx["decision"] == "patrol"

    def test_detect_and_pursue(self):
        tree = make_patrol_tree()
        ctx: dict = {"threats": ["intruder"], "threat_in_range": False}
        tree.tick(ctx)
        assert ctx["decision"] == "pursue"

    def test_engage_when_in_range(self):
        tree = make_patrol_tree()
        ctx: dict = {"threats": ["intruder"], "threat_in_range": True}
        tree.tick(ctx)
        assert ctx["decision"] == "engage"


class TestVehicleTree:
    def test_pick_destination_default(self):
        tree = make_vehicle_tree()
        ctx: dict = {}
        tree.tick(ctx)
        assert ctx["decision"] == "pick_destination"

    def test_drive_with_waypoints(self):
        tree = make_vehicle_tree()
        ctx: dict = {"waypoints": [(10, 20)]}
        tree.tick(ctx)
        assert ctx["decision"] == "drive"

    def test_park_at_destination(self):
        tree = make_vehicle_tree()
        ctx: dict = {"at_destination": True, "waypoints": [(10, 20)]}
        tree.tick(ctx)
        assert ctx["decision"] == "park"


class TestHostileTree:
    def test_regroup_default(self):
        tree = make_hostile_tree()
        ctx: dict = {"time": 0.0}
        tree.tick(ctx)
        assert ctx["decision"] == "regroup"

    def test_retreat_when_hurt(self):
        tree = make_hostile_tree()
        ctx: dict = {"health": 0.1, "retreat_threshold": 0.3, "time": 0.0}
        tree.tick(ctx)
        assert ctx["decision"] == "retreat"

    def test_engage_threat_in_range(self):
        tree = make_hostile_tree()
        ctx: dict = {"health": 1.0, "threat_in_range": True, "time": 100.0}
        tree.tick(ctx)
        # First tick with fresh cooldown -> seek_cover
        assert ctx["decision"] == "seek_cover"
        # Tick again within cooldown -> engage
        ctx2: dict = {"health": 1.0, "threat_in_range": True, "time": 101.0}
        tree.tick(ctx2)
        assert ctx2["decision"] == "engage"

    def test_approach_distant_threat(self):
        tree = make_hostile_tree()
        ctx: dict = {
            "health": 1.0,
            "threats": ["target_1"],
            "threat_in_range": False,
            "time": 0.0,
        }
        tree.tick(ctx)
        assert ctx["decision"] == "approach"
