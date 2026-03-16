# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the mission objective system."""

from __future__ import annotations

import copy
import pytest

from tritium_lib.sim_engine.objectives import (
    DynamicEvent,
    MissionObjective,
    ObjectiveEngine,
    ObjectiveStatus,
    ObjectiveType,
    TriggerCondition,
    OBJECTIVE_TEMPLATES,
    _evaluate_trigger,
    _count_units_in_radius,
)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------

class TestObjectiveType:
    def test_all_values(self):
        expected = {
            "eliminate", "capture", "defend", "escort", "extract",
            "destroy", "collect", "survive", "stealth", "patrol",
            "rescue", "sabotage",
        }
        assert {t.value for t in ObjectiveType} == expected

    def test_count(self):
        assert len(ObjectiveType) == 12

    def test_str_enum(self):
        assert ObjectiveType.ELIMINATE == "eliminate"


class TestObjectiveStatus:
    def test_all_values(self):
        expected = {"locked", "available", "active", "completed", "failed", "optional"}
        assert {s.value for s in ObjectiveStatus} == expected

    def test_count(self):
        assert len(ObjectiveStatus) == 6


# ---------------------------------------------------------------------------
# TriggerCondition tests
# ---------------------------------------------------------------------------

class TestTriggerCondition:
    def test_create(self):
        t = TriggerCondition("time_elapsed", {"seconds": 10.0})
        assert t.condition_type == "time_elapsed"
        assert t.params["seconds"] == 10.0

    def test_default_params(self):
        t = TriggerCondition("all_hostiles_dead")
        assert t.params == {}


# ---------------------------------------------------------------------------
# MissionObjective tests
# ---------------------------------------------------------------------------

class TestMissionObjective:
    def test_defaults(self):
        obj = MissionObjective(
            objective_id="test",
            name="Test",
            description="A test objective",
            objective_type=ObjectiveType.ELIMINATE,
        )
        assert obj.status == ObjectiveStatus.LOCKED
        assert obj.target_position is None
        assert obj.target_id is None
        assert obj.radius == 20.0
        assert obj.time_limit is None
        assert obj.required is True
        assert obj.points == 100
        assert obj.prerequisites == []
        assert obj.progress == 0.0
        assert obj.on_complete == []

    def test_custom_fields(self):
        obj = MissionObjective(
            objective_id="cap1",
            name="Capture",
            description="Capture the hill",
            objective_type=ObjectiveType.CAPTURE,
            target_position=(50.0, 50.0),
            radius=30.0,
            time_limit=120.0,
            required=False,
            points=250,
            prerequisites=["pre1"],
        )
        assert obj.target_position == (50.0, 50.0)
        assert obj.radius == 30.0
        assert obj.time_limit == 120.0
        assert not obj.required
        assert obj.points == 250


# ---------------------------------------------------------------------------
# DynamicEvent tests
# ---------------------------------------------------------------------------

class TestDynamicEvent:
    def test_create(self):
        ev = DynamicEvent(
            event_id="e1",
            name="Boom",
            trigger=TriggerCondition("time_elapsed", {"seconds": 5.0}),
            actions=[{"action": "spawn_units", "params": {"count": 3}}],
        )
        assert ev.one_shot is True
        assert ev.fired is False
        assert len(ev.actions) == 1


# ---------------------------------------------------------------------------
# Trigger evaluation tests
# ---------------------------------------------------------------------------

class TestEvaluateTrigger:
    def test_time_elapsed_true(self):
        t = TriggerCondition("time_elapsed", {"seconds": 10.0})
        assert _evaluate_trigger(t, {"elapsed": 15.0}) is True

    def test_time_elapsed_false(self):
        t = TriggerCondition("time_elapsed", {"seconds": 10.0})
        assert _evaluate_trigger(t, {"elapsed": 5.0}) is False

    def test_units_killed_true(self):
        t = TriggerCondition("units_killed", {"count": 5})
        assert _evaluate_trigger(t, {"units_killed": 5}) is True

    def test_units_killed_by_alliance(self):
        t = TriggerCondition("units_killed", {"count": 3, "alliance": "hostile"})
        world = {"units_killed_by_alliance": {"hostile": 4, "friendly": 1}}
        assert _evaluate_trigger(t, world) is True

    def test_units_killed_by_alliance_not_enough(self):
        t = TriggerCondition("units_killed", {"count": 3, "alliance": "hostile"})
        world = {"units_killed_by_alliance": {"hostile": 2}}
        assert _evaluate_trigger(t, world) is False

    def test_position_reached(self):
        t = TriggerCondition("position_reached", {"target": (10.0, 10.0), "radius": 5.0})
        world = {"units": [{"id": "u1", "pos": (11.0, 11.0)}]}
        assert _evaluate_trigger(t, world) is True

    def test_position_reached_too_far(self):
        t = TriggerCondition("position_reached", {"target": (10.0, 10.0), "radius": 1.0})
        world = {"units": [{"id": "u1", "pos": (100.0, 100.0)}]}
        assert _evaluate_trigger(t, world) is False

    def test_position_reached_specific_unit(self):
        t = TriggerCondition("position_reached", {"target": (10.0, 10.0), "radius": 5.0, "unit_id": "u2"})
        world = {"units": [
            {"id": "u1", "pos": (10.0, 10.0)},
            {"id": "u2", "pos": (100.0, 100.0)},
        ]}
        assert _evaluate_trigger(t, world) is False

    def test_structure_destroyed(self):
        t = TriggerCondition("structure_destroyed", {"structure_id": "bridge"})
        assert _evaluate_trigger(t, {"structures_destroyed": {"bridge", "wall"}}) is True
        assert _evaluate_trigger(t, {"structures_destroyed": {"wall"}}) is False

    def test_wave_completed(self):
        t = TriggerCondition("wave_completed", {"wave": 3})
        assert _evaluate_trigger(t, {"wave": 3}) is True
        assert _evaluate_trigger(t, {"wave": 2}) is False

    def test_objective_completed(self):
        t = TriggerCondition("objective_completed", {"objective_id": "obj_1"})
        assert _evaluate_trigger(t, {"completed_objectives": {"obj_1"}}) is True
        assert _evaluate_trigger(t, {"completed_objectives": set()}) is False

    def test_unit_health_below(self):
        t = TriggerCondition("unit_health_below", {"unit_id": "hero", "threshold": 0.3})
        world = {"units": [{"id": "hero", "health": 0.2}]}
        assert _evaluate_trigger(t, world) is True
        world = {"units": [{"id": "hero", "health": 0.5}]}
        assert _evaluate_trigger(t, world) is False

    def test_all_hostiles_dead(self):
        t = TriggerCondition("all_hostiles_dead")
        assert _evaluate_trigger(t, {"hostiles_alive": 0}) is True
        assert _evaluate_trigger(t, {"hostiles_alive": 3}) is False

    def test_zone_entered(self):
        t = TriggerCondition("zone_entered", {"zone_center": (50.0, 50.0), "zone_radius": 10.0})
        world = {"units": [{"id": "u1", "pos": (52.0, 52.0)}]}
        assert _evaluate_trigger(t, world) is True

    def test_zone_entered_alliance_filter(self):
        t = TriggerCondition("zone_entered", {
            "zone_center": (50.0, 50.0), "zone_radius": 10.0, "alliance": "hostile",
        })
        world = {"units": [{"id": "u1", "pos": (50.0, 50.0), "alliance": "friendly"}]}
        assert _evaluate_trigger(t, world) is False

    def test_unknown_trigger_returns_false(self):
        t = TriggerCondition("nonexistent_type", {})
        assert _evaluate_trigger(t, {}) is False


# ---------------------------------------------------------------------------
# Count units in radius tests
# ---------------------------------------------------------------------------

class TestCountUnitsInRadius:
    def test_basic(self):
        world = {"units": [
            {"id": "a", "pos": (10.0, 10.0), "alliance": "friendly"},
            {"id": "b", "pos": (10.5, 10.5), "alliance": "friendly"},
            {"id": "c", "pos": (100.0, 100.0), "alliance": "friendly"},
        ]}
        assert _count_units_in_radius((10.0, 10.0), 5.0, world) == 2

    def test_alliance_filter(self):
        world = {"units": [
            {"id": "a", "pos": (0.0, 0.0), "alliance": "friendly"},
            {"id": "b", "pos": (0.0, 0.0), "alliance": "hostile"},
        ]}
        assert _count_units_in_radius((0.0, 0.0), 5.0, world, alliance="friendly") == 1

    def test_empty(self):
        assert _count_units_in_radius((0.0, 0.0), 10.0, {}) == 0


# ---------------------------------------------------------------------------
# ObjectiveEngine — basic operations
# ---------------------------------------------------------------------------

class TestObjectiveEngineBasic:
    def test_add_objective(self):
        engine = ObjectiveEngine()
        obj = MissionObjective(
            objective_id="o1", name="Test", description="...",
            objective_type=ObjectiveType.ELIMINATE,
        )
        engine.add_objective(obj)
        assert "o1" in engine.objectives

    def test_add_event(self):
        engine = ObjectiveEngine()
        ev = DynamicEvent(
            event_id="e1", name="Test",
            trigger=TriggerCondition("time_elapsed", {"seconds": 5.0}),
        )
        engine.add_event(ev)
        assert len(engine.events) == 1

    def test_get_active_empty(self):
        engine = ObjectiveEngine()
        assert engine.get_active() == []

    def test_get_available(self):
        engine = ObjectiveEngine()
        obj = MissionObjective(
            objective_id="o1", name="Test", description="...",
            objective_type=ObjectiveType.ELIMINATE,
            status=ObjectiveStatus.AVAILABLE,
        )
        engine.add_objective(obj)
        assert len(engine.get_available()) == 1

    def test_all_required_complete_empty(self):
        engine = ObjectiveEngine()
        assert engine.all_required_complete() is False

    def test_any_required_failed_empty(self):
        engine = ObjectiveEngine()
        assert engine.any_required_failed() is False

    def test_total_points(self):
        engine = ObjectiveEngine()
        o1 = MissionObjective(
            objective_id="o1", name="A", description="...",
            objective_type=ObjectiveType.ELIMINATE,
            status=ObjectiveStatus.COMPLETED, points=100,
        )
        o2 = MissionObjective(
            objective_id="o2", name="B", description="...",
            objective_type=ObjectiveType.ELIMINATE,
            status=ObjectiveStatus.ACTIVE, points=200,
        )
        engine.add_objective(o1)
        engine.add_objective(o2)
        assert engine.total_points() == 100


# ---------------------------------------------------------------------------
# ObjectiveEngine — unlock and activation
# ---------------------------------------------------------------------------

class TestObjectiveEngineUnlock:
    def test_unlock_no_prereqs(self):
        engine = ObjectiveEngine()
        obj = MissionObjective(
            objective_id="o1", name="Test", description="...",
            objective_type=ObjectiveType.ELIMINATE,
            status=ObjectiveStatus.LOCKED,
        )
        engine.add_objective(obj)
        changes = engine.tick(0.1, {})
        types = [c["type"] for c in changes]
        assert "objective_unlocked" in types
        assert "objective_activated" in types

    def test_unlock_with_prereqs(self):
        engine = ObjectiveEngine()
        o1 = MissionObjective(
            objective_id="o1", name="First", description="...",
            objective_type=ObjectiveType.ELIMINATE,
            status=ObjectiveStatus.COMPLETED,
        )
        o2 = MissionObjective(
            objective_id="o2", name="Second", description="...",
            objective_type=ObjectiveType.CAPTURE,
            prerequisites=["o1"],
        )
        engine.add_objective(o1)
        engine.add_objective(o2)
        changes = engine.tick(0.1, {})
        assert any(c["type"] == "objective_unlocked" and c["objective_id"] == "o2" for c in changes)

    def test_stays_locked_prereqs_incomplete(self):
        engine = ObjectiveEngine()
        o1 = MissionObjective(
            objective_id="o1", name="First", description="...",
            objective_type=ObjectiveType.ELIMINATE,
            status=ObjectiveStatus.ACTIVE,
        )
        o2 = MissionObjective(
            objective_id="o2", name="Second", description="...",
            objective_type=ObjectiveType.CAPTURE,
            prerequisites=["o1"],
        )
        engine.add_objective(o1)
        engine.add_objective(o2)
        changes = engine.tick(0.1, {"total_hostiles": 10, "units_killed": 0})
        assert not any(c.get("objective_id") == "o2" and c["type"] == "objective_unlocked" for c in changes)
        assert o2.status == ObjectiveStatus.LOCKED

    def test_unlock_trigger(self):
        engine = ObjectiveEngine()
        obj = MissionObjective(
            objective_id="o1", name="Test", description="...",
            objective_type=ObjectiveType.ELIMINATE,
            unlock_triggers=[TriggerCondition("time_elapsed", {"seconds": 10.0})],
        )
        engine.add_objective(obj)
        # Not enough time
        engine.tick(5.0, {})
        assert obj.status == ObjectiveStatus.LOCKED
        # Enough time
        engine.tick(6.0, {})
        assert obj.status in (ObjectiveStatus.AVAILABLE, ObjectiveStatus.ACTIVE)


# ---------------------------------------------------------------------------
# ObjectiveEngine — failure
# ---------------------------------------------------------------------------

class TestObjectiveEngineFailure:
    def test_time_limit_failure(self):
        engine = ObjectiveEngine()
        obj = MissionObjective(
            objective_id="o1", name="Timed", description="...",
            objective_type=ObjectiveType.CAPTURE,
            target_position=(50.0, 50.0),
            status=ObjectiveStatus.AVAILABLE,
            time_limit=10.0,
        )
        engine.add_objective(obj)
        # Activate
        engine.tick(0.1, {"units": []})
        assert obj.status == ObjectiveStatus.ACTIVE
        # Exceed time limit
        for _ in range(120):
            engine.tick(0.1, {"units": []})
        assert obj.status == ObjectiveStatus.FAILED

    def test_fail_trigger(self):
        engine = ObjectiveEngine()
        obj = MissionObjective(
            objective_id="o1", name="Stealth", description="...",
            objective_type=ObjectiveType.STEALTH,
            status=ObjectiveStatus.AVAILABLE,
            fail_triggers=[TriggerCondition("units_killed", {"count": 1})],
        )
        engine.add_objective(obj)
        engine.tick(0.1, {"units_killed": 0})
        assert obj.status == ObjectiveStatus.ACTIVE
        changes = engine.tick(0.1, {"units_killed": 2})
        assert any(c["type"] == "objective_failed" for c in changes)

    def test_any_required_failed(self):
        engine = ObjectiveEngine()
        obj = MissionObjective(
            objective_id="o1", name="Critical", description="...",
            objective_type=ObjectiveType.DEFEND,
            status=ObjectiveStatus.FAILED,
            required=True,
        )
        engine.add_objective(obj)
        assert engine.any_required_failed() is True

    def test_stealth_detected_fails(self):
        engine = ObjectiveEngine()
        obj = MissionObjective(
            objective_id="s1", name="Sneak", description="...",
            objective_type=ObjectiveType.STEALTH,
            status=ObjectiveStatus.AVAILABLE,
            target_position=(100.0, 100.0),
        )
        engine.add_objective(obj)
        engine.tick(0.1, {"player_detected": False})
        assert obj.status == ObjectiveStatus.ACTIVE
        changes = engine.tick(0.1, {"player_detected": True})
        assert obj.status == ObjectiveStatus.FAILED


# ---------------------------------------------------------------------------
# ObjectiveEngine — progress and completion
# ---------------------------------------------------------------------------

class TestObjectiveEngineProgress:
    def test_eliminate_progress(self):
        engine = ObjectiveEngine()
        obj = MissionObjective(
            objective_id="e1", name="Kill All", description="...",
            objective_type=ObjectiveType.ELIMINATE,
            status=ObjectiveStatus.AVAILABLE,
        )
        engine.add_objective(obj)
        engine.tick(0.1, {"total_hostiles": 10, "units_killed": 5})
        assert obj.progress == pytest.approx(0.5)

    def test_eliminate_specific_target(self):
        engine = ObjectiveEngine()
        obj = MissionObjective(
            objective_id="e1", name="Kill Boss", description="...",
            objective_type=ObjectiveType.ELIMINATE,
            target_id="boss_1",
            status=ObjectiveStatus.AVAILABLE,
        )
        engine.add_objective(obj)
        engine.tick(0.1, {"dead_unit_ids": set()})
        assert obj.progress == 0.0
        engine.tick(0.1, {"dead_unit_ids": {"boss_1"}})
        assert obj.status == ObjectiveStatus.COMPLETED

    def test_capture_progress(self):
        engine = ObjectiveEngine()
        obj = MissionObjective(
            objective_id="c1", name="Capture Hill", description="...",
            objective_type=ObjectiveType.CAPTURE,
            target_position=(50.0, 50.0),
            radius=20.0,
            status=ObjectiveStatus.AVAILABLE,
        )
        engine.add_objective(obj)
        world = {"units": [
            {"id": "f1", "pos": (50.0, 50.0), "alliance": "friendly"},
            {"id": "f2", "pos": (55.0, 50.0), "alliance": "friendly"},
        ]}
        # Tick many times to accumulate progress
        for _ in range(100):
            engine.tick(0.1, world)
        assert obj.progress > 0.0

    def test_capture_contested(self):
        engine = ObjectiveEngine()
        obj = MissionObjective(
            objective_id="c1", name="Capture Hill", description="...",
            objective_type=ObjectiveType.CAPTURE,
            target_position=(50.0, 50.0),
            radius=20.0,
            status=ObjectiveStatus.AVAILABLE,
        )
        engine.add_objective(obj)
        # Build some progress
        world_clear = {"units": [
            {"id": "f1", "pos": (50.0, 50.0), "alliance": "friendly"},
        ]}
        for _ in range(20):
            engine.tick(0.1, world_clear)
        progress_before = obj.progress
        # Enemies arrive — progress should decay
        world_contested = {"units": [
            {"id": "f1", "pos": (50.0, 50.0), "alliance": "friendly"},
            {"id": "h1", "pos": (50.0, 50.0), "alliance": "hostile"},
        ]}
        engine.tick(1.0, world_contested)
        assert obj.progress < progress_before

    def test_extract_completion(self):
        engine = ObjectiveEngine()
        obj = MissionObjective(
            objective_id="x1", name="Extract", description="...",
            objective_type=ObjectiveType.EXTRACT,
            target_position=(0.0, 0.0),
            radius=10.0,
            status=ObjectiveStatus.AVAILABLE,
        )
        engine.add_objective(obj)
        world = {"units": [
            {"id": "f1", "pos": (3.0, 3.0), "alliance": "friendly"},
        ]}
        engine.tick(0.1, world)
        assert obj.status == ObjectiveStatus.COMPLETED

    def test_destroy_progress(self):
        engine = ObjectiveEngine()
        obj = MissionObjective(
            objective_id="d1", name="Destroy Bridge", description="...",
            objective_type=ObjectiveType.DESTROY,
            target_id="bridge",
            status=ObjectiveStatus.AVAILABLE,
        )
        engine.add_objective(obj)
        engine.tick(0.1, {"structures_destroyed": set()})
        assert obj.progress == 0.0
        engine.tick(0.1, {"structures_destroyed": {"bridge"}})
        assert obj.status == ObjectiveStatus.COMPLETED

    def test_collect_progress(self):
        engine = ObjectiveEngine()
        obj = MissionObjective(
            objective_id="col1", name="Collect Intel", description="...",
            objective_type=ObjectiveType.COLLECT,
            status=ObjectiveStatus.AVAILABLE,
        )
        engine.add_objective(obj)
        engine.tick(0.1, {"items_collected": 3, "items_needed": 5})
        assert obj.progress == pytest.approx(0.6)
        engine.tick(0.1, {"items_collected": 5, "items_needed": 5})
        assert obj.status == ObjectiveStatus.COMPLETED

    def test_survive_progress(self):
        engine = ObjectiveEngine()
        obj = MissionObjective(
            objective_id="sv1", name="Survive", description="...",
            objective_type=ObjectiveType.SURVIVE,
            time_limit=10.0,
            status=ObjectiveStatus.AVAILABLE,
        )
        engine.add_objective(obj)
        # Use enough ticks to clearly exceed the time limit
        for _ in range(60):
            engine.tick(0.2, {})
        # Should be completed (12s elapsed with 10s limit)
        assert obj.status == ObjectiveStatus.COMPLETED

    def test_patrol_progress(self):
        engine = ObjectiveEngine()
        obj = MissionObjective(
            objective_id="p1", name="Patrol", description="...",
            objective_type=ObjectiveType.PATROL,
            status=ObjectiveStatus.AVAILABLE,
        )
        engine.add_objective(obj)
        world = {"patrol_waypoints": ["wp1", "wp2", "wp3"], "patrol_visited": {"wp1"}}
        engine.tick(0.1, world)
        assert obj.progress == pytest.approx(1.0 / 3.0)

    def test_rescue_completion(self):
        engine = ObjectiveEngine()
        obj = MissionObjective(
            objective_id="r1", name="Rescue", description="...",
            objective_type=ObjectiveType.RESCUE,
            target_id="hostage_1",
            status=ObjectiveStatus.AVAILABLE,
        )
        engine.add_objective(obj)
        engine.tick(0.1, {"rescued_units": set()})
        assert obj.progress == 0.0
        engine.tick(0.1, {"rescued_units": {"hostage_1"}})
        assert obj.status == ObjectiveStatus.COMPLETED

    def test_sabotage_completion(self):
        engine = ObjectiveEngine()
        obj = MissionObjective(
            objective_id="sab1", name="Sabotage", description="...",
            objective_type=ObjectiveType.SABOTAGE,
            target_id="reactor",
            status=ObjectiveStatus.AVAILABLE,
        )
        engine.add_objective(obj)
        engine.tick(0.1, {"sabotaged_targets": set()})
        assert obj.progress == 0.0
        engine.tick(0.1, {"sabotaged_targets": {"reactor"}})
        assert obj.status == ObjectiveStatus.COMPLETED


# ---------------------------------------------------------------------------
# ObjectiveEngine — dynamic events
# ---------------------------------------------------------------------------

class TestDynamicEvents:
    def test_event_fires(self):
        engine = ObjectiveEngine()
        ev = DynamicEvent(
            event_id="boom", name="Explosion",
            trigger=TriggerCondition("time_elapsed", {"seconds": 5.0}),
            actions=[{"action": "destroy_structure", "params": {"id": "wall"}}],
        )
        engine.add_event(ev)
        changes = engine.tick(6.0, {})
        fired = [c for c in changes if c["type"] == "event_fired"]
        assert len(fired) == 1
        assert fired[0]["event_id"] == "boom"
        assert ev.fired is True

    def test_one_shot_only_fires_once(self):
        engine = ObjectiveEngine()
        ev = DynamicEvent(
            event_id="e1", name="Once",
            trigger=TriggerCondition("time_elapsed", {"seconds": 1.0}),
            actions=[],
            one_shot=True,
        )
        engine.add_event(ev)
        engine.tick(2.0, {})
        assert ev.fired is True
        changes = engine.tick(1.0, {})
        fired = [c for c in changes if c["type"] == "event_fired"]
        assert len(fired) == 0

    def test_repeating_event(self):
        engine = ObjectiveEngine()
        ev = DynamicEvent(
            event_id="e1", name="Repeating",
            trigger=TriggerCondition("time_elapsed", {"seconds": 1.0}),
            actions=[],
            one_shot=False,
        )
        engine.add_event(ev)
        engine.tick(2.0, {})
        changes = engine.tick(1.0, {})
        fired = [c for c in changes if c["type"] == "event_fired"]
        assert len(fired) == 1

    def test_event_on_objective_complete(self):
        engine = ObjectiveEngine()
        obj = MissionObjective(
            objective_id="kill_all", name="Kill All", description="...",
            objective_type=ObjectiveType.ELIMINATE,
            target_id="boss",
            status=ObjectiveStatus.AVAILABLE,
        )
        ev = DynamicEvent(
            event_id="reinforcements", name="Reinforcements",
            trigger=TriggerCondition("objective_completed", {"objective_id": "kill_all"}),
            actions=[{"action": "spawn_units", "params": {"count": 5}}],
        )
        engine.add_objective(obj)
        engine.add_event(ev)
        # Complete the objective
        engine.tick(0.1, {"dead_unit_ids": {"boss"}})
        # Now tick again so the event sees the completed objective
        changes = engine.tick(0.1, {"dead_unit_ids": {"boss"}})
        fired = [c for c in changes if c["type"] == "event_fired"]
        assert len(fired) == 1


# ---------------------------------------------------------------------------
# ObjectiveEngine — chaining
# ---------------------------------------------------------------------------

class TestObjectiveChaining:
    def test_chain_three(self):
        engine = ObjectiveEngine()
        o1 = MissionObjective(
            objective_id="a", name="First", description="...",
            objective_type=ObjectiveType.ELIMINATE,
            target_id="t1",
            status=ObjectiveStatus.AVAILABLE,
        )
        o2 = MissionObjective(
            objective_id="b", name="Second", description="...",
            objective_type=ObjectiveType.DESTROY,
            target_id="s1",
            prerequisites=["a"],
        )
        o3 = MissionObjective(
            objective_id="c", name="Third", description="...",
            objective_type=ObjectiveType.EXTRACT,
            target_position=(0.0, 0.0),
            radius=10.0,
            prerequisites=["b"],
        )
        engine.add_objective(o1)
        engine.add_objective(o2)
        engine.add_objective(o3)

        # Complete first
        engine.tick(0.1, {"dead_unit_ids": {"t1"}})
        assert o1.status == ObjectiveStatus.COMPLETED
        # Second should unlock
        engine.tick(0.1, {"structures_destroyed": set()})
        assert o2.status == ObjectiveStatus.ACTIVE
        assert o3.status == ObjectiveStatus.LOCKED
        # Complete second
        engine.tick(0.1, {"structures_destroyed": {"s1"}})
        assert o2.status == ObjectiveStatus.COMPLETED
        # Third should unlock and be completable
        world = {"units": [{"id": "f1", "pos": (0.0, 0.0), "alliance": "friendly"}]}
        engine.tick(0.1, world)
        assert o3.status == ObjectiveStatus.COMPLETED
        assert engine.all_required_complete()

    def test_all_required_complete(self):
        engine = ObjectiveEngine()
        o1 = MissionObjective(
            objective_id="req1", name="Required", description="...",
            objective_type=ObjectiveType.ELIMINATE,
            status=ObjectiveStatus.COMPLETED, required=True,
        )
        o2 = MissionObjective(
            objective_id="opt1", name="Optional", description="...",
            objective_type=ObjectiveType.COLLECT,
            status=ObjectiveStatus.ACTIVE, required=False,
        )
        engine.add_objective(o1)
        engine.add_objective(o2)
        assert engine.all_required_complete() is True


# ---------------------------------------------------------------------------
# ObjectiveEngine — to_three_js
# ---------------------------------------------------------------------------

class TestToThreeJs:
    def test_structure(self):
        engine = ObjectiveEngine()
        obj = MissionObjective(
            objective_id="o1", name="Capture Zone", description="...",
            objective_type=ObjectiveType.CAPTURE,
            target_position=(50.0, 50.0),
            radius=20.0,
            status=ObjectiveStatus.ACTIVE,
            progress=0.5,
        )
        engine.add_objective(obj)
        data = engine.to_three_js()
        assert "markers" in data
        assert "zones" in data
        assert "progress_bars" in data
        assert data["all_complete"] is False
        assert data["any_failed"] is False
        assert len(data["markers"]) == 1
        assert len(data["zones"]) == 1
        assert data["zones"][0]["progress"] == 0.5

    def test_locked_objectives_hidden(self):
        engine = ObjectiveEngine()
        obj = MissionObjective(
            objective_id="o1", name="Secret", description="...",
            objective_type=ObjectiveType.ELIMINATE,
            status=ObjectiveStatus.LOCKED,
            target_position=(10.0, 10.0),
        )
        engine.add_objective(obj)
        data = engine.to_three_js()
        assert len(data["markers"]) == 0
        assert len(data["progress_bars"]) == 0

    def test_total_points_in_output(self):
        engine = ObjectiveEngine()
        obj = MissionObjective(
            objective_id="o1", name="Done", description="...",
            objective_type=ObjectiveType.ELIMINATE,
            status=ObjectiveStatus.COMPLETED,
            points=250,
        )
        engine.add_objective(obj)
        data = engine.to_three_js()
        assert data["total_points"] == 250

    def test_colors_match_status(self):
        engine = ObjectiveEngine()
        for status in [ObjectiveStatus.ACTIVE, ObjectiveStatus.COMPLETED, ObjectiveStatus.FAILED]:
            obj = MissionObjective(
                objective_id=f"o_{status.value}", name=status.value, description="...",
                objective_type=ObjectiveType.PATROL,
                status=status,
                target_position=(0.0, 0.0),
            )
            engine.add_objective(obj)
        data = engine.to_three_js()
        colors = {m["status"]: m["color"] for m in data["markers"]}
        assert colors["active"] == "#fcee0a"
        assert colors["completed"] == "#05ffa1"
        assert colors["failed"] == "#ff2a6d"


# ---------------------------------------------------------------------------
# Template tests
# ---------------------------------------------------------------------------

class TestObjectiveTemplates:
    def test_all_templates_exist(self):
        expected = {"assault_chain", "defense_chain", "stealth_chain", "rescue_chain"}
        assert expected == set(OBJECTIVE_TEMPLATES.keys())

    def test_assault_chain_objectives(self):
        objs = OBJECTIVE_TEMPLATES["assault_chain"]["objectives"]
        ids = [o.objective_id for o in objs]
        assert ids == ["assault_1", "assault_2", "assault_3", "assault_4"]

    def test_defense_chain_has_events(self):
        evts = OBJECTIVE_TEMPLATES["defense_chain"]["events"]
        assert len(evts) >= 3

    def test_stealth_chain_types(self):
        objs = OBJECTIVE_TEMPLATES["stealth_chain"]["objectives"]
        types = [o.objective_type for o in objs]
        assert ObjectiveType.STEALTH in types
        assert ObjectiveType.SABOTAGE in types

    def test_rescue_chain_fail_trigger(self):
        objs = OBJECTIVE_TEMPLATES["rescue_chain"]["objectives"]
        escort = [o for o in objs if o.objective_type == ObjectiveType.ESCORT][0]
        assert len(escort.fail_triggers) > 0

    def test_load_template(self):
        engine = ObjectiveEngine()
        engine.load_template("assault_chain")
        assert len(engine.objectives) == 4
        assert len(engine.events) >= 1

    def test_load_template_unknown(self):
        engine = ObjectiveEngine()
        with pytest.raises(KeyError):
            engine.load_template("nonexistent_template")

    def test_load_template_deep_copy(self):
        """Loading a template should not mutate the global template."""
        engine = ObjectiveEngine()
        engine.load_template("assault_chain")
        engine.objectives["assault_1"].status = ObjectiveStatus.COMPLETED
        original = OBJECTIVE_TEMPLATES["assault_chain"]["objectives"][0]
        assert original.status == ObjectiveStatus.AVAILABLE

    def test_assault_chain_playthrough(self):
        """Simulate a full assault chain playthrough."""
        engine = ObjectiveEngine()
        engine.load_template("assault_chain")

        # Phase 1: Kill all hostiles
        world: dict = {"total_hostiles": 5, "units_killed": 0, "dead_unit_ids": set()}
        engine.tick(0.1, world)
        assert engine.objectives["assault_1"].status == ObjectiveStatus.ACTIVE

        # Kill all
        world["units_killed"] = 5
        engine.tick(0.1, world)
        assert engine.objectives["assault_1"].status == ObjectiveStatus.COMPLETED

        # Phase 2: Capture building
        world["units"] = [
            {"id": "f1", "pos": (150.0, 80.0), "alliance": "friendly"},
            {"id": "f2", "pos": (150.0, 80.0), "alliance": "friendly"},
        ]
        engine.tick(0.1, world)
        assert engine.objectives["assault_2"].status == ObjectiveStatus.ACTIVE
        for _ in range(200):
            engine.tick(0.1, world)
        assert engine.objectives["assault_2"].status == ObjectiveStatus.COMPLETED

        # Phase 3: Collect intel
        world["items_collected"] = 1
        world["items_needed"] = 1
        engine.tick(0.1, world)
        engine.tick(0.1, world)
        assert engine.objectives["assault_3"].status == ObjectiveStatus.COMPLETED

        # Phase 4: Extract
        world["units"] = [
            {"id": "f1", "pos": (0.0, 0.0), "alliance": "friendly"},
        ]
        engine.tick(0.1, world)
        engine.tick(0.1, world)
        assert engine.objectives["assault_4"].status == ObjectiveStatus.COMPLETED
        assert engine.all_required_complete()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_escort_no_unit_found(self):
        engine = ObjectiveEngine()
        obj = MissionObjective(
            objective_id="esc1", name="Escort VIP", description="...",
            objective_type=ObjectiveType.ESCORT,
            target_id="vip",
            target_position=(100.0, 100.0),
            status=ObjectiveStatus.AVAILABLE,
        )
        engine.add_objective(obj)
        # No VIP in world
        engine.tick(0.1, {"units": []})
        assert obj.progress == 0.0

    def test_defend_no_target_position(self):
        engine = ObjectiveEngine()
        obj = MissionObjective(
            objective_id="d1", name="Defend", description="...",
            objective_type=ObjectiveType.DEFEND,
            target_position=None,
            status=ObjectiveStatus.AVAILABLE,
        )
        engine.add_objective(obj)
        engine.tick(0.1, {"units": [{"id": "f1", "pos": (0, 0), "alliance": "friendly"}]})
        # Should not crash, progress stays 0
        assert obj.progress == 0.0

    def test_multiple_ticks_accumulate_time(self):
        engine = ObjectiveEngine()
        engine.tick(1.0, {})
        engine.tick(2.0, {})
        engine.tick(3.0, {})
        assert engine._total_elapsed == pytest.approx(6.0)

    def test_completed_objective_not_reprocessed(self):
        engine = ObjectiveEngine()
        obj = MissionObjective(
            objective_id="o1", name="Done", description="...",
            objective_type=ObjectiveType.ELIMINATE,
            status=ObjectiveStatus.COMPLETED,
            points=100,
        )
        engine.add_objective(obj)
        # tick should not re-activate or re-complete
        changes = engine.tick(0.1, {})
        assert obj.status == ObjectiveStatus.COMPLETED
        completion_events = [c for c in changes if c["type"] == "objective_completed"]
        assert len(completion_events) == 0

    def test_failed_objective_not_reprocessed(self):
        engine = ObjectiveEngine()
        obj = MissionObjective(
            objective_id="o1", name="Failed", description="...",
            objective_type=ObjectiveType.ELIMINATE,
            status=ObjectiveStatus.FAILED,
        )
        engine.add_objective(obj)
        changes = engine.tick(0.1, {})
        fail_events = [c for c in changes if c["type"] == "objective_failed"]
        assert len(fail_events) == 0

    def test_on_complete_actions_returned(self):
        engine = ObjectiveEngine()
        obj = MissionObjective(
            objective_id="o1", name="Kill Boss", description="...",
            objective_type=ObjectiveType.ELIMINATE,
            target_id="boss",
            status=ObjectiveStatus.AVAILABLE,
            on_complete=[{"action": "spawn_units", "params": {"count": 10}}],
        )
        engine.add_objective(obj)
        changes = engine.tick(0.1, {"dead_unit_ids": {"boss"}})
        completed = [c for c in changes if c["type"] == "objective_completed"]
        assert len(completed) == 1
        assert completed[0]["on_complete"] == [{"action": "spawn_units", "params": {"count": 10}}]

    def test_patrol_no_waypoints(self):
        engine = ObjectiveEngine()
        obj = MissionObjective(
            objective_id="p1", name="Patrol", description="...",
            objective_type=ObjectiveType.PATROL,
            status=ObjectiveStatus.AVAILABLE,
        )
        engine.add_objective(obj)
        engine.tick(0.1, {"patrol_waypoints": [], "patrol_visited": set()})
        assert obj.status == ObjectiveStatus.COMPLETED
