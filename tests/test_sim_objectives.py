# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for objectives.py — ObjectiveEngine, MissionObjective, TriggerCondition,
DynamicEvent, and mission-chain triggers."""

import pytest
from tritium_lib.sim_engine.objectives import (
    ObjectiveEngine,
    MissionObjective,
    ObjectiveType,
    ObjectiveStatus,
    TriggerCondition,
    DynamicEvent,
    OBJECTIVE_TEMPLATES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_active_obj(
    oid: str,
    obj_type: ObjectiveType = ObjectiveType.ELIMINATE,
    required: bool = True,
    points: int = 100,
    time_limit: float | None = None,
    target_position=None,
    target_id: str | None = None,
    prerequisites=None,
) -> MissionObjective:
    """Return an ACTIVE objective for testing."""
    obj = MissionObjective(
        objective_id=oid,
        name=oid.upper(),
        description="Test objective",
        objective_type=obj_type,
        status=ObjectiveStatus.ACTIVE,
        required=required,
        points=points,
        time_limit=time_limit,
        target_position=target_position,
        target_id=target_id,
        prerequisites=prerequisites or [],
    )
    return obj


def _base_world(**kwargs) -> dict:
    return {
        "elapsed": 0.0,
        "units_killed": 0,
        "units_killed_by_alliance": {},
        "units": [],
        "structures_destroyed": set(),
        "wave": 0,
        "completed_objectives": set(),
        "hostiles_alive": 5,
        **kwargs,
    }


# ---------------------------------------------------------------------------
# ObjectiveEngine — construction and basic queries
# ---------------------------------------------------------------------------

class TestObjectiveEngineInit:
    def test_empty_engine(self):
        eng = ObjectiveEngine()
        assert eng.objectives == {}
        assert eng.events == []

    def test_add_objective(self):
        eng = ObjectiveEngine()
        obj = _make_active_obj("o1")
        eng.add_objective(obj)
        assert "o1" in eng.objectives

    def test_add_event(self):
        eng = ObjectiveEngine()
        ev = DynamicEvent(
            event_id="ev1",
            name="Ambush",
            trigger=TriggerCondition(condition_type="time_elapsed", params={"seconds": 10.0}),
        )
        eng.add_event(ev)
        assert len(eng.events) == 1

    def test_all_required_complete_false_when_empty(self):
        eng = ObjectiveEngine()
        assert eng.all_required_complete() is False

    def test_any_required_failed_false_when_empty(self):
        eng = ObjectiveEngine()
        assert eng.any_required_failed() is False

    def test_total_points_zero_when_none_complete(self):
        eng = ObjectiveEngine()
        eng.add_objective(_make_active_obj("o1"))
        assert eng.total_points() == 0


class TestObjectiveEngineQueries:
    def test_get_active_returns_active(self):
        eng = ObjectiveEngine()
        eng.add_objective(_make_active_obj("o1"))
        eng.add_objective(MissionObjective(
            objective_id="o2", name="O2", description="",
            objective_type=ObjectiveType.CAPTURE,
            status=ObjectiveStatus.LOCKED,
        ))
        active = eng.get_active()
        assert len(active) == 1
        assert active[0].objective_id == "o1"

    def test_get_available(self):
        eng = ObjectiveEngine()
        obj = MissionObjective(
            objective_id="o1", name="O1", description="",
            objective_type=ObjectiveType.PATROL,
            status=ObjectiveStatus.AVAILABLE,
        )
        eng.add_objective(obj)
        available = eng.get_available()
        assert len(available) == 1

    def test_all_required_complete_true(self):
        eng = ObjectiveEngine()
        obj = _make_active_obj("o1")
        obj.status = ObjectiveStatus.COMPLETED
        eng.add_objective(obj)
        assert eng.all_required_complete() is True

    def test_any_required_failed_true(self):
        eng = ObjectiveEngine()
        obj = _make_active_obj("o1")
        obj.status = ObjectiveStatus.FAILED
        eng.add_objective(obj)
        assert eng.any_required_failed() is True

    def test_total_points_sums_completed(self):
        eng = ObjectiveEngine()
        o1 = _make_active_obj("o1", points=100)
        o1.status = ObjectiveStatus.COMPLETED
        o2 = _make_active_obj("o2", points=50)
        o2.status = ObjectiveStatus.COMPLETED
        o3 = _make_active_obj("o3", points=200)  # still active
        eng.add_objective(o1)
        eng.add_objective(o2)
        eng.add_objective(o3)
        assert eng.total_points() == 150


# ---------------------------------------------------------------------------
# Trigger evaluation
# ---------------------------------------------------------------------------

class TestTriggerEvaluation:
    """Test that check_triggers correctly evaluates world state."""

    def test_time_elapsed_trigger_fires(self):
        eng = ObjectiveEngine()
        ev = DynamicEvent(
            event_id="ev1",
            name="Timed Event",
            trigger=TriggerCondition("time_elapsed", {"seconds": 5.0}),
        )
        eng.add_event(ev)
        world = _base_world(elapsed=6.0)
        changes = eng.check_triggers(world)
        fired = [c for c in changes if c.get("type") == "event_fired"]
        assert len(fired) == 1
        assert fired[0]["event_id"] == "ev1"

    def test_time_elapsed_trigger_does_not_fire_early(self):
        eng = ObjectiveEngine()
        ev = DynamicEvent(
            event_id="ev1",
            name="Timed Event",
            trigger=TriggerCondition("time_elapsed", {"seconds": 10.0}),
        )
        eng.add_event(ev)
        world = _base_world(elapsed=3.0)
        changes = eng.check_triggers(world)
        fired = [c for c in changes if c.get("type") == "event_fired"]
        assert len(fired) == 0

    def test_one_shot_event_does_not_fire_twice(self):
        eng = ObjectiveEngine()
        ev = DynamicEvent(
            event_id="ev1",
            name="One-Shot",
            trigger=TriggerCondition("time_elapsed", {"seconds": 1.0}),
            one_shot=True,
        )
        eng.add_event(ev)
        world = _base_world(elapsed=5.0)
        eng.check_triggers(world)
        changes2 = eng.check_triggers(world)
        fired2 = [c for c in changes2 if c.get("type") == "event_fired"]
        assert len(fired2) == 0

    def test_units_killed_trigger(self):
        eng = ObjectiveEngine()
        ev = DynamicEvent(
            event_id="ev1",
            name="Mass Kill",
            trigger=TriggerCondition("units_killed", {"count": 3}),
        )
        eng.add_event(ev)
        world = _base_world(units_killed=5)
        changes = eng.check_triggers(world)
        fired = [c for c in changes if c.get("type") == "event_fired"]
        assert len(fired) == 1

    def test_all_hostiles_dead_trigger(self):
        eng = ObjectiveEngine()
        ev = DynamicEvent(
            event_id="ev1",
            name="Victory",
            trigger=TriggerCondition("all_hostiles_dead"),
        )
        eng.add_event(ev)
        world = _base_world(hostiles_alive=0)
        changes = eng.check_triggers(world)
        fired = [c for c in changes if c.get("type") == "event_fired"]
        assert len(fired) == 1

    def test_structure_destroyed_trigger(self):
        eng = ObjectiveEngine()
        ev = DynamicEvent(
            event_id="ev1",
            name="Bridge Down",
            trigger=TriggerCondition("structure_destroyed", {"structure_id": "bridge_01"}),
        )
        eng.add_event(ev)
        world = _base_world(structures_destroyed={"bridge_01", "barn_02"})
        changes = eng.check_triggers(world)
        fired = [c for c in changes if c.get("type") == "event_fired"]
        assert len(fired) == 1

    def test_position_reached_trigger(self):
        eng = ObjectiveEngine()
        ev = DynamicEvent(
            event_id="ev1",
            name="Reached Goal",
            trigger=TriggerCondition("position_reached", {
                "target": (100.0, 100.0),
                "radius": 10.0,
            }),
        )
        eng.add_event(ev)
        world = _base_world(units=[{"id": "u1", "pos": (102.0, 98.0), "health": 1.0, "alliance": "friendly"}])
        changes = eng.check_triggers(world)
        fired = [c for c in changes if c.get("type") == "event_fired"]
        assert len(fired) == 1

    def test_zone_entered_trigger(self):
        eng = ObjectiveEngine()
        ev = DynamicEvent(
            event_id="ev1",
            name="Zone Alert",
            trigger=TriggerCondition("zone_entered", {
                "zone_center": (200.0, 200.0),
                "zone_radius": 50.0,
                "alliance": "hostile",
            }),
        )
        eng.add_event(ev)
        world = _base_world(units=[
            {"id": "h1", "pos": (210.0, 205.0), "health": 1.0, "alliance": "hostile"},
        ])
        changes = eng.check_triggers(world)
        fired = [c for c in changes if c.get("type") == "event_fired"]
        assert len(fired) == 1

    def test_wave_completed_trigger(self):
        eng = ObjectiveEngine()
        ev = DynamicEvent(
            event_id="ev1",
            name="Wave 3 Done",
            trigger=TriggerCondition("wave_completed", {"wave": 3}),
        )
        eng.add_event(ev)
        world = _base_world(wave=3)
        changes = eng.check_triggers(world)
        fired = [c for c in changes if c.get("type") == "event_fired"]
        assert len(fired) == 1


# ---------------------------------------------------------------------------
# Objective prerequisite chains
# ---------------------------------------------------------------------------

class TestObjectivePrerequisiteChain:
    def test_locked_objective_unlocks_when_prereq_complete(self):
        eng = ObjectiveEngine()
        o1 = MissionObjective(
            objective_id="o1", name="First", description="",
            objective_type=ObjectiveType.CAPTURE,
            status=ObjectiveStatus.COMPLETED,
        )
        o2 = MissionObjective(
            objective_id="o2", name="Second", description="",
            objective_type=ObjectiveType.EXTRACT,
            status=ObjectiveStatus.LOCKED,
            prerequisites=["o1"],
        )
        eng.add_objective(o1)
        eng.add_objective(o2)
        world = _base_world()
        changes = eng.check_triggers(world)
        types = [c["type"] for c in changes]
        assert "objective_unlocked" in types or "objective_activated" in types

    def test_locked_objective_stays_locked_when_prereq_not_met(self):
        eng = ObjectiveEngine()
        o1 = MissionObjective(
            objective_id="o1", name="First", description="",
            objective_type=ObjectiveType.CAPTURE,
            status=ObjectiveStatus.ACTIVE,  # not yet complete
        )
        o2 = MissionObjective(
            objective_id="o2", name="Second", description="",
            objective_type=ObjectiveType.EXTRACT,
            status=ObjectiveStatus.LOCKED,
            prerequisites=["o1"],
        )
        eng.add_objective(o1)
        eng.add_objective(o2)
        world = _base_world()
        eng.check_triggers(world)
        assert o2.status == ObjectiveStatus.LOCKED

    def test_fail_trigger_fails_objective(self):
        eng = ObjectiveEngine()
        obj = _make_active_obj("o1")
        obj.fail_triggers = [TriggerCondition("all_hostiles_dead")]
        eng.add_objective(obj)
        world = _base_world(hostiles_alive=0)
        changes = eng.check_triggers(world)
        failed = [c for c in changes if c.get("type") == "objective_failed"]
        assert len(failed) == 1
        assert obj.status == ObjectiveStatus.FAILED


# ---------------------------------------------------------------------------
# Objective progress via tick
# ---------------------------------------------------------------------------

class TestObjectiveTick:
    def test_tick_increments_elapsed(self):
        eng = ObjectiveEngine()
        eng.add_objective(_make_active_obj("o1"))
        eng.tick(1.0, _base_world())
        assert eng._total_elapsed == pytest.approx(1.0)

    def test_eliminate_progress_from_kills(self):
        eng = ObjectiveEngine()
        obj = _make_active_obj("o1", obj_type=ObjectiveType.ELIMINATE)
        eng.add_objective(obj)
        world = _base_world(units_killed=5, total_hostiles=10)
        eng.tick(0.1, world)
        assert obj.progress == pytest.approx(0.5)

    def test_eliminate_all_kills_completes(self):
        eng = ObjectiveEngine()
        obj = _make_active_obj("o1", obj_type=ObjectiveType.ELIMINATE)
        eng.add_objective(obj)
        world = _base_world(units_killed=10, total_hostiles=10, hostiles_alive=0)
        changes = eng.tick(0.1, world)
        completed = [c for c in changes if c.get("type") == "objective_completed"]
        assert len(completed) > 0

    def test_survive_progress_over_time(self):
        eng = ObjectiveEngine()
        obj = _make_active_obj("o1", obj_type=ObjectiveType.SURVIVE, time_limit=10.0)
        eng.add_objective(obj)
        # Tick 11 seconds total
        for _ in range(110):
            changes = eng.tick(0.1, _base_world())
        completed = any(
            c.get("type") == "objective_completed"
            for c in changes
        )
        assert completed or obj.status == ObjectiveStatus.COMPLETED

    def test_non_survive_time_limit_fails(self):
        eng = ObjectiveEngine()
        obj = _make_active_obj("o1", obj_type=ObjectiveType.CAPTURE, time_limit=2.0)
        eng.add_objective(obj)
        for _ in range(25):
            eng.tick(0.1, _base_world())
        assert obj.status == ObjectiveStatus.FAILED

    def test_capture_progress_with_friendly_units(self):
        eng = ObjectiveEngine()
        obj = _make_active_obj(
            "o1",
            obj_type=ObjectiveType.CAPTURE,
            target_position=(100.0, 100.0),
        )
        obj.radius = 30.0
        eng.add_objective(obj)
        world = _base_world(units=[
            {"id": "f1", "pos": (100.0, 100.0), "health": 1.0, "alliance": "friendly"},
            {"id": "f2", "pos": (105.0, 100.0), "health": 1.0, "alliance": "friendly"},
        ])
        for _ in range(30):
            eng.tick(0.1, world)
        assert obj.progress > 0.0

    def test_destroy_objective_completes_on_structure_destroyed(self):
        eng = ObjectiveEngine()
        obj = _make_active_obj("o1", obj_type=ObjectiveType.DESTROY, target_id="ammo_depot")
        eng.add_objective(obj)
        world = _base_world(structures_destroyed={"ammo_depot"})
        changes = eng.tick(0.1, world)
        completed = [c for c in changes if c.get("type") == "objective_completed"]
        assert len(completed) > 0

    def test_collect_progress(self):
        eng = ObjectiveEngine()
        obj = _make_active_obj("o1", obj_type=ObjectiveType.COLLECT)
        eng.add_objective(obj)
        world = _base_world(items_collected=4, items_needed=8)
        eng.tick(0.1, world)
        assert obj.progress == pytest.approx(0.5)

    def test_points_awarded_on_completion(self):
        eng = ObjectiveEngine()
        obj = _make_active_obj("o1", obj_type=ObjectiveType.ELIMINATE, points=250)
        eng.add_objective(obj)
        world = _base_world(units_killed=10, total_hostiles=10)
        eng.tick(0.1, world)
        if obj.status == ObjectiveStatus.COMPLETED:
            assert eng.total_points() == 250


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

class TestObjectiveTemplates:
    def test_templates_dict_not_empty(self):
        assert len(OBJECTIVE_TEMPLATES) > 0

    def test_all_templates_have_objectives_key(self):
        for name, tmpl in OBJECTIVE_TEMPLATES.items():
            assert "objectives" in tmpl, f"Template '{name}' missing 'objectives' key"

    def test_load_template_assault_chain(self):
        if "assault_chain" not in OBJECTIVE_TEMPLATES:
            pytest.skip("assault_chain template not present")
        eng = ObjectiveEngine()
        eng.load_template("assault_chain")
        assert len(eng.objectives) > 0

    def test_load_template_unknown_raises(self):
        eng = ObjectiveEngine()
        with pytest.raises(KeyError):
            eng.load_template("nonexistent_template_xyz")

    def test_loaded_template_objectives_are_independent(self):
        """Loading the same template twice should give independent objective instances."""
        if "assault_chain" not in OBJECTIVE_TEMPLATES:
            pytest.skip("assault_chain template not present")
        eng1 = ObjectiveEngine()
        eng1.load_template("assault_chain")
        eng2 = ObjectiveEngine()
        eng2.load_template("assault_chain")
        # Modifying one should not affect the other
        for obj in eng1.objectives.values():
            obj.status = ObjectiveStatus.COMPLETED
        for obj in eng2.objectives.values():
            assert obj.status != ObjectiveStatus.COMPLETED


# ---------------------------------------------------------------------------
# to_three_js export
# ---------------------------------------------------------------------------

class TestObjectiveToThreeJs:
    def test_to_three_js_returns_dict(self):
        eng = ObjectiveEngine()
        result = eng.to_three_js()
        assert isinstance(result, dict)

    def test_to_three_js_has_expected_keys(self):
        eng = ObjectiveEngine()
        result = eng.to_three_js()
        assert "markers" in result
        assert "zones" in result
        assert "progress_bars" in result

    def test_active_objective_with_position_appears_in_markers(self):
        eng = ObjectiveEngine()
        obj = _make_active_obj("o1", target_position=(200.0, 300.0))
        eng.add_objective(obj)
        result = eng.to_three_js()
        ids = [m["id"] for m in result.get("markers", [])]
        assert "o1" in ids
