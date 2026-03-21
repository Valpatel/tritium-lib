# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for AI behavior system wired into game_server.py.

Verifies that:
- UnitAISystem registers units and builds correct behavior tree types
- BT decisions are produced correctly per unit type/alliance
- Steering behaviors (flee, wander, seek_cover) mutate unit positions
- FormationMover is created for squads with waypoints
- Formation slot targets are applied to non-combat units
- game_tick() includes 'ai_behaviors' in the frame
- AI state includes expected fields per unit
"""

from __future__ import annotations

import math
import time
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers — lightweight stubs so tests run without a full world
# ---------------------------------------------------------------------------

def _make_unit(uid: str, alliance: str = "friendly", unit_type: str = "infantry",
               health: float = 100.0, position: tuple = (100.0, 100.0),
               ammo: int = -1, suppression: float = 0.0) -> MagicMock:
    """Create a minimal Unit-like mock."""
    unit = MagicMock()
    unit.unit_id = uid
    unit.name = uid
    unit.alliance = MagicMock()
    unit.alliance.value = alliance
    unit.unit_type = MagicMock()
    unit.unit_type.value = unit_type
    unit.position = position
    unit.heading = 0.0
    unit.squad_id = None
    unit.state = MagicMock()
    unit.state.health = health
    unit.state.is_alive = True
    unit.state.status = "idle"
    unit.state.ammo = ammo
    unit.state.suppression = suppression
    unit.stats = MagicMock()
    unit.stats.max_health = 100.0
    unit.stats.speed = 5.0
    unit.stats.attack_range = 30.0
    unit.stats.detection_range = 50.0
    unit.is_alive = MagicMock(return_value=True)
    unit.effective_speed = MagicMock(return_value=5.0)
    return unit


def _make_world(units: dict) -> MagicMock:
    """Create a minimal World-like mock."""
    world = MagicMock()
    world.units = units
    world.squads = {}
    world.destruction = None
    world.sim_time = 0.0
    return world


# ---------------------------------------------------------------------------
# Import the system under test
# ---------------------------------------------------------------------------

from tritium_lib.sim_engine.demos.game_server import UnitAISystem
from tritium_lib.sim_engine.ai.behavior_tree import (
    make_patrol_tree, make_hostile_tree, make_civilian_tree,
)
from tritium_lib.sim_engine.ai.formations import FormationType, FormationMover


# ---------------------------------------------------------------------------
# UnitAISystem.register_unit
# ---------------------------------------------------------------------------

class TestRegisterUnit:
    def test_friendly_infantry_gets_patrol_tree(self) -> None:
        ai = UnitAISystem()
        ai.register_unit("u1", "infantry", "friendly")
        assert "u1" in ai._trees
        # Patrol tree should produce "patrol" or "idle" decision when no threats
        ctx = {"threats": [], "waypoints": True, "threat_in_range": False,
               "health": 1.0, "at_destination": False, "recently_threatened": False,
               "time": 0.0}
        ai._trees["u1"].tick(ctx)
        assert ctx.get("decision") in ("patrol", "idle", "engage", "pursue")

    def test_hostile_gets_hostile_tree(self) -> None:
        ai = UnitAISystem()
        ai.register_unit("h1", "infantry", "hostile")
        assert "h1" in ai._trees
        ctx = {"threats": [], "waypoints": False, "threat_in_range": False,
               "health": 1.0, "at_destination": False, "recently_threatened": False,
               "time": 0.0, "enemies": [], "enemy_in_range": False, "in_cover": False,
               "ammo_ratio": 1.0, "enemies_visible": 0, "allies_nearby": 0,
               "is_flanking": False, "is_suppressed": False, "squad_members": False,
               "engage_duration": 0.0, "stall_threshold": 8.0}
        ai._trees["h1"].tick(ctx)
        # Hostile with no threats should regroup
        assert ctx.get("decision") == "regroup"

    def test_duplicate_register_is_idempotent(self) -> None:
        ai = UnitAISystem()
        ai.register_unit("u1", "infantry", "friendly")
        tree_before = ai._trees["u1"]
        ai.register_unit("u1", "infantry", "friendly")
        # Same tree object — not replaced
        assert ai._trees["u1"] is tree_before

    def test_all_supported_unit_types(self) -> None:
        ai = UnitAISystem()
        for ut in ("infantry", "sniper", "heavy", "engineer", "medic", "scout"):
            ai.register_unit(f"u_{ut}", ut, "friendly")
            assert f"u_{ut}" in ai._trees


# ---------------------------------------------------------------------------
# UnitAISystem.register_squad
# ---------------------------------------------------------------------------

class TestRegisterSquad:
    def test_squad_with_valid_waypoints_creates_mover(self) -> None:
        ai = UnitAISystem()
        waypoints = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)]
        ai.register_squad("squad_alpha", "friendly", waypoints)
        assert "squad_alpha" in ai._formation_movers
        mover = ai._formation_movers["squad_alpha"]
        assert isinstance(mover, FormationMover)
        assert mover.formation == FormationType.WEDGE

    def test_squad_with_single_waypoint_skipped(self) -> None:
        ai = UnitAISystem()
        ai.register_squad("squad_alpha", "friendly", [(0.0, 0.0)])
        assert "squad_alpha" not in ai._formation_movers

    def test_squad_with_no_waypoints_skipped(self) -> None:
        ai = UnitAISystem()
        ai.register_squad("squad_alpha", "friendly", [])
        assert "squad_alpha" not in ai._formation_movers

    def test_duplicate_squad_register_is_idempotent(self) -> None:
        ai = UnitAISystem()
        waypoints = [(0.0, 0.0), (50.0, 0.0)]
        ai.register_squad("s1", "friendly", waypoints)
        mover_before = ai._formation_movers["s1"]
        ai.register_squad("s1", "friendly", [(0.0, 0.0), (99.0, 0.0)])
        assert ai._formation_movers["s1"] is mover_before


# ---------------------------------------------------------------------------
# BT decision correctness
# ---------------------------------------------------------------------------

class TestBehaviorTreeDecisions:
    def test_hostile_retreats_when_low_health(self) -> None:
        ai = UnitAISystem()
        ai.register_unit("h1", "infantry", "hostile")
        ctx = ai._bt_contexts["h1"]
        ctx.update({
            "time": 0.0, "threats": [{"id": "u1", "pos": (110.0, 100.0), "dist": 10.0}],
            "threat_in_range": True, "health": 0.1, "retreat_threshold": 0.3,
            "waypoints": False, "at_destination": False, "recently_threatened": True,
            "enemies": [{"id": "u1"}], "enemy_in_range": True, "in_cover": False,
            "ammo_ratio": 1.0, "enemies_visible": 1, "allies_nearby": 0,
            "is_flanking": False, "is_suppressed": False, "squad_members": False,
            "engage_duration": 0.0, "stall_threshold": 8.0,
        })
        ai._trees["h1"].tick(ctx)
        assert ctx.get("decision") == "retreat"

    def test_friendly_unit_seeks_cover_when_threat_in_range(self) -> None:
        """Friendly infantry uses make_friendly_tree: first seeks cover (cooldown-gated),
        then engages when cooldown is active."""
        ai = UnitAISystem()
        ai.register_unit("p1", "infantry", "friendly")
        ctx = ai._bt_contexts["p1"]
        ctx.update({
            "time": 0.0,
            "threats": [{"id": "h1", "pos": (120.0, 100.0), "dist": 20.0}],
            "threat_in_range": True,
            "health": 1.0,
            "retreat_threshold": 0.3,
            "waypoints": True,
            "at_destination": False,
            "recently_threatened": True,
        })
        ai._trees["p1"].tick(ctx)
        # First tick: seek_cover (cooldown starts), subsequent ticks: engage
        assert ctx.get("decision") == "seek_cover"

    def test_friendly_unit_approaches_out_of_range_threat(self) -> None:
        """Friendly infantry uses make_friendly_tree: approaches detected threats
        that are out of weapon range (unlike patrol tree which uses 'pursue')."""
        ai = UnitAISystem()
        ai.register_unit("p1", "infantry", "friendly")
        ctx = ai._bt_contexts["p1"]
        ctx.update({
            "time": 0.0,
            "threats": [{"id": "h1", "pos": (200.0, 100.0), "dist": 100.0}],
            "threat_in_range": False,
            "health": 1.0,
            "retreat_threshold": 0.3,
            "waypoints": True,
            "at_destination": False,
            "recently_threatened": True,
        })
        ai._trees["p1"].tick(ctx)
        assert ctx.get("decision") == "approach"

    def test_patrol_unit_patrols_when_no_threats(self) -> None:
        ai = UnitAISystem()
        ai.register_unit("p1", "infantry", "friendly")
        ctx = ai._bt_contexts["p1"]
        ctx.update({
            "time": 0.0, "threats": [], "threat_in_range": False,
            "health": 1.0, "retreat_threshold": 0.3,
            "waypoints": True, "at_destination": False,
            "recently_threatened": False,
        })
        ai._trees["p1"].tick(ctx)
        assert ctx.get("decision") == "patrol"

    def test_civilian_flees_threat(self) -> None:
        ai = UnitAISystem()
        ai.register_unit("c1", "medic", "friendly")  # medic uses civilian tree
        # Override with civilian tree for clarity
        ai._trees["c1"] = make_civilian_tree()
        ctx = {"time": 0.0,
               "threats": [{"id": "h1", "pos": (110.0, 100.0), "dist": 10.0}],
               "health": 1.0, "retreat_threshold": 0.3,
               "recently_threatened": True}
        ai._trees["c1"].tick(ctx)
        assert ctx.get("decision") == "flee"


# ---------------------------------------------------------------------------
# Steering — flee moves unit away from threat
# ---------------------------------------------------------------------------

class TestSteeringBehaviors:
    def test_flee_moves_unit_away(self) -> None:
        ai = UnitAISystem()
        unit = _make_unit("u1", "friendly", position=(100.0, 100.0))
        threat_pos = (110.0, 100.0)  # threat is to the right

        threats = [{"id": "h1", "pos": threat_pos, "dist": 10.0,
                    "vel": (0.0, 0.0), "health": 1.0}]
        new_pos, _ = ai._apply_steering(
            "u1", unit, "flee", threats, [], {}, dt=0.1,
        )
        # Should move left (away from threat at x=110)
        assert new_pos[0] < 100.0

    def test_wander_changes_position(self) -> None:
        ai = UnitAISystem()
        unit = _make_unit("u1", "friendly", position=(100.0, 100.0))
        ai._wander_velocities["u1"] = (3.0, 0.0)

        new_pos, _ = ai._apply_steering(
            "u1", unit, "wander", [], [], {}, dt=0.1,
        )
        # Wander should produce some movement
        dx = new_pos[0] - 100.0
        dy = new_pos[1] - 100.0
        assert math.hypot(dx, dy) > 0.0

    def test_no_steering_for_engage_decision(self) -> None:
        """Engage leaves position unchanged — world tick handles it."""
        ai = UnitAISystem()
        unit = _make_unit("u1", "friendly", position=(100.0, 100.0))
        threats = [{"id": "h1", "pos": (120.0, 100.0), "dist": 20.0,
                    "vel": (0.0, 0.0), "health": 1.0}]
        new_pos, _ = ai._apply_steering(
            "u1", unit, "engage", threats, [], {}, dt=0.1,
        )
        assert new_pos == (100.0, 100.0)

    def test_seek_cover_moves_toward_obstacle(self) -> None:
        ai = UnitAISystem()
        unit = _make_unit("u1", "friendly", position=(100.0, 100.0))
        # Obstacle between unit and threat
        obstacles = [((110.0, 100.0), 4.0)]
        threats = [{"id": "h1", "pos": (130.0, 100.0), "dist": 30.0,
                    "vel": (0.0, 0.0), "health": 1.0}]

        new_pos, _ = ai._apply_steering(
            "u1", unit, "seek_cover", threats, obstacles, {}, dt=0.5,
        )
        # Should move from (100, 100) — cover is on the far side of obstacle from threat
        assert new_pos != (100.0, 100.0)


# ---------------------------------------------------------------------------
# Full tick — produces ai_state dict
# ---------------------------------------------------------------------------

class TestFullTick:
    def test_tick_produces_ai_state_for_all_alive_units(self) -> None:
        ai = UnitAISystem()
        units = {
            "alpha_1": _make_unit("alpha_1", "friendly", position=(200.0, 200.0)),
            "tango_1": _make_unit("tango_1", "hostile", position=(280.0, 280.0)),
        }
        world = _make_world(units)
        ai_state = ai.tick(0.1, world)
        assert "alpha_1" in ai_state
        assert "tango_1" in ai_state

    def test_tick_ai_state_has_required_fields(self) -> None:
        ai = UnitAISystem()
        units = {"u1": _make_unit("u1", "friendly", position=(100.0, 100.0))}
        world = _make_world(units)
        ai_state = ai.tick(0.1, world)
        assert "u1" in ai_state
        entry = ai_state["u1"]
        assert "decision" in entry
        assert "formation" in entry
        assert "in_cover" in entry
        assert "threat_count" in entry
        assert "engage_timer" in entry

    def test_tick_with_none_world_returns_empty(self) -> None:
        ai = UnitAISystem()
        result = ai.tick(0.1, None)
        assert result == {}

    def test_engage_timer_increments_while_attacking(self) -> None:
        ai = UnitAISystem()
        unit = _make_unit("u1", "friendly", position=(100.0, 100.0))
        unit.state.status = "attacking"
        units = {"u1": unit}
        world = _make_world(units)

        ai.tick(0.1, world)
        ai.tick(0.1, world)
        assert ai._engage_timers.get("u1", 0.0) > 0.0

    def test_engage_timer_resets_when_not_attacking(self) -> None:
        ai = UnitAISystem()
        unit = _make_unit("u1", "friendly", position=(100.0, 100.0))
        unit.state.status = "attacking"
        units = {"u1": unit}
        world = _make_world(units)

        ai.tick(0.1, world)
        assert ai._engage_timers["u1"] > 0.0

        unit.state.status = "idle"
        ai.tick(0.1, world)
        assert ai._engage_timers["u1"] == 0.0

    def test_units_auto_registered_on_tick(self) -> None:
        """Units not pre-registered should be auto-registered during tick."""
        ai = UnitAISystem()
        units = {"new_unit": _make_unit("new_unit", "hostile", position=(50.0, 50.0))}
        world = _make_world(units)
        ai_state = ai.tick(0.1, world)
        assert "new_unit" in ai_state
        assert "new_unit" in ai._trees


# ---------------------------------------------------------------------------
# Formation integration
# ---------------------------------------------------------------------------

class TestFormationIntegration:
    def test_formation_mover_ticked_per_squad(self) -> None:
        ai = UnitAISystem()
        waypoints = [(100.0, 100.0), (200.0, 100.0), (300.0, 100.0)]
        ai.register_squad("alpha", "friendly", waypoints)
        mover = ai._formation_movers["alpha"]

        # Tick the mover directly to ensure it advances
        member_positions = {"u1": (100.0, 100.0), "u2": (98.0, 100.0)}
        targets1 = mover.tick(0.5, member_positions)
        targets2 = mover.tick(0.5, member_positions)
        # Leader position should advance
        assert mover._leader_pos != (100.0, 100.0) or mover.is_complete()

    def test_formation_slot_in_ai_state(self) -> None:
        ai = UnitAISystem()
        waypoints = [(100.0, 100.0), (200.0, 100.0)]
        ai.register_squad("squad_a", "friendly", waypoints)

        unit = _make_unit("m1", "friendly", position=(100.0, 100.0))
        unit.squad_id = "squad_a"
        units = {"m1": unit}

        squad_mock = MagicMock()
        squad_mock.members = ["m1"]

        world = _make_world(units)
        world.squads = {"squad_a": squad_mock}

        ai_state = ai.tick(0.1, world)
        assert "m1" in ai_state
        # Unit should have been targeted by a formation slot
        assert ai_state["m1"]["in_formation_slot"] is True

    def test_to_three_js_includes_formation_movers(self) -> None:
        ai = UnitAISystem()
        ai.register_squad("alpha", "friendly", [(0.0, 0.0), (100.0, 0.0)])
        three_js = ai.to_three_js()
        assert "formation_movers" in three_js
        assert "alpha" in three_js["formation_movers"]
        fm = three_js["formation_movers"]["alpha"]
        assert "formation" in fm
        assert "progress" in fm
        assert "complete" in fm


# ---------------------------------------------------------------------------
# game_tick integration — frame includes ai_behaviors
# ---------------------------------------------------------------------------

class TestGameTickIntegration:
    def test_game_tick_frame_has_ai_behaviors(self) -> None:
        """game_tick() must include 'ai_behaviors' key when unit_ai is wired."""
        from tritium_lib.sim_engine.demos.game_server import build_full_game, game_tick

        gs = build_full_game("urban_combat")
        assert gs.unit_ai is not None, "unit_ai should be initialized by build_full_game"

        frame = game_tick(gs, dt=0.1)
        assert "ai_behaviors" in frame, "frame must contain ai_behaviors"
        ai_data = frame["ai_behaviors"]
        assert "units" in ai_data
        assert "formation_movers" in ai_data

    def test_game_tick_ai_behaviors_covers_all_units(self) -> None:
        from tritium_lib.sim_engine.demos.game_server import build_full_game, game_tick

        gs = build_full_game("urban_combat")
        frame = game_tick(gs, dt=0.1)
        ai_units = frame["ai_behaviors"]["units"]
        alive_ids = {uid for uid, u in gs.world.units.items() if u.is_alive()}
        # All alive units must appear in ai_behaviors
        for uid in alive_ids:
            assert uid in ai_units, f"unit {uid} missing from ai_behaviors"

    def test_game_tick_ai_behaviors_has_valid_decisions(self) -> None:
        from tritium_lib.sim_engine.demos.game_server import build_full_game, game_tick

        valid_decisions = {
            "idle", "wander", "patrol", "pursue", "engage", "flee", "hide",
            "retreat", "seek_cover", "regroup", "approach", "park",
            "drive", "pick_destination",
        }
        gs = build_full_game("urban_combat")
        frame = game_tick(gs, dt=0.1)
        for uid, info in frame["ai_behaviors"]["units"].items():
            assert info["decision"] in valid_decisions, (
                f"unit {uid} has unknown decision: {info['decision']!r}"
            )

    def test_game_tick_ai_behaviors_stable_over_multiple_ticks(self) -> None:
        """Multiple ticks should not crash and ai_behaviors should remain present."""
        from tritium_lib.sim_engine.demos.game_server import build_full_game, game_tick

        gs = build_full_game("urban_combat")
        for _ in range(5):
            frame = game_tick(gs, dt=0.1)
            assert "ai_behaviors" in frame


# ---------------------------------------------------------------------------
# to_three_js serialisation
# ---------------------------------------------------------------------------

class TestToThreeJs:
    def test_to_three_js_returns_dict_with_units_key(self) -> None:
        ai = UnitAISystem()
        units = {"u1": _make_unit("u1", position=(10.0, 10.0))}
        world = _make_world(units)
        ai.tick(0.1, world)
        result = ai.to_three_js()
        assert isinstance(result, dict)
        assert "units" in result

    def test_to_three_js_empty_before_tick(self) -> None:
        ai = UnitAISystem()
        result = ai.to_three_js()
        assert result["units"] == {}
        assert result["formation_movers"] == {}
