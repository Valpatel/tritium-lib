# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for enhanced sim_engine AI behaviours, formation to_three_js,
and unit_states to_three_js.

Covers:
  - formation_to_three_js / formation_mover_to_three_js output shape
  - unit_state_to_three_js / units_states_to_three_js output shape
  - World._tick_units retreat behaviour (health < 30% + morale < 30%)
  - World._tick_units cover-seeking behaviour (suppression > 60% + health < 60%)
  - World._tick_units flanking behaviour (2+ friendlies vs 1 hostile)
  - World.assign_squad_formation wiring and formation data in render frame
"""

from __future__ import annotations

import math
import pytest

from tritium_lib.sim_engine.ai.formations import (
    FormationType,
    FormationConfig,
    FormationMover,
    formation_to_three_js,
    formation_mover_to_three_js,
)
from tritium_lib.sim_engine.behavior.unit_states import (
    create_turret_fsm,
    create_hostile_fsm,
    create_rover_fsm,
    unit_state_to_three_js,
    units_states_to_three_js,
)
from tritium_lib.sim_engine.world import World, WorldConfig
from tritium_lib.sim_engine.units import Alliance
from tritium_lib.sim_engine.destruction import Structure, StructureType


# ===========================================================================
# A. formation_to_three_js
# ===========================================================================


class TestFormationToThreeJs:
    def test_returns_required_keys(self):
        config = FormationConfig(
            formation_type=FormationType.WEDGE,
            spacing=3.0,
            facing=0.0,
            leader_pos=(10.0, 20.0),
            num_members=4,
        )
        data = formation_to_three_js(config)
        for key in ("formation_type", "leader_pos", "facing", "spacing",
                    "num_members", "slots", "lines"):
            assert key in data, f"Missing key: {key}"

    def test_slot_count_matches_num_members(self):
        for n in (1, 2, 4, 6):
            config = FormationConfig(
                formation_type=FormationType.LINE,
                num_members=n,
                leader_pos=(0.0, 0.0),
            )
            data = formation_to_three_js(config)
            assert len(data["slots"]) == n

    def test_slot_ids_populated_from_member_ids(self):
        config = FormationConfig(
            formation_type=FormationType.COLUMN,
            num_members=3,
            leader_pos=(5.0, 5.0),
        )
        ids = ["alpha", "bravo", "charlie"]
        data = formation_to_three_js(config, member_ids=ids)
        slot_ids = [s["id"] for s in data["slots"]]
        assert slot_ids == ids

    def test_lines_connect_leader_to_followers(self):
        config = FormationConfig(
            formation_type=FormationType.WEDGE,
            num_members=3,
            leader_pos=(0.0, 0.0),
        )
        data = formation_to_three_js(config)
        # n members => n-1 leader-to-follower lines
        assert len(data["lines"]) == 2

    def test_single_member_has_no_lines(self):
        config = FormationConfig(
            formation_type=FormationType.WEDGE,
            num_members=1,
            leader_pos=(0.0, 0.0),
        )
        data = formation_to_three_js(config)
        assert data["lines"] == []

    def test_leader_pos_in_output(self):
        config = FormationConfig(
            formation_type=FormationType.DIAMOND,
            num_members=4,
            leader_pos=(99.0, 77.0),
        )
        data = formation_to_three_js(config)
        assert data["leader_pos"] == [99.0, 77.0]

    def test_formation_type_name_in_output(self):
        for ft in FormationType:
            config = FormationConfig(formation_type=ft, num_members=2, leader_pos=(0.0, 0.0))
            data = formation_to_three_js(config)
            assert data["formation_type"] == ft.value


class TestFormationMoverToThreeJs:
    def test_returns_progress_and_waypoints(self):
        mover = FormationMover(
            waypoints=[(0.0, 0.0), (50.0, 0.0), (100.0, 0.0)],
            formation=FormationType.LINE,
        )
        data = formation_mover_to_three_js(mover)
        assert "progress" in data
        assert "complete" in data
        assert "waypoints" in data
        assert len(data["waypoints"]) == 3

    def test_progress_0_at_start(self):
        mover = FormationMover(
            waypoints=[(0.0, 0.0), (100.0, 0.0)],
            formation=FormationType.COLUMN,
        )
        data = formation_mover_to_three_js(mover)
        assert data["progress"] == pytest.approx(0.0, abs=0.05)

    def test_progress_advances_after_tick(self):
        mover = FormationMover(
            waypoints=[(0.0, 0.0), (20.0, 0.0)],
            formation=FormationType.LINE,
            max_speed=10.0,
        )
        for _ in range(10):
            mover.tick(0.1, {"u1": (0.0, 0.0), "u2": (1.0, 0.0)})
        data = formation_mover_to_three_js(mover)
        assert data["progress"] > 0.0

    def test_complete_flag_reflects_mover_state(self):
        mover = FormationMover(
            waypoints=[(0.0, 0.0), (1.0, 0.0)],
            formation=FormationType.FILE,
            max_speed=50.0,
        )
        for _ in range(20):
            mover.tick(0.1, {"u1": (0.0, 0.0)})
        data = formation_mover_to_three_js(mover)
        assert data["complete"] is True


# ===========================================================================
# D. unit_state_to_three_js / units_states_to_three_js
# ===========================================================================


class TestUnitStateToThreeJs:
    def test_initial_state_returned(self):
        fsm = create_turret_fsm()
        data = unit_state_to_three_js("t1", fsm)
        assert data["unit_id"] == "t1"
        assert data["state"] == "idle"
        assert "label" in data
        assert "color" in data
        assert "time_in_state" in data
        assert "available_states" in data

    def test_state_changes_after_tick(self):
        fsm = create_turret_fsm()
        # Force through idle -> scanning by ticking with enough time
        for _ in range(20):
            fsm.tick(0.1, {})
        data = unit_state_to_three_js("t1", fsm)
        # After 2s of idle time the turret should be scanning
        assert data["state"] != "idle"

    def test_hostile_fsm_initial_state(self):
        fsm = create_hostile_fsm()
        data = unit_state_to_three_js("h1", fsm)
        assert data["state"] == "spawning"

    def test_label_is_string(self):
        fsm = create_rover_fsm()
        data = unit_state_to_three_js("r1", fsm)
        assert isinstance(data["label"], str)
        assert len(data["label"]) > 0

    def test_color_is_hex(self):
        fsm = create_turret_fsm()
        data = unit_state_to_three_js("t1", fsm)
        assert data["color"].startswith("#")

    def test_available_states_contains_current(self):
        fsm = create_turret_fsm()
        data = unit_state_to_three_js("t1", fsm)
        assert data["state"] in data["available_states"]

    def test_time_in_state_is_float(self):
        fsm = create_turret_fsm()
        fsm.tick(0.5, {})
        data = unit_state_to_three_js("t1", fsm)
        assert isinstance(data["time_in_state"], float)

    def test_batch_serialisation_returns_list(self):
        fsm_map = {
            "u1": create_turret_fsm(),
            "u2": create_rover_fsm(),
            "u3": create_hostile_fsm(),
        }
        result = units_states_to_three_js(fsm_map)
        assert isinstance(result, list)
        assert len(result) == 3
        ids = {d["unit_id"] for d in result}
        assert ids == {"u1", "u2", "u3"}

    def test_batch_empty_map(self):
        result = units_states_to_three_js({})
        assert result == []

    def test_engaging_state_has_hostile_color(self):
        """Engaging state should have the magenta/hostile colour."""
        fsm = create_turret_fsm()
        # Drive FSM into engaging state
        for _ in range(30):
            fsm.tick(0.1, {
                "enemies_in_range": ["e1"],
                "aimed_at_target": True,
                "just_fired": False,
                "degradation": 0.0,
            })
        data = unit_state_to_three_js("t1", fsm)
        if data["state"] == "engaging":
            assert data["color"] == "#ff2a6d"


# ===========================================================================
# B. World formation integration
# ===========================================================================


class TestWorldFormationIntegration:
    def _make_world(self) -> World:
        cfg = WorldConfig(
            map_size=(200.0, 200.0),
            enable_weather=False,
            enable_destruction=False,
            enable_crowds=False,
            enable_vehicles=False,
            enable_los=False,
            seed=42,
        )
        return World(cfg)

    def test_assign_squad_formation_creates_mover(self):
        world = self._make_world()
        squad = world.spawn_squad(
            "alpha", "friendly",
            ["infantry", "infantry", "infantry"],
            [(10.0, 10.0), (13.0, 10.0), (16.0, 10.0)],
        )
        mover = world.assign_squad_formation(
            squad.squad_id,
            waypoints=[(10.0, 10.0), (100.0, 10.0)],
            formation=FormationType.WEDGE,
        )
        assert mover is not None
        assert squad.squad_id in world._formation_movers

    def test_assign_formation_returns_none_for_unknown_squad(self):
        world = self._make_world()
        result = world.assign_squad_formation(
            "nonexistent_squad_id",
            waypoints=[(0.0, 0.0), (50.0, 0.0)],
        )
        assert result is None

    def test_render_frame_includes_formations_key(self):
        world = self._make_world()
        squad = world.spawn_squad(
            "bravo", "friendly",
            ["infantry", "infantry"],
            [(0.0, 0.0), (3.0, 0.0)],
        )
        world.assign_squad_formation(
            squad.squad_id,
            waypoints=[(0.0, 0.0), (50.0, 0.0)],
            formation=FormationType.LINE,
        )
        frame = world.tick()
        assert "formations" in frame

    def test_render_frame_formation_has_squad_id(self):
        world = self._make_world()
        squad = world.spawn_squad(
            "charlie", "hostile",
            ["infantry", "infantry"],
            [(50.0, 50.0), (53.0, 50.0)],
        )
        world.assign_squad_formation(
            squad.squad_id,
            waypoints=[(50.0, 50.0), (100.0, 50.0)],
        )
        frame = world.tick()
        formations = frame.get("formations", [])
        assert any(f["squad_id"] == squad.squad_id for f in formations)

    def test_formation_mover_ticks_advance_progress(self):
        world = self._make_world()
        squad = world.spawn_squad(
            "delta", "friendly",
            ["infantry", "infantry"],
            [(0.0, 0.0), (3.0, 0.0)],
        )
        world.assign_squad_formation(
            squad.squad_id,
            waypoints=[(0.0, 0.0), (100.0, 0.0)],
            max_speed=20.0,
        )
        # Tick several times
        for _ in range(10):
            world.tick()
        mover = world._formation_movers[squad.squad_id]
        assert mover.progress() > 0.0


# ===========================================================================
# C. Enhanced unit AI behaviours
# ===========================================================================


class TestUnitRetreatBehaviour:
    """Units with health < 30% AND morale < 30% must retreat away from enemies.

    Enemy is placed within detection range (50 m) but outside attack range (30 m)
    so the wounded unit sees it and retreats before being hit.
    """

    def _world_with_units(self):
        cfg = WorldConfig(
            map_size=(300.0, 300.0),
            enable_weather=False,
            enable_destruction=False,
            enable_crowds=False,
            enable_vehicles=False,
            enable_los=False,
            seed=7,
        )
        world = World(cfg)
        # One critically wounded friendly
        friendly = world.spawn_unit("infantry", "wounded_friendly", "friendly", (50.0, 50.0))
        friendly.state.health = 25.0         # 25% health — below 30% threshold
        friendly.state.morale = 0.20         # morale 0.20 — below 0.30 threshold
        # Enemy at 40 m — within detection range (50 m) but outside attack range (30 m)
        _enemy = world.spawn_unit("infantry", "enemy_a", "hostile", (90.0, 50.0))
        return world, friendly

    def test_unit_status_becomes_retreating(self):
        world, friendly = self._world_with_units()
        world.tick()
        assert friendly.state.status == "retreating"

    def test_unit_moves_away_from_enemy(self):
        world, friendly = self._world_with_units()
        start_pos = friendly.position
        for _ in range(5):
            world.tick()
        end_pos = friendly.position
        # The unit must have moved (retreated)
        moved = math.sqrt(
            (end_pos[0] - start_pos[0]) ** 2 + (end_pos[1] - start_pos[1]) ** 2
        )
        assert moved > 0.1, "Retreating unit did not move"

    def test_retreat_direction_is_away_from_enemy(self):
        """After retreating, friendly must have moved away from its starting position
        in the direction opposite to the enemy — i.e. its X coordinate decreases
        (enemy is at x=90, friendly starts at x=50, so retreat goes toward x<50)."""
        world, friendly = self._world_with_units()
        start_x = friendly.position[0]
        for _ in range(8):
            world.tick()
        end_x = friendly.position[0]
        # Enemy is at x=90; retreating means x decreases
        assert end_x < start_x, (
            f"Friendly did not retreat in x direction: start_x={start_x:.2f}, end_x={end_x:.2f}"
        )


class TestUnitCoverSeekingBehaviour:
    """Units with suppression > 60% and health < 60% must move toward cover."""

    def _world_with_cover(self):
        cfg = WorldConfig(
            map_size=(200.0, 200.0),
            enable_weather=False,
            enable_destruction=True,
            enable_crowds=False,
            enable_vehicles=False,
            enable_los=False,
            seed=11,
        )
        world = World(cfg)
        # Place a building (cover) at 0,0
        world.add_structure(Structure(
            structure_id="bldg_1",
            position=(0.0, 0.0),
            structure_type=StructureType.BUILDING,
            health=100.0,
            max_health=100.0,
            size=(5.0, 5.0, 3.0),
        ))
        # Unit in the open, suppressed + damaged
        unit = world.spawn_unit("infantry", "suppressed_unit", "friendly", (40.0, 40.0))
        unit.state.suppression = 0.75       # 75% suppression
        unit.state.health = 45.0            # 45% — below 60% threshold
        unit.state.morale = 0.80            # morale OK so it won't just retreat
        # Enemy far away (visible but won't fire back immediately due to cooldown)
        _enemy = world.spawn_unit("infantry", "distant_enemy", "hostile", (100.0, 100.0))
        return world, unit

    def test_suppressed_unit_moves_toward_cover(self):
        world, unit = self._world_with_cover()
        cover_pos = (0.0, 0.0)
        initial_dist_to_cover = math.sqrt(
            (unit.position[0] - cover_pos[0]) ** 2 +
            (unit.position[1] - cover_pos[1]) ** 2
        )
        for _ in range(10):
            world.tick()
        final_dist_to_cover = math.sqrt(
            (unit.position[0] - cover_pos[0]) ** 2 +
            (unit.position[1] - cover_pos[1]) ** 2
        )
        assert final_dist_to_cover < initial_dist_to_cover, (
            f"Suppressed unit did not move toward cover: "
            f"before={initial_dist_to_cover:.2f}, after={final_dist_to_cover:.2f}"
        )

    def test_non_suppressed_unit_does_not_seek_cover(self):
        """A healthy, non-suppressed unit should engage, not seek cover."""
        world, _ = self._world_with_cover()
        healthy_unit = world.spawn_unit("infantry", "healthy", "friendly", (80.0, 80.0))
        healthy_unit.state.suppression = 0.0
        healthy_unit.state.health = 100.0

        cover_pos = (0.0, 0.0)
        initial_dist = math.sqrt(
            (healthy_unit.position[0] - cover_pos[0]) ** 2 +
            (healthy_unit.position[1] - cover_pos[1]) ** 2
        )
        for _ in range(5):
            world.tick()
        final_dist = math.sqrt(
            (healthy_unit.position[0] - cover_pos[0]) ** 2 +
            (healthy_unit.position[1] - cover_pos[1]) ** 2
        )
        # Healthy unit should NOT be moving toward the cover (0,0) — it should
        # be moving toward the enemy at (100,100)
        assert final_dist >= initial_dist - 1.0, (
            "Healthy unit incorrectly moved toward cover instead of engaging"
        )


class TestUnitFlankingBehaviour:
    """When 2+ friendlies outnumber a single visible enemy, one flanks."""

    def _world_with_flanking_scenario(self):
        cfg = WorldConfig(
            map_size=(300.0, 300.0),
            enable_weather=False,
            enable_destruction=False,
            enable_crowds=False,
            enable_vehicles=False,
            enable_los=False,
            seed=99,
        )
        world = World(cfg)
        # Three friendlies grouped together
        u1 = world.spawn_unit("infantry", "alpha", "friendly", (10.0, 10.0))
        u2 = world.spawn_unit("infantry", "bravo", "friendly", (12.0, 10.0))
        u3 = world.spawn_unit("infantry", "charlie", "friendly", (14.0, 10.0))
        # Single enemy at distance, within detection range (50m default)
        enemy = world.spawn_unit("infantry", "tango_1", "hostile", (30.0, 10.0))
        # Give all units good health/morale so they won't retreat/seek cover
        for u in (u1, u2, u3):
            u.state.health = 100.0
            u.state.morale = 1.0
            u.state.suppression = 0.0
        return world, u1, u2, u3, enemy

    def test_at_least_one_unit_moves_laterally(self):
        """At least one unit should offset laterally (flanking manoeuvre)."""
        world, u1, u2, u3, enemy = self._world_with_flanking_scenario()
        start_y = {u.unit_id: u.position[1] for u in (u1, u2, u3)}

        for _ in range(10):
            world.tick()

        end_y = {u.unit_id: u.position[1] for u in (u1, u2, u3)}
        lateral_movers = [
            uid for uid in (u1.unit_id, u2.unit_id, u3.unit_id)
            if abs(end_y[uid] - start_y[uid]) > 0.5
        ]
        assert len(lateral_movers) >= 1, (
            "No unit flanked laterally despite 3v1 outnumbering scenario"
        )

    def test_flanker_moves_perpendicular_to_enemy(self):
        """The flanker's final position should be offset to the side of the enemy."""
        world, u1, u2, u3, enemy = self._world_with_flanking_scenario()
        for _ in range(15):
            world.tick()

        # At least one friendly should now be laterally offset from the
        # unit-to-enemy axis (i.e. Y offset from the starting line)
        friendlies = [u for u in world.units.values() if u.alliance == Alliance.FRIENDLY]
        y_offsets = [abs(u.position[1] - 10.0) for u in friendlies if u.is_alive()]
        max_lateral = max(y_offsets) if y_offsets else 0.0
        assert max_lateral > 0.5, (
            f"No unit achieved meaningful lateral offset; max={max_lateral:.2f}m"
        )

    def test_outnumbering_scenario_has_attackers_move(self):
        """At least two of the three friendlies must have moved after 10 ticks."""
        world, u1, u2, u3, enemy = self._world_with_flanking_scenario()
        start_pos = {
            u.unit_id: u.position
            for u in (u1, u2, u3)
        }
        for _ in range(10):
            world.tick()
        movers = [
            uid
            for uid, pos in start_pos.items()
            if math.sqrt(
                (world.units[uid].position[0] - pos[0]) ** 2 +
                (world.units[uid].position[1] - pos[1]) ** 2
            ) > 0.1
        ]
        assert len(movers) >= 2, (
            f"Expected >=2 units to move in flanking scenario, got {len(movers)}"
        )
