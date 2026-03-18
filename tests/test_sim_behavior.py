# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for sim_engine behavior, combat, and world modules moved from SC."""

from __future__ import annotations

import math
import pytest

from tritium_lib.sim_engine.core.entity import SimulationTarget
from tritium_lib.sim_engine.core.state_machine import StateMachine, State


# ---------------------------------------------------------------------------
# Behavior: unit_states — FSM creation and transitions
# ---------------------------------------------------------------------------


class TestUnitStates:
    """Test FSM creation and state transitions."""

    def test_create_turret_fsm(self):
        from tritium_lib.sim_engine.behavior.unit_states import create_turret_fsm

        sm = create_turret_fsm()
        assert isinstance(sm, StateMachine)
        assert sm.current_state == "idle"
        # Should have at least: idle, scanning, tracking, engaging, cooldown
        assert "idle" in sm.state_names
        assert "scanning" in sm.state_names
        assert "tracking" in sm.state_names
        assert "engaging" in sm.state_names
        assert "cooldown" in sm.state_names

    def test_create_rover_fsm(self):
        from tritium_lib.sim_engine.behavior.unit_states import create_rover_fsm

        sm = create_rover_fsm()
        assert sm.current_state == "idle"
        assert "patrolling" in sm.state_names
        assert "pursuing" in sm.state_names
        assert "retreating" in sm.state_names

    def test_create_drone_fsm(self):
        from tritium_lib.sim_engine.behavior.unit_states import create_drone_fsm

        sm = create_drone_fsm()
        assert sm.current_state == "idle"
        assert "scouting" in sm.state_names
        assert "orbiting" in sm.state_names

    def test_create_hostile_fsm(self):
        from tritium_lib.sim_engine.behavior.unit_states import create_hostile_fsm

        sm = create_hostile_fsm()
        assert sm.current_state == "spawning"
        assert "advancing" in sm.state_names
        assert "engaging" in sm.state_names
        assert "flanking" in sm.state_names
        assert "fleeing" in sm.state_names

    def test_create_fsm_for_type_turret(self):
        from tritium_lib.sim_engine.behavior.unit_states import create_fsm_for_type

        sm = create_fsm_for_type("turret")
        assert sm is not None
        assert sm.current_state == "idle"

    def test_create_fsm_for_type_hostile(self):
        from tritium_lib.sim_engine.behavior.unit_states import create_fsm_for_type

        sm = create_fsm_for_type("hostile_person")
        assert sm is not None
        assert sm.current_state == "spawning"

    def test_create_fsm_for_type_unknown(self):
        from tritium_lib.sim_engine.behavior.unit_states import create_fsm_for_type

        sm = create_fsm_for_type("unknown_type_xyz")
        assert sm is None

    def test_turret_fsm_transition_idle_to_scanning(self):
        from tritium_lib.sim_engine.behavior.unit_states import create_turret_fsm

        sm = create_turret_fsm()
        # Tick with no enemies — transition via condition (always from idle)
        sm.tick(2.0, {})
        assert sm.current_state == "scanning"

    def test_hostile_fsm_transition_advancing_to_engaging(self):
        from tritium_lib.sim_engine.behavior.unit_states import create_hostile_fsm

        sm = create_hostile_fsm()
        # Move past spawning
        sm.tick(2.0, {})
        assert sm.current_state == "advancing"
        # Tick with enemies in range and weapon range
        ctx = {"enemies_in_range": ["e1"], "enemy_in_weapon_range": True}
        sm.tick(0.1, ctx)
        assert sm.current_state == "engaging"


# ---------------------------------------------------------------------------
# Behavior: unit_missions — NPCMission creation
# ---------------------------------------------------------------------------


class TestUnitMissions:
    """Test UnitMissionSystem."""

    def test_create_mission_system(self):
        from tritium_lib.sim_engine.behavior.unit_missions import UnitMissionSystem

        ms = UnitMissionSystem()
        assert ms is not None
        assert ms.IDLE_CHECK_INTERVAL == 2.0

    def test_assign_friendly_mission(self):
        from tritium_lib.sim_engine.behavior.unit_missions import UnitMissionSystem

        ms = UnitMissionSystem()
        t = SimulationTarget(
            target_id="rover1",
            name="Rover Alpha",
            alliance="friendly",
            asset_type="rover",
            position=(10, 20),
            speed=5.0,
        )
        mission = ms.assign_starter_mission(t)
        assert mission is not None
        assert "type" in mission
        assert mission["type"] in ("patrol", "scout", "sweep", "hold", "escort")

    def test_assign_hostile_mission(self):
        from tritium_lib.sim_engine.behavior.unit_missions import UnitMissionSystem

        ms = UnitMissionSystem()
        t = SimulationTarget(
            target_id="h1",
            name="Hostile 1",
            alliance="hostile",
            asset_type="person",
            position=(50, 50),
            speed=3.0,
        )
        mission = ms.assign_starter_mission(t)
        assert mission is not None
        assert mission["type"] in ("assault", "infiltrate", "scout", "advance")

    def test_backstory_scripted(self):
        from tritium_lib.sim_engine.behavior.unit_missions import UnitMissionSystem

        ms = UnitMissionSystem()
        t = SimulationTarget(
            target_id="t1",
            name="Turret Alpha",
            alliance="friendly",
            asset_type="turret",
            position=(0, 0),
        )
        story = ms.generate_backstory_scripted(t)
        assert isinstance(story, str)
        assert len(story) > 10

    def test_npc_mission_creation(self):
        from tritium_lib.sim_engine.behavior.npc import NPCMission

        m = NPCMission(
            mission_type="commute",
            origin=(0, 0),
            destination=(100, 100),
        )
        assert m.mission_type == "commute"
        assert m.completed is False


# ---------------------------------------------------------------------------
# Behavior: behaviors — UnitBehaviors lookup
# ---------------------------------------------------------------------------


class TestBehaviors:
    """Test UnitBehaviors class."""

    def test_import(self):
        from tritium_lib.sim_engine.behavior.behaviors import UnitBehaviors

        assert UnitBehaviors is not None


# ---------------------------------------------------------------------------
# Combat: combat — Projectile and CombatSystem
# ---------------------------------------------------------------------------


class TestCombat:
    """Test CombatSystem and Projectile."""

    def test_projectile_creation(self):
        from tritium_lib.sim_engine.combat.combat import Projectile

        p = Projectile(
            id="p1",
            source_id="turret1",
            source_name="Turret Alpha",
            target_id="hostile1",
            position=(0, 0),
            target_pos=(10, 10),
            speed=80.0,
            damage=15.0,
        )
        assert p.id == "p1"
        assert p.hit is False
        assert p.missed is False
        assert p.z_height == 0.0  # not a mortar

    def test_projectile_mortar_arc(self):
        from tritium_lib.sim_engine.combat.combat import Projectile

        p = Projectile(
            id="m1",
            source_id="turret1",
            source_name="Turret Alpha",
            target_id="hostile1",
            position=(0, 0),
            target_pos=(100, 0),
            is_mortar=True,
            arc_peak=20.0,
            flight_progress=0.5,
            total_flight_dist=100.0,
        )
        # At midpoint, z = 4 * peak * 0.5 * 0.5 = peak
        assert abs(p.z_height - 20.0) < 0.01

    def test_projectile_to_dict(self):
        from tritium_lib.sim_engine.combat.combat import Projectile

        p = Projectile(
            id="p2",
            source_id="s1",
            source_name="Source",
            target_id="t1",
            position=(5, 10),
            target_pos=(15, 20),
        )
        d = p.to_dict()
        assert d["id"] == "p2"
        assert d["position"]["x"] == 5
        assert d["position"]["y"] == 10

    def test_combat_system_creation(self):
        from tritium_lib.sim_engine.combat.combat import CombatSystem

        class DummyBus:
            def publish(self, topic, data):
                pass

        cs = CombatSystem(event_bus=DummyBus())
        assert cs.projectile_count == 0

    def test_combat_system_fire(self):
        from tritium_lib.sim_engine.combat.combat import CombatSystem
        import time

        class DummyBus:
            def __init__(self):
                self.events = []
            def publish(self, topic, data):
                self.events.append((topic, data))

        bus = DummyBus()
        cs = CombatSystem(event_bus=bus)

        source = SimulationTarget(
            target_id="turret1",
            name="Turret Alpha",
            alliance="friendly",
            asset_type="turret",
            position=(0, 0),
            weapon_range=50.0,
            weapon_damage=15.0,
            weapon_cooldown=1.0,
        )
        source.last_fired = 0  # ensure can fire

        target = SimulationTarget(
            target_id="hostile1",
            name="Hostile 1",
            alliance="hostile",
            asset_type="person",
            position=(10, 0),
        )

        proj = cs.fire(source, target)
        assert proj is not None
        assert cs.projectile_count == 1
        assert proj.source_id == "turret1"
        assert proj.target_id == "hostile1"


# ---------------------------------------------------------------------------
# Combat: weapons — Weapon and WeaponSystem
# ---------------------------------------------------------------------------


class TestWeapons:
    """Test Weapon and WeaponSystem."""

    def test_weapon_defaults(self):
        from tritium_lib.sim_engine.combat.weapons import Weapon

        w = Weapon()
        assert w.name == "nerf_blaster"
        assert w.damage == 10.0
        assert w.ammo == 30

    def test_weapon_system_equip(self):
        from tritium_lib.sim_engine.combat.weapons import WeaponSystem

        ws = WeaponSystem()
        ws.equip("unit1", "turret")
        w = ws.get_weapon("unit1")
        assert w is not None
        assert w.name == "nerf_turret_gun"
        assert w.damage == 15.0

    def test_weapon_catalog(self):
        from tritium_lib.sim_engine.combat.weapons import WEAPON_CATALOG

        assert "nerf_rifle" in WEAPON_CATALOG
        assert "nerf_shotgun" in WEAPON_CATALOG
        assert WEAPON_CATALOG["nerf_rifle"].damage == 12.0


# ---------------------------------------------------------------------------
# Combat: squads — Squad formation calculation
# ---------------------------------------------------------------------------


class TestSquads:
    """Test Squad and SquadManager."""

    def test_squad_wedge_offsets(self):
        from tritium_lib.sim_engine.combat.squads import Squad

        sq = Squad(
            squad_id="sq1",
            member_ids=["leader", "f1", "f2", "f3"],
            leader_id="leader",
            formation="wedge",
        )
        offsets = sq.get_formation_offsets()
        assert "leader" in offsets
        assert offsets["leader"] == (0.0, 0.0)
        assert "f1" in offsets
        # f1 should be offset from leader
        assert offsets["f1"] != (0.0, 0.0)

    def test_squad_line_offsets(self):
        from tritium_lib.sim_engine.combat.squads import Squad

        sq = Squad(
            squad_id="sq2",
            member_ids=["leader", "f1", "f2"],
            leader_id="leader",
            formation="line",
        )
        offsets = sq.get_formation_offsets()
        # Line formation: all y offsets should be 0
        for mid in ("f1", "f2"):
            assert offsets[mid][1] == 0.0

    def test_squad_circle_offsets(self):
        from tritium_lib.sim_engine.combat.squads import Squad, FORMATION_SPACING

        sq = Squad(
            squad_id="sq3",
            member_ids=["leader", "f1", "f2", "f3"],
            leader_id="leader",
            formation="circle",
        )
        offsets = sq.get_formation_offsets()
        # All followers should be FORMATION_SPACING from origin
        for mid in ("f1", "f2", "f3"):
            dx, dy = offsets[mid]
            dist = math.hypot(dx, dy)
            assert abs(dist - FORMATION_SPACING) < 0.01

    def test_squad_manager_creation(self):
        from tritium_lib.sim_engine.combat.squads import SquadManager

        sm = SquadManager()
        assert sm.get_squad("nonexistent") is None


# ---------------------------------------------------------------------------
# World: pathfinding — plan_path and grid_find_path
# ---------------------------------------------------------------------------


class TestPathfinding:
    """Test pathfinding functions."""

    def test_plan_path_stationary(self):
        from tritium_lib.sim_engine.world.pathfinding import plan_path

        result = plan_path((0, 0), (10, 10), "turret")
        assert result is None  # stationary

    def test_plan_path_flying(self):
        from tritium_lib.sim_engine.world.pathfinding import plan_path

        result = plan_path((0, 0), (10, 10), "drone")
        assert result == [(0, 0), (10, 10)]

    def test_plan_path_fallback(self):
        from tritium_lib.sim_engine.world.pathfinding import plan_path

        result = plan_path((0, 0), (50, 50), "rover")
        # Without street graph or terrain, should return direct path
        assert result == [(0, 0), (50, 50)]

    def test_smooth_path_collinear(self):
        from tritium_lib.sim_engine.world.grid_pathfinder import smooth_path

        # Three collinear points should reduce to two
        path = [(0, 0), (5, 5), (10, 10)]
        result = smooth_path(path)
        assert len(result) == 2
        assert result[0] == (0, 0)
        assert result[-1] == (10, 10)

    def test_smooth_path_non_collinear(self):
        from tritium_lib.sim_engine.world.grid_pathfinder import smooth_path

        # Non-collinear points should all be kept
        path = [(0, 0), (5, 0), (5, 5)]
        result = smooth_path(path)
        assert len(result) == 3

    def test_profile_for_unit(self):
        from tritium_lib.sim_engine.world.grid_pathfinder import profile_for_unit

        # Drone should be aerial
        assert profile_for_unit("drone") == "aerial"
        # Person should be pedestrian
        assert profile_for_unit("person") == "pedestrian"

    def test_movement_profiles_exist(self):
        from tritium_lib.sim_engine.world.grid_pathfinder import PROFILES

        assert "pedestrian" in PROFILES
        assert "light_vehicle" in PROFILES
        assert "heavy_vehicle" in PROFILES
        assert "aerial" in PROFILES


# ---------------------------------------------------------------------------
# World: cover — CoverSystem
# ---------------------------------------------------------------------------


class TestCover:
    """Test CoverSystem."""

    def test_cover_system_creation(self):
        from tritium_lib.sim_engine.world.cover import CoverSystem

        cs = CoverSystem()
        assert cs is not None

    def test_cover_bonus_no_cover(self):
        from tritium_lib.sim_engine.world.cover import CoverSystem

        cs = CoverSystem()
        bonus = cs.get_cover_bonus((5, 5), (20, 20))
        assert bonus == 0.0

    def test_cover_bonus_with_cover(self):
        from tritium_lib.sim_engine.world.cover import CoverSystem, CoverObject

        cs = CoverSystem()
        cs.add_cover(CoverObject(position=(5, 5), radius=3.0, cover_value=0.6))
        # Target at cover position, attacker from (20, 20)
        bonus = cs.get_cover_bonus((5, 5), (20, 20))
        # Cover is between target and attacker (same position), should give bonus
        assert bonus >= 0.0

    def test_cover_tick(self):
        from tritium_lib.sim_engine.world.cover import CoverSystem, CoverObject

        cs = CoverSystem()
        cs.add_cover(CoverObject(position=(10, 10), radius=5.0, cover_value=0.5))

        t = SimulationTarget(
            target_id="t1",
            name="Test",
            alliance="friendly",
            asset_type="rover",
            position=(10, 10),
        )
        cs.tick(0.1, {"t1": t})
        # Should have cached cover value
        assert cs.get_cover_reduction("t1") > 0.0


# ---------------------------------------------------------------------------
# World: vision — VisionSystem basic checks
# ---------------------------------------------------------------------------


class TestVision:
    """Test VisionSystem."""

    def test_vision_system_creation(self):
        from tritium_lib.sim_engine.world.vision import VisionSystem

        vs = VisionSystem()
        assert vs is not None

    def test_sighting_report(self):
        from tritium_lib.sim_engine.world.vision import SightingReport

        sr = SightingReport(
            observer_id="cam1",
            target_id="hostile1",
            observer_type="camera",
            confidence=0.9,
        )
        assert sr.observer_id == "cam1"
        assert sr.confidence == 0.9


# ---------------------------------------------------------------------------
# World: sensors — SensorSimulator
# ---------------------------------------------------------------------------


class TestSensors:
    """Test SensorSimulator."""

    def test_sensor_device(self):
        from tritium_lib.sim_engine.world.sensors import SensorDevice

        sd = SensorDevice(
            sensor_id="s1",
            name="Motion 1",
            sensor_type="motion",
            position=(10, 20),
            radius=5.0,
        )
        assert sd.sensor_id == "s1"
        assert sd.active is False

    def test_sensor_simulator_creation(self):
        from tritium_lib.sim_engine.world.sensors import SensorSimulator

        class DummyBus:
            def publish(self, topic, data):
                pass

        ss = SensorSimulator(event_bus=DummyBus())
        assert len(ss.sensors) == 0

    def test_sensor_add_and_query(self):
        from tritium_lib.sim_engine.world.sensors import SensorSimulator

        class DummyBus:
            def publish(self, topic, data):
                pass

        ss = SensorSimulator(event_bus=DummyBus())
        ss.add_sensor("s1", "Motion 1", "motion", (10, 20), 5.0)
        assert len(ss.sensors) == 1
        assert ss.sensors[0].sensor_id == "s1"
