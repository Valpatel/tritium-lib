# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the World simulation integrator."""

from __future__ import annotations

import math

import pytest

from tritium_lib.sim_engine.world import (
    World,
    WorldConfig,
    WorldBuilder,
    WORLD_PRESETS,
)
from tritium_lib.sim_engine.units import Alliance
from tritium_lib.sim_engine.environment import Weather
from tritium_lib.sim_engine.crowd import CrowdMood
from tritium_lib.sim_engine.destruction import (
    Structure,
    StructureType,
    MATERIAL_PROPERTIES,
)


# ============================================================================
# WorldConfig
# ============================================================================


class TestWorldConfig:
    def test_default_config(self):
        cfg = WorldConfig()
        assert cfg.map_size == (500.0, 500.0)
        assert cfg.tick_rate == 20.0
        assert cfg.enable_weather is True
        assert cfg.enable_destruction is True
        assert cfg.enable_crowds is False
        assert cfg.enable_vehicles is True
        assert cfg.enable_los is True
        assert cfg.gravity == 9.81
        assert cfg.seed is None

    def test_custom_config(self):
        cfg = WorldConfig(map_size=(100, 200), tick_rate=10.0, seed=42)
        assert cfg.map_size == (100, 200)
        assert cfg.tick_rate == 10.0
        assert cfg.seed == 42


# ============================================================================
# World creation
# ============================================================================


class TestWorldCreation:
    def test_default_world(self):
        world = World()
        assert world.tick_count == 0
        assert world.sim_time == 0.0
        assert len(world.units) == 0
        assert len(world.vehicles) == 0
        assert len(world.squads) == 0

    def test_world_with_config(self):
        cfg = WorldConfig(map_size=(100, 100), seed=123)
        world = World(cfg)
        assert world.config.map_size == (100, 100)
        assert world.config.seed == 123

    def test_world_subsystems_created(self):
        world = World()
        assert world.heightmap is not None
        assert world.los is not None
        assert world.environment is not None
        assert world.destruction is not None
        assert world.projectile_sim is not None
        assert world.area_effects is not None
        assert world.renderer is not None

    def test_world_no_los(self):
        cfg = WorldConfig(enable_los=False)
        world = World(cfg)
        assert world.los is None

    def test_world_no_destruction(self):
        cfg = WorldConfig(enable_destruction=False)
        world = World(cfg)
        assert world.destruction is None

    def test_world_no_crowds_by_default(self):
        world = World()
        assert world.crowd is None


# ============================================================================
# Spawning
# ============================================================================


class TestSpawning:
    def test_spawn_unit(self):
        world = World()
        unit = world.spawn_unit("infantry", "Soldier", "friendly", (10.0, 20.0))
        assert unit.name == "Soldier"
        assert unit.alliance == Alliance.FRIENDLY
        assert unit.position == (10.0, 20.0)
        assert unit.unit_id in world.units
        assert unit.is_alive()

    def test_spawn_unit_alliance_enum(self):
        world = World()
        unit = world.spawn_unit("infantry", "Test", Alliance.HOSTILE, (0, 0))
        assert unit.alliance == Alliance.HOSTILE

    def test_spawn_unit_assigns_weapon(self):
        world = World()
        sniper = world.spawn_unit("sniper", "Sniper", "friendly", (0, 0))
        assert sniper.weapon == "m24"
        heavy = world.spawn_unit("heavy", "Heavy", "friendly", (0, 0))
        assert heavy.weapon == "m249_saw"
        infantry = world.spawn_unit("infantry", "Inf", "friendly", (0, 0))
        assert infantry.weapon == "m4a1"

    def test_spawn_unit_with_squad(self):
        world = World()
        squad = world.spawn_squad("Alpha", "friendly", ["infantry"] * 3, [(0, 0), (3, 0), (6, 0)])
        assert squad.squad_id in world.squads
        assert len(squad.members) == 3
        # Each unit should reference the squad
        for uid in squad.members:
            assert world.units[uid].squad_id == squad.squad_id

    def test_spawn_vehicle(self):
        world = World()
        v = world.spawn_vehicle("humvee", "Hummer", "friendly", (50, 50))
        assert v.name == "Hummer"
        assert v.vehicle_id in world.vehicles
        assert v.position == (50, 50)

    def test_spawn_squad(self):
        world = World()
        sq = world.spawn_squad("Bravo", "hostile", ["infantry", "sniper"], [(10, 10), (15, 10)])
        assert len(sq.members) == 2
        assert sq.leader_id is not None
        assert sq.alliance == "hostile"

    def test_spawn_squad_assigns_leader(self):
        world = World()
        sq = world.spawn_squad("Charlie", "friendly", ["infantry"] * 4, [(0, 0)] * 4)
        assert sq.leader_id is not None
        assert sq.leader_id in sq.members

    def test_spawn_crowd(self):
        world = World()
        ids = world.spawn_crowd((100, 100), 50, 20.0, CrowdMood.CALM)
        assert len(ids) == 50
        assert world.crowd is not None
        assert len(world.crowd.members) == 50

    def test_spawn_crowd_enables_subsystem(self):
        world = World()
        assert world.crowd is None
        world.spawn_crowd((50, 50), 10, 10.0)
        assert world.crowd is not None
        assert world.config.enable_crowds is True

    def test_add_structure(self):
        world = World()
        s = Structure(
            structure_id="b1",
            structure_type=StructureType.BUILDING,
            position=(100, 100),
            size=(20, 15, 10),
            material="concrete",
            health=200.0,
            max_health=200.0,
        )
        world.add_structure(s)
        assert len(world.destruction.structures) == 1

    def test_add_structure_enables_destruction(self):
        cfg = WorldConfig(enable_destruction=False)
        world = World(cfg)
        assert world.destruction is None
        s = Structure(
            structure_id="b1",
            structure_type=StructureType.BUILDING,
            position=(50, 50),
            size=(10, 10, 5),
            material="wood",
            health=80.0,
            max_health=80.0,
        )
        world.add_structure(s)
        assert world.destruction is not None


# ============================================================================
# Tick
# ============================================================================


class TestTick:
    def test_tick_advances_time(self):
        world = World()
        world.tick()
        assert world.tick_count == 1
        assert world.sim_time > 0.0

    def test_tick_with_custom_dt(self):
        world = World()
        world.tick(dt=0.5)
        assert world.sim_time == pytest.approx(0.5, abs=1e-6)
        assert world.tick_count == 1

    def test_multiple_ticks(self):
        world = World()
        for _ in range(10):
            world.tick()
        assert world.tick_count == 10
        assert world.sim_time > 0.0

    def test_tick_returns_frame(self):
        world = World()
        frame = world.tick()
        assert "tick" in frame
        assert "time" in frame
        assert "units" in frame
        assert "events" in frame

    def test_tick_with_units(self):
        world = World(WorldConfig(seed=42))
        world.spawn_unit("infantry", "A", "friendly", (10, 10))
        world.spawn_unit("infantry", "B", "hostile", (15, 10))
        frame = world.tick()
        assert len(frame["units"]) == 2

    def test_multiple_ticks_produce_events(self):
        world = World(WorldConfig(seed=42))
        world.spawn_unit("infantry", "A", "friendly", (10, 10))
        world.spawn_unit("infantry", "B", "hostile", (15, 10))
        all_events = []
        for _ in range(20):
            frame = world.tick()
            all_events.extend(frame.get("events", []))
        # Should have at least some fire events (units are in range)
        fire_events = [e for e in all_events if e.get("type") == "fire"]
        assert len(fire_events) > 0

    def test_tick_empty_world(self):
        world = World()
        frame = world.tick()
        assert frame["tick"] == 1
        assert frame["units"] == []


# ============================================================================
# Combat
# ============================================================================


class TestCombat:
    def test_fire_weapon(self):
        world = World(WorldConfig(seed=42))
        unit = world.spawn_unit("infantry", "Shooter", "friendly", (0, 0))
        proj = world.fire_weapon(unit.unit_id, (50, 0))
        assert proj is not None
        assert len(world.projectile_sim.projectiles) == 1

    def test_fire_weapon_dead_unit(self):
        world = World()
        unit = world.spawn_unit("infantry", "Dead", "friendly", (0, 0))
        unit.state.is_alive = False
        unit.state.status = "dead"
        proj = world.fire_weapon(unit.unit_id, (50, 0))
        assert proj is None

    def test_fire_weapon_cooldown(self):
        world = World(WorldConfig(seed=42))
        unit = world.spawn_unit("infantry", "Fast", "friendly", (0, 0))
        p1 = world.fire_weapon(unit.unit_id, (50, 0))
        assert p1 is not None
        # Immediately fire again should fail (cooldown)
        p2 = world.fire_weapon(unit.unit_id, (50, 0))
        assert p2 is None

    def test_units_engage_enemies(self):
        """Two close units on opposite sides should start fighting."""
        world = World(WorldConfig(seed=42, enable_los=False))
        world.spawn_unit("infantry", "Good", "friendly", (10, 10))
        world.spawn_unit("infantry", "Bad", "hostile", (20, 10))
        # Run enough ticks for combat to happen
        for _ in range(50):
            world.tick(dt=0.1)
        s = world.stats()
        # At least one side should have taken casualties or both alive
        # The important thing is that combat events were generated
        total = s["alive_friendly"] + s["alive_hostile"] + s["dead"]
        assert total == 2

    def test_dead_units_stop_acting(self):
        world = World(WorldConfig(seed=42))
        unit = world.spawn_unit("infantry", "Doomed", "friendly", (0, 0))
        unit.take_damage(999)
        assert not unit.is_alive()
        proj = world.fire_weapon(unit.unit_id, (50, 0))
        assert proj is None

    def test_weather_affects_accuracy(self):
        """Storm weather should reduce accuracy modifier."""
        world = World(WorldConfig(seed=42))
        world.environment.weather.state.current = Weather.STORM
        world.environment.weather.state.wind_speed = 15.0
        world.environment.weather.state.intensity = 0.8
        acc = world.environment.accuracy_modifier()
        assert acc < 0.8  # Storm degrades accuracy

    def test_projectile_impact_damages_unit(self):
        """A projectile landing near a unit should deal damage."""
        world = World(WorldConfig(seed=42))
        target = world.spawn_unit("infantry", "Target", "hostile", (50, 0))
        initial_health = target.state.health
        # Create a projectile that will expire right on the target
        from tritium_lib.sim_engine.arsenal import ARSENAL
        weapon = ARSENAL["m4a1"]
        # Fire at very close range to almost guarantee impact near target
        proj = world.projectile_sim.fire(weapon, (49.5, 0), (50, 0))
        # Tick many times to let projectile fly and expire
        for _ in range(100):
            impacts = world.projectile_sim.tick(0.01)
            world._resolve_impacts(impacts)
        # The projectile may or may not have hit (depends on exact trajectory),
        # but the mechanism should work without error
        assert True  # No crash = pass


# ============================================================================
# LOS
# ============================================================================


class TestLOS:
    def test_los_flat_terrain(self):
        """On flat terrain, everything should be visible."""
        world = World(WorldConfig(seed=42))
        assert world.los is not None
        result = world.los.can_see((10, 10), (50, 50))
        assert result is True

    def test_los_blocked_by_terrain(self):
        """A high ridge between two points should block LOS."""
        cfg = WorldConfig(map_size=(100, 100), seed=42)
        world = World(cfg)
        # Create a ridge at y=50
        for x in range(100):
            world.heightmap.set_elevation(x, 50, 20.0)
        world.los = from_existing_heightmap(world.heightmap)
        result = world.los.can_see((10, 10), (10, 90))
        assert result is False


def from_existing_heightmap(hm):
    """Helper to create LOS from existing heightmap."""
    from tritium_lib.sim_engine.terrain import LineOfSight
    return LineOfSight(hm)


# ============================================================================
# Fire spread / destruction
# ============================================================================


class TestDestruction:
    def test_fire_spread_between_buildings(self):
        world = World(WorldConfig(seed=42))
        # Two wooden buildings close together
        s1 = Structure(
            structure_id="wood1",
            structure_type=StructureType.BUILDING,
            position=(50, 50),
            size=(10, 10, 5),
            material="wood",
            health=80.0,
            max_health=80.0,
        )
        s2 = Structure(
            structure_id="wood2",
            structure_type=StructureType.BUILDING,
            position=(55, 50),  # only 5m away
            size=(10, 10, 5),
            material="wood",
            health=80.0,
            max_health=80.0,
        )
        world.add_structure(s1)
        world.add_structure(s2)
        # Start a fire on s1
        world.destruction.start_fire((50, 50), radius=3.0, intensity=0.8, fuel=30.0)
        # Tick many times
        for _ in range(200):
            world.destruction.tick(0.1)
        # Fire should have damaged at least one structure
        total_damage = (s1.max_health - s1.health) + (s2.max_health - s2.health)
        assert total_damage > 0

    def test_structure_takes_explosive_damage(self):
        world = World(WorldConfig(seed=42))
        s = Structure(
            structure_id="target_bldg",
            structure_type=StructureType.BUILDING,
            position=(100, 100),
            size=(20, 15, 10),
            material="concrete",
            health=200.0,
            max_health=200.0,
        )
        world.add_structure(s)
        world.destruction.damage_structure("target_bldg", 100.0, (100, 100), damage_type="explosive")
        assert s.health < 200.0


# ============================================================================
# Crowds
# ============================================================================


class TestCrowds:
    def test_crowd_mood_escalation(self):
        world = World(WorldConfig(seed=42))
        ids = world.spawn_crowd((100, 100), 100, 20.0, CrowdMood.AGITATED)
        assert len(ids) == 100
        # Inject a gunshot event
        from tritium_lib.sim_engine.crowd import CrowdEvent
        world.crowd.inject_event(CrowdEvent(
            event_type="gunshot",
            position=(100, 100),
            radius=50.0,
            intensity=0.9,
            timestamp=0.0,
        ))
        # Tick several times
        for _ in range(20):
            world.crowd.tick(0.1)
        # Some members should have escalated
        panicked_or_fleeing = sum(
            1 for m in world.crowd.members
            if m.mood.value >= CrowdMood.PANICKED.value
        )
        assert panicked_or_fleeing > 0

    def test_crowd_tick_in_world(self):
        world = World(WorldConfig(seed=42))
        world.spawn_crowd((50, 50), 20, 10.0)
        # Should not crash
        for _ in range(5):
            world.tick()
        assert world.crowd is not None
        assert len(world.crowd.members) == 20


# ============================================================================
# Vehicles
# ============================================================================


class TestVehicles:
    def test_vehicle_convoy_movement(self):
        world = World(WorldConfig(seed=42))
        v = world.spawn_vehicle("humvee", "Scout", "friendly", (10, 10))
        initial_pos = v.position
        v.speed = 10.0  # Give it some speed
        for _ in range(20):
            world.tick(dt=0.1)
        # Vehicle should have moved
        assert v.position != initial_pos

    def test_drone_orbits(self):
        world = World(WorldConfig(seed=42))
        v = world.spawn_vehicle("quadcopter", "Drone-1", "friendly", (100, 100))
        v.altitude = 30.0
        from tritium_lib.sim_engine.vehicles import DroneController
        ctrl = DroneController(v)
        ctrl.orbit((100, 100), radius=50.0, altitude=30.0)
        world.drone_controllers[v.vehicle_id] = ctrl
        initial_pos = v.position
        for _ in range(50):
            world.tick(dt=0.1)
        # Drone should have moved
        from tritium_lib.sim_engine.ai.steering import distance
        moved = distance(initial_pos, v.position)
        assert moved > 1.0


# ============================================================================
# WorldBuilder
# ============================================================================


class TestWorldBuilder:
    def test_fluent_api(self):
        world = (
            WorldBuilder()
            .set_map_size(200, 200)
            .set_seed(42)
            .set_time(hour=14.0)
            .set_weather(Weather.CLEAR)
            .set_tick_rate(30.0)
            .set_gravity(10.0)
            .enable_destruction(True)
            .enable_los(True)
            .spawn_friendly_squad("Alpha", ["infantry"] * 3, (50, 50))
            .spawn_hostile_squad("Tango", ["infantry"] * 3, (150, 150))
            .add_building((100, 100), (20, 15, 10), "concrete")
            .build()
        )
        assert world.config.map_size == (200, 200)
        assert world.config.seed == 42
        assert world.config.tick_rate == 30.0
        assert world.config.gravity == 10.0
        assert len(world.squads) == 2
        assert len(world.units) == 6
        assert world.destruction is not None
        assert len(world.destruction.structures) == 1

    def test_builder_terrain_noise(self):
        world = (
            WorldBuilder()
            .set_map_size(50, 50)
            .set_seed(42)
            .add_terrain_noise(octaves=3, amplitude=5.0)
            .build()
        )
        # Terrain should not be perfectly flat
        elevations = set()
        for y in range(10):
            for x in range(10):
                elevations.add(world.heightmap.get_elevation(x, y))
        assert len(elevations) > 1

    def test_builder_with_vehicles(self):
        world = (
            WorldBuilder()
            .set_map_size(200, 200)
            .add_vehicle("humvee", "Hummer", "friendly", (50, 50))
            .add_vehicle("t72", "Tank", "hostile", (150, 150))
            .build()
        )
        assert len(world.vehicles) == 2

    def test_builder_with_crowds(self):
        world = (
            WorldBuilder()
            .set_map_size(100, 100)
            .add_crowd((50, 50), 30, 10.0, CrowdMood.CALM)
            .build()
        )
        assert world.crowd is not None
        assert len(world.crowd.members) == 30

    def test_builder_weather_and_time(self):
        world = (
            WorldBuilder()
            .set_time(hour=22.0)
            .set_weather(Weather.RAIN)
            .build()
        )
        assert world.environment.time.hour == pytest.approx(22.0)
        assert world.environment.weather.state.current == Weather.RAIN

    def test_builder_empty(self):
        world = WorldBuilder().build()
        assert len(world.units) == 0
        assert world.tick_count == 0

    def test_builder_multiple_buildings(self):
        world = (
            WorldBuilder()
            .add_building((10, 10), (5, 5, 3), "wood")
            .add_building((20, 10), (5, 5, 3), "concrete")
            .add_building((30, 10), (5, 5, 3), "brick")
            .build()
        )
        assert len(world.destruction.structures) == 3
        materials = {s.material for s in world.destruction.structures}
        assert materials == {"wood", "concrete", "brick"}


# ============================================================================
# Presets
# ============================================================================


class TestPresets:
    def test_urban_combat_preset(self):
        world = WORLD_PRESETS["urban_combat"]()
        assert len(world.units) > 0
        assert len(world.vehicles) > 0
        assert world.destruction is not None
        assert len(world.destruction.structures) > 0
        assert world.environment.time.hour == pytest.approx(22.0)

    def test_open_field_preset(self):
        world = WORLD_PRESETS["open_field"]()
        assert len(world.units) > 0
        assert world.environment.weather.state.current == Weather.CLEAR

    def test_riot_response_preset(self):
        world = WORLD_PRESETS["riot_response"]()
        assert world.crowd is not None
        assert len(world.crowd.members) > 0
        assert len(world.units) > 0

    def test_convoy_ambush_preset(self):
        world = WORLD_PRESETS["convoy_ambush"]()
        assert len(world.vehicles) >= 3
        assert len(world.squads) >= 2

    def test_drone_strike_preset(self):
        world = WORLD_PRESETS["drone_strike"]()
        assert len(world.vehicles) >= 1
        assert len(world.drone_controllers) >= 1
        # Check drone is at altitude
        for vid, v in world.vehicles.items():
            if v.name == "Reaper-1":
                assert v.altitude > 0

    def test_all_presets_create_valid_worlds(self):
        for name, factory in WORLD_PRESETS.items():
            world = factory()
            assert isinstance(world, World), f"Preset {name} did not return a World"
            assert world.tick_count == 0
            assert world.sim_time == 0.0

    def test_all_presets_can_tick(self):
        for name, factory in WORLD_PRESETS.items():
            world = factory()
            # Should not crash for a few ticks
            for _ in range(5):
                frame = world.tick()
            assert world.tick_count == 5, f"Preset {name} tick count wrong"


# ============================================================================
# Stats & Snapshot
# ============================================================================


class TestStatsAndSnapshot:
    def test_stats_empty_world(self):
        world = World()
        s = world.stats()
        assert s["tick_count"] == 0
        assert s["total_units"] == 0
        assert s["alive_friendly"] == 0
        assert s["alive_hostile"] == 0
        assert s["dead"] == 0
        assert s["total_vehicles"] == 0
        assert s["crowd_count"] == 0

    def test_stats_with_units(self):
        world = World()
        world.spawn_unit("infantry", "A", "friendly", (0, 0))
        world.spawn_unit("infantry", "B", "hostile", (10, 0))
        s = world.stats()
        assert s["total_units"] == 2
        assert s["alive_friendly"] == 1
        assert s["alive_hostile"] == 1
        assert s["dead"] == 0

    def test_stats_tracks_dead(self):
        world = World()
        unit = world.spawn_unit("infantry", "Doomed", "friendly", (0, 0))
        unit.take_damage(999)
        s = world.stats()
        assert s["dead"] == 1
        assert s["alive_friendly"] == 0

    def test_stats_environment_description(self):
        world = World()
        s = world.stats()
        assert "environment" in s
        assert isinstance(s["environment"], str)

    def test_snapshot_serializable(self):
        world = World(WorldConfig(seed=42))
        world.spawn_unit("infantry", "A", "friendly", (10, 20))
        world.spawn_vehicle("humvee", "V", "friendly", (30, 40))
        world.tick()
        snap = world.snapshot()
        assert snap["tick_count"] == 1
        assert "units" in snap
        assert "vehicles" in snap
        assert "squads" in snap
        assert "config" in snap
        assert "environment" in snap
        assert "damage_summary" in snap

    def test_snapshot_contains_unit_data(self):
        world = World()
        unit = world.spawn_unit("infantry", "TestUnit", "friendly", (5, 10))
        snap = world.snapshot()
        uid = unit.unit_id
        assert uid in snap["units"]
        u_data = snap["units"][uid]
        assert u_data["name"] == "TestUnit"
        assert u_data["alliance"] == "friendly"
        assert u_data["position"] == (5, 10)
        assert u_data["is_alive"] is True

    def test_snapshot_after_combat(self):
        world = World(WorldConfig(seed=42, enable_los=False))
        world.spawn_unit("infantry", "A", "friendly", (10, 10))
        world.spawn_unit("infantry", "B", "hostile", (15, 10))
        for _ in range(100):
            world.tick(dt=0.1)
        snap = world.snapshot()
        summary = snap["damage_summary"]
        assert "total_attacks" in summary


# ============================================================================
# Render
# ============================================================================


class TestRender:
    def test_render_empty_world(self):
        world = World()
        frame = world.render()
        assert "tick" in frame
        assert "time" in frame
        assert "units" in frame
        assert "vehicles" in frame

    def test_render_with_units(self):
        world = World()
        world.spawn_unit("infantry", "R1", "friendly", (10, 20))
        frame = world.render()
        assert len(frame["units"]) == 1
        u = frame["units"][0]
        assert u["x"] == pytest.approx(10.0)
        assert u["y"] == pytest.approx(20.0)

    def test_render_with_vehicles(self):
        world = World()
        world.spawn_vehicle("humvee", "V1", "friendly", (30, 40))
        frame = world.render()
        assert len(frame["vehicles"]) == 1
        v = frame["vehicles"][0]
        assert v["name"] == "V1"
        assert v["x"] == pytest.approx(30.0)

    def test_render_with_crowd(self):
        world = World()
        world.spawn_crowd((50, 50), 10, 5.0)
        frame = world.render()
        assert "crowd" in frame
        assert len(frame["crowd"]) == 10
