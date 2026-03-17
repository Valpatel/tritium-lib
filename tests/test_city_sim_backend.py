# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the city simulation backend (CitySim).

Verifies that the Python-side city simulation correctly uses sim_engine
modules and produces valid Three.js-compatible frame data.
"""

from __future__ import annotations

import math
import pytest

from tritium_lib.sim_engine.demos.city_sim_backend import (
    CitySim,
    CivilianAgent,
    CityVehicle,
    CityEntity,
)
from tritium_lib.sim_engine.environment import Weather
from tritium_lib.sim_engine.crowd import CrowdMood


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def city() -> CitySim:
    """A fully set-up city sim with deterministic seed."""
    sim = CitySim(width=500.0, height=400.0, seed=42, hour=10.0)
    sim.setup()
    return sim


@pytest.fixture
def small_city() -> CitySim:
    """A smaller city for faster tests."""
    sim = CitySim(width=200.0, height=200.0, seed=99, hour=14.0)
    sim.setup()
    return sim


# ---------------------------------------------------------------------------
# Construction / Setup
# ---------------------------------------------------------------------------

class TestCitySimSetup:
    """Tests for CitySim initialization and setup."""

    def test_construction_defaults(self):
        sim = CitySim()
        assert sim.width == 500.0
        assert sim.height == 400.0
        assert sim.tick_count == 0
        assert sim.sim_time == 0.0
        assert not sim._is_setup

    def test_construction_custom(self):
        sim = CitySim(width=300, height=250, seed=7, hour=18.0, weather=Weather.RAIN)
        assert sim.width == 300.0
        assert sim.height == 250.0
        assert sim.seed == 7

    def test_setup_creates_map(self, city: CitySim):
        assert city.map_data is not None
        assert city.map_data.width == 500.0
        assert city.map_data.height == 400.0

    def test_setup_creates_roads(self, city: CitySim):
        assert len(city.roads) > 0

    def test_setup_creates_buildings(self, city: CitySim):
        assert len(city.buildings) > 0

    def test_setup_creates_trees(self, city: CitySim):
        assert len(city.trees) > 0

    def test_setup_creates_civilians(self, city: CitySim):
        assert len(city.civilians) == 25

    def test_setup_creates_cars(self, city: CitySim):
        cars = [v for v in city.city_vehicles if v.vehicle_type == "car"]
        assert len(cars) == 12

    def test_setup_creates_taxis(self, city: CitySim):
        taxis = [v for v in city.city_vehicles if v.vehicle_type == "taxi"]
        assert len(taxis) == 3

    def test_setup_creates_police(self, city: CitySim):
        assert len(city.police_units) == 10

    def test_setup_creates_crowd(self, city: CitySim):
        assert city.crowd is not None
        assert len(city.crowd.members) == 20

    def test_setup_factions(self, city: CitySim):
        assert "police" in city.diplomacy.factions
        assert "civilians" in city.diplomacy.factions
        assert "protestors" in city.diplomacy.factions

    def test_setup_is_setup_flag(self, city: CitySim):
        assert city._is_setup

    def test_setup_idempotent(self, city: CitySim):
        """Calling setup again should not crash (though not recommended)."""
        initial_civs = len(city.civilians)
        city._is_setup = False
        city.setup()
        # Will add more entities, but should not crash
        assert len(city.civilians) >= initial_civs


# ---------------------------------------------------------------------------
# Tick
# ---------------------------------------------------------------------------

class TestCitySimTick:
    """Tests for the tick/step method."""

    def test_tick_returns_frame(self, city: CitySim):
        frame = city.tick(dt=0.05)
        assert isinstance(frame, dict)
        assert frame["type"] == "city_frame"

    def test_tick_advances_time(self, city: CitySim):
        city.tick(0.1)
        assert city.tick_count == 1
        assert abs(city.sim_time - 0.1) < 1e-6

    def test_multiple_ticks(self, city: CitySim):
        for _ in range(10):
            city.tick(0.05)
        assert city.tick_count == 10
        assert abs(city.sim_time - 0.5) < 1e-6

    def test_tick_auto_setup(self):
        """Calling tick on an unsetup sim should auto-setup."""
        sim = CitySim(width=200, height=200, seed=1)
        frame = sim.tick(0.05)
        assert sim._is_setup
        assert isinstance(frame, dict)

    def test_tick_civilians_move(self, city: CitySim):
        # Record initial positions
        initial_positions = [(c.position[0], c.position[1]) for c in city.civilians]
        for _ in range(20):
            city.tick(0.1)
        # At least some civilians should have moved
        moved = 0
        for i, civ in enumerate(city.civilians):
            if abs(civ.position[0] - initial_positions[i][0]) > 0.1 or \
               abs(civ.position[1] - initial_positions[i][1]) > 0.1:
                moved += 1
        assert moved > 0, "No civilians moved after 20 ticks"

    def test_tick_vehicles_move(self, city: CitySim):
        initial_positions = [(v.position[0], v.position[1]) for v in city.city_vehicles]
        for _ in range(20):
            city.tick(0.1)
        moved = 0
        for i, veh in enumerate(city.city_vehicles):
            if abs(veh.position[0] - initial_positions[i][0]) > 0.1 or \
               abs(veh.position[1] - initial_positions[i][1]) > 0.1:
                moved += 1
        assert moved > 0, "No vehicles moved after 20 ticks"

    def test_tick_crowd_updates(self, city: CitySim):
        assert city.crowd is not None
        initial_positions = [m.position for m in city.crowd.members]
        for _ in range(10):
            city.tick(0.1)
        moved = sum(
            1 for i, m in enumerate(city.crowd.members)
            if abs(m.position[0] - initial_positions[i][0]) > 0.01 or
               abs(m.position[1] - initial_positions[i][1]) > 0.01
        )
        assert moved > 0, "No crowd members moved"


# ---------------------------------------------------------------------------
# Frame output
# ---------------------------------------------------------------------------

class TestFrameOutput:
    """Tests for the to_frame() output structure."""

    def test_frame_has_required_keys(self, city: CitySim):
        frame = city.to_frame()
        required = [
            "type", "tick", "sim_time", "map", "environment",
            "civilians", "vehicles", "police", "crowd",
            "buildings", "trees", "roads", "destruction",
            "events", "stats",
        ]
        for key in required:
            assert key in frame, f"Missing key: {key}"

    def test_frame_map_dimensions(self, city: CitySim):
        frame = city.to_frame()
        assert frame["map"]["width"] == 500.0
        assert frame["map"]["height"] == 400.0

    def test_frame_environment(self, city: CitySim):
        frame = city.to_frame()
        env = frame["environment"]
        assert "hour" in env
        assert "weather" in env
        assert "light_level" in env
        assert "temperature" in env
        assert "visibility" in env

    def test_frame_civilians_format(self, city: CitySim):
        frame = city.to_frame()
        assert len(frame["civilians"]) == 25
        civ = frame["civilians"][0]
        assert "id" in civ
        assert "x" in civ
        assert "y" in civ
        assert "heading" in civ
        assert "type" in civ
        assert civ["type"] == "civilian"

    def test_frame_vehicles_format(self, city: CitySim):
        frame = city.to_frame()
        assert len(frame["vehicles"]) == 15  # 12 cars + 3 taxis
        veh = frame["vehicles"][0]
        assert "id" in veh
        assert "x" in veh
        assert "y" in veh
        assert "heading" in veh
        assert "speed" in veh
        assert "type" in veh
        assert "color" in veh

    def test_frame_police_format(self, city: CitySim):
        frame = city.to_frame()
        assert len(frame["police"]) == 10
        cop = frame["police"][0]
        assert "id" in cop
        assert "x" in cop
        assert "y" in cop
        assert "health" in cop
        assert cop["type"] == "police"
        assert cop["alliance"] == "friendly"

    def test_frame_crowd_data(self, city: CitySim):
        frame = city.to_frame()
        crowd = frame["crowd"]
        assert "members" in crowd
        assert "stats" in crowd
        assert crowd["stats"]["total"] == 20

    def test_frame_buildings_data(self, city: CitySim):
        frame = city.to_frame()
        assert len(frame["buildings"]) > 0
        bldg = frame["buildings"][0]
        assert "id" in bldg
        assert "x" in bldg
        assert "y" in bldg
        assert "width" in bldg
        assert "height" in bldg
        assert "material" in bldg

    def test_frame_trees_data(self, city: CitySim):
        frame = city.to_frame()
        assert len(frame["trees"]) > 0
        tree = frame["trees"][0]
        assert "id" in tree
        assert "x" in tree
        assert "y" in tree
        assert "height" in tree

    def test_frame_roads_data(self, city: CitySim):
        frame = city.to_frame()
        assert len(frame["roads"]) > 0
        road = frame["roads"][0]
        assert len(road) >= 2
        assert "x" in road[0]
        assert "y" in road[0]

    def test_frame_stats(self, city: CitySim):
        frame = city.to_frame()
        stats = frame["stats"]
        assert stats["total_civilians"] == 25
        assert stats["total_vehicles"] == 15
        assert stats["total_police"] == 10
        assert stats["crowd_count"] == 20


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:
    """Tests for the stats() method."""

    def test_stats_keys(self, city: CitySim):
        s = city.stats()
        expected_keys = [
            "tick", "sim_time", "total_civilians", "walking_civilians",
            "idle_civilians", "total_vehicles", "moving_vehicles",
            "total_police", "crowd_count", "crowd_mood",
            "total_buildings", "total_trees", "environment",
        ]
        for key in expected_keys:
            assert key in s, f"Missing stat: {key}"

    def test_stats_values(self, city: CitySim):
        s = city.stats()
        assert s["total_civilians"] == 25
        assert s["total_vehicles"] == 15
        assert s["total_police"] == 10
        assert s["crowd_count"] == 20


# ---------------------------------------------------------------------------
# CivilianAgent
# ---------------------------------------------------------------------------

class TestCivilianAgent:
    """Tests for the CivilianAgent dataclass."""

    def test_walking_toward_target(self):
        civ = CivilianAgent(
            agent_id="civ_1",
            position=(0.0, 0.0),
            target=(100.0, 0.0),
            speed=5.0,
        )
        import random
        rng = random.Random(42)
        civ.tick(1.0, rng, (200, 200))
        # Should have moved toward target
        assert civ.position[0] > 0

    def test_reaches_target_goes_idle(self):
        civ = CivilianAgent(
            agent_id="civ_1",
            position=(99.0, 0.0),
            target=(100.0, 0.0),
            speed=5.0,
        )
        import random
        rng = random.Random(42)
        civ.tick(1.0, rng, (200, 200))
        assert civ.state == "idle"

    def test_idle_resumes_walking(self):
        civ = CivilianAgent(
            agent_id="civ_1",
            position=(50.0, 50.0),
            target=(50.0, 50.0),
            speed=1.0,
            state="idle",
            _idle_timer=0.01,
        )
        import random
        rng = random.Random(42)
        civ.tick(0.1, rng, (200, 200))
        assert civ.state == "walking"

    def test_to_dict(self):
        civ = CivilianAgent(
            agent_id="civ_1",
            position=(10.5, 20.3),
            target=(50.0, 50.0),
            speed=1.2,
        )
        d = civ.to_dict()
        assert d["id"] == "civ_1"
        assert d["type"] == "civilian"
        assert d["x"] == 10.5
        assert d["y"] == 20.3


# ---------------------------------------------------------------------------
# CityVehicle
# ---------------------------------------------------------------------------

class TestCityVehicle:
    """Tests for the CityVehicle dataclass."""

    def test_drives_along_route(self):
        import random
        rng = random.Random(42)
        veh = CityVehicle(
            vehicle_id="car_1",
            vehicle_type="car",
            position=(50.0, 50.0),
            max_speed=10.0,
            route=[(100.0, 50.0), (100.0, 100.0)],
        )
        for _ in range(50):
            veh.tick(0.1, rng, (200, 200))
        # Should have moved toward first waypoint
        assert veh.position[0] > 50.0

    def test_generates_new_route_when_empty(self):
        import random
        rng = random.Random(42)
        veh = CityVehicle(
            vehicle_id="car_1",
            vehicle_type="car",
            position=(50.0, 50.0),
            max_speed=10.0,
            route=[],
        )
        veh.tick(0.1, rng, (200, 200))
        assert len(veh.route) > 0

    def test_to_dict(self):
        veh = CityVehicle(
            vehicle_id="car_1",
            vehicle_type="taxi",
            position=(10.0, 20.0),
            color="#ffcc00",
        )
        d = veh.to_dict()
        assert d["id"] == "car_1"
        assert d["type"] == "taxi"
        assert d["color"] == "#ffcc00"

    def test_stopped_timer(self):
        import random
        rng = random.Random(42)
        veh = CityVehicle(
            vehicle_id="car_1",
            vehicle_type="car",
            position=(50.0, 50.0),
            _stopped_timer=1.0,
        )
        pos_before = veh.position
        veh.tick(0.5, rng, (200, 200))
        assert veh.speed == 0.0
        assert veh.position == pos_before


# ---------------------------------------------------------------------------
# CityEntity
# ---------------------------------------------------------------------------

class TestCityEntity:
    """Tests for the CityEntity dataclass."""

    def test_creation(self):
        e = CityEntity(
            entity_id="tree_1",
            entity_type="tree",
            position=(10.0, 20.0),
            size=(2.0, 2.0, 8.0),
        )
        assert e.entity_id == "tree_1"
        assert e.entity_type == "tree"
        assert e.size == (2.0, 2.0, 8.0)


# ---------------------------------------------------------------------------
# Subsystem integration
# ---------------------------------------------------------------------------

class TestSubsystemIntegration:
    """Tests that real sim_engine subsystems are wired up correctly."""

    def test_environment_is_real(self, city: CitySim):
        """Environment should be a real Environment instance."""
        from tritium_lib.sim_engine.environment import Environment
        assert isinstance(city.environment, Environment)

    def test_destruction_is_real(self, city: CitySim):
        from tritium_lib.sim_engine.destruction import DestructionEngine
        assert isinstance(city.destruction, DestructionEngine)
        assert len(city.destruction.structures) > 0

    def test_crowd_is_real(self, city: CitySim):
        from tritium_lib.sim_engine.crowd import CrowdSimulator
        assert isinstance(city.crowd, CrowdSimulator)

    def test_detection_is_real(self, city: CitySim):
        from tritium_lib.sim_engine.detection import DetectionEngine
        assert isinstance(city.detection, DetectionEngine)

    def test_comms_is_real(self, city: CitySim):
        from tritium_lib.sim_engine.comms import CommsSimulator
        assert isinstance(city.comms, CommsSimulator)

    def test_medical_is_real(self, city: CitySim):
        from tritium_lib.sim_engine.medical import MedicalEngine
        assert isinstance(city.medical, MedicalEngine)

    def test_logistics_is_real(self, city: CitySim):
        from tritium_lib.sim_engine.logistics import LogisticsEngine
        assert isinstance(city.logistics, LogisticsEngine)

    def test_scoring_is_real(self, city: CitySim):
        from tritium_lib.sim_engine.scoring import ScoringEngine
        assert isinstance(city.scoring, ScoringEngine)

    def test_diplomacy_is_real(self, city: CitySim):
        from tritium_lib.sim_engine.factions import DiplomacyEngine
        assert isinstance(city.diplomacy, DiplomacyEngine)

    def test_territory_is_real(self, city: CitySim):
        from tritium_lib.sim_engine.territory import TerritoryControl, InfluenceMap
        assert isinstance(city.territory, TerritoryControl)
        assert isinstance(city.influence_map, InfluenceMap)

    def test_objectives_is_real(self, city: CitySim):
        from tritium_lib.sim_engine.objectives import ObjectiveEngine
        assert isinstance(city.objectives, ObjectiveEngine)

    def test_engineering_is_real(self, city: CitySim):
        from tritium_lib.sim_engine.fortifications import EngineeringEngine
        assert isinstance(city.engineering, EngineeringEngine)

    def test_narrator_is_real(self, city: CitySim):
        from tritium_lib.sim_engine.commander import BattleNarrator
        assert isinstance(city.narrator, BattleNarrator)

    def test_telemetry_is_real(self, city: CitySim):
        from tritium_lib.sim_engine.telemetry import TelemetrySession
        assert isinstance(city.telemetry, TelemetrySession)

    def test_tactics_is_real(self, city: CitySim):
        from tritium_lib.sim_engine.ai.tactics import TacticsEngine
        assert isinstance(city.tactics, TacticsEngine)

    def test_renderer_is_real(self, city: CitySim):
        from tritium_lib.sim_engine.renderer import SimRenderer
        assert isinstance(city.renderer, SimRenderer)


# ---------------------------------------------------------------------------
# Weather / Time control
# ---------------------------------------------------------------------------

class TestWeatherTimeControl:
    """Tests for weather and time manipulation."""

    def test_set_weather(self, city: CitySim):
        city.set_weather(Weather.RAIN)
        assert city.environment.weather.state.current == Weather.RAIN

    def test_set_time(self, city: CitySim):
        city.set_time(22.0)
        assert abs(city.environment.time.hour - 22.0) < 0.01

    def test_set_time_wraps(self, city: CitySim):
        city.set_time(25.0)
        assert abs(city.environment.time.hour - 1.0) < 0.01

    def test_time_advances_with_ticks(self, city: CitySim):
        initial_hour = city.environment.time.hour
        # Tick for a simulated minute
        for _ in range(60):
            city.tick(1.0)
        # Time should have advanced (60 seconds = 1/60 hour)
        assert city.environment.time.hour != initial_hour


# ---------------------------------------------------------------------------
# Crowd events
# ---------------------------------------------------------------------------

class TestCrowdEvents:
    """Tests for injecting crowd events."""

    def test_inject_gunshot(self, city: CitySim):
        city.inject_crowd_event("gunshot", (250.0, 200.0), radius=30.0, intensity=0.8)
        assert len(city.events) == 1
        assert city.events[0]["event_type"] == "gunshot"

    def test_inject_speech(self, city: CitySim):
        city.inject_crowd_event("speech", (250.0, 200.0), radius=20.0, intensity=0.5)
        assert len(city.events) == 1

    def test_inject_without_crowd(self):
        """Injecting event when no crowd exists should be safe."""
        sim = CitySim(width=200, height=200, seed=1)
        sim.inject_crowd_event("gunshot", (100, 100))
        # No crash, event list is empty (crowd not set up)
        assert len(sim.events) == 0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Tests that the simulation is deterministic for a given seed."""

    def test_same_seed_same_output(self):
        sim1 = CitySim(seed=42)
        sim1.setup()
        for _ in range(10):
            sim1.tick(0.05)
        frame1 = sim1.to_frame()

        sim2 = CitySim(seed=42)
        sim2.setup()
        for _ in range(10):
            sim2.tick(0.05)
        frame2 = sim2.to_frame()

        assert frame1["tick"] == frame2["tick"]
        assert frame1["sim_time"] == frame2["sim_time"]
        assert len(frame1["civilians"]) == len(frame2["civilians"])
        assert len(frame1["vehicles"]) == len(frame2["vehicles"])

    def test_different_seed_different_output(self):
        sim1 = CitySim(seed=42)
        sim1.setup()
        sim1.tick(0.05)

        sim2 = CitySim(seed=99)
        sim2.setup()
        sim2.tick(0.05)

        # Civilians should be in different positions
        civs1 = sim1.to_frame()["civilians"]
        civs2 = sim2.to_frame()["civilians"]
        # At least one civilian should differ
        different = any(
            abs(c1["x"] - c2["x"]) > 0.1 or abs(c1["y"] - c2["y"]) > 0.1
            for c1, c2 in zip(civs1, civs2)
        )
        assert different


# ---------------------------------------------------------------------------
# MapGenerator integration
# ---------------------------------------------------------------------------

class TestMapGenIntegration:
    """Tests that MapGenerator is correctly used."""

    def test_map_has_features(self, city: CitySim):
        assert city.map_data is not None
        assert len(city.map_data.features) > 0

    def test_map_has_roads(self, city: CitySim):
        assert len(city.map_data.roads) > 0

    def test_map_has_spawn_points(self, city: CitySim):
        assert "police" in city.map_data.spawn_points
        assert "civilian" in city.map_data.spawn_points
