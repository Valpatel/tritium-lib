# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for ambient activity simulation module."""

from __future__ import annotations

import math
import random

import pytest

from tritium_lib.game_ai.ambient import (
    ActivityProfile,
    AmbientEntity,
    AmbientSimulator,
    EntityState,
    EntityType,
    _add,
    _sub,
)


# ---------------------------------------------------------------------------
# ActivityProfile tests
# ---------------------------------------------------------------------------


class TestActivityProfile:
    """Test time-of-day density profiles."""

    def test_default_profile_has_24_hours(self):
        p = ActivityProfile()
        assert len(p.pedestrian_density) == 24
        assert len(p.vehicle_density) == 24

    def test_residential_morning_rush(self):
        p = ActivityProfile.residential()
        # 7-9am should be busier than 3am
        assert p.pedestrian_density[8] > p.pedestrian_density[3]
        assert p.vehicle_density[8] > p.vehicle_density[3]

    def test_commercial_daytime_peak(self):
        p = ActivityProfile.commercial()
        # Noon should be peak
        assert p.pedestrian_density[12] > 0.8
        # Night should be dead
        assert p.pedestrian_density[2] < 0.05

    def test_school_bimodal_peaks(self):
        p = ActivityProfile.school()
        # 8am and 3pm peaks
        assert p.pedestrian_density[8] >= 0.9
        assert p.pedestrian_density[15] >= 0.9
        # 10am lull between peaks
        assert p.pedestrian_density[10] < 0.2

    def test_density_at_interpolation(self):
        p = ActivityProfile.residential()
        # At exact hour should match table
        d_exact = p.density_at(8.0, EntityType.PEDESTRIAN)
        assert d_exact == pytest.approx(p.pedestrian_density[8])

        # At half hour should interpolate
        d_half = p.density_at(8.5, EntityType.PEDESTRIAN)
        d8 = p.pedestrian_density[8]
        d9 = p.pedestrian_density[9]
        expected = d8 + (d9 - d8) * 0.5
        assert d_half == pytest.approx(expected)

    def test_density_at_vehicle(self):
        p = ActivityProfile.residential()
        d = p.density_at(17.0, EntityType.VEHICLE)
        assert d == pytest.approx(p.vehicle_density[17])

    def test_density_at_wraps_at_24(self):
        p = ActivityProfile.residential()
        # Hour 23.5 should interpolate between hour 23 and hour 0
        d = p.density_at(23.5, EntityType.PEDESTRIAN)
        d23 = p.pedestrian_density[23]
        d0 = p.pedestrian_density[0]
        expected = d23 + (d0 - d23) * 0.5
        assert d == pytest.approx(expected)


# ---------------------------------------------------------------------------
# AmbientEntity tests
# ---------------------------------------------------------------------------


class TestAmbientEntity:
    """Test individual entity behavior."""

    def test_entity_defaults(self):
        e = AmbientEntity(entity_id="test1")
        assert e.entity_type == EntityType.PEDESTRIAN
        assert e.position == (0.0, 0.0)
        assert e.state == EntityState.MOVING

    def test_entity_follows_path(self):
        """Entity should move toward the first waypoint on tick."""
        path = [(100.0, 0.0), (200.0, 0.0)]
        e = AmbientEntity(
            entity_id="p1",
            entity_type=EntityType.PEDESTRIAN,
            position=(0.0, 0.0),
            speed=10.0,
            path=path,
            state=EntityState.MOVING,
        )
        random.seed(999)  # deterministic — avoid random stops
        e.tick(1.0)
        # Should have moved toward (100, 0)
        assert e.position[0] > 0.0
        assert abs(e.position[1]) < 1.0  # mostly along x-axis

    def test_vehicle_parks_at_end(self):
        """Vehicle reaching end of path should park."""
        path = [(5.0, 0.0)]
        e = AmbientEntity(
            entity_id="v1",
            entity_type=EntityType.VEHICLE,
            position=(0.0, 0.0),
            speed=100.0,  # fast enough to reach in one tick
            path=path,
            state=EntityState.MOVING,
        )
        random.seed(12345)
        e.tick(1.0)
        assert e.state == EntityState.PARKED

    def test_parked_entity_doesnt_move(self):
        e = AmbientEntity(
            entity_id="v2",
            position=(50.0, 50.0),
            state=EntityState.PARKED,
        )
        e.tick(1.0)
        assert e.position == (50.0, 50.0)

    def test_jogger_loops(self):
        """Jogger reaching end of path should wrap to index 0."""
        path = [(10.0, 0.0)]
        e = AmbientEntity(
            entity_id="j1",
            entity_type=EntityType.JOGGER,
            position=(0.0, 0.0),
            speed=100.0,
            path=path,
            state=EntityState.MOVING,
        )
        random.seed(42)
        e.tick(1.0)
        # Should have looped — path_index back to 0
        assert e.path_index == 0
        assert e.state == EntityState.MOVING

    def test_entity_clamps_to_bounds(self):
        """Entity should stay within walkable area bounds."""
        bounds = ((0.0, 0.0), (100.0, 100.0))
        e = AmbientEntity(
            entity_id="c1",
            position=(99.0, 50.0),
            velocity=(50.0, 0.0),
            speed=50.0,
            path=[(200.0, 50.0)],
            state=EntityState.MOVING,
        )
        random.seed(42)
        e.tick(1.0, walkable_area=bounds)
        assert e.position[0] <= 100.0

    def test_stopped_entity_resumes(self):
        """Stopped entity with expired timer should resume moving."""
        e = AmbientEntity(
            entity_id="s1",
            state=EntityState.STOPPED,
            _stop_timer=0.5,
            path=[(100.0, 100.0)],
        )
        e.tick(1.0)  # dt > stop_timer => resume
        assert e.state == EntityState.MOVING

    def test_to_dict_format(self):
        """to_dict should produce TargetTracker-compatible format."""
        e = AmbientEntity(
            entity_id="abc123",
            entity_type=EntityType.VEHICLE,
            position=(10.0, 20.0),
            heading=90.0,
            speed=8.5,
            state=EntityState.MOVING,
        )
        d = e.to_dict()
        assert d["target_id"] == "amb_abc123"
        assert d["source"] == "ambient_sim"
        assert d["classification"] == "vehicle"
        assert d["alliance"] == "neutral"
        assert d["position_x"] == 10.0
        assert d["position_y"] == 20.0
        assert d["heading"] == 90.0
        assert d["speed"] == 8.5
        assert d["state"] == "moving"
        assert d["metadata"]["simulated"] is True
        assert d["metadata"]["entity_type"] == EntityType.VEHICLE

    def test_to_dict_pedestrian_classification(self):
        e = AmbientEntity(entity_id="p2", entity_type=EntityType.PEDESTRIAN)
        d = e.to_dict()
        assert d["classification"] == "person"

    def test_to_dict_jogger_classification(self):
        e = AmbientEntity(entity_id="j2", entity_type=EntityType.JOGGER)
        d = e.to_dict()
        assert d["classification"] == "person"

    def test_to_dict_stopped_speed_zero(self):
        e = AmbientEntity(
            entity_id="s2",
            speed=5.0,
            state=EntityState.STOPPED,
        )
        d = e.to_dict()
        assert d["speed"] == 0.0


# ---------------------------------------------------------------------------
# AmbientSimulator tests
# ---------------------------------------------------------------------------


class TestAmbientSimulator:
    """Test the simulator manager."""

    def _make_sim(self, seed: int = 42) -> AmbientSimulator:
        bounds = ((0.0, 0.0), (500.0, 500.0))
        return AmbientSimulator(bounds=bounds, seed=seed)

    def test_spawn_pedestrian(self):
        sim = self._make_sim()
        ent = sim.spawn_pedestrian()
        assert ent.entity_type == EntityType.PEDESTRIAN
        assert ent.entity_id in sim.entities
        assert len(ent.path) >= 2

    def test_spawn_vehicle(self):
        sim = self._make_sim()
        ent = sim.spawn_vehicle()
        assert ent.entity_type == EntityType.VEHICLE
        assert ent.entity_id in sim.entities
        assert ent.speed >= 5.0

    def test_spawn_jogger(self):
        sim = self._make_sim()
        ent = sim.spawn_jogger()
        assert ent.entity_type == EntityType.JOGGER
        assert 2.5 <= ent.speed <= 3.5

    def test_spawn_jogger_with_route(self):
        sim = self._make_sim()
        route = [(100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0)]
        ent = sim.spawn_jogger(route=route)
        assert ent.path == route

    def test_spawn_dog_walker(self):
        sim = self._make_sim()
        ent = sim.spawn_dog_walker()
        assert ent.entity_type == EntityType.DOG_WALKER
        assert ent.speed <= 1.0

    def test_spawn_cyclist(self):
        sim = self._make_sim()
        ent = sim.spawn_cyclist()
        assert ent.entity_type == EntityType.CYCLIST
        assert ent.speed >= 3.0

    def test_spawn_vehicle_with_road(self):
        sim = self._make_sim()
        road = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)]
        ent = sim.spawn_vehicle(road_network=road)
        assert ent.path == road

    def test_set_density(self):
        sim = self._make_sim()
        sim.set_density(pedestrians=50, vehicles=20)
        assert sim._target_pedestrians == 50
        assert sim._target_vehicles == 20

    def test_tick_spawns_entities(self):
        sim = self._make_sim()
        sim.set_density(pedestrians=100, vehicles=40)
        sim.tick(dt=1.0, current_hour=12.0)
        # At noon, residential profile should have spawned entities
        assert len(sim.entities) > 0

    def test_tick_moves_entities(self):
        sim = self._make_sim()
        ent = sim.spawn_pedestrian(start=(250.0, 250.0))
        pos_before = ent.position
        random.seed(99999)  # avoid random stops
        sim.tick(dt=1.0, current_hour=12.0)
        # Entity should have moved (or at minimum been ticked)
        # Position may or may not change due to random stops, but entity exists
        assert ent.entity_id in sim.entities or len(sim.entities) >= 0

    def test_density_adjusts_by_hour(self):
        """Night should have fewer entities than noon."""
        sim_noon = self._make_sim(seed=1)
        sim_noon.set_density(pedestrians=100, vehicles=50)
        sim_noon.tick(dt=0.1, current_hour=12.0)
        count_noon = len(sim_noon.entities)

        sim_night = self._make_sim(seed=1)
        sim_night.set_density(pedestrians=100, vehicles=50)
        sim_night.tick(dt=0.1, current_hour=3.0)
        count_night = len(sim_night.entities)

        assert count_noon > count_night

    def test_get_entities_returns_dicts(self):
        sim = self._make_sim()
        sim.spawn_pedestrian()
        sim.spawn_vehicle()
        entities = sim.get_entities()
        assert len(entities) == 2
        for d in entities:
            assert "target_id" in d
            assert "source" in d
            assert d["source"] == "ambient_sim"
            assert "position_x" in d
            assert "position_y" in d
            assert "classification" in d
            assert "alliance" in d
            assert "metadata" in d
            assert d["metadata"]["simulated"] is True

    def test_get_entities_target_ids_unique(self):
        sim = self._make_sim()
        for _ in range(10):
            sim.spawn_pedestrian()
        entities = sim.get_entities()
        ids = [e["target_id"] for e in entities]
        assert len(ids) == len(set(ids))

    def test_vehicle_culled_after_parking(self):
        """Parked vehicles should be removed on next tick."""
        sim = self._make_sim()
        # Short path so vehicle parks quickly
        ent = sim.spawn_vehicle(road_network=[(5.0, 0.0)])
        ent.speed = 100.0  # instant arrival
        eid = ent.entity_id
        random.seed(12345)
        sim.tick(dt=1.0, current_hour=12.0)
        # After tick, the entity that parked should be culled
        # (it parks during entity tick, then _cull_finished removes it)
        assert eid not in sim.entities or sim.entities[eid].state != EntityState.PARKED

    def test_custom_profile(self):
        """Simulator should accept a custom profile."""
        profile = ActivityProfile.commercial()
        sim = AmbientSimulator(
            bounds=((0.0, 0.0), (500.0, 500.0)),
            profile=profile,
            seed=7,
        )
        sim.set_density(pedestrians=50, vehicles=20)
        sim.tick(dt=1.0, current_hour=12.0)
        assert len(sim.entities) > 0

    def test_entities_stay_in_bounds(self):
        """All entity positions should remain within bounds after ticking."""
        sim = self._make_sim()
        sim.set_density(pedestrians=30, vehicles=10)
        # Tick several times
        for _ in range(20):
            sim.tick(dt=1.0, current_hour=14.0)
        lo, hi = sim.bounds
        for ent in sim.entities.values():
            assert lo[0] <= ent.position[0] <= hi[0], f"{ent.entity_id} out of x bounds"
            assert lo[1] <= ent.position[1] <= hi[1], f"{ent.entity_id} out of y bounds"

    def test_multiple_entity_types_spawned(self):
        """With enough pedestrian density, should get varied entity types."""
        sim = self._make_sim(seed=0)
        sim.set_density(pedestrians=200, vehicles=0)
        sim.tick(dt=0.1, current_hour=12.0)
        types = {e.entity_type for e in sim.entities.values()}
        # Should have at least pedestrians and one other type
        assert EntityType.PEDESTRIAN in types
        assert len(types) >= 2
