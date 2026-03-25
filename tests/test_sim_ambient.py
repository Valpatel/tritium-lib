# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.sim_engine.ai.ambient — AmbientSimulator and related."""

import math
import pytest

from tritium_lib.sim_engine.ai.ambient import (
    AmbientSimulator,
    AmbientEntity,
    ActivityProfile,
    EntityState,
    EntityType,
)


class TestActivityProfile:
    def test_residential_factory(self):
        p = ActivityProfile.residential()
        assert isinstance(p.pedestrian_density, dict)
        assert len(p.pedestrian_density) == 24

    def test_commercial_factory(self):
        p = ActivityProfile.commercial()
        # Commercial should be busier midday than residential
        assert p.pedestrian_density[12] > 0.5

    def test_school_factory(self):
        p = ActivityProfile.school()
        # School peak at 8am
        assert p.pedestrian_density[8] >= 0.8

    def test_density_at_interpolation(self):
        p = ActivityProfile.residential()
        # At exact hour boundary
        d0 = p.density_at(8.0, EntityType.PEDESTRIAN)
        assert d0 == p.pedestrian_density[8]
        # Midway between hours should be interpolated
        d_mid = p.density_at(8.5, EntityType.PEDESTRIAN)
        d8 = p.pedestrian_density[8]
        d9 = p.pedestrian_density[9]
        expected = d8 + (d9 - d8) * 0.5
        assert abs(d_mid - expected) < 0.001

    def test_density_at_vehicle(self):
        p = ActivityProfile.residential()
        d = p.density_at(12.0, EntityType.VEHICLE)
        assert d == p.vehicle_density[12]

    def test_density_wraps_at_24(self):
        p = ActivityProfile.residential()
        d = p.density_at(23.5, EntityType.PEDESTRIAN)
        d23 = p.pedestrian_density[23]
        d0 = p.pedestrian_density[0]
        expected = d23 + (d0 - d23) * 0.5
        assert abs(d - expected) < 0.001


class TestEntityState:
    def test_values(self):
        assert EntityState.MOVING == "moving"
        assert EntityState.STOPPED == "stopped"
        assert EntityState.PARKED == "parked"
        assert EntityState.WAITING == "waiting"


class TestEntityType:
    def test_values(self):
        assert EntityType.PEDESTRIAN == "pedestrian"
        assert EntityType.VEHICLE == "vehicle"
        assert EntityType.CYCLIST == "cyclist"
        assert EntityType.JOGGER == "jogger"
        assert EntityType.DOG_WALKER == "dog_walker"


class TestAmbientEntity:
    def test_default_construction(self):
        e = AmbientEntity()
        assert e.entity_type == EntityType.PEDESTRIAN
        assert e.state == EntityState.MOVING
        assert e.position == (0.0, 0.0)

    def test_to_dict_has_fields(self):
        e = AmbientEntity(entity_id="abc", entity_type=EntityType.PEDESTRIAN,
                          position=(10.0, 20.0), speed=1.2)
        d = e.to_dict()
        assert d["target_id"] == "amb_abc"
        assert d["source"] == "ambient_sim"
        assert d["alliance"] == "neutral"
        assert d["position_x"] == 10.0
        assert d["position_y"] == 20.0
        assert d["classification"] == "person"

    def test_vehicle_classification(self):
        e = AmbientEntity(entity_id="v1", entity_type=EntityType.VEHICLE)
        d = e.to_dict()
        assert d["classification"] == "vehicle"

    def test_tick_moves_along_path(self):
        e = AmbientEntity(
            entity_id="p1", entity_type=EntityType.PEDESTRIAN,
            position=(0.0, 0.0), speed=10.0,
            path=[(100.0, 0.0)], path_index=0,
        )
        e.tick(1.0)
        # Should have moved toward (100, 0)
        assert e.position[0] > 0.0

    def test_tick_parked_no_move(self):
        e = AmbientEntity(
            entity_id="v1", entity_type=EntityType.VEHICLE,
            position=(5.0, 5.0), speed=10.0,
            state=EntityState.PARKED,
        )
        e.tick(1.0)
        assert e.position == (5.0, 5.0)

    def test_tick_clamps_to_bounds(self):
        e = AmbientEntity(
            entity_id="p1", position=(0.0, 0.0), speed=100.0,
            path=[(-1000.0, 0.0)], path_index=0,
        )
        bounds = ((0.0, 0.0), (500.0, 500.0))
        e.tick(1.0, walkable_area=bounds)
        assert e.position[0] >= 0.0


class TestAmbientSimulator:
    def test_construction(self):
        sim = AmbientSimulator(bounds=((0.0, 0.0), (500.0, 500.0)))
        assert sim.entities == {}
        assert sim.profile is not None

    def test_construction_with_seed(self):
        sim1 = AmbientSimulator(bounds=((0.0, 0.0), (500.0, 500.0)), seed=42)
        sim2 = AmbientSimulator(bounds=((0.0, 0.0), (500.0, 500.0)), seed=42)
        p1 = sim1.spawn_pedestrian()
        p2 = sim2.spawn_pedestrian()
        assert p1.speed == p2.speed

    def test_spawn_pedestrian(self):
        sim = AmbientSimulator(bounds=((0.0, 0.0), (500.0, 500.0)), seed=1)
        ent = sim.spawn_pedestrian()
        assert ent.entity_type == EntityType.PEDESTRIAN
        assert ent.entity_id in sim.entities
        assert len(ent.path) > 0

    def test_spawn_vehicle(self):
        sim = AmbientSimulator(bounds=((0.0, 0.0), (500.0, 500.0)), seed=1)
        ent = sim.spawn_vehicle()
        assert ent.entity_type == EntityType.VEHICLE
        assert ent.speed >= 5.0  # Vehicle speed range

    def test_spawn_jogger(self):
        sim = AmbientSimulator(bounds=((0.0, 0.0), (500.0, 500.0)), seed=1)
        ent = sim.spawn_jogger()
        assert ent.entity_type == EntityType.JOGGER

    def test_spawn_dog_walker(self):
        sim = AmbientSimulator(bounds=((0.0, 0.0), (500.0, 500.0)), seed=1)
        ent = sim.spawn_dog_walker()
        assert ent.entity_type == EntityType.DOG_WALKER

    def test_spawn_cyclist(self):
        sim = AmbientSimulator(bounds=((0.0, 0.0), (500.0, 500.0)), seed=1)
        ent = sim.spawn_cyclist()
        assert ent.entity_type == EntityType.CYCLIST

    def test_get_entities_returns_dicts(self):
        sim = AmbientSimulator(bounds=((0.0, 0.0), (500.0, 500.0)), seed=1)
        sim.spawn_pedestrian()
        sim.spawn_vehicle()
        entities = sim.get_entities()
        assert len(entities) == 2
        assert all(isinstance(e, dict) for e in entities)

    def test_set_density(self):
        sim = AmbientSimulator(bounds=((0.0, 0.0), (500.0, 500.0)), seed=1)
        sim.set_density(pedestrians=10, vehicles=5)
        assert sim._target_pedestrians == 10
        assert sim._target_vehicles == 5

    def test_tick_spawns_and_moves(self):
        sim = AmbientSimulator(bounds=((0.0, 0.0), (500.0, 500.0)), seed=1)
        sim.set_density(pedestrians=10, vehicles=5)
        sim.tick(1.0, current_hour=12.0)
        assert len(sim.entities) > 0

    def test_tick_empty_density_no_spawn(self):
        sim = AmbientSimulator(bounds=((0.0, 0.0), (500.0, 500.0)), seed=1)
        sim.set_density(pedestrians=0, vehicles=0)
        sim.tick(1.0, current_hour=12.0)
        assert len(sim.entities) == 0

    def test_culling_removes_parked_vehicles(self):
        sim = AmbientSimulator(bounds=((0.0, 0.0), (500.0, 500.0)), seed=1)
        ent = sim.spawn_vehicle()
        ent.state = EntityState.PARKED
        sim._cull_finished()
        assert ent.entity_id not in sim.entities
