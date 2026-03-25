# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.sim_engine.behavior.npc — NPCManager and related."""

import pytest
from types import SimpleNamespace

from tritium_lib.sim_engine.behavior.npc import (
    NPCManager,
    NPCMission,
    NPC_VEHICLE_TYPES,
    traffic_density,
    MISSION_TYPES,
)


def _make_engine(map_bounds=200.0):
    """Create a mock SimulationEngine for NPCManager."""
    engine = SimpleNamespace()
    engine._map_bounds = map_bounds
    engine._street_graph = None
    engine._targets = {}
    engine.spawners_paused = False

    def add_target(t):
        engine._targets[t.target_id] = t

    def get_targets():
        return list(engine._targets.values())

    engine.add_target = add_target
    engine.get_targets = get_targets
    return engine


class TestTrafficDensity:
    def test_rush_hour_high(self):
        assert traffic_density(8) >= 0.8

    def test_night_low(self):
        assert traffic_density(3) <= 0.10

    def test_midday_moderate(self):
        d = traffic_density(12)
        assert 0.3 <= d <= 0.7

    def test_all_hours_valid(self):
        for h in range(24):
            d = traffic_density(h)
            assert 0.0 <= d <= 1.0

    def test_wraps_at_24(self):
        assert traffic_density(24) == traffic_density(0)
        assert traffic_density(25) == traffic_density(1)


class TestNPCMission:
    def test_basic_construction(self):
        m = NPCMission(
            mission_type="commute",
            origin=(0.0, 0.0),
            destination=(100.0, 100.0),
        )
        assert m.mission_type == "commute"
        assert not m.completed

    def test_all_mission_types(self):
        assert "commute" in MISSION_TYPES
        assert "patrol" in MISSION_TYPES
        assert "delivery" in MISSION_TYPES
        assert "drive_through" in MISSION_TYPES
        assert "walk" in MISSION_TYPES


class TestNPCVehicleTypes:
    def test_sedan_exists(self):
        assert "sedan" in NPC_VEHICLE_TYPES
        assert NPC_VEHICLE_TYPES["sedan"]["speed"] > 0

    def test_all_types_have_names(self):
        for vtype, data in NPC_VEHICLE_TYPES.items():
            assert "names" in data
            assert len(data["names"]) > 0
            assert "speed" in data
            assert data["speed"] > 0

    def test_emergency_vehicles(self):
        assert "police" in NPC_VEHICLE_TYPES
        assert "ambulance" in NPC_VEHICLE_TYPES


class TestNPCManagerConstruction:
    def test_basic_creation(self):
        engine = _make_engine()
        mgr = NPCManager(engine)
        assert mgr.npc_count == 0
        assert mgr.enabled

    def test_custom_limits(self):
        engine = _make_engine()
        mgr = NPCManager(engine, max_vehicles=50, max_pedestrians=100)
        assert mgr.max_vehicles == 50
        assert mgr.max_pedestrians == 100


class TestNPCManagerSpawnVehicle:
    def test_spawn_vehicle(self):
        engine = _make_engine()
        mgr = NPCManager(engine)
        t = mgr.spawn_vehicle()
        assert t is not None
        assert t.alliance == "neutral"
        assert t.asset_type == "vehicle"
        assert mgr.npc_count == 1

    def test_spawn_specific_vehicle_type(self):
        engine = _make_engine()
        mgr = NPCManager(engine)
        t = mgr.spawn_vehicle(vehicle_type="police")
        assert t is not None
        vtype = mgr.get_vehicle_type(t.target_id)
        assert vtype == "police"

    def test_spawn_at_capacity_returns_none(self):
        engine = _make_engine()
        mgr = NPCManager(engine, max_vehicles=2)
        mgr.spawn_vehicle()
        mgr.spawn_vehicle()
        result = mgr.spawn_vehicle()
        assert result is None

    def test_spawn_has_waypoints(self):
        engine = _make_engine()
        mgr = NPCManager(engine)
        t = mgr.spawn_vehicle()
        assert len(t.waypoints) > 0

    def test_spawn_creates_mission(self):
        engine = _make_engine()
        mgr = NPCManager(engine)
        t = mgr.spawn_vehicle()
        m = mgr.get_mission(t.target_id)
        assert m is not None
        assert m.mission_type in MISSION_TYPES


class TestNPCManagerSpawnPedestrian:
    def test_spawn_pedestrian(self):
        engine = _make_engine()
        mgr = NPCManager(engine)
        t = mgr.spawn_pedestrian()
        assert t is not None
        assert t.alliance == "neutral"
        assert t.asset_type == "person"

    def test_pedestrian_has_walking_speed(self):
        engine = _make_engine()
        mgr = NPCManager(engine)
        t = mgr.spawn_pedestrian()
        assert 0.5 <= t.speed <= 3.0

    def test_pedestrian_at_capacity_returns_none(self):
        engine = _make_engine()
        mgr = NPCManager(engine, max_pedestrians=1)
        mgr.spawn_pedestrian()
        result = mgr.spawn_pedestrian()
        assert result is None


class TestNPCManagerBinding:
    def test_bind_to_track(self):
        engine = _make_engine()
        mgr = NPCManager(engine)
        t = mgr.spawn_vehicle()
        assert mgr.bind_to_track(t.target_id, "cot", "track_123")
        assert mgr.is_bound(t.target_id)

    def test_bind_unknown_target(self):
        engine = _make_engine()
        mgr = NPCManager(engine)
        assert not mgr.bind_to_track("unknown_id", "cot", "track_123")

    def test_unbind(self):
        engine = _make_engine()
        mgr = NPCManager(engine)
        t = mgr.spawn_vehicle()
        mgr.bind_to_track(t.target_id, "cot", "track_123")
        mgr.unbind(t.target_id)
        assert not mgr.is_bound(t.target_id)

    def test_get_binding(self):
        engine = _make_engine()
        mgr = NPCManager(engine)
        t = mgr.spawn_vehicle()
        mgr.bind_to_track(t.target_id, "mqtt", "sensor_5")
        binding = mgr.get_binding(t.target_id)
        assert binding["source"] == "mqtt"
        assert binding["track_id"] == "sensor_5"


class TestNPCManagerLifecycle:
    def test_reset_clears_all(self):
        engine = _make_engine()
        mgr = NPCManager(engine)
        mgr.spawn_vehicle()
        mgr.spawn_vehicle()
        mgr.spawn_pedestrian()
        assert mgr.npc_count == 3
        mgr.reset()
        assert mgr.npc_count == 0

    def test_remove_unit(self):
        engine = _make_engine()
        mgr = NPCManager(engine)
        t = mgr.spawn_vehicle()
        tid = t.target_id
        mgr.remove_unit(tid)
        assert mgr.get_mission(tid) is None
        assert mgr.get_vehicle_type(tid) is None

    def test_remove_unknown_unit_is_safe(self):
        engine = _make_engine()
        mgr = NPCManager(engine)
        mgr.remove_unit("nonexistent")  # Should not raise

    def test_enabled_toggle(self):
        engine = _make_engine()
        mgr = NPCManager(engine)
        assert mgr.enabled
        mgr.enabled = False
        assert not mgr.enabled
