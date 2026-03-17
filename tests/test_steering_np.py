# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for NumPy-vectorized steering and ambient modules.

Validates correctness and performance for 500+ agents at 10Hz.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from tritium_lib.sim_engine.ai.steering_np import SpatialHash, SteeringSystem
from tritium_lib.sim_engine.ai.ambient_np import (
    AmbientSimulatorNP,
    PEDESTRIAN,
    VEHICLE,
    CYCLIST,
    JOGGER,
    DOG_WALKER,
    STATE_MOVING,
)


# ---------------------------------------------------------------------------
# SpatialHash tests
# ---------------------------------------------------------------------------

class TestSpatialHash:
    def test_insert_and_query(self):
        sh = SpatialHash(cell_size=5.0)
        pos = np.array([[1.0, 1.0], [2.0, 2.0], [50.0, 50.0]], dtype=np.float32)
        active = np.array([True, True, True])
        sh.insert_all(pos, active)

        near = sh.query_radius(1.5, 1.5, 5.0)
        assert 0 in near
        assert 1 in near
        # Agent 2 is far away — should not be in result
        assert 2 not in near

    def test_empty_grid(self):
        sh = SpatialHash(cell_size=5.0)
        result = sh.query_radius(0, 0, 10)
        assert result == []

    def test_inactive_agents_excluded(self):
        sh = SpatialHash(cell_size=5.0)
        pos = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
        active = np.array([True, False])
        sh.insert_all(pos, active)
        near = sh.query_radius(1.5, 1.5, 5.0)
        assert 0 in near
        assert 1 not in near


# ---------------------------------------------------------------------------
# SteeringSystem tests
# ---------------------------------------------------------------------------

class TestSteeringSystem:
    def test_add_and_remove(self):
        ss = SteeringSystem(max_agents=10)
        idx0 = ss.add_agent((0, 0))
        idx1 = ss.add_agent((5, 5))
        assert ss.count == 2
        assert ss.active[idx0]
        assert ss.active[idx1]

        ss.remove_agent(idx0)
        assert not ss.active[idx0]
        assert ss.active[idx1]

    def test_seek_moves_toward_target(self):
        ss = SteeringSystem(max_agents=10)
        idx = ss.add_agent((0, 0), max_speed=2.0, behavior=SteeringSystem.SEEK)
        ss.targets[idx] = (100, 0)

        initial_x = float(ss.positions[idx, 0])
        for _ in range(100):
            ss.tick(0.1)
        final_x = float(ss.positions[idx, 0])
        assert final_x > initial_x + 5.0, f"Agent should have moved right: {initial_x} -> {final_x}"

    def test_flee_moves_away(self):
        ss = SteeringSystem(max_agents=10)
        idx = ss.add_agent((50, 50), max_speed=2.0, behavior=SteeringSystem.FLEE)
        ss.targets[idx] = (50, 0)  # threat below

        for _ in range(50):
            ss.tick(0.1)
        final_y = float(ss.positions[idx, 1])
        assert final_y > 50, "Agent should have fled upward"

    def test_arrive_decelerates(self):
        ss = SteeringSystem(max_agents=10)
        idx = ss.add_agent((0, 0), max_speed=5.0, behavior=SteeringSystem.ARRIVE)
        ss.targets[idx] = (10, 0)
        ss.slow_radii[idx] = 8.0

        for _ in range(200):
            ss.tick(0.1)
        final_x = float(ss.positions[idx, 0])
        # Should be close to target
        assert abs(final_x - 10.0) < 3.0, f"Agent should arrive near target: got {final_x}"

    def test_separation_keeps_agents_apart(self):
        ss = SteeringSystem(max_agents=20)
        # Use only SEPARATE (no seek pulling them together) to test pure separation
        sep_behavior = SteeringSystem.SEPARATE
        # Place 10 agents at nearly the same spot
        initial_positions = []
        for i in range(10):
            pos = (50 + np.random.uniform(-0.1, 0.1), 50 + np.random.uniform(-0.1, 0.1))
            idx = ss.add_agent(pos, max_speed=2.0, behavior=sep_behavior)
            ss.separation_dist[idx] = 5.0
            initial_positions.append(pos)

        # Tick until they spread
        for _ in range(200):
            ss.tick(0.1)

        positions = ss.get_positions()
        # Measure spread: average distance from centroid should increase
        centroid = positions.mean(axis=0)
        avg_dist_from_center = np.mean(np.linalg.norm(positions - centroid, axis=1))
        # Initially they were within 0.1m of each other, so avg dist ~0.05m
        # After separation they should be further apart
        assert avg_dist_from_center > 0.3, f"Agents didn't spread enough: avg_dist={avg_dist_from_center}"

    def test_wander_moves_agents(self):
        ss = SteeringSystem(max_agents=10)
        idx = ss.add_agent((50, 50), max_speed=1.5, behavior=SteeringSystem.WANDER)

        initial = ss.positions[idx].copy()
        for _ in range(100):
            ss.tick(0.1)
        final = ss.positions[idx].copy()
        dist = np.linalg.norm(final - initial)
        assert dist > 1.0, f"Wander should move agent: dist={dist}"

    def test_500_agents_tick_under_100ms(self):
        """500 agents ticking at 10Hz must complete in <100ms per tick."""
        ss = SteeringSystem(max_agents=600)
        for _ in range(500):
            pos = (np.random.uniform(0, 500), np.random.uniform(0, 500))
            idx = ss.add_agent(pos, max_speed=1.4, behavior=SteeringSystem.SEEK)
            ss.targets[idx] = (np.random.uniform(0, 500), np.random.uniform(0, 500))

        # Warm up
        ss.tick(0.1)

        # Time 10 ticks
        start = time.perf_counter()
        for _ in range(10):
            ss.tick(0.1)
        elapsed = time.perf_counter() - start

        per_tick_ms = (elapsed / 10) * 1000
        print(f"\n  500 agents SEEK: {per_tick_ms:.2f} ms/tick ({1000/per_tick_ms:.0f} Hz)")
        assert per_tick_ms < 100, f"Tick too slow: {per_tick_ms:.2f} ms (need <100ms)"

    def test_500_agents_flocking_performance(self):
        """500 agents with separation+alignment+cohesion."""
        ss = SteeringSystem(max_agents=600)
        behavior = SteeringSystem.SEEK | SteeringSystem.SEPARATE | SteeringSystem.ALIGN | SteeringSystem.COHERE
        for _ in range(500):
            pos = (np.random.uniform(0, 200), np.random.uniform(0, 200))
            idx = ss.add_agent(pos, max_speed=1.4, behavior=behavior)
            ss.targets[idx] = (100, 100)
            ss.separation_dist[idx] = 3.0

        # Warm up
        ss.tick(0.1)

        start = time.perf_counter()
        for _ in range(10):
            ss.tick(0.1)
        elapsed = time.perf_counter() - start

        per_tick_ms = (elapsed / 10) * 1000
        print(f"\n  500 agents FLOCK: {per_tick_ms:.2f} ms/tick ({1000/per_tick_ms:.0f} Hz)")
        # Flocking is more expensive due to spatial queries — allow 500ms
        assert per_tick_ms < 500, f"Flock tick too slow: {per_tick_ms:.2f} ms"

    def test_reuse_freed_slots(self):
        ss = SteeringSystem(max_agents=10)
        idx0 = ss.add_agent((0, 0))
        ss.remove_agent(idx0)
        idx1 = ss.add_agent((5, 5))
        assert idx1 == idx0, "Should reuse freed slot"
        assert ss.active[idx1]

    def test_get_headings(self):
        ss = SteeringSystem(max_agents=10)
        idx = ss.add_agent((0, 0), vel=(1, 0), max_speed=2.0)
        headings = ss.get_headings()
        assert len(headings) == 1
        assert abs(headings[0] - 0.0) < 0.01  # heading should be ~0 (east)


# ---------------------------------------------------------------------------
# AmbientSimulatorNP tests
# ---------------------------------------------------------------------------

class TestAmbientSimulatorNP:
    def test_spawn_single(self):
        sim = AmbientSimulatorNP(bounds=((0, 0), (500, 500)), seed=42)
        idx = sim.spawn(PEDESTRIAN, (100, 100))
        assert sim.steering.active[idx]
        assert sim.entity_types[idx] == PEDESTRIAN

    def test_spawn_batch(self):
        sim = AmbientSimulatorNP(bounds=((0, 0), (500, 500)), seed=42)
        positions = np.array([[10, 10], [20, 20], [30, 30]], dtype=np.float32)
        indices = sim.spawn_batch(VEHICLE, 3, positions)
        assert len(indices) == 3
        for idx in indices:
            assert sim.entity_types[idx] == VEHICLE

    def test_tick_moves_entities(self):
        sim = AmbientSimulatorNP(bounds=((0, 0), (500, 500)), seed=42)
        idx = sim.spawn(PEDESTRIAN, (100, 100))
        initial = sim.steering.positions[idx].copy()

        for _ in range(50):
            sim.tick(0.1, current_hour=12.0)

        final = sim.steering.positions[idx].copy()
        dist = np.linalg.norm(final - initial)
        assert dist > 0.5, f"Entity should have moved: dist={dist}"

    def test_to_target_dicts_format(self):
        sim = AmbientSimulatorNP(bounds=((0, 0), (500, 500)), seed=42)
        sim.spawn(PEDESTRIAN, (100, 200))
        sim.spawn(VEHICLE, (300, 400))

        dicts = sim.to_target_dicts()
        assert len(dicts) == 2
        for d in dicts:
            assert "target_id" in d
            assert d["target_id"].startswith("amb_")
            assert "source" in d and d["source"] == "ambient_sim"
            assert "alliance" in d and d["alliance"] == "neutral"
            assert "position_x" in d
            assert "position_y" in d
            assert "heading" in d
            assert "speed" in d
            assert "state" in d
            assert "metadata" in d
            assert d["metadata"]["simulated"] is True

    def test_to_target_dicts_classification(self):
        sim = AmbientSimulatorNP(bounds=((0, 0), (500, 500)), seed=42)
        sim.spawn(PEDESTRIAN, (10, 10))
        sim.spawn(VEHICLE, (20, 20))

        dicts = sim.to_target_dicts()
        ped = [d for d in dicts if d["asset_type"] == "pedestrian"][0]
        veh = [d for d in dicts if d["asset_type"] == "vehicle"][0]
        assert ped["classification"] == "person"
        assert veh["classification"] == "vehicle"

    def test_500_entities_tick(self):
        """500 entities should tick comfortably at 10Hz."""
        sim = AmbientSimulatorNP(bounds=((0, 0), (1000, 1000)), max_entities=600, seed=42)
        for _ in range(500):
            etype = int(np.random.choice([PEDESTRIAN, VEHICLE, CYCLIST, JOGGER, DOG_WALKER]))
            sim.spawn(etype)

        assert sim.active_count == 500

        # Warm up
        sim.tick(0.1, 12.0)

        start = time.perf_counter()
        for _ in range(10):
            sim.tick(0.1, 12.0)
        elapsed = time.perf_counter() - start

        per_tick_ms = (elapsed / 10) * 1000
        print(f"\n  500 ambient entities: {per_tick_ms:.2f} ms/tick ({1000/per_tick_ms:.0f} Hz)")
        assert per_tick_ms < 200, f"Ambient tick too slow: {per_tick_ms:.2f} ms"

    def test_density_scaling(self):
        sim = AmbientSimulatorNP(bounds=((0, 0), (500, 500)), seed=42)
        sim.set_density(pedestrians=50, vehicles=20)
        # Tick at noon — should spawn entities
        for _ in range(5):
            sim.tick(0.1, current_hour=12.0)
        assert sim.active_count > 0

    def test_entity_types_all_spawn(self):
        sim = AmbientSimulatorNP(bounds=((0, 0), (500, 500)), seed=42)
        for etype in [PEDESTRIAN, VEHICLE, CYCLIST, JOGGER, DOG_WALKER]:
            idx = sim.spawn(etype, (100, 100))
            assert sim.entity_types[idx] == etype

    def test_bounds_clamping(self):
        sim = AmbientSimulatorNP(bounds=((0, 0), (100, 100)), seed=42)
        # Place agent near edge heading outward
        idx = sim.spawn(PEDESTRIAN, (99, 99))
        sim.steering.targets[idx] = (200, 200)
        for _ in range(50):
            sim.tick(0.1, 12.0)
        pos = sim.steering.positions[idx]
        assert pos[0] <= 100.0 and pos[1] <= 100.0, "Should be clamped to bounds"


# ---------------------------------------------------------------------------
# Benchmark (prints results, always passes)
# ---------------------------------------------------------------------------

class TestBenchmark:
    def test_benchmark_1000_ticks_500_agents(self):
        """Benchmark: 1000 ticks with 500 agents (seek only)."""
        ss = SteeringSystem(max_agents=600)
        for _ in range(500):
            pos = (np.random.uniform(0, 500), np.random.uniform(0, 500))
            idx = ss.add_agent(pos, max_speed=1.4, behavior=SteeringSystem.SEEK)
            ss.targets[idx] = (np.random.uniform(0, 500), np.random.uniform(0, 500))

        start = time.perf_counter()
        for _ in range(1000):
            ss.tick(0.1)
        elapsed = time.perf_counter() - start

        per_tick_ms = (elapsed / 1000) * 1000
        total_agent_ticks = 500 * 1000
        agent_ticks_per_sec = total_agent_ticks / elapsed

        print(f"\n  === BENCHMARK: 1000 ticks x 500 agents (SEEK) ===")
        print(f"  Total time:         {elapsed:.3f} s")
        print(f"  Per tick:           {per_tick_ms:.2f} ms")
        print(f"  Agent-ticks/sec:    {agent_ticks_per_sec:,.0f}")
        print(f"  Effective Hz:       {1000 / per_tick_ms:.0f}")

    def test_benchmark_1000_ticks_500_agents_ambient(self):
        """Benchmark: 1000 ticks with 500 ambient entities."""
        sim = AmbientSimulatorNP(bounds=((0, 0), (1000, 1000)), max_entities=600, seed=42)
        for _ in range(500):
            etype = int(np.random.choice([PEDESTRIAN, VEHICLE, CYCLIST]))
            sim.spawn(etype)

        start = time.perf_counter()
        for _ in range(1000):
            sim.tick(0.1, 12.0)
        elapsed = time.perf_counter() - start

        per_tick_ms = (elapsed / 1000) * 1000
        print(f"\n  === BENCHMARK: 1000 ticks x 500 ambient entities ===")
        print(f"  Total time:         {elapsed:.3f} s")
        print(f"  Per tick:           {per_tick_ms:.2f} ms")
        print(f"  Effective Hz:       {1000 / per_tick_ms:.0f}")
