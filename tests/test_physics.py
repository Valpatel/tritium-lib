# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for sim_engine.physics: collision detection, momentum, vehicle dynamics."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from tritium_lib.sim_engine.physics import CollisionEvent, PhysicsWorld, RigidBody, VehiclePhysics


# ---------------------------------------------------------------------------
# Collision basics
# ---------------------------------------------------------------------------


class TestCollision:
    """Two bodies collide and bounce."""

    def test_head_on_collision_bounces(self):
        """Two equal-mass bodies approaching each other should bounce apart."""
        world = PhysicsWorld(max_bodies=16)
        a = world.add_body(pos=(0, 0), vel=(5, 0), mass=1.0, radius=1.0, restitution=1.0)
        b = world.add_body(pos=(1.5, 0), vel=(-5, 0), mass=1.0, radius=1.0, restitution=1.0)

        events = world.tick(0.0)  # dt=0 so positions don't move, just detect overlap

        assert len(events) >= 1
        ev = events[0]
        assert {ev.body_a, ev.body_b} == {a, b}
        assert ev.impulse > 0
        assert ev.relative_speed > 0

        # After resolution, velocities should have swapped direction.
        assert world.velocities[a][0] < 0, "Body A should bounce leftward"
        assert world.velocities[b][0] > 0, "Body B should bounce rightward"

    def test_unequal_mass_collision(self):
        """Heavy body should barely change velocity; light body gets flung."""
        world = PhysicsWorld(max_bodies=16)
        heavy = world.add_body(pos=(0, 0), vel=(2, 0), mass=100.0, radius=1.0, restitution=0.5)
        light = world.add_body(pos=(1.5, 0), vel=(-1, 0), mass=1.0, radius=1.0, restitution=0.5)

        world.tick(0.0)

        # Heavy body should still be moving roughly rightward.
        assert world.velocities[heavy][0] > 0
        # Light body should be flung rightward faster than before.
        assert world.velocities[light][0] > 1.0

    def test_collision_event_fields(self):
        """CollisionEvent should have valid point, normal, impulse."""
        world = PhysicsWorld(max_bodies=16)
        world.add_body(pos=(0, 0), vel=(3, 0), mass=1.0, radius=1.0)
        world.add_body(pos=(1.5, 0), vel=(-3, 0), mass=1.0, radius=1.0)

        events = world.tick(0.0)
        assert len(events) == 1
        ev = events[0]
        assert isinstance(ev, CollisionEvent)
        assert ev.point.shape == (2,)
        assert ev.normal.shape == (2,)
        assert ev.impulse >= 0
        assert ev.relative_speed >= 0

    def test_no_collision_when_apart(self):
        """Bodies far apart should not collide."""
        world = PhysicsWorld(max_bodies=16)
        world.add_body(pos=(0, 0), vel=(1, 0), mass=1.0, radius=1.0)
        world.add_body(pos=(100, 0), vel=(-1, 0), mass=1.0, radius=1.0)

        events = world.tick(0.01)
        assert len(events) == 0


# ---------------------------------------------------------------------------
# Static bodies
# ---------------------------------------------------------------------------


class TestStaticBodies:
    """Static bodies don't move when hit."""

    def test_static_body_stays_put(self):
        """A static body should not change position or velocity after collision."""
        world = PhysicsWorld(max_bodies=16)
        wall = world.add_body(pos=(5, 0), vel=(0, 0), mass=1000.0, radius=2.0, static=True)
        ball = world.add_body(pos=(3.5, 0), vel=(10, 0), mass=1.0, radius=1.0, restitution=0.8)

        wall_pos_before = world.positions[wall].copy()
        world.tick(0.0)

        np.testing.assert_array_equal(world.velocities[wall], [0, 0])
        np.testing.assert_allclose(world.positions[wall], wall_pos_before, atol=1e-6)
        # Ball should bounce back.
        assert world.velocities[ball][0] < 0

    def test_two_static_bodies_no_event(self):
        """Two overlapping static bodies should not generate events."""
        world = PhysicsWorld(max_bodies=16)
        world.add_body(pos=(0, 0), static=True, radius=2.0)
        world.add_body(pos=(1, 0), static=True, radius=2.0)

        events = world.tick(0.01)
        assert len(events) == 0


# ---------------------------------------------------------------------------
# Explosion
# ---------------------------------------------------------------------------


class TestExplosion:
    """apply_explosion pushes nearby bodies."""

    def test_explosion_pushes_bodies(self):
        """Bodies near explosion center should gain velocity away from center."""
        world = PhysicsWorld(max_bodies=16)
        a = world.add_body(pos=(3, 0), vel=(0, 0), mass=1.0, radius=0.5)
        b = world.add_body(pos=(0, 4), vel=(0, 0), mass=1.0, radius=0.5)
        far = world.add_body(pos=(100, 100), vel=(0, 0), mass=1.0, radius=0.5)

        world.apply_explosion(center=(0, 0), radius=10.0, force=100.0)
        world.tick(0.1)

        # Body A should be pushed rightward (+x).
        assert world.velocities[a][0] > 0
        # Body B should be pushed upward (+y).
        assert world.velocities[b][1] > 0
        # Far body should be unaffected.
        np.testing.assert_allclose(world.velocities[far], [0, 0], atol=1e-3)

    def test_explosion_does_not_move_static(self):
        """Static bodies should not be affected by explosions."""
        world = PhysicsWorld(max_bodies=16)
        wall = world.add_body(pos=(2, 0), static=True, radius=1.0)

        world.apply_explosion(center=(0, 0), radius=10.0, force=1000.0)
        world.tick(0.1)

        np.testing.assert_allclose(world.velocities[wall], [0, 0], atol=1e-6)


# ---------------------------------------------------------------------------
# Vehicle dynamics
# ---------------------------------------------------------------------------


class TestVehiclePhysics:
    """VehiclePhysics bicycle model."""

    def test_straight_line_acceleration(self):
        """Full throttle should increase speed and move position forward."""
        v = VehiclePhysics(max_speed=20.0)
        v.throttle = 1.0
        v.steering = 0.0

        for _ in range(100):
            v.tick(0.016)

        assert v.speed > 5.0
        assert v.position[0] > 0  # Moved rightward (heading=0).

    def test_turning_changes_heading(self):
        """Steering input should change heading over time."""
        v = VehiclePhysics()
        v.throttle = 0.5
        v.steering = 1.0  # Full right.
        initial_heading = v.heading

        for _ in range(50):
            v.tick(0.016)

        assert v.heading != initial_heading
        # With positive steering and positive speed, heading should increase.
        assert v.heading > initial_heading

    def test_no_turn_at_zero_speed(self):
        """Vehicle at rest should not change heading regardless of steering."""
        v = VehiclePhysics()
        v.throttle = 0.0
        v.steering = 1.0
        initial_heading = v.heading

        for _ in range(50):
            v.tick(0.016)

        assert v.heading == pytest.approx(initial_heading, abs=1e-6)

    def test_braking(self):
        """Negative throttle should decelerate a moving vehicle."""
        v = VehiclePhysics()
        v.throttle = 1.0
        for _ in range(60):
            v.tick(0.016)
        speed_before = v.speed
        assert speed_before > 1.0

        v.throttle = -1.0
        for _ in range(60):
            v.tick(0.016)

        assert v.speed < speed_before

    def test_speed_clamped(self):
        """Speed should not exceed max_speed."""
        v = VehiclePhysics(max_speed=10.0)
        v.throttle = 1.0
        for _ in range(1000):
            v.tick(0.016)

        assert v.speed <= v.max_speed + 0.01

    def test_apply_collision_impulse(self):
        """apply_collision should change velocity and heading."""
        v = VehiclePhysics(mass=1000.0)
        v.throttle = 1.0
        for _ in range(30):
            v.tick(0.016)

        speed_before = v.speed
        # Slam from the side.
        v.apply_collision(
            impulse=np.array([0, 5000], dtype=np.float32),
            point=v.position.copy(),
        )
        # Velocity should now have a y-component.
        assert abs(v.velocity[1]) > 1.0


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


class TestPerformance:
    """Tick with many bodies should be fast enough for real-time sim."""

    def test_500_bodies_under_50ms(self):
        """500 active bodies should tick in under 50ms."""
        world = PhysicsWorld(max_bodies=1024, cell_size=5.0)
        rng = np.random.default_rng(42)

        for i in range(500):
            pos = rng.uniform(-100, 100, size=2).astype(np.float32)
            vel = rng.uniform(-5, 5, size=2).astype(np.float32)
            world.add_body(pos=pos, vel=vel, mass=1.0, radius=0.5)

        # Warm up.
        world.tick(0.016)

        t0 = time.perf_counter()
        for _ in range(10):
            world.tick(0.016)
        elapsed = (time.perf_counter() - t0) / 10

        assert elapsed < 0.05, f"Tick took {elapsed*1000:.1f}ms, want < 50ms"


# ---------------------------------------------------------------------------
# Integration: RigidBody helper
# ---------------------------------------------------------------------------


class TestRigidBody:
    """RigidBody dataclass and add_rigid_body convenience."""

    def test_add_rigid_body(self):
        world = PhysicsWorld(max_bodies=8)
        rb = RigidBody(
            position=np.array([1, 2], dtype=np.float32),
            velocity=np.array([3, 4], dtype=np.float32),
            mass=5.0,
            radius=2.0,
            restitution=0.7,
            is_static=False,
        )
        idx = world.add_rigid_body(rb)
        assert idx == 0
        np.testing.assert_allclose(world.positions[idx], [1, 2])
        np.testing.assert_allclose(world.velocities[idx], [3, 4])
        assert world.masses[idx] == pytest.approx(5.0)
        assert world.radii[idx] == pytest.approx(2.0)
        assert world.restitutions[idx] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and robustness."""

    def test_zero_dt_no_crash(self):
        world = PhysicsWorld(max_bodies=8)
        world.add_body(pos=(0, 0), vel=(1, 0))
        events = world.tick(0.0)
        assert isinstance(events, list)

    def test_empty_world_tick(self):
        world = PhysicsWorld(max_bodies=8)
        events = world.tick(0.016)
        assert events == []

    def test_grow_beyond_initial_capacity(self):
        world = PhysicsWorld(max_bodies=4)
        for i in range(10):
            world.add_body(pos=(i * 10, 0))
        assert world.count == 10
        assert world.active[:10].all()

    def test_remove_body(self):
        world = PhysicsWorld(max_bodies=8)
        a = world.add_body(pos=(0, 0))
        world.remove_body(a)
        assert not world.active[a]
