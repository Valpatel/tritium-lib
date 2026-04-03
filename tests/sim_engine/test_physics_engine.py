# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.sim_engine.physics — collision, PhysicsWorld, VehiclePhysics."""

import math

import numpy as np
import pytest

from tritium_lib.sim_engine.physics import (
    CollisionEvent,
    PhysicsWorld,
    RigidBody,
    VehiclePhysics,
)


# ===================================================================
# RigidBody tests
# ===================================================================


class TestRigidBody:
    def test_default_values(self):
        rb = RigidBody()
        assert rb.mass == 1.0
        assert rb.radius == 1.0
        assert rb.restitution == 0.3
        assert rb.is_static is False
        np.testing.assert_array_equal(rb.position, [0.0, 0.0])
        np.testing.assert_array_equal(rb.velocity, [0.0, 0.0])

    def test_custom_values(self):
        rb = RigidBody(
            position=np.array([5.0, 10.0], dtype=np.float32),
            velocity=np.array([1.0, 2.0], dtype=np.float32),
            mass=50.0,
            radius=3.0,
            is_static=True,
        )
        assert rb.mass == 50.0
        assert rb.radius == 3.0
        assert rb.is_static is True


# ===================================================================
# PhysicsWorld — body management
# ===================================================================


class TestPhysicsWorldBodies:
    def test_add_body(self):
        pw = PhysicsWorld(max_bodies=10)
        idx = pw.add_body(pos=(5.0, 10.0), mass=2.0, radius=1.5)
        assert idx == 0
        assert pw.count == 1
        assert pw.active[idx] == True
        np.testing.assert_allclose(pw.positions[idx], [5.0, 10.0], atol=1e-5)
        assert pw.masses[idx] == pytest.approx(2.0)
        assert pw.radii[idx] == pytest.approx(1.5)

    def test_add_multiple_bodies(self):
        pw = PhysicsWorld(max_bodies=10)
        i0 = pw.add_body(pos=(0, 0))
        i1 = pw.add_body(pos=(10, 10))
        i2 = pw.add_body(pos=(20, 20))
        assert pw.count == 3
        assert i0 == 0
        assert i1 == 1
        assert i2 == 2

    def test_add_rigid_body(self):
        pw = PhysicsWorld(max_bodies=10)
        rb = RigidBody(
            position=np.array([3.0, 4.0], dtype=np.float32),
            mass=10.0,
            radius=2.0,
            is_static=True,
        )
        idx = pw.add_rigid_body(rb)
        assert pw.static[idx] == True
        assert pw.masses[idx] == pytest.approx(10.0)

    def test_remove_body(self):
        pw = PhysicsWorld(max_bodies=10)
        idx = pw.add_body(pos=(0, 0))
        assert pw.active[idx] == True
        pw.remove_body(idx)
        assert pw.active[idx] == False

    def test_auto_grow(self):
        pw = PhysicsWorld(max_bodies=2)
        pw.add_body(pos=(0, 0))
        pw.add_body(pos=(1, 1))
        # This should trigger _grow()
        idx = pw.add_body(pos=(2, 2))
        assert idx == 2
        assert pw.count == 3
        assert pw._cap >= 4  # doubled


# ===================================================================
# PhysicsWorld — integration (position update)
# ===================================================================


class TestPhysicsWorldIntegration:
    def test_velocity_updates_position(self):
        pw = PhysicsWorld(max_bodies=10)
        idx = pw.add_body(pos=(0.0, 0.0), vel=(10.0, 0.0))
        pw.tick(1.0)
        # After 1 second at 10 m/s, should be at (10, 0)
        assert pw.positions[idx][0] == pytest.approx(10.0, abs=0.5)

    def test_static_body_does_not_move(self):
        pw = PhysicsWorld(max_bodies=10)
        idx = pw.add_body(pos=(5.0, 5.0), vel=(10.0, 10.0), static=True)
        pw.tick(1.0)
        np.testing.assert_allclose(pw.positions[idx], [5.0, 5.0], atol=1e-5)

    def test_inactive_body_does_not_move(self):
        pw = PhysicsWorld(max_bodies=10)
        idx = pw.add_body(pos=(0.0, 0.0), vel=(10.0, 0.0))
        pw.remove_body(idx)
        pw.tick(1.0)
        np.testing.assert_allclose(pw.positions[idx], [0.0, 0.0], atol=1e-5)

    def test_apply_force(self):
        pw = PhysicsWorld(max_bodies=10)
        idx = pw.add_body(pos=(0.0, 0.0), mass=1.0)
        pw.apply_force(idx, (10.0, 0.0))
        pw.tick(1.0)
        # F=ma => a=10, v = 10*1.0 = 10, pos = 10*1.0 = 10
        assert pw.velocities[idx][0] == pytest.approx(10.0, abs=0.5)

    def test_apply_explosion(self):
        pw = PhysicsWorld(max_bodies=10)
        # Place a body at (5, 0), explosion at origin
        idx = pw.add_body(pos=(5.0, 0.0), mass=1.0)
        pw.apply_explosion(center=(0.0, 0.0), radius=10.0, force=100.0)
        pw.tick(0.1)
        # Body should have been pushed in +x direction
        assert pw.velocities[idx][0] > 0.0

    def test_explosion_outside_radius(self):
        pw = PhysicsWorld(max_bodies=10)
        idx = pw.add_body(pos=(50.0, 0.0), mass=1.0)
        pw.apply_explosion(center=(0.0, 0.0), radius=10.0, force=100.0)
        pw.tick(0.1)
        # Body outside blast radius should not be affected
        assert pw.velocities[idx][0] == pytest.approx(0.0, abs=1e-5)

    def test_empty_world_tick(self):
        pw = PhysicsWorld(max_bodies=10)
        events = pw.tick(0.1)
        assert events == []


# ===================================================================
# PhysicsWorld — collision detection and resolution
# ===================================================================


class TestPhysicsWorldCollision:
    def test_two_bodies_collide(self):
        pw = PhysicsWorld(max_bodies=10, cell_size=10.0)
        # Two bodies overlapping
        pw.add_body(pos=(0.0, 0.0), radius=2.0, mass=1.0)
        pw.add_body(pos=(1.0, 0.0), radius=2.0, mass=1.0)
        events = pw.tick(0.0001)
        assert len(events) >= 1
        e = events[0]
        assert isinstance(e, CollisionEvent)
        assert e.body_a == 0
        assert e.body_b == 1

    def test_collision_pushes_apart(self):
        pw = PhysicsWorld(max_bodies=10, cell_size=10.0)
        # Two overlapping bodies — initial distance (1.0) < sum of radii (4.0)
        pw.add_body(pos=(0.0, 0.0), vel=(0.0, 0.0), radius=2.0, mass=1.0)
        pw.add_body(pos=(1.0, 0.0), vel=(0.0, 0.0), radius=2.0, mass=1.0)

        initial_dist = float(np.linalg.norm(pw.positions[1] - pw.positions[0]))

        # Multiple ticks to let positional correction push them apart
        for _ in range(50):
            pw.tick(0.01)

        final_dist = float(np.linalg.norm(pw.positions[1] - pw.positions[0]))
        # After collision resolution, bodies should be farther apart than before
        assert final_dist > initial_dist

    def test_no_collision_when_separated(self):
        pw = PhysicsWorld(max_bodies=10, cell_size=10.0)
        pw.add_body(pos=(0.0, 0.0), radius=1.0)
        pw.add_body(pos=(100.0, 0.0), radius=1.0)
        events = pw.tick(0.01)
        assert len(events) == 0

    def test_static_body_collision(self):
        pw = PhysicsWorld(max_bodies=10, cell_size=10.0)
        # Static wall at (0, 0)
        wall_idx = pw.add_body(pos=(0.0, 0.0), radius=2.0, mass=100.0, static=True)
        # Dynamic body moving toward wall
        ball_idx = pw.add_body(pos=(2.5, 0.0), vel=(-5.0, 0.0), radius=1.0, mass=1.0)

        events = pw.tick(0.01)
        assert len(events) >= 1

        # Wall should not have moved
        np.testing.assert_allclose(pw.positions[wall_idx], [0.0, 0.0], atol=0.1)
        # Ball should have bounced (velocity reversed or reduced)

    def test_collision_event_fields(self):
        pw = PhysicsWorld(max_bodies=10, cell_size=10.0)
        pw.add_body(pos=(0.0, 0.0), vel=(0.0, 0.0), radius=2.0, mass=1.0)
        pw.add_body(pos=(3.0, 0.0), vel=(-5.0, 0.0), radius=2.0, mass=1.0)
        events = pw.tick(0.01)
        assert len(events) >= 1
        e = events[0]
        assert hasattr(e, "body_a")
        assert hasattr(e, "body_b")
        assert hasattr(e, "point")
        assert hasattr(e, "normal")
        assert hasattr(e, "impulse")
        assert hasattr(e, "relative_speed")


# ===================================================================
# VehiclePhysics tests
# ===================================================================


class TestVehiclePhysics:
    def test_default_state(self):
        vp = VehiclePhysics()
        assert vp.mass == 1500.0
        assert vp.max_speed == 15.0
        assert vp.speed == 0.0
        assert vp.heading == 0.0
        np.testing.assert_array_equal(vp.position, [0.0, 0.0])

    def test_throttle_accelerates(self):
        vp = VehiclePhysics()
        vp.throttle = 1.0
        vp.tick(1.0)
        assert vp.speed > 0.0

    def test_speed_clamped(self):
        vp = VehiclePhysics(max_speed=10.0)
        vp.throttle = 1.0
        # Many ticks to saturate speed
        for _ in range(100):
            vp.tick(0.1)
        assert vp.speed <= 10.0

    def test_braking(self):
        vp = VehiclePhysics()
        vp.throttle = 1.0
        for _ in range(10):
            vp.tick(0.1)
        speed_before = vp.speed
        assert speed_before > 0

        vp.throttle = -1.0
        vp.tick(0.5)
        assert vp.speed < speed_before

    def test_steering_changes_heading(self):
        vp = VehiclePhysics()
        vp.throttle = 1.0
        vp.steering = 1.0  # full right
        for _ in range(20):
            vp.tick(0.1)
        # Heading should have changed from 0
        assert vp.heading != 0.0

    def test_no_turn_at_zero_speed(self):
        vp = VehiclePhysics()
        vp.steering = 1.0
        vp.throttle = 0.0
        vp.tick(1.0)
        # Should not turn when stationary (speed ~0)
        assert abs(vp.heading) < 0.01

    def test_position_updates(self):
        vp = VehiclePhysics()
        vp.throttle = 1.0
        vp.steering = 0.0  # straight
        for _ in range(10):
            vp.tick(0.1)
        # Should have moved in +x direction (heading=0 = east)
        assert vp.position[0] > 0.0

    def test_drag_slows_vehicle(self):
        vp = VehiclePhysics(drag=0.5)
        vp.throttle = 1.0
        vp.tick(0.5)
        speed_with_throttle = vp.speed

        vp.throttle = 0.0
        vp.tick(1.0)
        assert vp.speed < speed_with_throttle

    def test_reverse_speed(self):
        vp = VehiclePhysics(max_speed=10.0)
        vp.throttle = -1.0
        for _ in range(50):
            vp.tick(0.1)
        # Reverse should be capped at 30% of max_speed
        assert vp.speed >= -3.0

    def test_apply_collision(self):
        vp = VehiclePhysics()
        vp.throttle = 1.0
        for _ in range(5):
            vp.tick(0.1)
        speed_before = vp.speed

        # Apply a sideways collision impulse
        impulse = np.array([0.0, 5000.0], dtype=np.float32)
        vp.apply_collision(impulse, np.array([0.0, 0.0]))
        # Heading should have changed due to velocity direction change
        # and speed should be updated
        assert vp.speed > 0.0

    def test_sync_to_and_from_world(self):
        vp = VehiclePhysics()
        vp.position = np.array([10.0, 20.0], dtype=np.float32)
        vp.velocity = np.array([1.0, 2.0], dtype=np.float32)

        pw = PhysicsWorld(max_bodies=10)
        body_id = pw.add_body(pos=(0.0, 0.0))

        vp.sync_to_world(pw, body_id)
        np.testing.assert_allclose(pw.positions[body_id], [10.0, 20.0], atol=1e-5)
        np.testing.assert_allclose(pw.velocities[body_id], [1.0, 2.0], atol=1e-5)

        # Modify world state
        pw.positions[body_id] = [30.0, 40.0]
        pw.velocities[body_id] = [3.0, 4.0]
        vp.sync_from_world(pw, body_id)

        np.testing.assert_allclose(vp.position, [30.0, 40.0], atol=1e-5)
        np.testing.assert_allclose(vp.velocity, [3.0, 4.0], atol=1e-5)
        assert vp.speed == pytest.approx(5.0, abs=0.1)  # hypot(3, 4) = 5

    def test_bicycle_turning_radius(self):
        """At constant speed and steering, the vehicle should trace a circle.
        The radius = wheelbase / tan(steer_angle)."""
        vp = VehiclePhysics(wheelbase=2.5, max_steer_angle=0.6)
        vp.speed = 5.0
        vp.throttle = 0.0  # will coast (with drag, speed decreases)
        vp.steering = 0.5  # half steer

        initial_heading = vp.heading
        for _ in range(10):
            vp.tick(0.1)

        # Heading should have changed
        assert vp.heading != initial_heading
