# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Simple vehicle dynamics using a bicycle model.

Provides realistic turning arcs, acceleration, braking, and drag without
requiring a full rigid-body integrator.  Designed to feed position/velocity
into PhysicsWorld for collision handling.

Copyright 2026 Valpatel Software LLC — AGPL-3.0
"""

from __future__ import annotations

import math

import numpy as np


class VehiclePhysics:
    """Bicycle-model vehicle dynamics.

    Parameters
    ----------
    mass : float
        Vehicle mass in kg.
    max_speed : float
        Maximum forward speed in m/s.
    max_steer_angle : float
        Maximum front-wheel steering angle in radians.
    wheelbase : float
        Distance between front and rear axles in meters.
    acceleration : float
        Maximum acceleration in m/s^2.
    drag : float
        Linear drag coefficient (fraction of speed lost per second).
    brake_decel : float
        Deceleration when braking in m/s^2.
    """

    def __init__(
        self,
        mass: float = 1500.0,
        max_speed: float = 15.0,
        max_steer_angle: float = 0.6,
        wheelbase: float = 2.5,
        acceleration: float = 6.0,
        drag: float = 0.5,
        brake_decel: float = 10.0,
    ) -> None:
        self.mass = mass
        self.max_speed = max_speed
        self.max_steer_angle = max_steer_angle
        self.wheelbase = wheelbase
        self.acceleration = acceleration
        self.drag = drag
        self.brake_decel = brake_decel

        # State
        self.position = np.zeros(2, dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.heading = 0.0  # radians, 0 = east (+x)
        self.speed = 0.0  # signed scalar along heading

        # Controls (set each frame by AI or player input)
        self.throttle = 0.0  # -1 (full reverse) to 1 (full forward)
        self.steering = 0.0  # -1 (full left) to 1 (full right)

    # -- Tick ----------------------------------------------------------------

    def tick(self, dt: float) -> None:
        """Advance vehicle state by *dt* seconds using bicycle model.

        Updates heading, speed, velocity, and position based on current
        throttle and steering inputs.
        """
        # Effective steering angle.
        steer_angle = self.steering * self.max_steer_angle

        # Acceleration / braking.
        if self.throttle >= 0:
            # Forward throttle.
            self.speed += self.throttle * self.acceleration * dt
        else:
            # Braking / reverse.
            if self.speed > 0:
                self.speed -= abs(self.throttle) * self.brake_decel * dt
                self.speed = max(self.speed, 0.0)
            else:
                # Allow reverse at reduced acceleration.
                self.speed += self.throttle * self.acceleration * 0.5 * dt

        # Drag.
        self.speed *= 1.0 - self.drag * dt

        # Clamp speed.
        self.speed = max(-self.max_speed * 0.3, min(self.speed, self.max_speed))

        # Bicycle model: heading rate = (speed / wheelbase) * tan(steer_angle).
        if abs(self.speed) > 1e-4:
            turn_rate = (self.speed / self.wheelbase) * math.tan(steer_angle)
            self.heading += turn_rate * dt

        # Heading vector.
        hx = math.cos(self.heading)
        hy = math.sin(self.heading)

        # Update velocity and position.
        self.velocity[0] = self.speed * hx
        self.velocity[1] = self.speed * hy
        self.position += self.velocity * dt

    # -- Collision interface -------------------------------------------------

    def apply_collision(self, impulse: np.ndarray, point: np.ndarray) -> None:
        """Apply an external collision impulse.

        Parameters
        ----------
        impulse : ndarray, shape (2,)
            Impulse vector in world space (kg*m/s).
        point : ndarray, shape (2,)
            Contact point (currently unused — no torque model).
        """
        # Convert impulse to velocity change: dv = impulse / mass.
        dv = impulse / self.mass
        self.velocity += dv.astype(np.float32)
        # Update scalar speed to match new velocity magnitude.
        self.speed = float(np.linalg.norm(self.velocity))
        # Update heading to match velocity direction if moving.
        if self.speed > 0.1:
            self.heading = float(math.atan2(self.velocity[1], self.velocity[0]))

    def sync_to_world(self, world: "PhysicsWorld", body_id: int) -> None:
        """Push vehicle state into a PhysicsWorld body."""
        world.positions[body_id] = self.position
        world.velocities[body_id] = self.velocity

    def sync_from_world(self, world: "PhysicsWorld", body_id: int) -> None:
        """Pull post-collision state from PhysicsWorld back into vehicle."""
        self.position[:] = world.positions[body_id]
        self.velocity[:] = world.velocities[body_id]
        self.speed = float(np.linalg.norm(self.velocity))
        if self.speed > 0.1:
            self.heading = float(math.atan2(self.velocity[1], self.velocity[0]))
