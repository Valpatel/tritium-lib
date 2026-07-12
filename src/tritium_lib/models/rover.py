# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Mini 4-wheel rover profile + differential-drive kinematics.

The second body in the multi-body fleet
(``docs/plans/multi-body-sil-framework.md``). A skid-steer / differential-drive
rover is driven by the SAME twist intent the robot dog uses —
``set_motors(left, right)`` — so it plugs into the identical
``HardwareInterface`` / autonomy seam with ZERO changes to the autonomy stack.
This module is the framework-free VOCABULARY + KINEMATICS a rover controller
(edge), the sim, and real hardware all agree on — the wheeled analogue of
:mod:`tritium_lib.models.quadruped`.

TWIST CONTRACT (identical to ``gait_core``/``QuadrupedProfile``): the two motor
commands ``left``/``right`` (each in ``[-1, 1]``) mean
``forward = (left + right) / 2`` and ``turn = (left - right)`` — a normalized
intent, not raw wheel velocities. ``forward`` scales to :attr:`max_speed_mps`,
``turn`` to :attr:`max_turn_dps` (skid-steer is nimble, so a rover turns faster
than the dog).

FRAME (identical to ``gait_core`` and ``isaac_quadruped_server``): Tritium
convention — ``x`` = east, ``y`` = north, ``heading`` in degrees with ``0`` =
north, increasing CLOCKWISE (north -> east -> south -> west). Forward motion adds
``(sin(h), cos(h))``; a positive turn (left wheel faster than right) increases
the heading. So a real rover and its Isaac/sim twin place identically on the map.
"""
from __future__ import annotations

import math

from pydantic import BaseModel, Field


class RoverProfile(BaseModel):
    """A mini rover's physical envelope: geometry, speed limits, and battery.

    Defaults describe a small 4-wheel skid-steer rover (~30 cm track, 1.2 m/s
    top speed, 100 Wh pack). Overridable per unit via config, exactly like
    :class:`~tritium_lib.models.quadruped.QuadrupedProfile`.
    """

    name: str = "rover"
    asset_type: str = "rover"
    wheel_count: int = Field(default=4, ge=2)
    track_width_m: float = Field(default=0.30, gt=0)   # left<->right wheel span
    wheel_radius_m: float = Field(default=0.06, gt=0)
    max_speed_mps: float = Field(default=1.2, gt=0)    # at |forward intent| = 1
    max_turn_dps: float = Field(default=180.0, gt=0)   # skid-steer is nimble
    battery_wh: float = Field(default=100.0, gt=0)
    idle_power_w: float = Field(default=8.0, gt=0)
    drive_power_w: float = Field(default=60.0, gt=0)   # electrical draw at top speed

    def twist(self, left: float, right: float) -> tuple[float, float]:
        """Motor commands (each clamped to ``[-1, 1]``) -> ``(forward_mps, turn_dps)``.

        ``forward = (left + right) / 2`` scaled to :attr:`max_speed_mps`;
        ``turn = (left - right)`` (clamped to ``[-1, 1]``) scaled to
        :attr:`max_turn_dps` — the same twist intent ``gait_core`` consumes.
        """
        left = max(-1.0, min(1.0, left))
        right = max(-1.0, min(1.0, right))
        forward_mps = 0.5 * (left + right) * self.max_speed_mps
        turn_intent = max(-1.0, min(1.0, left - right))
        turn_dps = turn_intent * self.max_turn_dps
        return forward_mps, turn_dps

    def step(
        self, x: float, y: float, heading_deg: float,
        left: float, right: float, dt: float,
    ) -> tuple[float, float, float]:
        """Integrate one tick of differential drive.

        Returns the new ``(x, y, heading_deg)`` in the Tritium frame (x=east,
        y=north, heading 0 = north increasing clockwise). Deterministic — no RNG.
        """
        forward_mps, turn_dps = self.twist(left, right)
        heading_deg = (heading_deg + turn_dps * dt) % 360.0
        h = math.radians(heading_deg)
        x += forward_mps * math.sin(h) * dt   # east
        y += forward_mps * math.cos(h) * dt   # north
        return x, y, heading_deg

    def drain_pct_per_s(self, forward_mps: float) -> float:
        """Fraction of the battery (0..1 scale) consumed per second.

        Standing burns :attr:`idle_power_w`; driving scales linearly with
        ``|forward_mps|`` up to :attr:`drive_power_w` at :attr:`max_speed_mps`.
        Mirrors :meth:`QuadrupedProfile.drain_pct_per_s`.
        """
        frac = min(1.0, abs(forward_mps) / self.max_speed_mps) if self.max_speed_mps else 0.0
        power_w = self.idle_power_w + frac * (self.drive_power_w - self.idle_power_w)
        return power_w / (self.battery_wh * 3600.0)
