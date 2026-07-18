# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Multirotor (quadcopter) body profile + thrust/attitude kinematics.

The third body in the multi-body fleet
(``docs/plans/multi-body-sil-framework.md``) and the first AERIAL one. This
module is the framework-free VOCABULARY + KINEMATICS a multirotor controller
(edge), the sim, and real hardware all agree on — the aerial analogue of
:mod:`tritium_lib.models.quadruped` and :mod:`tritium_lib.models.rover`.

SEAM: this body was the first to need more than the ground twist, so
:class:`~tritium_lib.models.body.ControlIntent` and the 6-DOF
:class:`~tritium_lib.models.body.BodyState` were born here — they now LIVE in
:mod:`tritium_lib.models.body` (the neutral single source of truth for the
body-agnostic seam, Track A step 2 of the plan doc) and are RE-EXPORTED here
unchanged for compatibility: ``from tritium_lib.models.multirotor import
ControlIntent, BodyState, G_MPS2`` keeps working forever.

FRAME (identical to ``rover``/``gait_core``): Tritium convention — ``x`` =
east, ``y`` = north, ``heading`` in degrees with ``0`` = north, increasing
CLOCKWISE. Forward motion adds ``(sin(h), cos(h))``. Aloft adds ``alt_m``
(meters above ground, up positive), ``pitch_deg`` (positive = nose up), and
``roll_deg`` (positive = right wing / right side down).

Scope: pure deterministic kinematics — thrust envelope, tilt-for-speed,
altitude/heading integration, battery drain. Takeoff/landing/RTL state
machines and MQTT wire telemetry stay in :mod:`tritium_lib.models.drone`
(:class:`~tritium_lib.models.drone.DroneState`,
:class:`~tritium_lib.models.drone.DroneTelemetry`); this module is what a
body CONTROLLER computes with, not what it publishes.
"""
from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field, model_validator

# Hoisted to the neutral seam module (Track A step 2); re-exported here so
# every pre-hoist import path (`from tritium_lib.models.multirotor import
# ControlIntent, BodyState, G_MPS2`) keeps working unchanged.
from .body import G_MPS2, BodyState, ControlIntent


class MultirotorProfile(BaseModel):
    """A multirotor's physical envelope: thrust, attitude, speeds, battery.

    Defaults describe a Mavic-3-class camera quadcopter (0.9 kg, 77 Wh pack,
    ~40 min hover). Overridable per unit via config, exactly like
    :class:`~tritium_lib.models.quadruped.QuadrupedProfile`.
    """

    profile: Literal["multirotor"] = "multirotor"
    name: str = "quadcopter"
    # What the drone publishes in MQTT telemetry; SC renders the glyph off
    # this ("drone" is the ecosystem's existing aerial asset vocabulary —
    # see models/tak_export.py, models/scenario.py).
    asset_type: str = "drone"
    rotor_count: int = Field(default=4, ge=3, le=12)
    rotor_layout: Literal["x", "plus", "coax"] = "x"
    mass_kg: float = Field(default=0.9, gt=0)
    max_thrust_n: float = Field(default=22.0, gt=0)  # all rotors, combined
    max_climb_mps: float = Field(default=5.0, gt=0)
    max_descent_mps: float = Field(default=3.0, gt=0)  # magnitude; descent is rate-limited
    max_speed_mps: float = Field(default=16.0, gt=0)  # horizontal, at full tilt
    max_tilt_deg: float = Field(default=30.0, gt=0, le=90)
    max_yaw_rate_dps: float = Field(default=120.0, gt=0)
    battery_wh: float = Field(default=77.0, gt=0)
    idle_power_w: float = Field(default=6.0, gt=0)  # landed, motors off, avionics up
    hover_power_w: float = Field(default=115.0, gt=0)  # flight baseline
    max_power_w: float = Field(default=260.0, gt=0)  # full climb / full tilt

    @model_validator(mode="after")
    def _envelope_consistent(self) -> "MultirotorProfile":
        """Thrust must beat weight (a multirotor that cannot hover is a
        brick) and power draws must be ordered idle <= hover <= max."""
        weight_n = self.mass_kg * G_MPS2
        if self.max_thrust_n <= weight_n:
            raise ValueError(
                f"max_thrust_n {self.max_thrust_n} must exceed weight "
                f"{weight_n:.2f} N (mass {self.mass_kg} kg) — cannot hover"
            )
        if not (self.idle_power_w <= self.hover_power_w <= self.max_power_w):
            raise ValueError(
                f"power draws must be ordered idle <= hover <= max, got "
                f"{self.idle_power_w} / {self.hover_power_w} / {self.max_power_w}"
            )
        return self

    def thrust_to_weight(self) -> float:
        """Thrust-to-weight ratio (> 1 by construction — see the validator)."""
        return self.max_thrust_n / (self.mass_kg * G_MPS2)

    def hover_thrust_frac(self) -> float:
        """Fraction of max thrust needed to hover (0..1 by construction)."""
        return (self.mass_kg * G_MPS2) / self.max_thrust_n

    def tilt_for_speed(self, requested_mps: float) -> float:
        """Body tilt (deg) the sim animates for a horizontal speed.

        Linear map to :attr:`max_tilt_deg` at :attr:`max_speed_mps`, clamped —
        the multirotor analogue of the quadruped's gait animation table: a
        visual/telemetry envelope both sim and hardware agree on, not CFD.
        """
        frac = min(1.0, abs(requested_mps) / self.max_speed_mps)
        return frac * self.max_tilt_deg

    def step(self, state: BodyState, intent: ControlIntent, dt: float) -> BodyState:
        """Integrate one tick of multirotor flight kinematics.

        Returns a NEW :class:`BodyState` in the Tritium frame. Heading turns
        first (like :meth:`RoverProfile.step <tritium_lib.models.rover.RoverProfile.step>`),
        then the body translates along the new heading. Climb and descent use
        their own asymmetric limits. Altitude clamps at the ground (0 m) —
        takeoff/landing state machines live in :mod:`tritium_lib.models.drone`,
        this is the in-flight integrator. Deterministic — no RNG.
        """
        heading_deg = (state.heading_deg + intent.turn * self.max_yaw_rate_dps * dt) % 360.0
        climb_mps = intent.climb * (
            self.max_climb_mps if intent.climb > 0 else self.max_descent_mps
        )
        alt_m = state.alt_m + climb_mps * dt
        if alt_m <= 0.0:
            alt_m, climb_mps = 0.0, 0.0  # on the ground; descent stops
        speed_mps = intent.forward * self.max_speed_mps
        h = math.radians(heading_deg)
        x = state.x + speed_mps * math.sin(h) * dt  # east
        y = state.y + speed_mps * math.cos(h) * dt  # north
        # A multirotor tilts INTO its motion: nose down (negative pitch) when
        # translating ahead, nose up when backing. Yaw is coordinated, so no roll.
        pitch_deg = -math.copysign(self.tilt_for_speed(speed_mps), speed_mps) if speed_mps else 0.0
        return BodyState(
            x=x, y=y, alt_m=alt_m, heading_deg=heading_deg,
            pitch_deg=pitch_deg, roll_deg=0.0,
            speed_mps=speed_mps, climb_mps=climb_mps,
        )

    def drain_pct_per_s(self, speed_mps: float | None, climb_mps: float = 0.0) -> float:
        """Fraction of the battery (0..1 scale) consumed per second.

        ``None`` means landed with motors off — burns :attr:`idle_power_w`.
        In flight the draw is :attr:`hover_power_w` plus a linear margin to
        :attr:`max_power_w` with effort (the larger of horizontal-speed and
        climb fractions; descending costs no more than hovering). Mirrors
        :meth:`QuadrupedProfile.drain_pct_per_s
        <tritium_lib.models.quadruped.QuadrupedProfile.drain_pct_per_s>` —
        a coarse envelope both sim and hardware agree on, not blade-element
        theory.
        """
        if speed_mps is None:
            power_w = self.idle_power_w
        else:
            speed_frac = min(1.0, abs(speed_mps) / self.max_speed_mps)
            climb_frac = min(1.0, max(0.0, climb_mps) / self.max_climb_mps)
            effort = max(speed_frac, climb_frac)
            power_w = self.hover_power_w + effort * (self.max_power_w - self.hover_power_w)
        return power_w / (self.battery_wh * 3600.0)


# Named default profiles. Treat as read-only — customize a unit with
# ``DEFAULT_QUADCOPTER.model_copy(update={...})``, never by mutation (the
# same read-only discipline as DEFAULT_GAITS in quadruped.py).
DEFAULT_QUADCOPTER: MultirotorProfile = MultirotorProfile()
DEFAULT_HEXROTOR: MultirotorProfile = MultirotorProfile(
    name="hexrotor",
    rotor_count=6,
    mass_kg=6.2,
    max_thrust_n=160.0,
    max_climb_mps=6.0,
    max_descent_mps=4.0,
    max_speed_mps=20.0,
    max_tilt_deg=35.0,
    battery_wh=460.0,
    idle_power_w=15.0,
    hover_power_w=900.0,
    max_power_w=2200.0,
)
