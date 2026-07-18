# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Fixed-wing body profile + airspeed/control-surface kinematics.

The fourth body in the multi-body fleet
(``docs/plans/multi-body-sil-framework.md``). A fixed-wing aircraft is driven
by the SAME :class:`~tritium_lib.models.body.ControlIntent` the multirotor
uses — ``forward`` picks an airspeed, ``turn`` a bank angle, ``climb`` a
vertical rate — so it plugs into the identical autonomy seam with ZERO
changes to the autonomy stack. This module is the framework-free
VOCABULARY + KINEMATICS a fixed-wing controller (edge), the sim, and real
hardware all agree on — the winged analogue of
:mod:`tritium_lib.models.multirotor`.

WHAT MAKES A WING DIFFERENT (and what the model enforces):

* It can NEVER fly below :attr:`FixedWingProfile.stall_speed_mps` — intent
  maps into the ``[stall, max]`` airspeed envelope, with ``forward = 0``
  meaning cruise (the natural trim), ``-1`` the stall floor, ``+1`` flat out.
* It turns by BANKING: heading rate is ``g * tan(bank) / v`` (the standard
  coordinated-turn relation), so the same turn intent yields a WIDER turn at
  higher airspeed — unlike the yaw-in-place multirotor.
* Its actuators are control SURFACES, described by a
  :class:`ControlSurfaceSpec` table exactly the way the quadruped carries a
  gait table.

FRAME + SEAM (identical to ``multirotor``/``rover``/``gait_core``): Tritium
convention — ``x`` = east, ``y`` = north, ``heading`` 0 = north increasing
clockwise; :class:`~tritium_lib.models.body.BodyState` carries the 6-DOF
pose (imported from the neutral seam module, not duplicated). Ground track
equals airspeed — no wind model at this tier; wind belongs to the physics
rig (Isaac), not the shared vocabulary.
"""
from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .body import G_MPS2, BodyState, ControlIntent


class ControlSurfaceSpec(BaseModel):
    """One control surface — deflection travel and actuator slew rate.

    ``travel_deg`` is the symmetric deflection limit (+/- from neutral);
    ``rate_dps`` is how fast the servo can slew it. The table below is the
    fixed-wing's analogue of the quadruped's gait table: the envelope a sim
    animation and a real servo both agree on.
    """

    travel_deg: float = Field(gt=0)  # symmetric deflection limit from neutral
    rate_dps: float = Field(gt=0)  # actuator slew rate


# Foam-trainer-class default surface table. Treat as read-only — profiles
# deep-copy it via the default factory below (same pattern as DEFAULT_GAITS).
DEFAULT_SURFACES: dict[str, ControlSurfaceSpec] = {
    "aileron": ControlSurfaceSpec(travel_deg=25.0, rate_dps=200.0),
    "elevator": ControlSurfaceSpec(travel_deg=20.0, rate_dps=180.0),
    "rudder": ControlSurfaceSpec(travel_deg=25.0, rate_dps=180.0),
}


def _default_surfaces() -> dict[str, ControlSurfaceSpec]:
    """Deep-copy the default surface table (avoids a shared mutable default)."""
    return {name: spec.model_copy() for name, spec in DEFAULT_SURFACES.items()}


class FixedWingProfile(BaseModel):
    """A fixed-wing aircraft's envelope: airspeeds, bank limit, surfaces, battery.

    Defaults describe a 2 kg foam-trainer-class electric plane (0.60 m^2
    wing, 9 m/s stall, 16 m/s cruise, ~48 min cruise endurance on an 88 Wh
    pack). Overridable per unit via config, exactly like
    :class:`~tritium_lib.models.multirotor.MultirotorProfile`.
    """

    profile: Literal["fixedwing"] = "fixedwing"
    name: str = "fixedwing"
    # What the plane publishes in MQTT telemetry; SC renders the glyph off this.
    asset_type: str = "fixed_wing"
    mass_kg: float = Field(default=2.0, gt=0)
    wing_area_m2: float = Field(default=0.60, gt=0)
    wingspan_m: float = Field(default=1.8, gt=0)
    stall_speed_mps: float = Field(default=9.0, gt=0)
    cruise_speed_mps: float = Field(default=16.0, gt=0)
    max_speed_mps: float = Field(default=28.0, gt=0)
    max_bank_deg: float = Field(default=45.0, gt=0, lt=90)
    max_climb_mps: float = Field(default=5.0, gt=0)
    max_sink_mps: float = Field(default=4.0, gt=0)  # magnitude; descent is rate-limited
    surfaces: dict[str, ControlSurfaceSpec] = Field(default_factory=_default_surfaces)
    battery_wh: float = Field(default=88.0, gt=0)
    idle_power_w: float = Field(default=4.0, gt=0)  # parked, avionics up
    cruise_power_w: float = Field(default=110.0, gt=0)  # flight baseline at cruise
    max_power_w: float = Field(default=350.0, gt=0)  # flat out / full climb

    @model_validator(mode="after")
    def _envelope_consistent(self) -> "FixedWingProfile":
        """Airspeeds must be ordered stall < cruise <= max, and power draws
        idle <= cruise <= max — a wing whose cruise sits below stall (or
        whose cruise draw beats full power) is a contract violation."""
        if not (self.stall_speed_mps < self.cruise_speed_mps <= self.max_speed_mps):
            raise ValueError(
                f"airspeeds must be ordered stall < cruise <= max, got "
                f"{self.stall_speed_mps} / {self.cruise_speed_mps} / {self.max_speed_mps}"
            )
        if not (self.idle_power_w <= self.cruise_power_w <= self.max_power_w):
            raise ValueError(
                f"power draws must be ordered idle <= cruise <= max, got "
                f"{self.idle_power_w} / {self.cruise_power_w} / {self.max_power_w}"
            )
        return self

    def wing_loading_kg_m2(self) -> float:
        """Wing loading (kg per square meter) — the classic handling figure."""
        return self.mass_kg / self.wing_area_m2

    def airspeed_for_intent(self, forward: float) -> float:
        """Map a forward intent (clamped to ``[-1, 1]``) into the airspeed envelope.

        Piecewise linear with ``0`` = :attr:`cruise_speed_mps` (the natural
        trim), ``-1`` = :attr:`stall_speed_mps`, ``+1`` = :attr:`max_speed_mps`.
        A wing can NEVER fly below stall, so this is the only way intent
        becomes airspeed — there is no zero-speed hover to fall back to.
        """
        forward = max(-1.0, min(1.0, forward))
        if forward >= 0.0:
            return self.cruise_speed_mps + forward * (self.max_speed_mps - self.cruise_speed_mps)
        return self.cruise_speed_mps + forward * (self.cruise_speed_mps - self.stall_speed_mps)

    def turn_radius_m(self, airspeed_mps: float, bank_deg: float | None = None) -> float:
        """Coordinated-turn radius ``v^2 / (g * tan(bank))`` in meters.

        ``bank_deg`` defaults to :attr:`max_bank_deg` (tightest allowed turn)
        and is clamped into ``(0, max_bank_deg]``; ``airspeed_mps`` is clamped
        into the flyable ``[stall, max]`` envelope — a radius quoted below
        stall would describe a falling plane, not a turning one.
        """
        bank_deg = self.max_bank_deg if bank_deg is None else bank_deg
        bank_deg = min(abs(bank_deg), self.max_bank_deg)
        v = max(self.stall_speed_mps, min(self.max_speed_mps, airspeed_mps))
        return (v * v) / (G_MPS2 * math.tan(math.radians(bank_deg)))

    def turn_rate_dps(self, airspeed_mps: float, bank_deg: float | None = None) -> float:
        """Coordinated-turn heading rate ``g * tan(bank) / v`` in deg/s.

        Same clamping as :meth:`turn_radius_m`; the two agree by construction
        (``omega = v / r``). Slower + steeper = faster heading change.
        """
        bank_deg = self.max_bank_deg if bank_deg is None else bank_deg
        bank_deg = min(abs(bank_deg), self.max_bank_deg)
        v = max(self.stall_speed_mps, min(self.max_speed_mps, airspeed_mps))
        return math.degrees(G_MPS2 * math.tan(math.radians(bank_deg)) / v)

    def step(self, state: BodyState, intent: ControlIntent, dt: float) -> BodyState:
        """Integrate one tick of fixed-wing flight kinematics.

        Returns a NEW :class:`~tritium_lib.models.body.BodyState` in the
        Tritium frame. ``turn`` intent banks the wing (roll = intent *
        :attr:`max_bank_deg`) and the heading rate follows the
        coordinated-turn relation — the wing rolls INTO the turn, unlike the
        flat-yawing multirotor. Pitch reports the flight-path angle
        (``asin(climb / v)``). Altitude clamps at the ground (0 m); ground
        track equals airspeed (no wind at this tier). Deterministic — no RNG.
        """
        v = self.airspeed_for_intent(intent.forward)
        bank_deg = intent.turn * self.max_bank_deg
        turn_dps = math.degrees(G_MPS2 * math.tan(math.radians(bank_deg)) / v)
        heading_deg = (state.heading_deg + turn_dps * dt) % 360.0
        climb_mps = intent.climb * (self.max_climb_mps if intent.climb > 0 else self.max_sink_mps)
        alt_m = state.alt_m + climb_mps * dt
        if alt_m <= 0.0:
            alt_m, climb_mps = 0.0, 0.0  # on the deck; descent stops
        h = math.radians(heading_deg)
        x = state.x + v * math.sin(h) * dt  # east
        y = state.y + v * math.cos(h) * dt  # north
        pitch_deg = math.degrees(math.asin(max(-1.0, min(1.0, climb_mps / v))))
        return BodyState(
            x=x, y=y, alt_m=alt_m, heading_deg=heading_deg,
            pitch_deg=pitch_deg, roll_deg=bank_deg,
            speed_mps=v, climb_mps=climb_mps,
        )

    def drain_pct_per_s(self, airspeed_mps: float | None, climb_mps: float = 0.0) -> float:
        """Fraction of the battery (0..1 scale) consumed per second.

        ``None`` means parked with the motor off — burns :attr:`idle_power_w`.
        In flight the draw is :attr:`cruise_power_w` plus a linear margin to
        :attr:`max_power_w` with effort (the larger of the over-cruise speed
        fraction and the climb fraction; descending costs no more than
        cruising). Mirrors :meth:`MultirotorProfile.drain_pct_per_s
        <tritium_lib.models.multirotor.MultirotorProfile.drain_pct_per_s>` —
        a coarse envelope, not a power curve.
        """
        if airspeed_mps is None:
            power_w = self.idle_power_w
        else:
            speed_span = self.max_speed_mps - self.cruise_speed_mps
            speed_frac = (
                min(1.0, max(0.0, airspeed_mps - self.cruise_speed_mps) / speed_span)
                if speed_span > 0 else 0.0
            )
            climb_frac = min(1.0, max(0.0, climb_mps) / self.max_climb_mps)
            effort = max(speed_frac, climb_frac)
            power_w = self.cruise_power_w + effort * (self.max_power_w - self.cruise_power_w)
        return power_w / (self.battery_wh * 3600.0)


# Named default profile. Treat as read-only — customize a unit with
# ``DEFAULT_FIXEDWING.model_copy(update={...})``, never by mutation.
DEFAULT_FIXEDWING: FixedWingProfile = FixedWingProfile()
