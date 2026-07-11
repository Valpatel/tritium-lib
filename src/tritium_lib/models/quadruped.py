# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Quadruped (robot dog) profile models — the shared gait vocabulary.

These models are the single schema shared by BOTH the simulator and a real
Unitree-Go2-class robot dog.  The sim's quadruped units animate body roll,
pitch, and bob from a :class:`GaitSpec` (roll oscillates at stride frequency,
pitch at twice stride frequency), pick a gait for a requested speed with
:meth:`QuadrupedProfile.gait_for_speed`, and drain battery per tick with
:meth:`QuadrupedProfile.drain_pct_per_s`.  A real dog reports the same
vocabulary in its telemetry to::

    tritium/{site}/robots/{device_id}/telemetry   (QoS 0, retain False)

``asset_type`` is what the dog publishes in that MQTT telemetry; SC renders
the dog glyph off this.  Because sim and hardware share the identical gait
table, a real dog commanded to a speed selects the same gait the sim would:
digital-twin parity.  This is why :data:`DEFAULT_GAITS` below is MIRRORED as
documented defaults in the robot-template example (the same way
``examples/robot-template/brain/turret.py`` mirrors the ``fire_control``
servo bounds): a gait the sim animates must be one the hardware could
physically execute.

"stand" (not moving) is deliberately NOT a gait — it is the caller's state.
A stationary dog burns ``idle_power_w``; gaits only describe locomotion.

See ``docs/MQTT-PROTOCOL.md`` for the robot telemetry topic grammar.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class GaitSpec(BaseModel):
    """One locomotion gait — speed, footfall rhythm, body motion, and cost.

    The oscillation amplitudes drive the sim's body animation: roll swings at
    ``stride_hz`` (legs alternate side to side), pitch at ``2 * stride_hz``
    (fore/aft pairs strike twice per stride cycle), and the body bobs
    vertically by ``bob_amp_m``.  ``power_w`` is the electrical draw the
    battery model integrates while this gait is active.
    """

    speed_mps: float = Field(gt=0)  # steady-state body speed
    stride_hz: float = Field(gt=0)  # footfall cycle frequency
    roll_amp_deg: float = Field(ge=0)  # body roll oscillation amplitude at stride freq
    pitch_amp_deg: float = Field(ge=0)  # body pitch oscillation amplitude (2x stride freq)
    bob_amp_m: float = Field(ge=0)  # vertical body bob amplitude
    power_w: float = Field(gt=0)  # electrical draw at this gait


# Go2-class default gait table.  MIRRORED as documented defaults in the
# robot-template example — do not retune one side without the other.
DEFAULT_GAITS: dict[str, GaitSpec] = {
    "walk": GaitSpec(
        speed_mps=0.7, stride_hz=1.6, roll_amp_deg=1.5,
        pitch_amp_deg=1.0, bob_amp_m=0.01, power_w=65.0,
    ),
    "trot": GaitSpec(
        speed_mps=1.6, stride_hz=2.6, roll_amp_deg=2.5,
        pitch_amp_deg=1.8, bob_amp_m=0.02, power_w=120.0,
    ),
    "bound": GaitSpec(
        speed_mps=3.0, stride_hz=3.2, roll_amp_deg=4.0,
        pitch_amp_deg=6.0, bob_amp_m=0.04, power_w=250.0,
    ),
}


def _default_gaits() -> dict[str, GaitSpec]:
    """Deep-copy the default gait table (avoids a shared mutable default)."""
    return {name: spec.model_copy() for name, spec in DEFAULT_GAITS.items()}


class QuadrupedProfile(BaseModel):
    """A robot dog's physical envelope: gait table, battery, and geometry.

    Defaults describe a Unitree-Go2-class dog (155 Wh pack, 0.40 m body
    height).  A profile travels with the unit — the sim engine instantiates
    one per quadruped unit, and a real dog's driver loads the same profile so
    both sides agree on which gait a requested speed maps to and how fast the
    battery drains.
    """

    profile: Literal["quadruped"] = "quadruped"
    name: str = "robot_dog"
    # What the dog publishes in MQTT telemetry; SC renders the dog glyph off this.
    asset_type: str = "robot_dog"
    leg_count: int = Field(default=4, ge=4, le=6)
    body_height_m: float = Field(default=0.40, gt=0)
    turn_rate_dps: float = Field(default=120.0, gt=0)
    battery_wh: float = Field(default=155.0, gt=0)  # Go2-class pack
    idle_power_w: float = Field(default=25.0, gt=0)
    default_gait: str = "trot"
    gaits: dict[str, GaitSpec] = Field(default_factory=_default_gaits)

    @model_validator(mode="after")
    def _default_gait_exists(self) -> "QuadrupedProfile":
        """``default_gait`` must name an entry in the gait table."""
        if self.default_gait not in self.gaits:
            raise ValueError(
                f"default_gait {self.default_gait!r} not in gaits "
                f"{sorted(self.gaits)}"
            )
        return self

    def gait_for_speed(self, requested_mps: float) -> tuple[str, GaitSpec]:
        """Pick the slowest gait that covers a requested body speed.

        Returns the ``(name, spec)`` of the SLOWEST gait whose ``speed_mps``
        is at least ``requested_mps`` (gaits sorted by speed, robust to dict
        declaration order).  If the request exceeds every gait, returns the
        fastest — the dog runs flat out rather than refusing.

        A request at or below zero maps to the slowest gait: "stand" (not
        moving) is the caller's state, not a gait, so callers gate on
        ``requested_mps > 0`` before asking for a gait at all.
        """
        by_speed = sorted(self.gaits.items(), key=lambda item: item[1].speed_mps)
        for name, spec in by_speed:
            if spec.speed_mps >= requested_mps:
                return name, spec
        return by_speed[-1]

    def drain_pct_per_s(self, gait_name: str | None) -> float:
        """Fraction of the battery (0..1 scale) consumed per second.

        ``gait_name`` selects a gait's ``power_w``; ``None`` means the dog is
        standing and burns ``idle_power_w``.  An unknown gait name raises
        ``KeyError`` — a driver asking for a gait the profile doesn't define
        is a contract violation, not a case to paper over.
        """
        power_w = self.gaits[gait_name].power_w if gait_name else self.idle_power_w
        return power_w / (self.battery_wh * 3600.0)
