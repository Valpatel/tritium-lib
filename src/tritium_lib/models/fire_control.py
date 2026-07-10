# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Fire-control command models — the wire contract for turret actuation.

These models are the single schema shared by BOTH the simulator and a real
physical Nerf-class turret.  When the sim fires a projectile, SC publishes a
``turret_aim`` command followed by a ``fire`` command to::

    tritium/{site}/robots/{robot_id}/command   (QoS 1, retain False)

A physical robot subscribed to that topic — the reference brain in
``examples/robot-template/`` — parses the *exact* keys these models emit
(``command``/``pan``/``tilt`` for aiming, ``command``/``target_id`` for
firing).  Because the sim and the real turret consume the identical schema, a
real robot subscribed to its command topic shadows the sim shot-for-shot:
digital-twin parity.  This is why the numeric bounds below MIRROR the servo
limits enforced in ``examples/robot-template/brain/turret.py``
(pan clamped to +/-90 deg, tilt clamped to -30..+60 deg): a command the sim
generates must be one the hardware could physically execute.

See ``docs/MQTT-PROTOCOL.md`` family 1a (robot command topic) for the topic
grammar.  The SC publisher is ``engine.comms.mqtt_bridge.MQTTBridge``
(``publish_turret_aim`` / ``publish_fire`` + the ``projectile_fired`` echo).
"""

from __future__ import annotations

from datetime import datetime, timezone
from math import atan2, degrees, hypot
from typing import Literal

from pydantic import BaseModel, Field, computed_field, field_validator

# Physical actuator limits mirrored from examples/robot-template/brain/turret.py.
# A generated command must stay inside what the real servos can reach.
PAN_MIN_DEG: float = -90.0
PAN_MAX_DEG: float = 90.0
TILT_MIN_DEG: float = -30.0
TILT_MAX_DEG: float = 60.0


def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string (default timestamp)."""
    return datetime.now(timezone.utc).isoformat()


class TurretAimCommand(BaseModel):
    """Aim command — pan/tilt the turret before firing.

    Wire form (as parsed by examples/robot-template/robot.py handle_command)::

        {"command": "turret_aim", "pan": 45.0, "tilt": -10.0}

    ``target_id`` is optional provenance (which tracked entity this aim is for)
    and is dropped from the wire payload when unset (``model_dump(exclude_none=True)``).
    ``pan``/``tilt`` bounds mirror the physical servo clamps in
    ``brain/turret.py`` so the sim can never command an angle the hardware
    cannot reach.
    """

    command: Literal["turret_aim"] = "turret_aim"
    pan: float = Field(ge=PAN_MIN_DEG, le=PAN_MAX_DEG)
    tilt: float = Field(ge=TILT_MIN_DEG, le=TILT_MAX_DEG)
    target_id: str | None = None
    timestamp: str = Field(default_factory=_now_iso)


class FireCommand(BaseModel):
    """Fire command — pull the trigger.

    Wire form (as parsed by examples/robot-template/robot.py handle_command)::

        {"command": "fire", "target_id": "det_person_3"}

    ``target_id`` is optional and dropped when unset.  ``burst`` is a
    forward-compatible extension (rounds per trigger pull); the reference brain
    ignores unknown keys and fires once, while a burst-capable turret may read
    it.  This keeps the schema a superset the real hardware tolerates.
    """

    command: Literal["fire"] = "fire"
    target_id: str | None = None
    burst: int = Field(default=1, ge=1, le=10)
    timestamp: str = Field(default_factory=_now_iso)


class FireSolution(BaseModel):
    """A computed aim solution: where to point and how far the target is."""

    pan: float
    tilt: float
    distance: float


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into the inclusive ``[low, high]`` range."""
    return max(low, min(high, value))


def compute_fire_solution(
    source_pos: tuple[float, float],
    target_pos: tuple[float, float],
    arc_peak: float = 0.0,
) -> FireSolution:
    """Compute a turret aim solution from shooter to target.

    Args:
        source_pos: ``(x, y)`` world position of the firing unit.
        target_pos: ``(x, y)`` world position of the target.
        arc_peak: Peak Z height of the projectile arc in world units
            (mortar/ballistic shots).  ``0`` for a flat, direct-fire shot.

    Returns:
        A :class:`FireSolution` with ``pan``/``tilt`` in degrees and
        ``distance`` in world units.

    Conventions:
        * ``pan`` uses the sim's compass convention — ``0 = +y`` (north),
          increasing clockwise — matching ``turret.heading`` in the SC combat
          behavior.  It is ``degrees(atan2(dx, dy))`` normalised to
          ``[-180, 180]`` then CLAMPED to ``[-90, 90]``.
        * ``tilt`` is the launch elevation for the ballistic arc,
          ``degrees(atan2(4 * arc_peak, distance))`` (0 for a flat shot),
          clamped to ``[-30, 60]``.

    The clamps mirror the physical servo limits in
    ``examples/robot-template/brain/turret.py`` — the sim will not emit an
    angle the real turret could not reach, preserving digital-twin parity.
    """
    dx = target_pos[0] - source_pos[0]
    dy = target_pos[1] - source_pos[1]
    distance = hypot(dx, dy)

    # Compass bearing: 0 = +y (north), clockwise. atan2(dx, dy) already
    # returns [-180, 180]; clamp into the turret's reachable pan window.
    pan = _clamp(degrees(atan2(dx, dy)), PAN_MIN_DEG, PAN_MAX_DEG)

    if arc_peak > 0.0:
        tilt = _clamp(degrees(atan2(4.0 * arc_peak, distance)), TILT_MIN_DEG, TILT_MAX_DEG)
    else:
        tilt = 0.0

    return FireSolution(pan=pan, tilt=tilt, distance=distance)


class WeaponStatus(BaseModel):
    """Weapon-status telemetry — the REVERSE of the command direction.

    Where :class:`TurretAimCommand` / :class:`FireCommand` flow SC -> robot on
    the command topic, ``WeaponStatus`` flows robot -> SC: the turret reports
    its own live ammo / reload / servo / fault state.  The robot publishes it
    to::

        tritium/{site}/robots/{device_id}/telemetry   (QoS 1, retain False)

    as a message of type ``"weapon_status"``, so SC can render magazine level,
    reload countdown, and fault indicators for a real Nerf-class turret.

    ``pan_deg`` / ``tilt_deg`` are the ACTUAL current servo positions and are
    clamped to the same physical limits (``PAN_MIN_DEG``..``PAN_MAX_DEG``,
    ``TILT_MIN_DEG``..``TILT_MAX_DEG``) that bound the command direction — a
    real servo cannot report an angle it cannot reach, so an out-of-range
    reading is pinned to the boundary rather than dropping the whole packet.

    See ``docs/MQTT-PROTOCOL.md`` for the robot telemetry topic grammar.
    """

    device_id: str
    weapon_id: str = "primary"
    ammo: int = Field(ge=0)
    max_ammo: int = Field(ge=1)
    reloading: bool = False
    reload_remaining_s: float = Field(default=0.0, ge=0.0)
    pan_deg: float
    tilt_deg: float
    fault: str | None = None
    ts: str = Field(default_factory=_now_iso)

    @field_validator("pan_deg")
    @classmethod
    def _clamp_pan(cls, value: float) -> float:
        """Pin the reported pan to the reachable servo window (mirrors the
        command direction's ``[PAN_MIN_DEG, PAN_MAX_DEG]`` bound)."""
        return _clamp(value, PAN_MIN_DEG, PAN_MAX_DEG)

    @field_validator("tilt_deg")
    @classmethod
    def _clamp_tilt(cls, value: float) -> float:
        """Pin the reported tilt to the reachable servo window (mirrors the
        command direction's ``[TILT_MIN_DEG, TILT_MAX_DEG]`` bound)."""
        return _clamp(value, TILT_MIN_DEG, TILT_MAX_DEG)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ammo_pct(self) -> float:
        """Fraction of the magazine remaining, ``0.0`` (empty) .. ``1.0`` (full).

        ``max_ammo >= 1`` is enforced, so this never divides by zero; the
        result is clamped to ``[0, 1]`` against a nonsensical over-full read.
        """
        return _clamp(self.ammo / self.max_ammo, 0.0, 1.0)
