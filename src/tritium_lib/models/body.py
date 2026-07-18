# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Body-agnostic seam vocabulary — the single source of truth.

Track A step 2 of ``docs/plans/multi-body-sil-framework.md``: ONE autonomy
stack drives a heterogeneous fleet (robot dog, rover, quadcopter, fixed-wing)
through a generalized :class:`BodyController` contract. This module is the
NEUTRAL home of that seam — it names no body and imports no body module:

* :class:`ControlIntent` — the normalized intent superset. Ground bodies read
  ``(forward, turn)`` (exactly the rover/gait_core twist); aerial bodies add
  ``climb``. All three are normalized to ``[-1, 1]`` and scale to the
  receiving profile's limits, so the autonomy stack emits ONE intent type and
  never imports a body. Zero intent = the body's natural steady state (a
  multirotor hovers, a fixed-wing holds cruise, a ground body stands).
* :class:`BodyState` — the full 6-DOF pose + velocity a body reports back.
  Ground bodies leave ``alt_m``/``pitch_deg``/``roll_deg``/``climb_mps`` at
  zero; nothing else changes.
* :class:`BodyController` — the ``command(intent)`` / ``get_state()`` pair
  every body controller implements. It is a :class:`typing.Protocol`
  (structural, ``runtime_checkable``) so sim bodies, Isaac TCP bodies, and
  real-hardware drivers satisfy it without inheriting from lib.
* :class:`SupportsTurret` / :class:`SupportsBattery` / :class:`SupportsImu` /
  :class:`SupportsGps` — the OPTIONAL hooks the framework doc keeps common
  across all bodies (turret/fire/battery/IMU/GPS). Narrow capability
  protocols, method names mirroring the edge tier's ``HardwareInterface``
  (``tritium-edge/ros2/tritium_quadruped/``), so an existing edge body
  satisfies them structurally. A body implements only what its hardware
  carries.

GROUND SPECIALIZATION (unchanged, deliberate): rover and quadruped KEEP their
2D vocabulary — ``set_motors(left, right)`` twist, gait tables, ``(x, y,
heading)`` state. Their 2D state embeds losslessly in :class:`BodyState` and
their motor twist maps 1:1 through :func:`intent_from_motors` /
:func:`motors_from_intent` — the seam generalizes AROUND them, it does not
rewrite them.

HISTORY / COMPAT: :class:`ControlIntent`, :class:`BodyState`, and
:data:`G_MPS2` were born in :mod:`tritium_lib.models.multirotor` (the first
aerial body forced the superset). They are hoisted here and RE-EXPORTED from
``multirotor`` unchanged, so every existing import path keeps working.

FRAME (identical to ``rover``/``gait_core``): Tritium convention — ``x`` =
east, ``y`` = north, ``heading`` in degrees with ``0`` = north, increasing
CLOCKWISE. Forward motion adds ``(sin(h), cos(h))``. Aloft adds ``alt_m``
(meters above ground, up positive), ``pitch_deg`` (positive = nose up), and
``roll_deg`` (positive = right wing / right side down).
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

# Standard gravity — shared by every aerial body model.
G_MPS2: float = 9.80665


class ControlIntent(BaseModel):
    """Body-agnostic control intent — the superset of the ground twist.

    All fields are normalized to ``[-1, 1]`` and scale to the receiving
    profile's limits (``forward`` -> max speed, ``turn`` -> max yaw/turn rate,
    ``climb`` -> max climb or descent rate). The all-zero default is the
    body's steady state: hover for a multirotor, cruise for a fixed-wing,
    standing for a ground body.
    """

    forward: float = Field(default=0.0, ge=-1.0, le=1.0)  # + ahead, - astern
    turn: float = Field(default=0.0, ge=-1.0, le=1.0)  # + clockwise (heading increases)
    climb: float = Field(default=0.0, ge=-1.0, le=1.0)  # + up, - down


class BodyState(BaseModel):
    """Full 6-DOF pose + velocity in the Tritium frame.

    Ground bodies fill only ``x``/``y``/``heading_deg``/``speed_mps`` (their
    2D state embeds losslessly); aerial bodies fill everything. This is the
    ``get_state()`` side of the body seam from
    ``docs/plans/multi-body-sil-framework.md``.
    """

    x: float = 0.0  # east, m
    y: float = 0.0  # north, m
    alt_m: float = Field(default=0.0, ge=0)  # above ground, m
    heading_deg: float = 0.0  # yaw: 0 = north, increasing clockwise
    pitch_deg: float = 0.0  # + nose up
    roll_deg: float = 0.0  # + right side down
    speed_mps: float = 0.0  # horizontal speed along heading (+ ahead)
    climb_mps: float = 0.0  # vertical rate, + up


@runtime_checkable
class BodyController(Protocol):
    """The generalized body seam — what EVERY body controller implements.

    The autonomy stack (perception -> world model -> planner -> decision)
    talks to a body ONLY through this pair: it emits a :class:`ControlIntent`
    and consumes a :class:`BodyState`, never a body-specific type. Structural
    and ``runtime_checkable`` — a kinematic sim body, an Isaac TCP body, and
    a real-hardware driver all satisfy it by shape, no lib inheritance.

    Optional hardware hooks (turret/fire/battery/IMU/GPS) live in the narrow
    ``Supports*`` protocols below — a body implements only what it carries.
    """

    def command(self, intent: ControlIntent) -> None:
        """Apply a control intent. The body scales it to its own envelope."""
        ...

    def get_state(self) -> BodyState:
        """Report the body's current 6-DOF pose + velocity (Tritium frame)."""
        ...


@runtime_checkable
class SupportsTurret(Protocol):
    """Optional hook: an aimable turret + firing mechanism.

    Mirrors the edge tier's ``HardwareInterface.set_turret`` /
    ``fire_trigger`` — the two travel together (aim, then fire).
    """

    def set_turret(self, pan: float, tilt: float) -> None:
        """Set turret pan/tilt angles in degrees."""
        ...

    def fire_trigger(self) -> None:
        """Activate the firing mechanism (nerf trigger)."""
        ...


@runtime_checkable
class SupportsBattery(Protocol):
    """Optional hook: battery charge telemetry."""

    def get_battery(self) -> float:
        """Battery level ``0.0``-``1.0``."""
        ...


@runtime_checkable
class SupportsImu(Protocol):
    """Optional hook: inertial measurement (orientation + acceleration).

    The reference shape is the edge tier's ``ImuState`` (roll/pitch/yaw in
    degrees, accelerations in m/s^2); the neutral tier deliberately does not
    pin the carrier class — presence of the hook is the contract.
    """

    def get_imu(self) -> Any:
        """Current IMU reading (roll/pitch/yaw degrees + accel m/s^2)."""
        ...


@runtime_checkable
class SupportsGps(Protocol):
    """Optional hook: global position fix."""

    def get_gps(self) -> tuple[float, float, float] | None:
        """``(latitude, longitude, altitude)`` or ``None`` without a fix."""
        ...


def intent_from_motors(left: float, right: float) -> ControlIntent:
    """Ground motor twist -> the neutral :class:`ControlIntent`.

    The EXACT contract ``rover``/``gait_core`` consume: motor commands (each
    clamped to ``[-1, 1]``) mean ``forward = (left + right) / 2`` and
    ``turn = left - right`` (clamped to ``[-1, 1]``). ``climb`` stays zero —
    ground bodies have no vertical intent. This is how a 2D body's existing
    ``set_motors`` vocabulary expresses itself through the generalized seam
    without changing the body.
    """
    left = max(-1.0, min(1.0, left))
    right = max(-1.0, min(1.0, right))
    return ControlIntent(
        forward=0.5 * (left + right),
        turn=max(-1.0, min(1.0, left - right)),
        climb=0.0,
    )


def motors_from_intent(intent: ControlIntent) -> tuple[float, float]:
    """The inverse map: :class:`ControlIntent` -> ``(left, right)`` motors.

    ``left = forward + turn / 2``, ``right = forward - turn / 2``, each
    clamped to ``[-1, 1]``. Exact round-trip with :func:`intent_from_motors`
    whenever the pair fits the motor envelope (``|forward| + |turn| / 2 <=
    1``); saturated demands clamp — a motor cannot exceed full throttle.
    ``climb`` is ignored: a ground body has no vertical actuator.
    """
    left = max(-1.0, min(1.0, intent.forward + 0.5 * intent.turn))
    right = max(-1.0, min(1.0, intent.forward - 0.5 * intent.turn))
    return left, right
