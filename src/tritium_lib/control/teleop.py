# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Human-in-the-loop teleop — the seam between a stick and a body.

This is the operator's half of the same junction :mod:`waypoint_follower`
serves for the planner: something upstream wants the body to move, and the
body can only obey a velocity.  A planner produces a polyline; a human
produces two analog axes.  Both land on :class:`TwistCommand`.

The axis shaping here (radial deadzone, cubic-blend expo) was extracted from
``tritium_quadruped.teleop_core`` rather than rewritten — it was already
correct and tested, but it lived inside a ROS2 package, so a rover controller
or the Command Center could not reach it without dragging ROS in.  It is pure
stdlib arithmetic, which makes ``tritium-lib`` its right home; ``teleop_core``
now re-exports these names so the ROS2 node keeps working unchanged.

What is genuinely new here is the part a batch-scripted teleop never needs and
a real one cannot ship without:

  * :class:`TeleopWatchdog` — a stale input decays to a stop.  Without it a
    dropped USB pad, a wedged browser tab or a dead link leaves the last twist
    latched and the body keeps driving.  This is the single property that
    separates a demo from something allowed near real hardware.
  * :class:`SlewLimiter` — bounds commanded acceleration, so slamming a stick
    ramps rather than steps.  Its ``emergency_stop`` deliberately bypasses the
    ramp: a limiter that also smooths the e-stop is a bug.
  * :func:`twist_command_from_intent` — the explicit sign bridge.
    :class:`~tritium_lib.models.body.ControlIntent` counts ``turn`` positive
    CLOCKWISE (heading increases); :class:`TwistCommand` counts
    ``angular_rps`` positive COUNTER-clockwise (REP-103, port).  Those are
    opposite, and ``body_node`` documents living with the mismatch rather than
    resolving it.  Crossing the two types without negating is a bug that shows
    up as a body turning the wrong way, so the negation lives in one named
    function instead of in every caller's head.

Stdlib only, so it imports on a bare Jetson next to the rest of the brain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from tritium_lib.control.waypoint_follower import TwistCommand

__all__ = [
    "AxisMap",
    "GamepadState",
    "SlewLimiter",
    "TeleopProfile",
    "TeleopWatchdog",
    "apply_deadzone",
    "apply_expo",
    "shape_axis",
    "twist_command_from_intent",
    "twist_from_stick",
]


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


# --------------------------------------------------------------------------
# axis shaping (extracted from tritium_quadruped.teleop_core)
# --------------------------------------------------------------------------


def apply_deadzone(value: float, deadzone: float) -> float:
    """Radial deadzone with continuous rescale.

    ``|value| <= deadzone`` -> exactly ``0.0`` (a resting stick never creeps);
    above it the remaining travel rescales to the full ``[0, 1]`` band so
    there is no step at the deadzone edge and full deflection still reaches
    exactly ``1.0``.  Input is clamped to ``[-1, 1]`` first, because some
    drivers report slightly outside it.
    """
    value = _clamp(value)
    deadzone = _clamp(deadzone, 0.0, 0.99)
    magnitude = abs(value)
    if magnitude <= deadzone:
        return 0.0
    rescaled = (magnitude - deadzone) / (1.0 - deadzone)
    return rescaled if value > 0.0 else -rescaled


def apply_expo(value: float, expo: float) -> float:
    """Exponential (cubic-blend) response shaping.

    ``y = (1 - expo) * x + expo * x^3`` — the standard RC-transmitter curve:
    strictly monotonic for ``expo`` in ``[0, 1]``, odd-symmetric, endpoints
    pinned (``+/-1 -> +/-1``).  Higher expo is softer around center for fine
    control with the same authority at full deflection.
    """
    value = _clamp(value)
    expo = _clamp(expo, 0.0, 1.0)
    return (1.0 - expo) * value + expo * value ** 3


def shape_axis(value: float, deadzone: float, expo: float) -> float:
    """Deadzone then expo — the full per-axis shaping chain."""
    return apply_expo(apply_deadzone(value, deadzone), expo)


# --------------------------------------------------------------------------
# stick -> twist
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AxisMap:
    """Which axis carries what, and which way round it reads.

    Pads disagree: joy_node reports stick-up as +1, SDL reports stick-down as
    +1.  Rather than a converter per driver, the inversion is data.
    """

    linear_axis: int = 1
    angular_axis: int = 0
    linear_inverted: bool = False
    angular_inverted: bool = True


@dataclass(frozen=True)
class TeleopProfile:
    """The envelope a stick is allowed to command.

    ``turbo_*`` are the raised ceilings while ``turbo_button`` is held; unset,
    turbo is a no-op rather than an error, so a profile stays valid on a pad
    with fewer buttons than the operator's.
    """

    max_linear_mps: float
    max_angular_rps: float
    deadzone: float = 0.10
    expo: float = 0.0
    axes: AxisMap = field(default_factory=AxisMap)
    enable_button: int | None = None
    turbo_button: int | None = None
    turbo_linear_mps: float | None = None
    turbo_angular_rps: float | None = None


@dataclass(frozen=True)
class GamepadState:
    """One polled frame from a pad.

    ``timestamp_s`` is when the frame was *read*, not when it is consumed —
    the watchdog needs the former to notice that reads have stopped.
    """

    axes: tuple[float, ...] = ()
    buttons: tuple[bool, ...] = ()
    timestamp_s: float = 0.0

    def axis(self, index: int) -> float:
        """Missing axis reads as centered — a short pad must not crash."""
        if 0 <= index < len(self.axes):
            return float(self.axes[index])
        return 0.0

    def button(self, index: int | None) -> bool:
        """Missing button reads as RELEASED.

        Fail-safe direction on purpose: an absent enable button must refuse to
        drive, never grant permission.
        """
        if index is None or not (0 <= index < len(self.buttons)):
            return False
        return bool(self.buttons[index])


def twist_from_stick(state: GamepadState, profile: TeleopProfile) -> TwistCommand:
    """Map one gamepad frame onto a body velocity.

    Enable (deadman) dominates: if the profile names an enable button and it
    is not held, the result is a stop regardless of the sticks.
    """
    if profile.enable_button is not None and not state.button(profile.enable_button):
        return TwistCommand.stop()

    axes = profile.axes
    linear_raw = state.axis(axes.linear_axis)
    angular_raw = state.axis(axes.angular_axis)
    if axes.linear_inverted:
        linear_raw = -linear_raw
    if axes.angular_inverted:
        angular_raw = -angular_raw

    turbo = profile.turbo_button is not None and state.button(profile.turbo_button)
    max_linear = profile.max_linear_mps
    max_angular = profile.max_angular_rps
    if turbo:
        # An unset turbo ceiling falls back to the normal one rather than
        # raising: the button existing is not a promise that it means anything.
        max_linear = profile.turbo_linear_mps or max_linear
        max_angular = profile.turbo_angular_rps or max_angular

    return TwistCommand(
        linear_mps=shape_axis(linear_raw, profile.deadzone, profile.expo) * max_linear,
        angular_rps=shape_axis(angular_raw, profile.deadzone, profile.expo) * max_angular,
    )


def twist_command_from_intent(
    intent, max_linear_mps: float, max_angular_rps: float
) -> TwistCommand:
    """``ControlIntent`` -> ``TwistCommand``, negating the yaw sign.

    ``ControlIntent.turn`` is positive CLOCKWISE; ``TwistCommand.angular_rps``
    is positive counter-clockwise (REP-103).  The negation is the whole reason
    this function exists — see the module docstring.
    """
    return TwistCommand(
        linear_mps=_clamp(intent.forward) * max_linear_mps,
        angular_rps=-_clamp(intent.turn) * max_angular_rps,
    )


# --------------------------------------------------------------------------
# safety
# --------------------------------------------------------------------------


class TeleopWatchdog:
    """Decays a command to a stop when input goes stale.

    Feed it every frame that arrives and poll it every control tick.  If no
    frame has arrived within ``timeout_s``, :meth:`poll` returns a stop and
    keeps returning one until a *fresh* frame revives it — a watchdog that
    un-trips on the clock alone would let a dead link stutter the body.

    ``timeout_s=None`` disables expiry, for a bench rig where a stop is more
    disruptive than a stale command.  That is a deliberate opt-out, not the
    default.
    """

    def __init__(self, timeout_s: float | None = 0.5) -> None:
        if timeout_s is not None and timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive or None")
        self.timeout_s = timeout_s
        self._twist = TwistCommand.stop()
        self._last_input_s: float | None = None

    def feed(self, twist: TwistCommand, now_s: float) -> TwistCommand:
        """Record a fresh command and return it."""
        self._twist = twist
        self._last_input_s = now_s
        return twist

    def poll(self, now_s: float) -> TwistCommand:
        """The command to issue right now, given how long input has been quiet."""
        if self._last_input_s is None:
            return TwistCommand.stop()
        if self.timeout_s is None:
            return self._twist
        if now_s - self._last_input_s > self.timeout_s:
            return TwistCommand.stop()
        return self._twist

    @property
    def expired(self) -> bool:
        return self._last_input_s is None


class SlewLimiter:
    """Bounds how fast a commanded velocity may change.

    Symmetric in both directions: the same bound applies accelerating and
    decelerating, because a body thrown from full speed to zero in one tick is
    as unphysical as the reverse.
    """

    def __init__(self, max_linear_accel: float, max_angular_accel: float) -> None:
        if max_linear_accel <= 0.0 or max_angular_accel <= 0.0:
            raise ValueError("acceleration limits must be positive")
        self.max_linear_accel = max_linear_accel
        self.max_angular_accel = max_angular_accel
        self._current = TwistCommand.stop()

    @staticmethod
    def _step(current: float, target: float, max_delta: float) -> float:
        delta = target - current
        if delta > max_delta:
            return current + max_delta
        if delta < -max_delta:
            return current - max_delta
        return target

    def limit(self, target: TwistCommand, dt_s: float) -> TwistCommand:
        """Advance the held command toward ``target`` by at most one tick."""
        if dt_s <= 0.0:
            # No time has passed, so nothing may change. Returning the target
            # here would make dt an unchecked bypass of the whole limiter.
            return self._current
        self._current = TwistCommand(
            linear_mps=self._step(
                self._current.linear_mps, target.linear_mps,
                self.max_linear_accel * dt_s),
            angular_rps=self._step(
                self._current.angular_rps, target.angular_rps,
                self.max_angular_accel * dt_s),
        )
        return self._current

    def emergency_stop(self) -> TwistCommand:
        """Snap to zero, ignoring the ramp, and reset the held state."""
        self._current = TwistCommand.stop()
        return self._current

    @property
    def current(self) -> TwistCommand:
        return self._current
