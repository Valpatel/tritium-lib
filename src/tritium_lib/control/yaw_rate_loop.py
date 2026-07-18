# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Closing the loop on yaw *rate* — the gap a position loop cannot see.

:mod:`tritium_lib.control.waypoint_follower` already closes a loop: it reads
the body's measured pose every step and re-aims.  That loop is correct and it
is not enough, because it assumes the body delivers the yaw rate it is asked
for.  A live Newton-stepped Go2 delivers roughly **12%** of it — the stride
mixer's authority is simply lower than the pure-pursuit demand.  A position
loop responds to that by asking for the same too-small turn again, every step,
forever: the body tracks a wide arc, misses the corner, and the route ends
*short* rather than *wrong*.  Which is exactly what was measured.

The fix is the standard one and it belongs one layer below the follower: an
inner rate loop.  Compare the yaw rate the body *achieved* against the yaw
rate it was *told*, and correct the demand by the difference.  Proportional
action gives immediate response; integral action is what actually solves a
persistent plant-gain deficit, because a constant fractional shortfall is a
constant error and only an integrator drives a constant error to zero.

Deliberately body-agnostic: the loop never learns "the Go2 delivers 12%".  It
observes whatever the body does and compensates for it, so the same object in
front of a rover, a tracked vehicle or a stronger-hipped quadruped needs no
retuning.  That is why integral action is preferred here over the obvious
alternative — multiplying the demand by a measured 1/0.12 gain constant, which
is a number that would be wrong on the next body, on a different surface, or
after a payload change.

Cascade, outermost first::

    planner  ->  PurePursuitFollower  ->  YawRateLoop  ->  differential_stride
     route        pose loop: where          rate loop:        the body's
                  to point                  how hard          actual mixer

Stdlib only, so it imports on a bare Jetson alongside the rest of the brain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "YawRateCorrection",
    "YawRateLoop",
    "yaw_rate_from_headings",
]


def yaw_rate_from_headings(
    previous_rad: float, current_rad: float, dt_s: float
) -> float:
    """Yaw rate implied by two headings ``dt_s`` apart, in rad/s.

    The wrap is the entire difficulty and the reason this is a named function
    rather than a subtraction at each call site.  A body crossing due-south
    goes from ``+179°`` to ``-179°``; the naive difference reads that 2° nudge
    as a 358° spin in the opposite direction, and a rate loop fed that number
    slams the stride to full opposite lock at the worst possible moment.

    ``dt_s == 0`` returns ``0.0`` rather than dividing: a repeated simulation
    timestamp is a normal event on a paused or re-entrant step callback, and
    an ``inf`` would poison the integrator permanently.
    """
    if dt_s < 0.0:
        raise ValueError(f"dt_s must be >= 0, got {dt_s}")
    if dt_s == 0.0:
        return 0.0
    delta = math.atan2(
        math.sin(current_rad - previous_rad), math.cos(current_rad - previous_rad)
    )
    return delta / dt_s


@dataclass(frozen=True)
class YawRateCorrection:
    """One tick of the rate loop, including what it did and why.

    Every term is exposed rather than just the output, because the offline
    graders in this package take only ground truth as input — when a live run
    steers badly, these fields are how the cause is told apart from the
    symptom (a saturated output and a growing integral are different faults
    with different fixes).
    """

    commanded_rps: float
    measured_rps: float
    compensated_rps: float
    error_rps: float
    integral: float
    saturated: bool


class YawRateLoop:
    """PI compensator on yaw rate, with conditional-integration anti-windup.

    ``max_output_rps`` is a demand ceiling, not a promise about the body: it
    exists so a weak plant cannot drive the mixer to a stride scale that
    trips the gait.  Set it above the follower's own ``max_angular_rps`` —
    the whole point is to ask for *more* than the follower wanted, so that
    what arrives is what the follower wanted.
    """

    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 6.0,
        max_output_rps: float = 4.0,
        integral_limit: float = 2.0,
    ) -> None:
        if kp < 0.0:
            raise ValueError(f"kp must be >= 0, got {kp}")
        if ki < 0.0:
            raise ValueError(f"ki must be >= 0, got {ki}")
        if max_output_rps <= 0.0:
            raise ValueError(f"max_output_rps must be > 0, got {max_output_rps}")
        if integral_limit < 0.0:
            raise ValueError(f"integral_limit must be >= 0, got {integral_limit}")
        self.kp = float(kp)
        self.ki = float(ki)
        self.max_output_rps = float(max_output_rps)
        self.integral_limit = float(integral_limit)
        self.integral = 0.0

    def reset(self) -> None:
        """Forget accumulated error — call between runs, not between ticks."""
        self.integral = 0.0

    def update(
        self, commanded_rps: float, measured_rps: float, dt_s: float
    ) -> YawRateCorrection:
        """Correct ``commanded_rps`` using what the body actually achieved."""
        if dt_s < 0.0:
            raise ValueError(f"dt_s must be >= 0, got {dt_s}")

        error = float(commanded_rps) - float(measured_rps)

        # The unsaturated demand, evaluated with the integral as it stands.
        # Whether this tick's error is allowed to move the integral is decided
        # below, from this value — integrating first and asking afterwards is
        # precisely the windup this guards against.
        raw = float(commanded_rps) + self.kp * error + self.ki * self.integral
        limit = self.max_output_rps
        saturated = abs(raw) > limit

        # Conditional integration: accumulate unless the output is already
        # against its stop AND this error pushes it further into the stop.
        # An integrator that keeps climbing while saturated buys authority
        # that physically cannot be delivered, then spends seconds unwinding
        # it after the error reverses — the body carries on turning long
        # after the follower asked it to straighten.
        winding_up = saturated and (raw > 0.0) == (error > 0.0)
        if dt_s > 0.0 and not winding_up:
            self.integral += error * dt_s
            self.integral = max(
                -self.integral_limit, min(self.integral_limit, self.integral)
            )
            raw = (
                float(commanded_rps) + self.kp * error + self.ki * self.integral
            )
            saturated = abs(raw) > limit

        compensated = max(-limit, min(limit, raw))
        return YawRateCorrection(
            commanded_rps=float(commanded_rps),
            measured_rps=float(measured_rps),
            compensated_rps=compensated,
            error_rps=error,
            integral=self.integral,
            saturated=saturated,
        )
