# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Heading hold — the regulator that keeps a walking body pointed somewhere.

Why this exists
---------------
The closed-loop gait stack now holds the body *upright* — measured live on
Newton (2026-07-18), 34 of 34 trials — but nothing in it regulates where the
body points.  The same live measurement shows the heading wandering by
roughly +/-15 degrees per run: every stride deposits a small yaw error (feet
slip, the two sides never load perfectly evenly), no feedback path opposes
it, and the errors integrate.  A body that cannot hold a commanded course
cannot navigate waypoints — after ten metres of honest walking it is simply
facing the wrong way — so the drift breaks the layer *above* locomotion even
though every layer below reads healthy.

This module is the textbook fix, deliberately not novel: a **PD regulator on
heading**.  Compare the heading the body *has* against the heading it was
*told*, wrap the difference the short way around the compass, and emit a
bounded turn-rate correction for the gait to fold into its existing turn
command.  It is the heading analogue of
:class:`~tritium_lib.control.attitude_stabilizer.AttitudeStabilizer` (which
regulates roll/pitch and deliberately leaves yaw alone) and it sits a layer
*above* :class:`~tritium_lib.control.yaw_rate_loop.YawRateLoop` (which makes
a demanded yaw RATE actually happen; this decides what rate to demand)::

    planner -> follower -> YawRegulator -> YawRateLoop -> stride mixer
                pose loop:   heading loop:    rate loop:      the body
                where to go  which way to     how hard to
                             face NOW         turn

Validation status, stated honestly
----------------------------------
**Closed-form and simulated only — NOT yet validated on live Newton.**  The
gains below settle this module's own lagged closed-loop plant; the +/-15
degree figure is a live measurement, but the fix has not yet walked.  The
physical assumption the whole law rests on: *the body converts a turn-rate
command into actual yaw rate promptly and monotonically* — first-order-ish
response, no sign reversals, no dead band wider than the drift being fought.
A live gait that delivers a fraction of the demanded rate (the measured 12%
plant of :mod:`tritium_lib.control.yaw_rate_loop`) needs that rate loop
UNDER this one; a plant with a sign reversal turns any heading regulator
into positive feedback, and no gain here can save it.

Layering contract
-----------------
Stateless and frozen, in the style of
:class:`~tritium_lib.control.step_reflex.StepReflex`: configuration in,
correction out, nothing remembered between ticks.  Composable without
changing undisturbed behavior — a body exactly on its commanded heading and
not yawing gets a correction of EXACTLY ``0.0`` (byte-identical, pinned by
test), so folding the correction into a healthy gait's turn command changes
nothing until there is an error to fight.

Conventions match the Tritium ground frame everywhere they can be checked:
heading in degrees, ``0`` = north, increasing CLOCKWISE; a POSITIVE
correction means "turn clockwise" and maps to positive
:class:`~tritium_lib.models.body.ControlIntent` ``turn`` (the contract
``motors_from_intent`` and the edge tier's ``twist_to_motors`` pin: left
side faster than right).  Heading error is wrapped to ``[-180, 180)`` so a
body at 359 commanded to 1 turns 2 degrees clockwise, never 358 the other
way.  Stdlib only — no numpy — so this imports on a bare Jetson.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "DEFAULT_KD",
    "DEFAULT_KP",
    "DEFAULT_MAX_CORRECTION_DPS",
    "YawCorrection",
    "YawRegulator",
    "heading_error_deg",
]

# Gains that settle the module's own lagged closed-loop plant (0.3 s rate
# lag, 50 Hz tick) from 30 degrees of error with visible kd benefit: kp
# alone overshoots, kp+kd arrives cleanly.  Chosen by sweep against that
# plant — a defensible starting point, not a calibration for any particular
# robot, and NOT yet validated on live Newton (see module docstring).
DEFAULT_KP = 1.5
DEFAULT_KD = 0.6

# Correction ceiling in deg/s.  A heading-hold trim exists to null a
# few-degree error, not to execute turns — the follower owns those — so the
# default deliberately leaves the gait most of its turn authority (a
# Go2-class body turns at roughly 60-90 deg/s flat out).  It also caps the
# damage of one bad compass read to a bounded, recoverable nudge.
DEFAULT_MAX_CORRECTION_DPS = 30.0


def heading_error_deg(
    measured_heading_deg: float, commanded_heading_deg: float
) -> float:
    """Signed heading error in degrees, wrapped to ``[-180, 180)``.

    ``commanded - measured``, folded the short way around the compass:
    measured 359 commanded 1 is ``+2`` (turn clockwise), never ``-358``.
    The wrap is the entire difficulty and the reason this is a named public
    function rather than a subtraction at each call site — the same lesson
    :func:`tritium_lib.control.yaw_rate_loop.yaw_rate_from_headings` records
    for rates.  Positive means the commanded heading lies CLOCKWISE of the
    measured one, matching the frame's positive-turn direction, so the error
    already carries the sign of the required correction.

    An exact 180-degree disagreement returns ``-180.0`` (the half-open
    interval's closed end): both ways round are equally short, and a
    deterministic pick beats a coin flip that would dither between ticks.
    """
    delta = float(commanded_heading_deg) - float(measured_heading_deg)
    return (delta + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class YawCorrection:
    """One tick of heading regulation, including what it saw and why.

    Every term is exposed rather than just the output — the same discipline
    as :class:`~tritium_lib.control.yaw_rate_loop.YawRateCorrection` —
    because when a live run steers badly, a saturated correction and a
    wrong-sign error are different faults with different fixes, and the log
    line has to be able to tell them apart.

    ``correction_dps`` is the bounded turn-rate demand in deg/s, positive
    clockwise.  ``yaw_rate_dps`` is the measured rate the damping term used
    (``0.0`` when the caller supplied none).  ``saturated`` reports whether
    the raw PD demand hit ``max_correction_dps`` — persistent saturation
    means the drift outruns the authority this trim was given.
    """

    measured_heading_deg: float
    commanded_heading_deg: float
    error_deg: float
    yaw_rate_dps: float
    correction_dps: float
    saturated: bool

    def turn_intent(self, turn_rate_dps: float) -> float:
        """The correction as a normalized ``ControlIntent.turn`` increment.

        ``turn_rate_dps`` is the body profile's full-scale turn rate — the
        same number the edge tier's ``twist_to_motors`` divides by — so the
        returned fraction is in ``[-1, 1]`` and a driver folds it in as::

            total = max(-1.0, min(1.0, gait_turn + corr.turn_intent(limit)))
            left, right = motors_from_intent(
                ControlIntent(forward=fwd, turn=total)
            )

        A zero correction returns exactly ``0.0``, so the fold-in leaves an
        undisturbed gait's turn command byte-identical.  For a driver that
        speaks :func:`~tritium_lib.control.waypoint_follower.differential_stride`
        instead, note that :class:`TwistCommand` is REP-103 (+yaw =
        counter-clockwise) while this correction is compass-positive
        (clockwise): the rad/s conversion is
        ``-math.radians(correction_dps)`` — the negation is the contract,
        dropping it turns every correction into positive feedback.
        """
        if turn_rate_dps <= 0.0:
            raise ValueError(
                f"turn_rate_dps must be > 0, got {turn_rate_dps}; a body "
                "with no turn authority cannot express any correction"
            )
        return max(-1.0, min(1.0, self.correction_dps / float(turn_rate_dps)))

    def as_dict(self) -> dict:
        return {
            "measured_heading_deg": self.measured_heading_deg,
            "commanded_heading_deg": self.commanded_heading_deg,
            "error_deg": self.error_deg,
            "yaw_rate_dps": self.yaw_rate_dps,
            "correction_dps": self.correction_dps,
            "saturated": self.saturated,
        }


@dataclass(frozen=True)
class YawRegulator:
    """PD heading-hold regulator for one body.

    Frozen and stateless — configuration in, correction out, nothing
    remembered between ticks.  Statelessness is what makes the layering
    safe: the regulator cannot drift, wind up, or disagree with itself
    across a reset, and two arms of an A/B fed the same measurements get
    the same corrections.  It also means the damping term uses a MEASURED
    yaw rate (gyro or solver read) rather than a finite difference the
    object would have to remember headings for; with no rate supplied the
    kd term is honestly zero, not secretly estimated.

    :param kp: proportional gain, (deg/s of correction) per degree of error.
    :param kd: damping gain against the measured yaw rate.  Zero disables
        damping; the closed-loop test in this module's suite demonstrates
        the overshoot that costs.
    :param max_correction_dps: hard ceiling on the emitted correction,
        deg/s.  Keeps a heading trim from stealing the gait's full turn
        authority and bounds the damage of a single bad compass read.
    """

    kp: float = DEFAULT_KP
    kd: float = DEFAULT_KD
    max_correction_dps: float = DEFAULT_MAX_CORRECTION_DPS

    def __post_init__(self) -> None:
        if self.kp < 0.0 or self.kd < 0.0:
            raise ValueError(
                f"gains must be non-negative (got kp={self.kp}, "
                f"kd={self.kd}); a negative gain is positive feedback that "
                "amplifies the very drift this regulator exists to null"
            )
        if self.max_correction_dps <= 0.0:
            raise ValueError(
                f"max_correction_dps must be positive, got "
                f"{self.max_correction_dps}; a zero ceiling means the "
                "regulator cannot act at all, and a regulator that silently "
                "cannot act should be configured off, not configured mute"
            )

    def correct(
        self,
        measured_heading_deg: float,
        commanded_heading_deg: float,
        *,
        measured_yaw_rate_dps: float | None = None,
        dt: float | None = None,
    ) -> YawCorrection:
        """One tick: measured and commanded heading in, bounded demand out.

        ``measured_yaw_rate_dps`` is the body's actual yaw rate (positive
        clockwise, matching the heading convention) from a gyro or the
        solver; supply it to enable the damping term.  ``dt`` is optional
        and, when given, enables a **deadbeat cap**: the toward-target
        component of the correction is additionally limited to
        ``|error| / dt``, the rate that closes the error in exactly one
        tick — anything hotter is guaranteed overshoot *within the tick*
        at coarse control rates, which no gain tuning can see.  The cap
        never touches a correction pointing away from the error (damping
        must stay free to oppose a spin through the setpoint).

        Zero error with zero (or absent) measured rate returns a correction
        of exactly ``0.0`` — the byte-identical no-op the layering contract
        promises.
        """
        if dt is not None and dt <= 0.0:
            raise ValueError(
                f"dt must be positive when supplied, got {dt}; a "
                "non-positive tick has no deadbeat rate to cap against"
            )
        error = heading_error_deg(measured_heading_deg, commanded_heading_deg)
        rate = (
            0.0 if measured_yaw_rate_dps is None else float(measured_yaw_rate_dps)
        )

        # PD with derivative on the MEASUREMENT: d(error)/dt = -rate for a
        # constant command, and derivative-on-error would kick hard at the
        # exact moment a follower slews the commanded heading — the same
        # reasoning AttitudeStabilizer records for its rate term.
        raw = self.kp * error - self.kd * rate

        # Deadbeat cap first (tighter near the setpoint), authority clamp
        # second — order matters only for the `saturated` report, which is
        # about the authority ceiling, not the tick geometry.
        if dt is not None:
            bound = abs(error) / dt
            if error > 0.0:
                raw = min(raw, bound)
            elif error < 0.0:
                raw = max(raw, -bound)
            # error == 0.0: no toward-target component exists to cap; any
            # remaining demand is pure damping and stays free.

        limit = self.max_correction_dps
        saturated = abs(raw) > limit
        correction = max(-limit, min(limit, raw))
        return YawCorrection(
            measured_heading_deg=float(measured_heading_deg),
            commanded_heading_deg=float(commanded_heading_deg),
            error_deg=error,
            yaw_rate_dps=rate,
            correction_dps=correction,
            saturated=saturated,
        )

    def hold(
        self,
        measured_heading_deg: float,
        commanded_heading_deg: float,
        gait_turn: float,
        turn_rate_dps: float,
        *,
        measured_yaw_rate_dps: float | None = None,
        dt: float | None = None,
    ) -> float:
        """Convenience fold-in: the gait's turn command with the correction.

        Exactly ``clamp(gait_turn + correct(...).turn_intent(turn_rate_dps))``
        — provided so the three-line driver idiom has one honest name.  With
        zero error and zero rate the addend is exactly ``0.0`` and the
        returned value is byte-identical to ``gait_turn`` (clamped only if
        the gait itself was already out of range, which is its own bug).
        """
        corr = self.correct(
            measured_heading_deg,
            commanded_heading_deg,
            measured_yaw_rate_dps=measured_yaw_rate_dps,
            dt=dt,
        )
        total = float(gait_turn) + corr.turn_intent(turn_rate_dps)
        return max(-1.0, min(1.0, total))


def _closed_loop_overshoot(
    kp: float,
    kd: float,
    *,
    error_deg: float = 30.0,
    tau_s: float = 0.3,
    dt_s: float = 0.02,
    duration_s: float = 8.0,
    max_correction_dps: float = 90.0,
) -> float:
    """Peak overshoot (deg) of the regulator against a lagged yaw plant.

    The plant is the simplest one that can overshoot at all: commanded rate
    reaches the body through a first-order lag ``tau_s`` (yaw inertia +
    stride quantization), heading integrates the achieved rate.  It exists
    so the damping claim in this module is demonstrated against dynamics
    rather than asserted — the test suite runs it with ``kd = 0`` and
    ``kd > 0`` and requires the overshoot ordering the docstring promises.
    Simulation only; the live plant will differ (see module docstring).
    """
    regulator = YawRegulator(
        kp=kp, kd=kd, max_correction_dps=max_correction_dps
    )
    commanded = float(error_deg)
    heading = 0.0
    rate = 0.0
    overshoot = 0.0
    alpha = dt_s / (tau_s + dt_s)  # first-order lag blend, unconditionally stable
    for _ in range(int(duration_s / dt_s)):
        corr = regulator.correct(
            heading, commanded, measured_yaw_rate_dps=rate,
        )
        rate += alpha * (corr.correction_dps - rate)
        heading += rate * dt_s
        overshoot = max(overshoot, heading - commanded)
    return overshoot
