# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Closed-loop attitude regulation for a legged or wheeled body.

Why this exists
---------------
The Go2 gait in ``tritium-addons/isaac_sim`` is **open-loop**: it replays a
joint table against the clock and never looks at the robot.  Measured over two
recorded ticks it stayed upright 4 of 6 trials and then 2 of 7 — same gait
file, same gains, same ground.  An open-loop gait has no mechanism that could
make those numbers agree, because nothing in the system is trying to keep the
body level; whether a stride recovers or compounds is decided by whatever tilt
it happened to start with.  The failures were not stumbles either — they were
full 180-degree inversions, the signature of a disturbance that grows every
stride instead of decaying.

This module supplies the missing feedback path.  It is the textbook fix and is
deliberately not novel: measure body attitude, run a **PD** law on roll and
pitch, and distribute the correction across the contact points as per-leg
height offsets.  That decomposition — attitude error in, per-foot Δz out — is
the standard virtual-model / balance-controller stage used on production
quadrupeds; the gait generator keeps owning *where* the feet go in the stride,
and this owns *how far down* each one reaches.

Design notes worth knowing
--------------------------
* **Sign discipline.**  A PD controller with a reversed sign is not a weak
  controller, it is positive feedback — it drives the body over faster than no
  controller at all.  Every convention below is pinned by a test that fails if
  it is flipped.
* **Conventions are REP-103** (ROS): body frame +X forward, +Y left, +Z up, and
  therefore **positive pitch is nose-down**.  That last one surprises people
  often enough to be worth stating twice; it is why the front legs *extend* to
  correct positive pitch.
* **Derivative on measurement.**  The rate term differentiates the measured
  angle, not the error.  With a zero setpoint these coincide, but the setpoint
  becomes non-zero the moment a body deliberately leans into a turn, and
  derivative-on-error would kick hard at exactly that moment.
* **Yaw is invisible.**  A walking body changes heading constantly and that is
  not a fall — the same reasoning as
  :mod:`tritium_lib.geo.body_attitude`, which measures whether the body *has*
  fallen.  This module tries to stop it.  They are the metric and the actuator
  of one idea and share its conventions.
* **Stdlib only** — no numpy — so this imports on a bare Jetson.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

__all__ = [
    "DEFAULT_KD",
    "DEFAULT_KP",
    "DEFAULT_MAX_CMD",
    "AttitudeCorrection",
    "AttitudeStabilizer",
    "LegPlacement",
    "roll_pitch_deg",
]

# Gains that settle the module's own unstable convergence plant from 8 deg to
# under 0.1 deg in 4 s with no overshoot past the initial condition, chosen by
# sweep.  They are a defensible starting point, not a calibration for any
# particular robot — a new plant should be swept again, and the convergence
# test is written against these values so a retune has to face it.
DEFAULT_KP = 0.8
DEFAULT_KD = 0.3

# The correction is a dimensionless lean command that becomes metres of foot
# travel once multiplied by a lever arm.  Clamping it keeps a single bad pose
# read — or the instant of an actual crash — from commanding a joint excursion
# that damages hardware or explodes the solver.
DEFAULT_MAX_CMD = 1.0


def roll_pitch_deg(quat_wxyz: Sequence[float]) -> tuple[float, float]:
    """Roll and pitch in degrees from a ``(w, x, y, z)`` quaternion.

    Yaw is discarded rather than returned, because no caller here should be
    tempted to regulate it: heading is the gait's business, not the balance
    controller's.  The quaternion need not be unit length — solver-read
    quaternions drift, and normalising is this function's job.
    """
    if len(quat_wxyz) != 4:
        raise ValueError(
            f"quaternion must have 4 components (w, x, y, z), got {len(quat_wxyz)}"
        )
    w, x, y, z = (float(c) for c in quat_wxyz)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        raise ValueError(
            "quaternion is all zeros, which is not a rotation. This usually means "
            "the pose read failed; treating it as level would hide that and hand "
            "the controller a fabricated zero error."
        )
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    # Standard ZYX (yaw-pitch-roll) extraction.
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    # asin's argument leaves [-1, 1] only through round-off; clamp so a body at
    # exactly +-90 deg pitch raises nothing.
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sin_pitch)))
    return math.degrees(roll), math.degrees(pitch)


@dataclass(frozen=True)
class LegPlacement:
    """Where a contact point sits in the body frame, in metres.

    Only the horizontal offsets matter: ``x`` is the lever arm about the pitch
    axis and ``y`` about the roll axis.  A leg's own height is irrelevant to how
    much moment it can apply, which is why it is absent.

    Passing the layout as data is what keeps this module body-agnostic — a
    quadruped, a hexapod and a four-corner rover suspension differ only in the
    tuple handed to :meth:`AttitudeCorrection.leg_height_offsets`.
    """

    name: str
    x: float
    y: float


@dataclass(frozen=True)
class AttitudeCorrection:
    """One control cycle's output.

    ``roll_cmd`` and ``pitch_cmd`` are restoring commands in **radians** that
    oppose the measured tilt; multiplying by a lever arm in metres turns them
    into metres of foot travel.  Angles and rates are reported in degrees
    because that is what a person reads off a trial log; only the control law
    is metric-radian internally.  The
    measured angles and rates ride along because a caller logging a trial wants
    the input that produced the output in the same record.
    """

    roll_deg: float
    pitch_deg: float
    roll_rate_dps: float
    pitch_rate_dps: float
    roll_cmd: float
    pitch_cmd: float

    def leg_height_offsets(
        self, legs: Iterable[LegPlacement]
    ) -> dict[str, float]:
        """Per-leg vertical offsets in metres, keyed by leg name.

        Positive extends the leg (foot reaches further down, that corner of the
        body rises).  The mapping is::

            dz = -pitch_cmd * x + roll_cmd * y

        The two signs differ, and that asymmetry is not a typo — it falls out of
        REP-103 defining positive roll as *left-side-up* but positive pitch as
        *nose-down*, so the two axes have opposite handedness with respect to
        "which end went up".  Working it through: a positive roll leans the body
        right-side-down, so the restoring command is negative and the right legs
        (``y < 0``) must extend, giving ``+roll_cmd * y``.  A positive pitch is
        nose-down, so its restoring command is likewise negative but now the
        *front* legs (``x > 0``) must extend, which needs ``-pitch_cmd * x``.
        Both signs are pinned by tests that fail if either is flipped.

        The offsets are mean-centred, so they carry zero net vertical component:
        this stage rotates the body without also raising or lowering it.  Ride
        height belongs to the gait, and an attitude controller that quietly
        changed it would fight its own stride generator.
        """
        placements = tuple(legs)
        if not placements:
            raise ValueError(
                "leg layout must contain at least one leg; an empty layout would "
                "silently discard the correction"
            )
        names = [leg.name for leg in placements]
        if len(set(names)) != len(names):
            raise ValueError(
                f"leg layout has duplicate names {sorted(set(n for n in names if names.count(n) > 1))}; "
                "offsets are keyed by name, so a duplicate would overwrite a real leg"
            )

        raw = [
            -self.pitch_cmd * leg.x + self.roll_cmd * leg.y for leg in placements
        ]
        mean = sum(raw) / len(raw)
        return {leg.name: value - mean for leg, value in zip(placements, raw)}


@dataclass
class AttitudeStabilizer:
    """PD balance controller on body roll and pitch.

    Stateful across calls: it remembers the previous attitude so it can
    estimate rates by finite difference when no gyro is available.  One
    instance drives one body; :meth:`reset` clears the history between trials.
    """

    kp: float = DEFAULT_KP
    kd: float = DEFAULT_KD
    max_cmd: float = DEFAULT_MAX_CMD
    _prev_roll: float | None = field(default=None, init=False, repr=False)
    _prev_pitch: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.kp < 0.0 or self.kd < 0.0:
            raise ValueError(
                f"gains must be non-negative (got kp={self.kp}, kd={self.kd}); a "
                "negative gain turns this into positive feedback that accelerates "
                "the fall it is meant to arrest"
            )
        if self.max_cmd <= 0.0:
            raise ValueError(f"max_cmd must be positive, got {self.max_cmd}")

    def reset(self) -> None:
        """Forget the previous sample, so the next call has no rate estimate.

        Call between trials.  Carrying a stale attitude across a teleport back
        to the start pose would manufacture an enormous phantom rate on the
        first step of the new run.
        """
        self._prev_roll = None
        self._prev_pitch = None

    def update(
        self,
        quat_wxyz: Sequence[float],
        dt: float,
        *,
        roll_rate_dps: float | None = None,
        pitch_rate_dps: float | None = None,
    ) -> AttitudeCorrection:
        """Run one control cycle against a measured body orientation.

        ``dt`` is the interval since the previous call, in seconds.  Supply
        ``roll_rate_dps`` / ``pitch_rate_dps`` when a gyro is available; a real
        rate measurement beats differencing a noisy pose read, and on the first
        call it is the only way to have a derivative term at all.
        """
        if dt <= 0.0:
            raise ValueError(
                f"dt must be positive, got {dt}; a non-positive step would make the "
                "rate estimate meaningless or divide by zero"
            )
        roll, pitch = roll_pitch_deg(quat_wxyz)

        # No prior sample means no derivative.  Reporting a rate of zero here is
        # correct rather than merely convenient: the alternative, differencing
        # against an assumed-level start, invents a large rate out of the body's
        # initial tilt and kicks the very first command hard.
        if roll_rate_dps is None:
            roll_rate_dps = (
                0.0 if self._prev_roll is None else (roll - self._prev_roll) / dt
            )
        if pitch_rate_dps is None:
            pitch_rate_dps = (
                0.0 if self._prev_pitch is None else (pitch - self._prev_pitch) / dt
            )

        self._prev_roll, self._prev_pitch = roll, pitch

        # The setpoint is level, so error = -measurement, and the command is
        # simply the negated PD sum.  Derivative is on the measurement, not the
        # error, which matters the moment a caller adds a non-zero setpoint.
        #
        # The law runs in RADIANS even though the reported angles are degrees.
        # That is a dimensional requirement, not a preference: the command gets
        # multiplied by a lever arm in metres to produce foot travel in metres,
        # and only radians make that product a small-angle arc length. Degrees
        # would also make every gain 57x hotter than it reads and saturate the
        # clamp at one degree of tilt.
        roll_cmd = self._clamp(
            -(self.kp * math.radians(roll)
              + self.kd * math.radians(roll_rate_dps))
        )
        pitch_cmd = self._clamp(
            -(self.kp * math.radians(pitch)
              + self.kd * math.radians(pitch_rate_dps))
        )

        return AttitudeCorrection(
            roll_deg=roll,
            pitch_deg=pitch,
            roll_rate_dps=roll_rate_dps,
            pitch_rate_dps=pitch_rate_dps,
            roll_cmd=roll_cmd,
            pitch_cmd=pitch_cmd,
        )

    def _clamp(self, value: float) -> float:
        return max(-self.max_cmd, min(self.max_cmd, value))
