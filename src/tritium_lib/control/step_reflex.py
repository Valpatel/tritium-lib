# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Capture-point stepping reflex — where to put a foot to stop a fall.

Why this exists
---------------
:mod:`tritium_lib.control.attitude_stabilizer` closed the loop on body
attitude and, measured live on Newton (2026-07-18), it holds the gait 100%
upright undisturbed — 34 of 34 trials.  But a push of more than about 5 N·s
still inverts the robot, because a foot-height trim is an *ankle strategy*: it
redistributes weight across feet that are already planted.  Once the body's
momentum carries its capture point outside the support polygon, no amount of
leaning re-captures it — the only recovery is to MOVE A FOOT under the fall.
That stepping reflex is what this module computes.  The trim answers "how hard
does each planted foot push"; this answers "which foot must leave the ground,
and where must it land".

The physics, stated honestly
----------------------------
Everything here is the **Linear Inverted Pendulum** model (Kajita) and its
**capture point** (Pratt 2006; the instantaneous divergent component of
motion of Takenaka/Englsberger):

* the body is a point mass at constant height ``z0`` over **flat ground**;
* legs are massless and a step is instantaneous — swing time, swing-leg
  dynamics and the momentum cost of moving the leg are all ignored;
* angular momentum about the CoM is ignored (no flywheel/torso term);
* under those assumptions the CoM diverges from an equilibrium with time
  constant ``sqrt(z0/g)``, and a point foot placed AT

      capture_pt = v_xy * sqrt(z0 / g)

  (an offset from the CoM's ground projection, in the direction of travel)
  brings the pendulum asymptotically to rest above that foot.  A foot short
  of it leaves residual divergence; a foot past it reverses the fall.

These assumptions are exactly wrong in all the usual ways — real swing takes
~150 ms, real ground has friction cones, a real Go2 torso carries angular
momentum — which is why the capture point is used as a *target*, not a
guarantee, and why the residual after reach-clamping is reported rather than
hidden.  **This module is validated against its own closed-form math only; it
has NOT yet been tested on live Newton.**  The 5 N·s figure above is the
measured failure threshold of the trim-only stack, i.e. the regime this
reflex exists to enter.

Layering contract (read this before wiring it in)
-------------------------------------------------
The reflex is an **optional, additive layer** over the existing trim:

* it never imports, wraps, or alters :class:`AttitudeStabilizer` — the
  undisturbed behavior that measures 100% stays byte-identical;
* it is **gated**: below :attr:`StepReflex.threshold_m` of capture-point
  excursion it decides "no step" and passes any trim offsets through
  untouched (the same object, not a copy — pinned by test);
* it emits a *decision* (which leg, what landing point), never joint
  commands.  The driver owns swing timing: a real gait can only lift a leg
  its phase allows, so the driver applies the decision at the next
  compatible swing slot and keeps re-reading it each tick — the reflex is
  stateless and recomputes from the current velocity every call.

How an Isaac/Newton driver gates it: each control tick, read root horizontal
velocity from the solver (or, for a scored A/B kick, the projected dv that
:func:`tritium_lib.control.disturbance.kick_landed` already computes),
call :meth:`StepReflex.decide` with the stance-foot layout, and act only when
``decision.step`` is not ``None``.  An undisturbed walk never crosses the
gate, so the shipped gait is untouched by construction.

Conventions match the rest of :mod:`tritium_lib.control`: REP-103 body frame
(+X forward, +Y left, +Z up), SI units, degrees only in logs.  Stdlib only —
no numpy — so this imports on a bare Jetson.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from tritium_lib.control.attitude_stabilizer import LegPlacement

__all__ = [
    "DEFAULT_CAPTURE_THRESHOLD_M",
    "GRAVITY_MPS2",
    "ReachLimits",
    "ReflexDecision",
    "StepDecision",
    "StepReflex",
    "capture_point",
    "step_target",
    "velocity_from_impulse",
]

# Standard gravity.  Numerically identical to models.body.G_MPS2, restated
# here so the control package keeps its stdlib-only import guarantee.
GRAVITY_MPS2: float = 9.80665

# Capture-point excursion below which no step fires.  Calibrated against the
# live-Newton measurement that motivated this module: the trim-only stack
# recovers pushes up to ~5 N·s, which on a Go2-class body (~12 kg, ~0.3 m ride
# height) is ~0.42 m/s of CoM velocity and therefore ~0.07 m of capture-point
# excursion.  Gating at 0.05 m keeps the reflex silent through the regime the
# trim already handles, with margin *below* the measured inversion threshold
# rather than at it — a reflex that waits for the trim's exact ceiling arrives
# as the fall becomes unrecoverable.
DEFAULT_CAPTURE_THRESHOLD_M: float = 0.05

Vec2 = tuple[float, float]


def capture_point(
    vel_xy: Sequence[float],
    com_height: float,
    g: float = GRAVITY_MPS2,
) -> Vec2:
    """Capture point offset from the CoM ground projection, in metres.

    ``vel_xy`` is the body's horizontal velocity in the body frame (m/s),
    ``com_height`` the CoM height above flat ground (m).  Returns
    ``v * sqrt(com_height / g)`` — the point a foot must reach to bring the
    linear inverted pendulum to rest (see module docstring for what that
    model ignores).  Zero velocity maps to ``(0, 0)``: the body is already
    captured and no step is needed.
    """
    if len(vel_xy) != 2:
        raise ValueError(
            f"vel_xy must have 2 components (vx, vy), got {len(vel_xy)}"
        )
    if com_height <= 0.0:
        raise ValueError(
            f"com_height must be positive, got {com_height}; a pendulum of "
            "non-positive length has no dynamics to capture"
        )
    if g <= 0.0:
        raise ValueError(f"g must be positive, got {g}")
    tc = math.sqrt(com_height / g)
    return (float(vel_xy[0]) * tc, float(vel_xy[1]) * tc)


def velocity_from_impulse(
    impulse_xy: Sequence[float],
    body_mass: float,
) -> Vec2:
    """CoM velocity change (m/s) from a horizontal push impulse (N·s).

    Plain ``J / m`` — the velocity an unopposed push imparts.  A real push
    into real foot contacts delivers only part of this (measured ~half; see
    :func:`tritium_lib.control.disturbance.kick_landed`), so when a measured
    velocity is available, prefer it — this helper is for sizing an expected
    response to a *commanded* kick.
    """
    if len(impulse_xy) != 2:
        raise ValueError(
            f"impulse_xy must have 2 components (Jx, Jy), got {len(impulse_xy)}"
        )
    if body_mass <= 0.0:
        raise ValueError(f"body_mass must be positive, got {body_mass}")
    return (float(impulse_xy[0]) / body_mass, float(impulse_xy[1]) / body_mass)


@dataclass(frozen=True)
class ReachLimits:
    """How far a foot may land from its neutral placement, per axis (m).

    A rectangular clamp about each leg's home position: the commanded landing
    point may deviate at most ``max_dx`` forward/back and ``max_dy``
    left/right from where that leg normally stands.  Rectangular rather than
    radial because a quadruped's reach genuinely differs by axis — the thigh
    sweeps far in X, the hip ab/adducts little in Y — and because a
    closed-form clamp is testable without a kinematics library.  The numbers
    are the caller's statement about its body; nothing here checks them
    against a real leg.
    """

    max_dx: float
    max_dy: float

    def __post_init__(self) -> None:
        if self.max_dx <= 0.0 or self.max_dy <= 0.0:
            raise ValueError(
                f"reach limits must be positive (got max_dx={self.max_dx}, "
                f"max_dy={self.max_dy}); a zero limit means the leg cannot "
                "step at all, and a reflex that silently cannot step should "
                "be configured off, not configured impossible"
            )


def step_target(
    stance_feet: Iterable[LegPlacement],
    capture_pt: Sequence[float],
    *,
    reach_limits: ReachLimits,
) -> tuple[str, Vec2]:
    """Which leg should step, and where it should land.

    Each candidate leg's landing point is the capture point clamped into that
    leg's reach rectangle (:class:`ReachLimits` about its home placement).
    The chosen leg is the one whose clamped landing point comes CLOSEST to
    the capture point — i.e. the leg that can actually arrest the fall, or
    failing full arrest, leave the least residual divergence.  For a lateral
    push that is naturally a leg on the push side: its reach rectangle is
    nearer the capture point, so its residual is smaller.

    Ties (an exactly symmetric pair straddling the capture point) resolve to
    the FIRST such leg in input order — strict-less-than comparison, so the
    result is deterministic and owned by the caller's leg ordering, not by
    dict or hash order.

    Coordinates are body-frame metres with the origin at the CoM ground
    projection, the same frame :class:`LegPlacement` already uses for the
    trim.  Returns ``(leg_name, (x, y))``.
    """
    feet = tuple(stance_feet)
    if not feet:
        raise ValueError(
            "stance_feet must contain at least one leg; with no foot "
            "available to move there is no step decision to make"
        )
    names = [leg.name for leg in feet]
    if len(set(names)) != len(names):
        raise ValueError(
            f"stance_feet has duplicate names "
            f"{sorted(set(n for n in names if names.count(n) > 1))}; the "
            "decision is reported by name, so a duplicate would be ambiguous"
        )
    if len(capture_pt) != 2:
        raise ValueError(
            f"capture_pt must have 2 components (x, y), got {len(capture_pt)}"
        )
    cx, cy = float(capture_pt[0]), float(capture_pt[1])

    best_residual = math.inf
    best_leg = feet[0]
    best_target: Vec2 = (feet[0].x, feet[0].y)
    for leg in feet:
        dx = max(-reach_limits.max_dx, min(reach_limits.max_dx, cx - leg.x))
        dy = max(-reach_limits.max_dy, min(reach_limits.max_dy, cy - leg.y))
        target = (leg.x + dx, leg.y + dy)
        residual = math.hypot(target[0] - cx, target[1] - cy)
        # Strict < keeps the earliest leg on an exact tie (see docstring).
        if residual < best_residual:
            best_residual = residual
            best_leg = leg
            best_target = target
    return best_leg.name, best_target


@dataclass(frozen=True)
class StepDecision:
    """One commanded recovery step.

    :param leg: name of the leg that should step (a key from the caller's
        :class:`LegPlacement` layout).
    :param foot_target: body-frame ``(x, y)`` landing point in metres,
        already clamped to the leg's reach.
    :param residual_m: distance from the clamped landing point to the
        capture point.  Zero means the step fully captures the fall under
        the LIP model; positive means the leg cannot reach far enough and
        divergence will remain — the driver should know that rather than
        discover it.
    """

    leg: str
    foot_target: Vec2
    residual_m: float

    def as_dict(self) -> dict:
        return {
            "leg": self.leg,
            "foot_target": list(self.foot_target),
            "residual_m": self.residual_m,
        }


@dataclass(frozen=True)
class ReflexDecision:
    """One tick's reflex output: capture state, gate verdict, optional step.

    ``leg_height_offsets`` is the trim dict passed through **untouched** —
    the very object the caller handed in, never copied, never modified.
    Below the gate ``step`` is ``None`` and this decision changes nothing
    about the tick; that pass-through-identity is the layering contract and
    is pinned by test.
    """

    capture_pt: Vec2
    capture_distance_m: float
    threshold_m: float
    step: StepDecision | None
    leg_height_offsets: Mapping[str, float] | None = None

    @property
    def stepping(self) -> bool:
        return self.step is not None

    def as_dict(self) -> dict:
        return {
            "capture_pt": list(self.capture_pt),
            "capture_distance_m": self.capture_distance_m,
            "threshold_m": self.threshold_m,
            "step": None if self.step is None else self.step.as_dict(),
            "leg_height_offsets": (
                None if self.leg_height_offsets is None
                else dict(self.leg_height_offsets)
            ),
        }


@dataclass(frozen=True)
class StepReflex:
    """Gated capture-point stepping reflex for one body.

    Frozen and stateless: configuration in, decision out, nothing remembered
    between ticks.  Statelessness is what makes the layering safe — the
    reflex cannot drift, wind up, or disagree with itself across a reset,
    and two arms of an A/B fed the same measurements get the same decisions.

    :param com_height_m: CoM height over flat ground, metres.  A required
        parameter rather than a default because baking one body's ride
        height into lib would quietly mis-scale every other body.
    :param threshold_m: capture-point excursion above which a step fires
        (strictly above; at or below the gate stays closed).  The default is
        calibrated against the measured trim-only envelope — see
        :data:`DEFAULT_CAPTURE_THRESHOLD_M`.
    :param g_mps2: gravity, override for non-Earth or scaled-physics sims.
    """

    com_height_m: float
    threshold_m: float = DEFAULT_CAPTURE_THRESHOLD_M
    g_mps2: float = GRAVITY_MPS2

    def __post_init__(self) -> None:
        if self.com_height_m <= 0.0:
            raise ValueError(
                f"com_height_m must be positive, got {self.com_height_m}"
            )
        if self.threshold_m < 0.0:
            raise ValueError(
                f"threshold_m must be >= 0, got {self.threshold_m}; to "
                "disable the reflex, do not call it"
            )
        if self.g_mps2 <= 0.0:
            raise ValueError(f"g_mps2 must be positive, got {self.g_mps2}")

    def decide(
        self,
        vel_xy: Sequence[float],
        stance_feet: Iterable[LegPlacement],
        *,
        reach_limits: ReachLimits,
        leg_height_offsets: Mapping[str, float] | None = None,
    ) -> ReflexDecision:
        """Run the gate against a measured horizontal velocity.

        ``vel_xy`` is the body's horizontal CoM velocity (m/s, body frame) —
        from the solver's root velocity, an estimator, or
        :func:`velocity_from_impulse` when only the commanded kick is known.
        ``leg_height_offsets`` is optional and opaque: whatever trim dict the
        stabilizer produced rides through unchanged so the driver handles one
        object, not two.

        Below the gate the decision carries ``step=None`` and the tick
        proceeds exactly as if this layer did not exist.  Above it, the
        decision names the stepping leg and its clamped landing point; the
        driver applies it at the next swing slot its gait allows.
        """
        cp = capture_point(vel_xy, self.com_height_m, self.g_mps2)
        distance = math.hypot(cp[0], cp[1])
        step: StepDecision | None = None
        if distance > self.threshold_m:
            leg, target = step_target(
                stance_feet, cp, reach_limits=reach_limits,
            )
            step = StepDecision(
                leg=leg,
                foot_target=target,
                residual_m=math.hypot(target[0] - cp[0], target[1] - cp[1]),
            )
        return ReflexDecision(
            capture_pt=cp,
            capture_distance_m=distance,
            threshold_m=self.threshold_m,
            step=step,
            leg_height_offsets=leg_height_offsets,
        )
