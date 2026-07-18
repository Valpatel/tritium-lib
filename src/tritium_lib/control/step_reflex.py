# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Capture-point stepping reflex — MEASURED: unfit to gate a walking gait.

    =====================================================================
    MEASURED VERDICT (live Newton, 2026-07-17/18): a velocity residual
    cannot separate a push from the gait on this body.  DO NOT WIRE
    StepReflex TO A WALKING BODY.  It remains coherent for a STANDING
    body, where the deviation gate degenerates to the absolute capture
    point correctly.  A walking body needs a different trigger entirely —
    contact/force sensing, or model-predicted vs actual state error.
    Numbers, derivation, and three caveats below.
    =====================================================================

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

Round 1 of what live measurement disproved: the ABSOLUTE gate
-------------------------------------------------------------
The first shipped version of this module gated on the capture point of the
body's **total** CoM velocity, with a default gate of 0.05 m, and its
docstring claimed an undisturbed walk never crosses the gate.  Live Newton
measurement (2026-07-17, matched back-to-back trials, **no push at all**)
proved that claim false and the design wrong:

* trim-only: **6/6 upright**.  trim + reflex: **0/6 upright** — median tilt
  179.97 degrees, flat on its back.  Fisher p ~ 0.002.  The reflex destroyed
  a working gait with no disturbance present.
* Root cause, measured: a healthy 1.2 m/s trot's total-velocity capture
  point peaks at **0.101–0.131 m** every stride — more than double the
  0.05 m gate.  ``v * sqrt(z/g)`` cannot distinguish "moving because
  walking" from "moving because pushed"; the gate was open essentially
  every stride and the reflex fought the gait continuously.
* Causal isolation: re-running undisturbed with the gate raised to 0.35 m
  opened it **0 ticks** and restored **6/6 upright**.  The actuation path is
  inert when the gate is shut; THE GATE ALONE caused the regression.
* No fixed absolute threshold fixes it: a 5 N·s push adds only
  **0.058–0.087 m** of capture-point excursion, while the walking carrier
  varies stride-to-stride by MORE than that — the disturbance is smaller
  than the carrier's own noise.  At an absolute gate of 0.28 m the reflex
  fired in only 2 of 8 push trials and both of those tumbled anyway
  (reverse causation: a body already falling is what raises the capture
  point that far).

Round 2: the CORRECTED deviation gate is disproven too
------------------------------------------------------
This version gates on the capture point of the **deviation** from the gait's
commanded velocity — ``measured − nominal`` — not on absolute velocity.  The
theory: a push moves the measured velocity while leaving the commanded
velocity untouched, so in deviation space the push appears at full size
while the walking carrier subtracts out as common mode.  The caller MUST
state what the gait commanded (``nominal_vel_xy``, required, no default).

Live physics then disproved the theory as well (Newton, 2026-07-17/18):

* With the **legal nominal** — the commanded 1.2 m/s, the only value a real
  caller has — the deviation capture point exceeded the 0.04 m gate on
  **100.0% of undisturbed walking ticks** (median 0.170 m, n=2719;
  corroborated on an independent run, n=1336).  Even the best-case nominal
  — the realised MEAN velocity, which no caller knows ahead of time —
  still exceeded the gate on **37.7%** of ticks.
* Live A/B, matched trials: baseline **6/6 upright** (median tilt
  9.22 deg) vs reflex **0/5 upright** (median tilt 179.98 deg), Fisher
  p = 0.0022.  The gate was open on **2350/2350 pooled ticks**; the gate
  signal's MINIMUM across a 470-tick window was 0.0996 m — 2.5x the
  threshold.  The gate never closed once.
* The separability floor, measured: staying quiet on 98% of undisturbed
  walking ticks needs a gate of **0.264 m** (0.098 m in the best-nominal
  case), against a weakest measured push signature of **0.058 m** — a
  floor **4.6x** (best case **1.7x**) LARGER than the signal it exists to
  catch.  Sweeping the nominal across 0.15–1.2 m/s never brings the
  floor/signal ratio under 1.7x, and CoM height cancels out of the ratio,
  so no ride-height retune can fix it either.

The mechanism: the redesign assumed the walking carrier is common mode.
Measured, it is not — this gait tracks its command loosely (caveat 3
below), so ``measured − commanded`` is dominated by the gait's own
tracking error, which dwarfs any push worth catching.  Retuning
``threshold_m`` is not a fix; it would be the third disproven variation of
the same idea.

Three caveats, so the verdict is not overstated either
------------------------------------------------------
1. **No visual evidence** exists for the corrected-gate run — the 0/5 is
   numeric telemetry (tilt + gate traces) only, never a watched render.
2. **The gate-shut control arm never ran for the corrected build.**  A
   trim-only arm with identical logging did run 6/6, which exonerates the
   instrumentation, but "corrected reflex with the gate forced shut" was
   never measured.
3. **The push signature may need re-measurement**: the 0.058–0.087 m band
   was measured on a gait realising only ~18% of its commanded speed, so
   the separability ratio above may rest on a signal floor that moves if
   the gait ever tracks its command tightly.

Under caveats 2–3 the honest claim is "disproven as shipped, on this body,
at this gait quality" — not "capture-point stepping can never work".  What
IS closed: on THIS gait, no threshold on THIS signal separates push from
walk.

Why the module survives at all
------------------------------
The LIP math below (:func:`capture_point`, :func:`step_target`) is
correct, closed-form, and reusable — deleting it would only invite an
unmeasured rewrite.  And for a STANDING body the design is coherent: the
honest nominal is ``(0, 0)``, the deviation gate degenerates exactly to
the absolute capture point, and there is no gait carrier to alias.  That
standing regime — plus telemetry/analysis reuse of the math — are the only
supported uses.

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

When the deviation gate opens, the emitted step still targets the capture
point of the **total** velocity: placing a foot there brings the whole body
to rest under the LIP model — an arrest-to-stand emergency policy.  Stop
walking, don't fall, let the driver restart the gait.  A gentler policy
exists (shift the gait's *planned* footfall by the deviation capture point,
preserving the walk — LIP dynamics are linear, so the error obeys the same
equation) but it requires the planned footfall as an input this module does
not have, and it is not shipped and not validated.  Note what the live data
does and does not establish: the gate's inertness when shut was causally
isolated (0.35 m control, above); the arrest step *helping* under a real
push has never been observed on live Newton.

Layering contract (read this before wiring it in)
-------------------------------------------------
The reflex is an **optional, additive layer** over the existing trim:

* it never imports, wraps, or alters :class:`AttitudeStabilizer`;
* it is **gated**: below :attr:`StepReflex.threshold_m` of
  deviation-capture-point excursion it decides "no step" and passes any trim
  offsets through untouched (the same object, not a copy — pinned by test);
* it emits a *decision* (which leg, what landing point), never joint
  commands.  The driver owns swing timing: a real gait can only lift a leg
  its phase allows, so the driver applies the decision at the next
  compatible swing slot and keeps re-reading it each tick — the reflex is
  stateless and recomputes from the current velocities every call.

How a driver gates it — for a STANDING body only: each control tick, read
root horizontal velocity from the solver (or, for a scored A/B kick, the
projected dv that :func:`tritium_lib.control.disturbance.kick_landed`
already computes), pass ``nominal_vel_xy=(0.0, 0.0)`` — honest for a body
commanded to stand still — call :meth:`StepReflex.decide` with the
stance-foot layout, and act only when ``decision.step`` is not ``None``.
For a WALKING body: do not wire this at all (measured verdict, top of this
docstring) — supplying the honest commanded velocity as the nominal was
measured NOT to rescue the gate (round 2), and feeding ``(0, 0)`` under a
walking gait re-creates the round-1 absolute failure by construction.  The
nominal stays REQUIRED with no default so a caller that cannot state it
gets a ``TypeError`` at the call site, not a silent fallback into either
measured failure.

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
    "DEFAULT_DEVIATION_THRESHOLD_M",
    "GRAVITY_MPS2",
    "ReachLimits",
    "ReflexDecision",
    "StepDecision",
    "StepReflex",
    "capture_point",
    "step_target",
    "velocity_deviation",
    "velocity_from_impulse",
]

# Standard gravity.  Numerically identical to models.body.G_MPS2, restated
# here so the control package keeps its stdlib-only import guarantee.
GRAVITY_MPS2: float = 9.80665

# Deviation-capture-point excursion below which no step fires, in metres of
# capture point computed from (measured − nominal) velocity.
#
# Sized for the STANDING regime — the only supported one (module docstring):
# 0.04 m sits ~30% under the weakest measured push signature (0.058 m, from
# the 5 N·s trim-ceiling push), so a standing body steps *before* the
# trim-only stack's inversion ceiling.  At Go2 ride height (~0.31 m, time
# constant ~0.178 s) it corresponds to a sustained velocity error of
# ~0.22 m/s.
#
# For a WALKING body there is NO valid value of this constant — that is a
# measurement, not caution.  The prior revision of this comment said the
# walking residual in deviation space "has NOT been measured" and capped the
# honest retuning headroom at ~0.058 m, past which "the velocity-residual
# signal itself cannot separate push from gait".  The measurement came back
# (live Newton, 2026-07-17/18, module docstring round 2) an order of
# magnitude past that cap: median residual 0.170 m against this 0.04 m gate
# (open on 100.0% of walking ticks, n=2719), with 0.264 m (0.098 m best
# case) needed to stay quiet on 98% of ticks — 4.6x/1.7x the push signal it
# must catch.  The escape clause triggered: do not retune this for walking;
# the trigger signal itself is wrong there.
DEFAULT_DEVIATION_THRESHOLD_M: float = 0.04

Vec2 = tuple[float, float]


def capture_point(
    vel_xy: Sequence[float],
    com_height: float,
    g: float = GRAVITY_MPS2,
) -> Vec2:
    """Capture point offset from the CoM ground projection, in metres.

    ``vel_xy`` is a horizontal velocity in the body frame (m/s),
    ``com_height`` the CoM height above flat ground (m).  Returns
    ``v * sqrt(com_height / g)`` — the point a foot must reach to bring the
    linear inverted pendulum to rest (see module docstring for what that
    model ignores).  Zero velocity maps to ``(0, 0)``: the body is already
    captured and no step is needed.

    This is a pure kinematic map and carries no gating judgement.  Feed it a
    TOTAL velocity and you get the arrest point; feed it a DEVIATION
    (:func:`velocity_deviation`) and you get the gate signal.  The distinction
    is the whole content of the 2026-07-17 live failure — see the module
    docstring.
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


def velocity_deviation(
    measured_vel_xy: Sequence[float],
    nominal_vel_xy: Sequence[float],
) -> Vec2:
    """``measured − nominal`` horizontal velocity, componentwise (m/s).

    The disturbance residual the gate acts on: what the body is actually
    doing minus what the gait commanded.  Direction is preserved — a body
    shoved left of its commanded track deviates ``+Y``; a body blocked or
    tripped below its commanded speed deviates ``−X`` (falling behind the
    gait is also a disturbance, and the capture point of that deviation
    points backward, where the recovery foot belongs).

    The nominal must be the caller's *statement* of the commanded gait
    velocity — never an estimate inferred from the same measurement stream,
    which would subtract the disturbance out of its own detector.

    Measured caution (live Newton, 2026-07-17/18): for a real walking gait
    this residual does NOT vanish — the gait tracks its command loosely, so
    the residual's capture point ran a median 0.170 m at the legal nominal,
    dwarfing the 0.058 m push signature it was meant to expose.  The math
    here is exact; it is the *premise* that the residual is small during a
    healthy walk that measured false.  See the module docstring, round 2.
    """
    if len(measured_vel_xy) != 2:
        raise ValueError(
            "measured_vel_xy must have 2 components (vx, vy), got "
            f"{len(measured_vel_xy)}"
        )
    if len(nominal_vel_xy) != 2:
        raise ValueError(
            "nominal_vel_xy must have 2 components (vx, vy), got "
            f"{len(nominal_vel_xy)}"
        )
    return (
        float(measured_vel_xy[0]) - float(nominal_vel_xy[0]),
        float(measured_vel_xy[1]) - float(nominal_vel_xy[1]),
    )


def velocity_from_impulse(
    impulse_xy: Sequence[float],
    body_mass: float,
) -> Vec2:
    """CoM velocity change (m/s) from a horizontal push impulse (N·s).

    Plain ``J / m`` — the velocity an unopposed push imparts.  A real push
    into real foot contacts delivers only part of this (measured ~half; see
    :func:`tritium_lib.control.disturbance.kick_landed`), so when a measured
    velocity is available, prefer it — this helper is for sizing an expected
    response to a *commanded* kick.  Because a push changes the measured
    velocity and not the nominal, this delta IS a deviation and can be
    compared against the gate directly.
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
    trim.  ``capture_pt`` here is the TOTAL-velocity capture point (the
    arrest target), not the deviation gate signal.  Returns
    ``(leg_name, (x, y))``.
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
    """One tick's reflex output: gate signal, gate verdict, optional step.

    ``deviation_distance_m`` — the magnitude of the deviation capture point
    — is the gated quantity; ``capture_pt`` is the TOTAL-velocity capture
    point the arrest step targets when the gate is open.  Both are reported
    every tick so telemetry can show what the gate saw and what it would
    have seen under the disproven absolute design.

    ``leg_height_offsets`` is the trim dict passed through **untouched** —
    the very object the caller handed in, never copied, never modified.
    Below the gate ``step`` is ``None`` and this decision changes nothing
    about the tick; that pass-through-identity is the layering contract and
    is pinned by test.
    """

    capture_pt: Vec2
    deviation_vel_xy: Vec2
    deviation_capture_pt: Vec2
    deviation_distance_m: float
    threshold_m: float
    step: StepDecision | None
    leg_height_offsets: Mapping[str, float] | None = None

    @property
    def stepping(self) -> bool:
        return self.step is not None

    def as_dict(self) -> dict:
        return {
            "capture_pt": list(self.capture_pt),
            "deviation_vel_xy": list(self.deviation_vel_xy),
            "deviation_capture_pt": list(self.deviation_capture_pt),
            "deviation_distance_m": self.deviation_distance_m,
            "threshold_m": self.threshold_m,
            "step": None if self.step is None else self.step.as_dict(),
            "leg_height_offsets": (
                None if self.leg_height_offsets is None
                else dict(self.leg_height_offsets)
            ),
        }


@dataclass(frozen=True)
class StepReflex:
    """Gated capture-point stepping reflex — STANDING bodies only (measured).

    MEASURED VERDICT (live Newton, 2026-07-17/18): do not wire this to a
    WALKING body.  With the legal nominal the deviation gate was open on
    100.0% of undisturbed walking ticks (median 0.170 m vs the 0.04 m
    gate), and the live A/B measured baseline 6/6 upright vs reflex 0/5
    (Fisher p = 0.0022, gate open 2350/2350 pooled ticks).  A velocity
    residual cannot separate a push from the gait on this body; walking
    push-recovery needs a different trigger (contact/force, or
    model-predicted vs actual state error).  The supported regime is a
    STANDING body, where the deviation degenerates to the absolute capture
    point correctly.  Full numbers and three caveats: module docstring.

    Frozen and stateless: configuration in, decision out, nothing remembered
    between ticks.  Statelessness is what makes the layering safe — the
    reflex cannot drift, wind up, or disagree with itself across a reset,
    and two arms of an A/B fed the same measurements get the same decisions.

    :param com_height_m: CoM height over flat ground, metres.  A required
        parameter rather than a default because baking one body's ride
        height into lib would quietly mis-scale every other body.
    :param threshold_m: DEVIATION-capture-point excursion above which a step
        fires (strictly above; at or below the gate stays closed).  Compared
        against ``|capture_point(measured − nominal)|``, never against the
        absolute capture point — the absolute form measured 0/6 upright on
        an undisturbed walk (module docstring, round 1).  Default:
        :data:`DEFAULT_DEVIATION_THRESHOLD_M`, sized for the STANDING
        regime; no walking value exists for it (measured — round 2), so
        retuning it for a walking body is not a supported escape.
    :param g_mps2: gravity, override for non-Earth or scaled-physics sims.
    """

    com_height_m: float
    threshold_m: float = DEFAULT_DEVIATION_THRESHOLD_M
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
        measured_vel_xy: Sequence[float],
        stance_feet: Iterable[LegPlacement],
        *,
        nominal_vel_xy: Sequence[float],
        reach_limits: ReachLimits,
        leg_height_offsets: Mapping[str, float] | None = None,
    ) -> ReflexDecision:
        """Run the gate against the deviation from the commanded velocity.

        Call this for a STANDING body only.  For a walking body no call is
        correct (measured, module docstring): passing the honest commanded
        velocity as the nominal left the gate open on 100.0% of undisturbed
        walking ticks and measured 0/5 upright against a 6/6 baseline
        (round 2), while passing ``(0, 0)`` under a walking gait re-creates
        the round-1 absolute failure by construction.

        ``measured_vel_xy`` is the body's horizontal CoM velocity (m/s, body
        frame) — from the solver's root velocity or an estimator.
        ``nominal_vel_xy`` is REQUIRED and keyword-only: the horizontal
        velocity the gait is currently commanding, stated by the caller —
        never inferred here.  There is no default because the only universal
        default is ``(0, 0)``, which is honest exactly when the gait
        genuinely commands standing still — the supported regime.

        The gate compares ``|capture_point(measured − nominal)|`` against
        :attr:`threshold_m`.  Below it, the decision carries ``step=None``
        and the tick proceeds exactly as if this layer did not exist —
        ``leg_height_offsets`` rides through as the identical object.  Above
        it, the decision names the stepping leg and its landing point,
        targeted at the TOTAL-velocity capture point (arrest to a stand; see
        the module docstring for why, and note the arrest step has never
        been observed *helping* on live Newton); the driver applies it at
        the next swing slot its gait allows.
        """
        if nominal_vel_xy is None:
            raise ValueError(
                "nominal_vel_xy is None; the reflex needs the caller's "
                "statement of the gait's commanded velocity to separate "
                "push from walk.  For a body commanded to stand still, the "
                "honest nominal is (0.0, 0.0) — passing that for a WALKING "
                "gait re-creates the measured 0/6 absolute-gate failure"
            )
        deviation = velocity_deviation(measured_vel_xy, nominal_vel_xy)
        dev_cp = capture_point(deviation, self.com_height_m, self.g_mps2)
        dev_distance = math.hypot(dev_cp[0], dev_cp[1])
        total_cp = capture_point(
            measured_vel_xy, self.com_height_m, self.g_mps2,
        )
        step: StepDecision | None = None
        if dev_distance > self.threshold_m:
            leg, target = step_target(
                stance_feet, total_cp, reach_limits=reach_limits,
            )
            step = StepDecision(
                leg=leg,
                foot_target=target,
                residual_m=math.hypot(
                    target[0] - total_cp[0], target[1] - total_cp[1],
                ),
            )
        return ReflexDecision(
            capture_pt=total_cp,
            deviation_vel_xy=deviation,
            deviation_capture_pt=dev_cp,
            deviation_distance_m=dev_distance,
            threshold_m=self.threshold_m,
            step=step,
            leg_height_offsets=leg_height_offsets,
        )
