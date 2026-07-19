# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Newton-native quadruped gait trajectory generator — pure kinematics.

Produces the time-parameterized 12-DOF joint-angle trajectory that makes a
Go2-class quadruped WALK under Isaac Newton physics.  Live Newton validation
showed the reliable actuation path is setting USD joint drive
``targetPositions`` per joint; this module is the lib-side (light-dep, pure
math) half that GENERATES those targets over a gait cycle — a separate GPU
Isaac driver applies them.  Per the copper-roof rule the kinematics live here
(reusable, stdlib/pydantic-only, Jetson-safe); nothing in this file imports
Isaac, USD, torch, or ROS.

Joint name scheme (12 revolute DOF, the exact dict keys returned)::

    {leg}_{joint}   for leg in FL, FR, RL, RR   (Front/Rear x Left/Right)
                    and joint in hip, thigh, calf

    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf"

All angles are RADIANS.  A USD driver converts to degrees for
``targetPositions`` with ``degrees = radians * 180 / pi``.

Conventions (validated stable STAND under Newton): hip 0 deg / thigh +50 deg /
calf -100 deg.  Positive thigh offset sweeps the leg rearward (stance
propulsion), negative reaches forward; a more negative calf angle flexes the
knee further, tucking the foot up.  Each leg runs the same swing/stance cycle
shifted by a per-leg phase offset — trot puts diagonal pairs anti-phase at
0.5, walk staggers the four feet at quarter-cycle intervals — so the body is
carried forward while every joint stays centered on the neutral stand within
bounded, physically plausible amplitudes.

Gait timing (speed, stride frequency) comes from
:data:`~tritium_lib.models.quadruped.DEFAULT_GAITS` — the single gait table
shared with the sim's body animation and the real dog's telemetry vocabulary.
This module never redefines those numbers.

Commanded vs realised speed — the ``speed`` contract (READ THIS)
----------------------------------------------------------------
``speed`` controls CADENCE (stride frequency), not ground speed, and the
trajectory it produces CANNOT reach the commanded ground speed by
construction.  Three facts, all verified live under Newton (2026-07):

1. **Stride length is speed-invariant — the terms cancel.**  Both
   ``stride_hz`` and the effective ``speed_mps`` scale by the same ratio
   ``speed / spec.speed_mps``, so the commanded stride length
   ``speed_mps / stride_hz`` collapses to the constant
   ``spec.speed_mps / spec.stride_hz`` at EVERY commanded speed
   (default trot: 1.6 / 2.6 = 0.6154 m).

2. **The amplitude clamp pins the realised stride.**  The thigh amplitude
   derived from that stride length (default trot: 0.769 rad) exceeds
   ``_THIGH_AMP_MAX`` (0.45 rad), so it saturates at every commanded trot
   speed and the achievable no-slip stride is pinned at
   ``2 * 0.45 * body_height_m`` = 0.36 m (default profile).

3. **Net: a hard no-slip ceiling of ~0.585 x commanded for default trot.**
   Even with perfect joint tracking and infinite friction the body can
   move at most ``achievable_stride * stride_hz`` = 0.585 x the commanded
   speed — verified live across 0.4-3.2 m/s with the clamp saturated at
   every point.  ``joint_targets_at(..., speed=1.2)`` can deliver at most
   ~0.702 m/s.  (Walk: 0.823x; bound: 0.384x; ratios are per-gait and
   per-profile — query them, below, instead of hardcoding.)

Live measurement (healthy kit): 48.4% of commanded realised in practice
(2.39x spread across trials), i.e. the sim tracks slightly under the 58.5%
theoretical ceiling — the ceiling explains ~80.5% of the measured shortfall.
**Do NOT "fix" this by raising ``_THIGH_AMP_MAX``**: a live sweep at x1.5
and x2.0 thigh amplitude made realised speed WORSE (medians 0.194 and
0.164 m/s vs 0.452 at x1.0) and tumbled the body (1/3 and 2/3 of trials
lost upright posture).  Amplitude, stiffness, damping, torque limit, and
friction all sit at or near their measured local optimum; the binding
constraint is the open-loop position-controlled trot itself.  Closing the
gap needs feedback control, not bigger constants — that module now exists:
:mod:`tritium_lib.control.gait_speed` (``GaitSpeedTracker`` — designed, NOT
yet live-validated; its docstring keeps the measured/designed ledger).

Callers that plan against ground speed must consult
:meth:`QuadrupedGaitCycle.no_slip_speed_mps` /
:func:`no_slip_speed_for` for the honest ceiling instead of assuming
``speed`` is realised.
"""

from __future__ import annotations

import math

from .quadruped import DEFAULT_GAITS, GaitSpec, QuadrupedProfile

# Leg order is fixed and canonical: Front-Left, Front-Right, Rear-Left,
# Rear-Right.  Joint order within a leg: hip (ab/adduction), thigh (hip
# pitch), calf (knee).
LEG_NAMES: tuple[str, ...] = ("FL", "FR", "RL", "RR")
JOINT_PARTS: tuple[str, ...] = ("hip", "thigh", "calf")
JOINT_NAMES: tuple[str, ...] = tuple(
    f"{leg}_{part}" for leg in LEG_NAMES for part in JOINT_PARTS
)

# The Newton-validated stable stand, in radians: hip 0 / thigh +50deg /
# calf -100deg.  Every trajectory this module emits is centered on this pose.
NEUTRAL_HIP_RAD: float = 0.0
NEUTRAL_THIGH_RAD: float = math.radians(50.0)
NEUTRAL_CALF_RAD: float = math.radians(-100.0)

NEUTRAL_STAND_RAD: dict[str, float] = {
    f"{leg}_{part}": angle
    for leg in LEG_NAMES
    for part, angle in (
        ("hip", NEUTRAL_HIP_RAD),
        ("thigh", NEUTRAL_THIGH_RAD),
        ("calf", NEUTRAL_CALF_RAD),
    )
}

# Per-leg phase offsets (fraction of the gait cycle) for each supported gait.
# trot:  diagonal pairs move together, the two diagonals anti-phase at 0.5.
# walk:  four-beat lateral-sequence walk — feet staggered at quarter cycles
#        in the classic FL -> RR -> FR -> RL footfall order.
# bound: front pair together, rear pair together, halves anti-phase.
GAIT_PHASE_OFFSETS: dict[str, dict[str, float]] = {
    "trot": {"FL": 0.0, "FR": 0.5, "RL": 0.5, "RR": 0.0},
    "walk": {"FL": 0.0, "FR": 0.5, "RL": 0.75, "RR": 0.25},
    "bound": {"FL": 0.0, "FR": 0.0, "RL": 0.5, "RR": 0.5},
}

# Duty factor = fraction of the cycle each foot spends in STANCE (on the
# ground pushing back).  Slower gaits keep more feet planted.
GAIT_DUTY_FACTOR: dict[str, float] = {
    "walk": 0.75,
    "trot": 0.55,
    "bound": 0.40,
}

# Bounded swing amplitudes (radians).  Chosen so every emitted angle stays
# well inside a Go2-class joint envelope while still being a visible,
# plausible step.  Thigh amplitude is derived from stride length below and
# clamped into [_THIGH_AMP_MIN, _THIGH_AMP_MAX].
_THIGH_AMP_MIN: float = 0.12
_THIGH_AMP_MAX: float = 0.45
_CALF_LIFT_RAD: float = 0.35  # extra knee tuck at mid-swing (foot clearance)
_HIP_AMP_MAX: float = 0.10  # lateral hip sway ceiling

# Plausible Go2-class joint envelopes (radians) — the generator guarantees
# every emitted angle stays inside these.  Exported so tests and drivers can
# assert against the same numbers.
JOINT_LIMITS_RAD: dict[str, tuple[float, float]] = {
    "hip": (-0.6, 0.6),
    "thigh": (0.0, 1.8),
    "calf": (-2.4, -1.2),
}


class QuadrupedGaitCycle:
    """One gait's 12-joint trajectory as a function of phase or time.

    Pure kinematics: given a phase in [0, 1) (or a wall-clock time ``t``,
    which is folded through the stride frequency), returns the 12 joint
    angles (radians) of a swing/stance gait cycle centered on the neutral
    stand.  In STANCE the thigh sweeps linearly rearward (foot planted,
    body carried forward) with the calf held at neutral; in SWING the thigh
    returns forward on a cosine profile while the calf tucks (sinusoidal
    lift, zero at both swing boundaries) so the foot clears the ground and
    lands back at the stand pose.  The trajectory is C0-continuous across
    the stance/swing boundary and exactly periodic.

    ``speed`` (m/s) scales stride frequency linearly through the gait's
    ``DEFAULT_GAITS`` operating point (``stride_hz`` at ``speed_mps``),
    clamped to [0.2x, 2.0x] of nominal so an absurd request degrades to a
    bounded cadence instead of a physically impossible one.

    ``speed`` is COMMANDED, not realised: it sets cadence only.  Because
    stride length is speed-invariant (the scaling cancels — see the module
    docstring) and the derived thigh amplitude saturates ``_THIGH_AMP_MAX``
    at the default profile, the emitted trajectory has a hard no-slip
    ground-speed ceiling of ~0.585 x commanded for default trot (0.823x
    walk, 0.384x bound), at every commanded speed.  Query
    :attr:`no_slip_speed_mps` / :attr:`speed_ceiling_ratio` /
    :attr:`thigh_amp_saturated` for the honest numbers for this instance.
    """

    def __init__(
        self,
        gait: str = "trot",
        profile: QuadrupedProfile | None = None,
        *,
        speed: float | None = None,
    ) -> None:
        self.profile = profile or QuadrupedProfile()
        if gait not in self.profile.gaits:
            raise KeyError(
                f"gait {gait!r} not in profile gaits {sorted(self.profile.gaits)}"
            )
        if gait not in GAIT_PHASE_OFFSETS:
            raise KeyError(
                f"gait {gait!r} has no phase-offset table "
                f"(known: {sorted(GAIT_PHASE_OFFSETS)})"
            )
        self.gait = gait
        self.spec: GaitSpec = self.profile.gaits[gait]
        self.phase_offsets = GAIT_PHASE_OFFSETS[gait]
        self.duty_factor = GAIT_DUTY_FACTOR.get(gait, 0.5)

        # Stride frequency scales linearly with commanded speed through the
        # gait table's operating point, clamped to a sane band.
        nominal_hz = self.spec.stride_hz
        if speed is not None and speed > 0.0:
            ratio = speed / self.spec.speed_mps
            ratio = min(max(ratio, 0.2), 2.0)
            self.stride_hz = nominal_hz * ratio
            self.speed_mps = self.spec.speed_mps * ratio
        else:
            self.stride_hz = nominal_hz
            self.speed_mps = self.spec.speed_mps
        self.period_s = 1.0 / self.stride_hz

        # Thigh sweep amplitude from stride length: each stance half-cycle
        # covers ~half the stride, and the leg pivots about the hip with the
        # body height as the effective leg length.  Clamped into the bounded
        # plausible band.
        stride_len_m = self.speed_mps / self.stride_hz
        raw_amp = stride_len_m / (2.0 * self.profile.body_height_m)
        # Kept pre-clamp so the commanded-vs-realised accessors below can
        # report whether (and by how much) the clamp binds.
        self.raw_thigh_amp_rad = raw_amp
        self.thigh_amp_rad = min(max(raw_amp, _THIGH_AMP_MIN), _THIGH_AMP_MAX)
        self.calf_lift_rad = _CALF_LIFT_RAD
        self.hip_amp_rad = min(math.radians(self.spec.roll_amp_deg), _HIP_AMP_MAX)

    # ------------------------------------------------------------------ #
    # Commanded-vs-realised speed contract (see module docstring)
    # ------------------------------------------------------------------ #

    @property
    def commanded_stride_m(self) -> float:
        """Stride length (m) the commanded speed implies at this cadence.

        Speed-INVARIANT by construction: cadence and effective speed scale
        by the same ratio, so this is always ``spec.speed_mps /
        spec.stride_hz`` regardless of the ``speed`` argument (default
        trot: 0.6154 m).
        """
        return self.speed_mps / self.stride_hz

    @property
    def achievable_stride_m(self) -> float:
        """Stride length (m) the CLAMPED thigh amplitude can actually sweep.

        No-slip kinematic model (the same one the amplitude was derived
        from, inverted): ``2 * thigh_amp_rad * body_height_m``.  With the
        default profile the clamp saturates for every built-in gait, so
        this is pinned at 0.36 m.
        """
        return 2.0 * self.thigh_amp_rad * self.profile.body_height_m

    @property
    def thigh_amp_saturated(self) -> bool:
        """True when the derived thigh amplitude hit a clamp bound.

        True for every built-in gait at the default profile — the visible
        symptom of the speed ceiling.  ``raw_thigh_amp_rad`` holds the
        unclamped value for diagnostics.
        """
        return self.thigh_amp_rad != self.raw_thigh_amp_rad

    @property
    def no_slip_speed_mps(self) -> float:
        """Theoretical ground-speed ceiling (m/s) of this trajectory.

        The body speed a PERFECT tracker with infinite friction would
        realise: ``achievable_stride_m * stride_hz``.  Live Newton runs on
        a healthy kit realise ~48.4% of commanded against this ~58.5%
        trot ceiling.  Planners must compare against this, not ``speed``.
        """
        return self.achievable_stride_m * self.stride_hz

    @property
    def speed_ceiling_ratio(self) -> float:
        """``no_slip_speed_mps / speed_mps`` — the honest fraction.

        Constant per gait+profile at every commanded speed (default
        profile: trot 0.585, walk 0.823, bound 0.384).  Equals 1.0 only
        when the amplitude clamp does not bind (e.g. a taller profile).
        """
        return self.no_slip_speed_mps / self.speed_mps

    # ------------------------------------------------------------------ #
    # Core kinematics
    # ------------------------------------------------------------------ #

    def _leg_angles(self, leg_phase: float, left_side: bool) -> tuple[float, float, float]:
        """(hip, thigh, calf) radians for one leg at its local phase [0, 1)."""
        duty = self.duty_factor
        if leg_phase < duty:
            # STANCE: foot planted, thigh sweeps linearly from forward reach
            # (-amp) to rearward push (+amp); calf holds the stand angle.
            s = leg_phase / duty  # 0..1 through stance
            thigh_off = self.thigh_amp_rad * (2.0 * s - 1.0)
            calf_off = 0.0
        else:
            # SWING: foot airborne, thigh returns rearward -> forward on a
            # cosine (matches +amp at liftoff, -amp at touchdown); calf tucks
            # with a sine lift that is exactly zero at both boundaries, so
            # the foot leaves and lands at the stand pose.
            s = (leg_phase - duty) / (1.0 - duty)  # 0..1 through swing
            thigh_off = self.thigh_amp_rad * math.cos(math.pi * s)
            calf_off = -self.calf_lift_rad * math.sin(math.pi * s)

        # Small lateral hip sway at stride frequency, mirrored across the
        # body midline (left legs +, right legs -).
        sway = self.hip_amp_rad * math.sin(2.0 * math.pi * leg_phase)
        hip = NEUTRAL_HIP_RAD + (sway if left_side else -sway)
        thigh = NEUTRAL_THIGH_RAD + thigh_off
        calf = NEUTRAL_CALF_RAD + calf_off
        return hip, thigh, calf

    def angles_at_phase(self, phase: float) -> dict[str, float]:
        """All 12 joint angles (radians) at a global cycle phase.

        ``phase`` is folded into [0, 1); each leg evaluates the shared
        swing/stance profile at ``phase + its gait offset``.
        """
        phase = phase % 1.0
        out: dict[str, float] = {}
        for leg in LEG_NAMES:
            leg_phase = (phase + self.phase_offsets[leg]) % 1.0
            left = leg.endswith("L")
            hip, thigh, calf = self._leg_angles(leg_phase, left)
            out[f"{leg}_hip"] = hip
            out[f"{leg}_thigh"] = thigh
            out[f"{leg}_calf"] = calf
        return out

    def angles_at_time(self, t: float) -> dict[str, float]:
        """All 12 joint angles (radians) at wall-clock time ``t`` seconds."""
        return self.angles_at_phase(t * self.stride_hz)

    def sample_cycle(self, steps: int = 32) -> list[tuple[float, dict[str, float]]]:
        """Sample one full gait cycle at ``steps`` evenly spaced phases.

        Returns ``[(phase, {joint: radians}), ...]`` with phases at
        ``i / steps`` for ``i`` in ``0..steps-1`` — a full period without
        duplicating the wrap point, ready for recording/replay or for a
        driver that plays the cycle as a lookup table.
        """
        if steps < 2:
            raise ValueError(f"steps must be >= 2, got {steps}")
        return [
            (i / steps, self.angles_at_phase(i / steps)) for i in range(steps)
        ]


# ---------------------------------------------------------------------- #
# Module-level convenience (default Go2-class profile)
# ---------------------------------------------------------------------- #

_CYCLE_CACHE: dict[tuple[str, float | None], QuadrupedGaitCycle] = {}


def joint_targets_at(
    t: float,
    *,
    gait: str = "trot",
    speed: float | None = None,
) -> dict[str, float]:
    """12 named joint drive targets (RADIANS) at time ``t`` seconds.

    Keys are exactly :data:`JOINT_NAMES` — ``"FL_hip"``, ``"FL_thigh"``,
    ``"FL_calf"``, then the same three for ``FR``, ``RL``, ``RR``.  Uses
    the default :class:`~tritium_lib.models.quadruped.QuadrupedProfile`
    (Go2-class ``DEFAULT_GAITS``); ``speed`` in m/s scales stride frequency
    through the gait's table entry.  A USD Newton driver applies these as
    joint drive ``targetPositions`` after converting to degrees
    (``radians * 180 / pi``).

    ``speed`` is COMMANDED cadence, not realised ground speed: the
    trajectory has a hard no-slip ceiling of ~0.585 x commanded for trot
    (see module docstring) — ``speed=1.2`` can deliver at most ~0.702 m/s
    even under perfect tracking.  Call :func:`no_slip_speed_for` with the
    same arguments for the honest ceiling.
    """
    key = (gait, speed)
    cycle = _CYCLE_CACHE.get(key)
    if cycle is None:
        cycle = QuadrupedGaitCycle(gait, speed=speed)
        _CYCLE_CACHE[key] = cycle
    return cycle.angles_at_time(t)


def no_slip_speed_for(
    *,
    gait: str = "trot",
    speed: float | None = None,
    profile: QuadrupedProfile | None = None,
) -> float:
    """Theoretical no-slip ground-speed ceiling (m/s) for a gait command.

    The honest counterpart to :func:`joint_targets_at`: for the same
    ``gait`` / ``speed`` arguments, returns the maximum body speed the
    emitted trajectory can realise under PERFECT joint tracking and
    infinite friction (``achievable_stride * stride_hz``).  Because stride
    length is speed-invariant and the thigh-amplitude clamp saturates at
    the default profile, this is ~0.585 x the (band-clamped) commanded
    speed for trot at EVERY speed — e.g. ``speed=1.2`` -> ~0.702 m/s.
    Live Newton runs on a healthy kit realise ~48.4% of commanded, i.e.
    they track slightly under this ceiling; the ceiling accounts for
    ~80.5% of the measured shortfall.  A planner that needs 1.2 m/s must
    check this BEFORE dispatching, not discover it in the field.
    """
    return QuadrupedGaitCycle(gait, profile, speed=speed).no_slip_speed_mps
