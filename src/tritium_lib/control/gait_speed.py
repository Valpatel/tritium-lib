# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Closed-loop gait speed — converge to the achievable, report the shortfall.

Why this exists (the measured defect it answers)
------------------------------------------------
:mod:`tritium_lib.models.gait_trajectory` states its own contract honestly:
``speed`` sets CADENCE, not ground speed.  Stride length is speed-invariant
by construction (the scaling cancels at every commanded speed), the derived
thigh amplitude saturates its clamp at the default profile, and the result
is a hard no-slip ceiling of ~0.585 x commanded for default trot (walk
0.823x, bound 0.384x) — with live Newton runs on a healthy kit realising
~48.4% of commanded (2.39x spread across trials).  Two "obvious" open-loop
fixes were swept live and MEASURED WORSE:

* **Bigger amplitude tumbles the body.**  Thigh amplitude at x1.5 and x2.0
  of the clamp gave realised-speed medians 0.194 and 0.164 m/s against the
  0.452 m/s baseline, with 1/3 and 2/3 of trials losing upright posture.
* **More cadence past a point buys slip, not speed.**  Doubling stride
  frequency covered LESS ground: 1.20 m -> 0.85 m.  The realised-speed-vs-
  cadence curve is NON-monotonic somewhere in (1x, 2x) nominal.

So commanded speed cannot be made honest by retuning a constant.  This
module is the named follow-up: a feedback loop on REALISED body speed that
adapts the two parameters it is allowed to adapt — cadence, and amplitude
strictly downward — inside tumble-safe authority limits, converges to the
achievable operating point when the setpoint is out of reach, and answers
"am I at the ceiling?" truthfully instead of winding up against physics.

What is MEASURED versus what is DESIGNED (read before trusting)
---------------------------------------------------------------
**Measured (live Newton, 2026-07, inherited from the trajectory module and
the amplitude/cadence sweeps):** the 0.585x trot ceiling; the ~48.4%
realisation; amplitude increases tumbling the body; cadence doubling
covering less ground; the 2.39x trial-to-trial spread of realised speed.

**Designed and NOT live-validated (this module, all of it):** the PI loop
on stride-averaged speed; the 1.25x default cadence authority; the slip
guard and its thresholds; the low-stop amplitude trim; the estimator
window.  The unit tests prove the arithmetic — convergence, saturation
behaviour, authority limits, determinism — on a scripted plant.  They are
NOT evidence the controller helps a real body.  This package has been
burned exactly there before: :mod:`tritium_lib.control.step_reflex` shipped
a plausible, well-tested gate that live measurement then disproved twice
(an undisturbed walk went 6/6 upright -> 0/6).  Two specific ways THIS
design could be disproven live, named in advance:

1. **The plant gain may not be identifiable.**  The yaw rate loop's live
   system-ID (stride-filter lane, 2026-07) found the yaw plant gain
   sign-unstable (-0.905..+0.829), which unsupported that rate-loop
   architecture entirely.  If the cadence->speed gain is likewise
   sign-unstable tick to tick, a PI on it is unsupported and this module
   must be verdict-banned for walking bodies exactly as StepReflex was.
   (The one measured point in its favour: realised speed tracked commanded
   cadence at a consistent ~48% across the healthy-kit sweep, which is a
   stable positive gain AT the nominal operating point.  Off-nominal is
   unmeasured.)
2. **The slip guard can false-latch.**  It infers "slip" from (demand up,
   stride-averaged speed down) — but terrain, a push, or the measured
   2.39x trial spread produce the same signature.  A false latch degrades
   to a conservative, self-releasing speed cap (it cannot tumble the
   body), which is why the asymmetric risk was accepted; it is still a
   designed guess until a live sweep locates the real slip knee.

Authority limits, and the measured basis for each
-------------------------------------------------
* **Cadence: [0.2x, 1.25x] nominal by default.**  The floor is the
  trajectory generator's own band floor.  The 1.25x head: 1.0x is
  measured-good and 2.0x measured-worse, so the ceiling must sit in the
  unmeasured band between; closing the measured tracking gap (48.4%
  realised vs the 58.5% no-slip ceiling) needs ~1.21x more realised
  speed, so 1.25x grants exactly that headroom and no more.  A quarter of
  the way toward the measured-negative doubling — a DESIGN choice, with
  the slip guard as the in-band check.  The hard cap is 2.0x because the
  generator ratio-clamps there: authority beyond it is demand the
  trajectory silently cannot express, i.e. a phantom stop to wind against.
* **Demand slew: rate-limited, 0.5 (m/s)/s by default.**  A cadence step
  is a discontinuity in every joint's commanded velocity — exactly the
  transient a tumble-safe loop must not inject — so the demand may move
  at most ``max_slew_mps_per_s * dt`` per tick (full band in ~3.4 s, ~9
  trot strides, at the default).  The slew is also what makes the slip
  guard possible at all: without it the PI reaches its stop in one tick
  and there is never a demand GRADIENT to attribute a speed drop to.
  The rate is a DESIGN choice, not a measured one.
* **Amplitude: strictly DOWNWARD, scale in [0.267, 1.0].**  Raising
  amplitude is the one direction measured to tumble the body, so this
  controller is structurally incapable of it: ``amp_scale`` never exceeds
  1.0 (the trajectory's own clamped amplitude).  The floor mirrors the
  generator's own band ratio (0.12 / 0.45 rad).  Downward trim engages
  only when the demand is pinned at the cadence FLOOR and the body still
  overshoots — the low-speed regime cadence alone cannot reach.
* **Integral: clamped at the value that ALONE saturates the demand**
  (``max_demand / ki``), the same derivation
  :class:`~tritium_lib.control.yaw_rate_tracker.YawRateTracker` uses.  A
  larger integral is windup by definition.

The law — shared, not re-invented
---------------------------------
The PI-with-unity-feedforward law and its conditional-integration
anti-windup live in :class:`~tritium_lib.control.yaw_rate_loop.YawRateLoop`
and are run here FUNCTIONALLY, exactly the way
:class:`~tritium_lib.control.yaw_rate_tracker.YawRateTracker` runs them in
the compass frame: the loop's parameter names say rad/s because that is its
native frame, but the math is unit-blind (``kp`` dimensionless, ``ki`` in
1/s), so this seam runs it in m/s without conversion.  Anti-windup is the
classic bug of this loop shape; a second hand-rolled copy is how the
classic bug ships twice.  Two stops the loop's symmetric clamp cannot see
— the asymmetric demand FLOOR (a gait has no negative cadence) and the
slew band — are handled by feeding the loop the tick's slew-narrowed
ceiling as its ``max_output`` and by evaluating the loop's own
conditional-integration predicate against the tick floor before it runs,
never by a divergent re-implementation of the law.

The GAINS are this domain's, not the yaw tracker's, and the difference is
load-bearing: that loop updates at ~60 Hz control ticks, this one at
stride cadence (~0.4 s), and the discrete PI-with-feedforward against a
static plant of gain ``g`` is stable only while ``g*ki*dt < 2*(1 - g*kp)``
(Jury criterion on the [error, integral] map).  The yaw pair (kp=1,
ki=6) VIOLATES that at the measured g~0.5 and per-stride dt (1.2 > 1.0)
— it was caught oscillating unstably in this module's own tests — so the
defaults here are kp=0.4, ki=1.5: margin ~5x at g=0.5, still stable at a
full-delivery plant (g=1) and out to ~1 s update periods.

State is threaded, not hidden: :meth:`GaitSpeedTracker.track` takes a
frozen :class:`GaitSpeedState` and returns the successor inside the
command.  A replay fed the same states reproduces the same commands byte
for byte; a reset is dropping a value; an A/B forks one recorded state
down two arms.  No wall clock is ever read — ``dt_s`` is always an
argument.

Measure speed over STRIDES, not ticks (the StepReflex lesson)
-------------------------------------------------------------
A trotting body's instantaneous velocity is dominated by stride surge —
the exact carrier that made StepReflex's tick-level velocity gate fire on
100% of healthy walking ticks.  Feed this loop tick-level velocity and it
inherits that failure by construction.  :class:`StrideSpeedEstimator`
exists to prevent it: chord displacement over a sliding window of at least
one full stride period (two recommended), so the surge integrates out
before the loop ever sees a number.  It reports ``None`` until the window
is genuinely spanned — a partial window aliases the surge, and an honest
``None`` beats a confident alias.  Known biases, stated: chord under-reads
on a curved path (the loop then over-demands slightly, bounded by its
authority clamp), and body sway is averaged out rather than modelled.
Call :meth:`GaitSpeedTracker.track` at stride cadence or slower — with the
default gains the discrete loop's stability margin wants an update period
under ~0.4 s at the measured ~0.5 plant gain.

Composing with the addons GaitScheduler (design target, not a dependency)
-------------------------------------------------------------------------
The Newton driver's seam is ``GaitScheduler(targets_fn, dt, t0, limits,
stabilize_fn=None, reflex_fn=None)`` with composition order PINNED as
reflex -> stabilizer -> convert/clamp.  This controller does NOT enter
that order: it is a TRAJECTORY-PARAMETER loop, not a per-step joint hook —
it must never be wired as ``reflex_fn`` or ``stabilize_fn`` (those receive
joint targets; this emits gait parameters).  It acts upstream, inside the
``targets_fn`` closure the driver already owns, which is why the scheduler
API needs no change.  Changing stride frequency mid-run through
``angles_at_time`` would JUMP the gait phase (phase = t * hz is not
continuous under an hz change) and kick every joint at once;
:class:`GaitPhaseClock` is the required adapter — it accumulates phase
piecewise so a cadence change is C0-continuous.  The driver-side idiom::

    cycle = QuadrupedGaitCycle("trot", speed=commanded_mps)
    clock = GaitPhaseClock(cycle.stride_hz)
    est = StrideSpeedEstimator(window_s=2.0 / cycle.stride_hz)
    tracker = GaitSpeedTracker(nominal_mps=cycle.spec.speed_mps)
    state = None
    base_amp = cycle.thigh_amp_rad          # the generator's clamped amp

    def targets_fn(t):                       # bound into GaitScheduler
        return cycle.angles_at_phase(clock.phase_at(t))

    # per control step (scheduler drives the body as before):
    t, targets_deg = sched.step(attitude=quat_wxyz)
    speed = est.update(t, x, y)              # None until a full window
    if speed is not None and t - t_last >= update_period_s:
        cmd = tracker.track(commanded_mps, speed, t - t_last, state=state)
        state = cmd.state
        cycle = QuadrupedGaitCycle("trot", speed=cmd.demand_mps)
        cycle.thigh_amp_rad = base_amp * cmd.amp_scale   # only ever <= base
        clock.retime(t, cycle.stride_hz)     # phase-continuous cadence change
        t_last = t
        if cmd.at_ceiling:
            ...  # surface cmd.shortfall_mps to the operator / planner

The reflex -> stabilizer -> convert/clamp order is untouched: the
stabilizer keeps trimming whatever trajectory this loop parameterizes.
Planners should still consult
:func:`tritium_lib.models.gait_trajectory.no_slip_speed_for` BEFORE
dispatch; this loop reports the ceiling it finds, it does not repeal it.

Conventions match the rest of :mod:`tritium_lib.control`: SI units,
body-agnostic (no joint names — amplitude leaves here as a dimensionless
downward scale), stdlib only, so it imports on a bare aarch64 Jetson.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from tritium_lib.control.yaw_rate_loop import YawRateLoop

__all__ = [
    "DEFAULT_MAX_CADENCE_SCALE",
    "DEFAULT_MIN_AMP_SCALE",
    "DEFAULT_MIN_CADENCE_SCALE",
    "GaitPhaseClock",
    "GaitSpeedCommand",
    "GaitSpeedState",
    "GaitSpeedTracker",
    "StrideSpeedEstimator",
]

# Cadence authority band, as multiples of the gait table's nominal speed.
# Floor = the trajectory generator's own ratio-clamp floor.  Ceiling: the
# unmeasured band between the measured-good 1.0x and the measured-WORSE
# 2.0x, opened just far enough (~1.21x needed) to null the measured
# tracking gap below the no-slip ceiling — see the module docstring's
# authority section.  DESIGNED, not measured.
DEFAULT_MIN_CADENCE_SCALE: float = 0.2
DEFAULT_MAX_CADENCE_SCALE: float = 1.25

# The generator's own amplitude band ratio (_THIGH_AMP_MIN / _THIGH_AMP_MAX
# = 0.12 / 0.45 rad).  The floor of the strictly-downward amplitude trim.
DEFAULT_MIN_AMP_SCALE: float = 0.12 / 0.45

# The trajectory generator ratio-clamps commanded speed to [0.2, 2.0] x
# nominal.  Demand authority outside that band is demand the trajectory
# silently cannot express — a phantom stop for the integrator to wind
# against — so the tracker's config refuses to exceed it.
_GENERATOR_SCALE_FLOOR: float = 0.2
_GENERATOR_SCALE_CEIL: float = 2.0


def _require_finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


@dataclass(frozen=True)
class GaitSpeedState:
    """The tracker's entire memory, as one frozen threaded value.

    :meth:`GaitSpeedTracker.track` never modifies the state it is given; it
    returns a successor.  A fresh ``GaitSpeedState()`` is the canonical
    cold start, so a driver's reset is ``state = GaitSpeedState()`` and
    nothing else.

    :param integral_mps: the PI integrator (integrated speed error, m).
    :param active_demand_mps: the demand emitted last tick — the demand the
        CURRENT measurement is attributed to by the slip guard, and the
        reference the slew limiter moves from.  ``None`` on a cold start
        (no attribution possible; the slew then references the clamped
        commanded speed).
    :param anchor_demand_mps: the slip guard's reference operating point,
        with ``anchor_speed_mps`` the stride-averaged speed measured
        there.  The anchor advances with the demand climb only while
        speed is NOT declining, and HOLDS through sub-threshold declines
        so they accumulate — that is what catches both a steep knee and a
        slow creep into slip; comparing consecutive ticks would see
        neither.
    :param slip_ceiling_mps: latched demand ceiling after slip evidence,
        ``None`` when unlatched.  Self-releases after
        ``slip_release_s`` of accumulated ``dt``.
    :param since_latch_s: time accumulated since the latch engaged.
    :param ceiling_ticks: consecutive ticks the tracker has been saturated
        high while still short of commanded — a consumer's persistence
        signal for "this is a steady ceiling, not a transient".
    """

    integral_mps: float = 0.0
    active_demand_mps: float | None = None
    anchor_demand_mps: float | None = None
    anchor_speed_mps: float | None = None
    slip_ceiling_mps: float | None = None
    since_latch_s: float = 0.0
    ceiling_ticks: int = 0


@dataclass(frozen=True)
class GaitSpeedCommand:
    """One tick of the speed loop, including what it did and why.

    Every term is exposed rather than just the output — the discipline of
    :class:`~tritium_lib.control.yaw_rate_tracker.TurnCorrection` — because
    when a live run walks slow, a saturated demand, a slip latch, and a
    wound integral are different faults with different fixes, and the log
    line has to tell them apart.

    ``demand_mps`` is what the driver feeds the trajectory generator as its
    ``speed`` argument (it sets cadence); ``cadence_scale`` is the same
    demand as a multiple of nominal, for logs.  ``amp_scale`` multiplies
    the generator's own clamped thigh amplitude and NEVER exceeds 1.0 —
    this controller is structurally unable to command the amplitude
    increase that was measured to tumble the body.

    The honest-shortfall contract: ``at_ceiling`` is True exactly when the
    demand is pinned at its authority ceiling this tick AND the body is
    still short of commanded by more than the deadband; ``shortfall_mps``
    is how short; ``ceiling_ticks`` counts consecutive such ticks so a
    consumer can distinguish a steady ceiling from a spin-up transient.
    ``slip_latched`` marks a ceiling lowered by the slip guard rather than
    by static authority.
    """

    commanded_mps: float
    measured_mps: float
    error_mps: float
    demand_mps: float
    cadence_scale: float
    amp_scale: float
    saturated: bool
    at_ceiling: bool
    ceiling_ticks: int
    shortfall_mps: float
    slip_latched: bool
    state: GaitSpeedState

    def as_dict(self) -> dict:
        return {
            "commanded_mps": self.commanded_mps,
            "measured_mps": self.measured_mps,
            "error_mps": self.error_mps,
            "demand_mps": self.demand_mps,
            "cadence_scale": self.cadence_scale,
            "amp_scale": self.amp_scale,
            "saturated": self.saturated,
            "at_ceiling": self.at_ceiling,
            "ceiling_ticks": self.ceiling_ticks,
            "shortfall_mps": self.shortfall_mps,
            "slip_latched": self.slip_latched,
            "integral_mps": self.state.integral_mps,
            "slip_ceiling_mps": self.state.slip_ceiling_mps,
        }


@dataclass(frozen=True)
class GaitSpeedTracker:
    """Feedback speed loop over a cadence-driven gait, with honest limits.

    Frozen configuration; all memory lives in a threaded
    :class:`GaitSpeedState` (module docstring: why).  The PI law and its
    anti-windup are :class:`~tritium_lib.control.yaw_rate_loop.YawRateLoop`
    run functionally in the m/s frame — never a second PID.

    NOT live-validated: this controller's helpfulness on a real body is
    designed, not measured (module docstring, and the StepReflex warning
    there).  What the unit tests DO prove: inertness when the setpoint is
    already met, convergence to the stop without windup when it is not,
    and that no emitted command ever exceeds the authority limits.

    :param nominal_mps: the gait table operating-point speed the trajectory
        generator ratios against (``spec.speed_mps`` — default trot: 1.6).
        Required, not defaulted: baking one body's number into lib would
        quietly mis-scale every other body, and the authority band is
        expressed as multiples of exactly this number.
    :param kp: proportional gain, dimensionless.
    :param ki: integral gain, 1/s.  The defaults (0.4, 1.5) are sized for
        STRIDE-cadence updates, not control ticks — the module docstring
        derives the stability criterion (``g*ki*dt < 2*(1 - g*kp)``) and
        why the yaw tracker's hotter pair is unstable at this dt.  Keep
        update periods at or under ~0.5 s with these defaults.
    :param max_slew_mps_per_s: demand rate limit.  A cadence step is a
        joint-velocity discontinuity (tumble risk) and the slip guard
        needs a demand gradient to attribute speed changes to; both want
        the demand to RAMP.  ``math.inf`` disables (and starves the slip
        guard — see the module docstring's authority section).
    :param min_cadence_scale / max_cadence_scale: demand authority band as
        multiples of ``nominal_mps``.  Must stay inside the generator's
        own [0.2, 2.0] ratio clamp (see module constants); the default
        ceiling of 1.25 is a design choice inside the unmeasured band —
        its basis is in the module docstring's authority section.
    :param min_amp_scale: floor of the strictly-downward amplitude trim.
        The trim can never raise amplitude (``amp_scale`` <= 1.0 always);
        that direction is the one measured to tumble the body.
    :param deadband_mps: shortfall below which ``at_ceiling`` stays False —
        a truth-telling deadband against measurement noise, not a control
        deadband (the PI still integrates small errors; that is its job).
        Sized as ~11% of the measured 0.452 m/s healthy-kit median —
        designed, not measured.
    :param integral_limit_mps: clamp on the integrated error.  ``None``
        derives ``max_demand / ki`` — the integral that ALONE saturates
        the demand; anything larger is windup by definition.
    :param slip_guard: enable the non-monotonicity guard.  Cadence past an
        unmeasured knee was measured to buy slip, not speed (2x cadence
        covered LESS ground); the guard latches the demand ceiling at the
        anchor — the last operating point whose speed was not declining —
        whenever demand has risen by at least ``slip_probe_delta_mps``
        past it while stride-averaged speed fell ``slip_tol_mps`` below
        it.  Because the anchor holds through sub-threshold declines, a
        persistent decline of ANY slope accumulates to a latch; the trade
        is that the latch then binds at the base of the decline
        (conservative).  Latches self-release after ``slip_release_s`` so
        a transient (bump, push, mismeasure) cannot cripple the gait
        permanently.  The failure mode is deliberately conservative: a
        false latch caps speed and self-releases; it cannot tumble the
        body.  Thresholds are designed, not measured.
    """

    nominal_mps: float
    kp: float = 0.4
    ki: float = 1.5
    max_slew_mps_per_s: float = 0.5
    min_cadence_scale: float = DEFAULT_MIN_CADENCE_SCALE
    max_cadence_scale: float = DEFAULT_MAX_CADENCE_SCALE
    min_amp_scale: float = DEFAULT_MIN_AMP_SCALE
    deadband_mps: float = 0.05
    integral_limit_mps: float | None = None
    slip_guard: bool = True
    slip_probe_delta_mps: float = 0.05
    slip_tol_mps: float = 0.08
    slip_release_s: float = 8.0

    def __post_init__(self) -> None:
        if not (math.isfinite(self.nominal_mps) and self.nominal_mps > 0.0):
            raise ValueError(
                f"nominal_mps must be a finite positive speed, got "
                f"{self.nominal_mps!r}; it is the gait table operating point "
                "the whole authority band is expressed against"
            )
        if self.kp < 0.0 or self.ki < 0.0:
            raise ValueError(
                f"gains must be non-negative (got kp={self.kp}, ki={self.ki});"
                " a negative gain is positive feedback that amplifies the "
                "very shortfall this loop exists to null"
            )
        if not self.max_slew_mps_per_s > 0.0:  # also catches NaN
            raise ValueError(
                f"max_slew_mps_per_s must be > 0, got "
                f"{self.max_slew_mps_per_s}; a zero slew freezes the demand "
                "forever — to disable rate limiting use math.inf"
            )
        if not (
            _GENERATOR_SCALE_FLOOR <= self.min_cadence_scale
            < self.max_cadence_scale <= _GENERATOR_SCALE_CEIL
        ):
            raise ValueError(
                "cadence authority must satisfy "
                f"{_GENERATOR_SCALE_FLOOR} <= min < max <= "
                f"{_GENERATOR_SCALE_CEIL} (got min={self.min_cadence_scale}, "
                f"max={self.max_cadence_scale}); the trajectory generator "
                "ratio-clamps to that band, so authority outside it is a "
                "phantom stop the integrator would wind against"
            )
        if not 0.0 < self.min_amp_scale <= 1.0:
            raise ValueError(
                f"min_amp_scale must be in (0, 1], got {self.min_amp_scale}; "
                "1.0 disables the downward trim, 0 would command a legless "
                "shuffle"
            )
        if self.deadband_mps < 0.0:
            raise ValueError(
                f"deadband_mps must be >= 0, got {self.deadband_mps}"
            )
        if (
            self.integral_limit_mps is not None
            and self.integral_limit_mps < 0.0
        ):
            raise ValueError(
                f"integral_limit_mps must be >= 0, got "
                f"{self.integral_limit_mps}"
            )
        if self.slip_probe_delta_mps <= 0.0 or self.slip_tol_mps <= 0.0:
            raise ValueError(
                "slip thresholds must be positive (got probe_delta="
                f"{self.slip_probe_delta_mps}, tol={self.slip_tol_mps}); a "
                "zero threshold latches on noise every tick"
            )
        if self.slip_release_s <= 0.0:
            raise ValueError(
                f"slip_release_s must be > 0, got {self.slip_release_s}; a "
                "latch that never releases turns one bad measurement into a "
                "permanent speed cap"
            )

    # ------------------------------------------------------------------ #
    # Derived authority
    # ------------------------------------------------------------------ #

    @property
    def min_demand_mps(self) -> float:
        """The demand floor (m/s) — the cadence band floor."""
        return self.min_cadence_scale * self.nominal_mps

    @property
    def max_demand_mps(self) -> float:
        """The static demand ceiling (m/s), before any slip latch."""
        return self.max_cadence_scale * self.nominal_mps

    @property
    def effective_integral_limit_mps(self) -> float:
        """The integral clamp in force (see ``integral_limit_mps``)."""
        if self.integral_limit_mps is not None:
            return self.integral_limit_mps
        if self.ki == 0.0:
            return 0.0  # no integral action — nothing worth remembering
        return self.max_demand_mps / self.ki

    # ------------------------------------------------------------------ #
    # The tick
    # ------------------------------------------------------------------ #

    def track(
        self,
        commanded_mps: float,
        measured_mps: float,
        dt_s: float,
        *,
        state: GaitSpeedState | None = None,
    ) -> GaitSpeedCommand:
        """One tick: commanded and measured speed in, gait command + state out.

        ``commanded_mps`` is the ground speed the caller wants (> 0 — a
        stop is the driver ending the gait, not this loop chasing zero
        cadence).  ``measured_mps`` is the body's realised ground speed,
        STRIDE-AVERAGED (:class:`StrideSpeedEstimator`, or displacement
        over a stride window) — tick-level instantaneous velocity carries
        the stride surge this loop must never see (module docstring).
        ``dt_s`` is the time since the previous ``track`` call; ``state``
        is the value from the previous tick's command, ``None`` for a cold
        start.  ``dt_s == 0`` is a proportional-only tick (integrator
        holds), matching the shared law.

        The inert contract, pinned by test: when the body already tracks
        the setpoint (``measured == commanded``, zero integral), the
        emitted demand is byte-equal to ``commanded_mps``, ``amp_scale``
        is exactly 1.0, and ``at_ceiling`` is False — the loop changes
        nothing about a gait that is already honest.
        """
        commanded = _require_finite("commanded_mps", commanded_mps)
        measured = _require_finite("measured_mps", measured_mps)
        dt = _require_finite("dt_s", dt_s)
        if commanded <= 0.0:
            raise ValueError(
                f"commanded_mps must be > 0, got {commanded}; commanding a "
                "stop is the driver's job (end the gait), not a setpoint "
                "this loop can reach with any positive cadence"
            )
        if measured < 0.0:
            raise ValueError(
                f"measured_mps must be >= 0, got {measured}; ground speed "
                "is a magnitude — a signed velocity belongs upstream"
            )
        if dt < 0.0:
            raise ValueError(f"dt_s must be >= 0, got {dt}")

        prior = GaitSpeedState() if state is None else state
        lo = self.min_demand_mps
        hi_static = self.max_demand_mps

        # -- Slip guard: release, then attribute, then (maybe) latch. ---- #
        slip_ceiling = prior.slip_ceiling_mps
        since_latch = prior.since_latch_s
        anchor_d = prior.anchor_demand_mps
        anchor_s = prior.anchor_speed_mps
        if self.slip_guard:
            if slip_ceiling is not None:
                since_latch += dt
                if since_latch >= self.slip_release_s:
                    # Self-release, and drop the (now stale) anchor: the
                    # world may have changed in slip_release_s; re-probe
                    # from a fresh baseline rather than a memory.
                    slip_ceiling = None
                    since_latch = 0.0
                    anchor_d = None
                    anchor_s = None
            if prior.active_demand_mps is not None:
                # The measurement this tick is attributed to the demand
                # emitted LAST tick — which is why track() must be called
                # at stride cadence or slower (module docstring).
                cur_d = prior.active_demand_mps
                if anchor_d is None or anchor_s is None:
                    anchor_d, anchor_s = cur_d, measured
                elif anchor_d - cur_d >= self.slip_probe_delta_mps:
                    # Demand moved DOWN meaningfully: re-anchor, no slip
                    # inference (less demand, less speed is the normal
                    # monotonic direction).
                    anchor_d, anchor_s = cur_d, measured
                elif cur_d - anchor_d >= self.slip_probe_delta_mps:
                    if anchor_s - measured >= self.slip_tol_mps:
                        # Meaningfully more demand, meaningfully less
                        # speed: past the knee.  Latch the ceiling at the
                        # anchor — the last operating point whose speed
                        # was still holding — and drop the anchor so the
                        # eventual release re-probes from fresh data.
                        slip_ceiling = max(lo, anchor_d)
                        since_latch = 0.0
                        anchor_d, anchor_s = None, None
                    elif measured >= anchor_s:
                        # Speed is NOT declining: advance the anchor with
                        # the climb.  (A sub-threshold DECLINE keeps the
                        # anchor, so slow declines accumulate to a latch
                        # instead of being re-anchored away.)
                        anchor_d, anchor_s = cur_d, measured
        hi = hi_static if slip_ceiling is None else min(
            hi_static, max(lo, slip_ceiling),
        )

        # -- Slew band: the demand may only RAMP from where it was. ------ #
        # (Module docstring: tumble rationale + the slip guard's gradient.)
        prev = prior.active_demand_mps
        if prev is None:
            # Cold start: reference the clamped commanded speed — the gait
            # the driver was presumably already running open-loop.
            prev = min(hi, max(lo, commanded))
        else:
            prev = min(hi_static, max(lo, prev))  # stale-state honesty
        slew = self.max_slew_mps_per_s * dt
        hi_tick = min(hi, prev + slew)
        lo_tick = max(lo, prev - slew)
        if lo_tick > hi_tick:
            # A slip latch below the reachable band: descend toward it at
            # the slew rate rather than jumping.
            lo_tick = hi_tick

        # -- The shared PI law, run functionally (never a second PID). --- #
        limit_i = self.effective_integral_limit_mps
        loop = YawRateLoop(
            kp=self.kp,
            ki=self.ki,
            max_output_rps=hi_tick,
            integral_limit=limit_i,
        )
        # Entry clamp keeps a stale state honest against the CURRENT
        # config — the loop itself only clamps when it integrates.
        loop.integral = max(-limit_i, min(limit_i, float(prior.integral_mps)))

        # The loop's conditional integration knows only its symmetric
        # +/-max_output stop — which this tick IS the slew-narrowed
        # ceiling, so high-side anti-windup (including "winding while
        # slew-limited") comes with the law.  The demand FLOOR is this
        # domain's (a gait has no negative cadence, and the slew bounds
        # deceleration too), so the same predicate — "already against the
        # stop and this error pushes further in" — is evaluated against
        # the tick floor here, and integration is withheld by passing
        # dt=0 (the loop's documented integrator-hold path).  Same law,
        # second stop.
        error = commanded - measured
        raw_pred = commanded + self.kp * error + self.ki * loop.integral
        eff_dt = dt
        if raw_pred < lo_tick and error < 0.0:
            eff_dt = 0.0
        corr = loop.update(commanded, measured, eff_dt)

        low_stop = corr.compensated_rps <= lo_tick
        demand = min(hi_tick, max(lo_tick, corr.compensated_rps))
        saturated = corr.saturated or low_stop

        # -- Honest ceiling report. -------------------------------------- #
        # "At the ceiling" means pinned against AUTHORITY (static or slip
        # latch) — being slew-limited mid-ramp is a transient, not a wall.
        at_ceiling = demand >= hi - 1e-12 and error > self.deadband_mps
        ceiling_ticks = prior.ceiling_ticks + 1 if at_ceiling else 0
        shortfall = max(0.0, error)

        # -- Amplitude: strictly downward, only at the cadence floor. ---- #
        # Engages when cadence AUTHORITY (not slew) is exhausted downward
        # and the body still overshoots — the low-speed regime.
        # Proportional, stateless, floor-clamped; never above 1.0 by
        # construction.
        amp_scale = 1.0
        if demand <= lo + 1e-12 and measured > commanded + self.deadband_mps:
            amp_scale = max(self.min_amp_scale, min(1.0, commanded / measured))

        next_state = GaitSpeedState(
            integral_mps=corr.integral,
            active_demand_mps=demand,
            anchor_demand_mps=anchor_d,
            anchor_speed_mps=anchor_s,
            slip_ceiling_mps=slip_ceiling,
            since_latch_s=since_latch,
            ceiling_ticks=ceiling_ticks,
        )
        return GaitSpeedCommand(
            commanded_mps=commanded,
            measured_mps=measured,
            error_mps=error,
            demand_mps=demand,
            cadence_scale=demand / self.nominal_mps,
            amp_scale=amp_scale,
            saturated=saturated,
            at_ceiling=at_ceiling,
            ceiling_ticks=ceiling_ticks,
            shortfall_mps=shortfall,
            slip_latched=slip_ceiling is not None,
            state=next_state,
        )


class StrideSpeedEstimator:
    """Stride-averaged ground speed from injected (t, x, y) samples.

    Chord displacement over a sliding time window: the newest sample
    against the oldest sample still spanning at least ``window_s``.  With
    the window an integer number of stride periods, the stride surge and
    lateral sway integrate out — the whole reason this class exists (the
    module docstring's StepReflex lesson).  Reports ``None`` until the
    buffer genuinely spans the window: a partial window aliases the surge,
    and an honest ``None`` beats a confident alias.

    Biases, stated: the chord under-reads on a curved path (a speed loop
    fed it then over-demands, bounded by its authority clamp), and speed
    is a magnitude — direction belongs to the yaw stack.

    Injected time only — ``t`` is always an argument, monotonicity is
    enforced (a rewound clock raises; an equal timestamp is ignored, the
    paused-sim case), and no wall clock is ever read.  Memory is bounded
    by the samples inside one window plus one boundary sample.
    """

    def __init__(self, window_s: float) -> None:
        window_s = _require_finite("window_s", window_s)
        if window_s <= 0.0:
            raise ValueError(
                f"window_s must be > 0, got {window_s}; use at least one "
                "full stride period (two recommended) so the stride surge "
                "averages out"
            )
        self.window_s = window_s
        self._samples: deque[tuple[float, float, float]] = deque()

    def reset(self) -> None:
        """Drop all samples — call between runs, not between ticks."""
        self._samples.clear()

    def update(self, t: float, x: float, y: float) -> float | None:
        """Add a position sample; return the windowed speed, or ``None``.

        ``t`` in seconds (any monotonic clock, injected), ``x``/``y`` in
        metres in any fixed planar frame.  Returns the chord speed over
        the trailing window once the samples span it, else ``None``.
        """
        t = _require_finite("t", t)
        x = _require_finite("x", x)
        y = _require_finite("y", y)
        if self._samples:
            last_t = self._samples[-1][0]
            if t < last_t:
                raise ValueError(
                    f"time went backwards: t={t} after t={last_t}; the "
                    "estimator requires a monotonic injected clock"
                )
            if t == last_t:
                return self.speed()  # paused clock: ignore, report as-is
        self._samples.append((t, x, y))
        # Trim, keeping one sample AT or beyond the window boundary so the
        # reported span always covers at least window_s once it can.
        cutoff = t - self.window_s
        while len(self._samples) >= 2 and self._samples[1][0] <= cutoff:
            self._samples.popleft()
        return self.speed()

    def speed(self) -> float | None:
        """The current windowed chord speed, or ``None`` if not yet spanned."""
        if len(self._samples) < 2:
            return None
        t0, x0, y0 = self._samples[0]
        t1, x1, y1 = self._samples[-1]
        span = t1 - t0
        if span < self.window_s:
            return None
        return math.hypot(x1 - x0, y1 - y0) / span


class GaitPhaseClock:
    """Phase-continuous gait clock across stride-frequency changes.

    ``QuadrupedGaitCycle.angles_at_time`` folds ``t * stride_hz`` — correct
    at a fixed cadence, but change ``stride_hz`` mid-run and the implied
    phase JUMPS, kicking all twelve joints at once (the exact transient a
    tumble-safe speed loop must not inject).  This clock integrates phase
    piecewise instead: within a segment ``phase(t) = phase0 + (t - t0) *
    hz``, and :meth:`retime` rebases the segment so phase is C0-continuous
    through every cadence change.  Drive the trajectory with
    ``cycle.angles_at_phase(clock.phase_at(t))`` — never ``angles_at_time``
    — whenever cadence can change.

    Injected time only; queries are pure within a segment (the same ``t``
    always yields the same phase), so a scheduler may sample it repeatedly.
    """

    def __init__(
        self, stride_hz: float, t0: float = 0.0, phase0: float = 0.0,
    ) -> None:
        self._hz = self._check_hz(stride_hz)
        self._t0 = _require_finite("t0", t0)
        self._phase0 = _require_finite("phase0", phase0)

    @staticmethod
    def _check_hz(stride_hz: float) -> float:
        stride_hz = _require_finite("stride_hz", stride_hz)
        if stride_hz <= 0.0:
            raise ValueError(
                f"stride_hz must be > 0, got {stride_hz}; a gait with no "
                "cadence has no phase to advance"
            )
        return stride_hz

    @property
    def stride_hz(self) -> float:
        """The cadence of the current segment, Hz."""
        return self._hz

    def phase_at(self, t: float) -> float:
        """Gait phase at time ``t`` (unwrapped; fold with ``% 1.0``)."""
        return self._phase0 + (_require_finite("t", t) - self._t0) * self._hz

    def retime(self, t: float, stride_hz: float) -> None:
        """Change cadence at time ``t`` with C0-continuous phase.

        The phase at ``t`` is identical immediately before and after the
        call (pinned by test); only the slope changes.
        """
        new_hz = self._check_hz(stride_hz)
        self._phase0 = self.phase_at(t)
        self._t0 = float(t)
        self._hz = new_hz
