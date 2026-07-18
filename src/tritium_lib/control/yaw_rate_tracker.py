# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""The actuator end of the yaw cascade — rate demand in, turn command out.

Why this exists
---------------
:class:`~tritium_lib.control.yaw_regulator.YawRegulator` decides what yaw
RATE to demand and states its assumption honestly: *the body converts a
turn-rate command to actual yaw rate promptly and monotonically — a weak
plant needs a rate loop underneath.*  The live Newton-stepped Go2 IS that
weak plant.  Its turn is an open-loop gait artifact — stride scaling, not a
servo — and it was measured delivering roughly **12%** of the commanded rate
(see :mod:`tritium_lib.control.yaw_rate_loop`).  Push
:attr:`~tritium_lib.control.yaw_regulator.YawCorrection.correction_dps`
straight through :meth:`~tritium_lib.control.yaw_regulator.YawCorrection.
turn_intent` on such a body and every layer above inherits the deficit: the
heading closes at an eighth of the intended pace, the follower times out,
the route ends short.  This module is the inner loop the regulator names as
required, at the ACTUATOR seam::

    planner -> follower -> YawRegulator -> YawRateTracker -> motors
                pose loop:   heading loop:    rate loop:       the body
                where to go  what rate to     what TURN
                             demand           COMMAND makes
                                              that rate happen

The law
-------
Unity feedforward plus PI trim, with conditional-integration anti-windup:
the demanded rate passes through scaled by the body profile's full-scale
turn rate (exactly the normalization ``turn_intent`` and the edge tier's
``twist_to_motors`` already perform), and a PI term on the measured
shortfall asks for MORE than the profile claims is needed, so that what
arrives is what was demanded.  Proportional action answers a transient;
integral action is what actually nulls a persistent plant-gain deficit,
because a constant fractional shortfall is a constant error and only an
integrator drives a constant error to zero.  The law is implemented ONCE,
in :class:`~tritium_lib.control.yaw_rate_loop.YawRateLoop`, and run here
functionally — anti-windup is the classic bug of this loop shape, and a
second hand-rolled copy is how the classic bug ships twice.  That loop's
parameter names say rad/s because that is its native frame, but the math is
unit-blind (``kp`` dimensionless, ``ki`` in 1/s), so this seam runs it in
the compass frame without conversion.

Why frozen, and why the state is threaded
-----------------------------------------
Every controller in this package is a frozen dataclass, and the package
docstring explains what that buys: no drift, no wind-up across a reset, two
arms of an A/B fed the same measurements get the same answer.  An
integrator seems to break that bargain — it exists to remember — so the
memory is made EXPLICIT instead of hidden: :meth:`YawRateTracker.track`
takes a :class:`YawRateState` and returns the successor state inside the
:class:`TurnCorrection`.  The configuration object stays frozen and
shareable; the state is a value the CALLER owns.  That is not style — it is
what keeps the integrator honest: a replay fed the same states reproduces
the same commands byte for byte, a reset is dropping a value rather than
trusting a ``reset()`` was called, an A/B can fork one recorded state down
two arms, and a checkpoint of the controller is a copy of one small frozen
record.  (:class:`~tritium_lib.control.yaw_rate_loop.YawRateLoop` keeps its
mutable ``self.integral`` because a live addon already drives it; that API
is preserved, not extended.)

Conventions
-----------
Compass frame end to end, matching :class:`YawRegulator` exactly: rates in
deg/s, POSITIVE = CLOCKWISE (heading increasing).  The emitted ``turn`` is
a normalized :class:`~tritium_lib.models.body.ControlIntent` ``turn``
command in ``[-max_turn, max_turn]``: positive turn = clockwise = left side
faster than right (the contract ``motors_from_intent`` and the edge tier's
``twist_to_motors`` pin).  A REP-103 gyro reports +yaw = COUNTER-clockwise;
the caller supplies ``measured_dps = -math.degrees(wz)`` — the negation is
the contract, dropping it turns the loop into positive feedback.

The driver idiom — the tracker OWNS the turn axis.  Feed it the TOTAL
demanded rate (route turn plus heading correction) and use its output as
the whole turn command; adding a separate feedforward turn elsewhere would
double the feedforward this loop already carries::

    corr = regulator.correct(heading, commanded,
                             measured_yaw_rate_dps=rate_dps, dt=dt)
    cmd = tracker.track(corr.correction_dps, rate_dps, dt, state=state)
    state = cmd.state
    left, right = motors_from_intent(
        ControlIntent(forward=fwd, turn=cmd.turn))

Validation status, stated honestly
----------------------------------
**Closed-form and simulated only — NOT validated on live Newton.**  The
plant assumption this law rests on: the body's yaw rate responds to the
turn command MONOTONICALLY with no sign reversal — arbitrary unknown gain
and first-order-ish lag are exactly what the integrator absorbs, but a
plant that turns the wrong way under some commands turns any rate loop into
positive feedback.  And the measurement must be the body's NET turn, not
its gait: a trotting body's raw gyro is dominated by stride rock (measured
0.567 rad/s of it against 0.800 of command), and the rad/s loop fed that
raw signal was measured making the body WORSE live.  On a legged body,
feed this loop through :class:`~tritium_lib.control.stride_filter.
StrideFilter` and accept its bandwidth bound — a loop cannot be faster than
the group delay in its own measurement path.

Stdlib only, so it imports on a bare Jetson alongside the rest of the brain.
"""

from __future__ import annotations

from dataclasses import dataclass

from tritium_lib.control.yaw_rate_loop import YawRateLoop

__all__ = [
    "TurnCorrection",
    "YawRateState",
    "YawRateTracker",
]


@dataclass(frozen=True)
class YawRateState:
    """The tracker's entire memory: the integrated rate error, in degrees.

    A frozen value, not a mutable box — :meth:`YawRateTracker.track` never
    modifies the state it is given, it returns a successor.  A fresh
    ``YawRateState()`` is the canonical cold start (zero integral), so a
    driver's reset is ``state = YawRateState()`` and nothing else.
    """

    integral_deg: float = 0.0


@dataclass(frozen=True)
class TurnCorrection:
    """One tick of the rate tracker, including what it did and why.

    Every term is exposed rather than just the output — the same discipline
    as :class:`~tritium_lib.control.yaw_regulator.YawCorrection` — because
    when a live run steers badly, a saturated command and a wound integral
    are different faults with different fixes, and the log line has to be
    able to tell them apart.

    ``turn`` is the actuator-level command: a normalized
    :class:`~tritium_lib.models.body.ControlIntent` ``turn`` in
    ``[-max_turn, max_turn]``, positive clockwise.  ``compensated_dps`` is
    the same demand before normalization, for comparison against
    ``demanded_dps`` (their ratio is how hard the loop is pushing).
    ``state`` is the successor state the caller threads into the next tick;
    ``saturated`` reports the demand hitting the actuator ceiling —
    persistent saturation means the demand outruns the body's authority,
    which no controller can fix.
    """

    demanded_dps: float
    measured_dps: float
    error_dps: float
    compensated_dps: float
    turn: float
    saturated: bool
    state: YawRateState

    def as_dict(self) -> dict:
        return {
            "demanded_dps": self.demanded_dps,
            "measured_dps": self.measured_dps,
            "error_dps": self.error_dps,
            "compensated_dps": self.compensated_dps,
            "turn": self.turn,
            "saturated": self.saturated,
            "integral_deg": self.state.integral_deg,
        }


@dataclass(frozen=True)
class YawRateTracker:
    """PI-with-feedforward yaw-rate tracker emitting a normalized turn command.

    Frozen configuration; the integral lives in a threaded
    :class:`YawRateState` (see module docstring for why).

    :param turn_rate_dps: the body profile's full-scale turn rate — the SAME
        number ``YawCorrection.turn_intent`` and the edge tier's
        ``twist_to_motors`` divide by, so ``turn = 1.0`` means "the rate the
        profile claims full stick delivers".  Required rather than defaulted
        for the reason :class:`~tritium_lib.control.step_reflex.StepReflex`
        gives for ``com_height_m``: baking one body's number into lib would
        quietly mis-scale every other body.
    :param kp: proportional gain on the rate error, dimensionless.
    :param ki: integral gain, 1/s.  The defaults are the pair the rad/s loop
        settles its measured-weak plant with; both are unit-invariant, so
        they carry to this frame unchanged.
    :param max_turn: actuator authority ceiling in ``(0, 1]``.  ``1.0`` is
        the physical stop; below it reserves turn authority for the gait's
        forward mixing (``left = forward + turn/2`` clamps, so full turn on
        top of full forward saturates a motor).
    :param integral_limit_deg: clamp on the integrated error, degrees.
        ``None`` (the default) derives ``max_turn * turn_rate_dps / ki`` —
        the integral that ALONE saturates the actuator.  A larger integral
        is windup by definition: it demands authority that physically cannot
        be delivered, then spends seconds unwinding after the error
        reverses.
    """

    turn_rate_dps: float
    kp: float = 1.0
    ki: float = 6.0
    max_turn: float = 1.0
    integral_limit_deg: float | None = None

    def __post_init__(self) -> None:
        if self.turn_rate_dps <= 0.0:
            raise ValueError(
                f"turn_rate_dps must be > 0, got {self.turn_rate_dps}; a "
                "body with no turn authority cannot express any command"
            )
        if self.kp < 0.0 or self.ki < 0.0:
            raise ValueError(
                f"gains must be non-negative (got kp={self.kp}, "
                f"ki={self.ki}); a negative gain is positive feedback that "
                "amplifies the very shortfall this loop exists to null"
            )
        if not 0.0 < self.max_turn <= 1.0:
            raise ValueError(
                f"max_turn must be in (0, 1], got {self.max_turn}; above 1 "
                "promises authority the motor envelope does not have, and 0 "
                "means the tracker cannot act at all — configure it off, "
                "not mute"
            )
        if (
            self.integral_limit_deg is not None
            and self.integral_limit_deg < 0.0
        ):
            raise ValueError(
                f"integral_limit_deg must be >= 0, got "
                f"{self.integral_limit_deg}"
            )

    @property
    def effective_integral_limit_deg(self) -> float:
        """The integral clamp in force (see ``integral_limit_deg``)."""
        if self.integral_limit_deg is not None:
            return self.integral_limit_deg
        if self.ki == 0.0:
            return 0.0  # no integral action — nothing worth remembering
        return self.max_turn * self.turn_rate_dps / self.ki

    def track(
        self,
        demanded_dps: float,
        measured_dps: float,
        dt_s: float,
        *,
        state: YawRateState | None = None,
    ) -> TurnCorrection:
        """One tick: demanded and measured rate in, turn command + state out.

        ``demanded_dps`` is the TOTAL demanded yaw rate (deg/s, positive
        clockwise) — e.g. ``YawCorrection.correction_dps``, plus any route
        turn the driver carries.  ``measured_dps`` is the body's achieved
        yaw rate in the same frame (stride-filtered on a legged body — see
        module docstring).  ``state`` is the value returned in the previous
        tick's :attr:`TurnCorrection.state`; ``None`` means a cold start.

        Zero demand with zero measured rate (and a zero integral) returns a
        ``turn`` of exactly ``0.0`` and an unchanged state — the
        byte-identical no-op the package's layering contract promises,
        pinned by test.
        """
        prior = YawRateState() if state is None else state
        limit_deg = self.effective_integral_limit_deg
        # The PI law and its anti-windup live in YawRateLoop (see module
        # docstring); this object threads the frozen state through it.  The
        # entry clamp keeps a stale state honest against the CURRENT config
        # — the loop itself only clamps when it integrates.
        loop = YawRateLoop(
            kp=self.kp,
            ki=self.ki,
            max_output_rps=self.max_turn * self.turn_rate_dps,
            integral_limit=limit_deg,
        )
        loop.integral = max(
            -limit_deg, min(limit_deg, float(prior.integral_deg))
        )
        corr = loop.update(float(demanded_dps), float(measured_dps), dt_s)
        return TurnCorrection(
            demanded_dps=corr.commanded_rps,
            measured_dps=corr.measured_rps,
            error_dps=corr.error_rps,
            compensated_dps=corr.compensated_rps,
            turn=corr.compensated_rps / self.turn_rate_dps,
            saturated=corr.saturated,
            state=YawRateState(integral_deg=corr.integral),
        )
