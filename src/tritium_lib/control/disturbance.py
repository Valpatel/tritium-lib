# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Deterministic disturbance injection and recovery scoring.

Why this exists
---------------
:mod:`tritium_lib.control.attitude_stabilizer` was built to stop a walking Go2
from inverting.  When it was finally A/B'd live against its own open-loop
control, **20 of 20 runs stayed upright in both arms** — the failure it was
built to fix simply did not occur that session.  Across three sessions the same
gait file produced upright rates of 67%, 29% and 100%, which means that "rate"
was never a property of the gait at all; it was a property of whatever tilt
each session happened to start with.

You cannot measure rejection of a disturbance that shows up by luck.  So stop
waiting for a fall and **cause** one: apply a known impulse at a known sim
time, identically in both arms, and score what happens *after* it.  That turns
an anecdote ("it fell twice on Tuesday") into a controlled experiment with a
manipulated variable.  This is standard practice for real balance controllers —
push-recovery testing — and it is the same discipline whether the body is in
Isaac or on a bench.

What is here
------------
:class:`Impulse` and :class:`DisturbanceSchedule` decide *when the kick lands*;
:func:`score_recovery` decides *whether the body came back*.  Neither knows
what applies the force — the simulator, a ROS effort controller, or a person
with a stick all drive the same objects.  The schedule hands back impulses and
the caller applies them.

Design notes worth knowing
--------------------------
* **A missed kick is louder than a failed recovery.**  The worst outcome of a
  disturbance experiment is not a body that falls over, it is a kick that never
  fired: both arms then score perfectly and the null result reads as success.
  So an impulse whose time is straddled by a slow frame still fires, and
  :attr:`DisturbanceSchedule.all_fired` lets a run report that it never landed
  rather than publishing a vacuous comparison.
* **Settling means staying below, not touching below.**  A body oscillating
  through the threshold has not recovered.  The first crossing is only settling
  if every later sample stays under too — the same "score the worst sample, not
  the mean" rule that :mod:`tritium_lib.geo.path_fidelity` applies to tracks.
* **Insufficient evidence is not success.**  A trace that ends while the body
  is quiet, but too soon to prove it stayed quiet, scores ``recovered=False``.
  A trace that never settles reports ``settle_time=None``, never a large
  sentinel — a sentinel averages into a summary and reads as a slow recovery.
* **Stdlib only** — no numpy — so this imports on a bare Jetson.

Units are SI: seconds, N·s for linear impulse, N·m·s for angular.  Tilt samples
are in **degrees**, matching :mod:`tritium_lib.geo.body_attitude`, which is
where they come from.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

__all__ = [
    "DisturbanceSchedule",
    "Impulse",
    "KickVerdict",
    "RecoveryScore",
    "kick_landed",
    "score_recovery",
]

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class Impulse:
    """A single push applied to a body's root at a fixed simulation time.

    Frozen on purpose: a schedule shared across trials must describe the same
    experiment every time, and an impulse mutated mid-run would mean the arms
    were no longer comparable.

    :param at_time: simulation seconds, measured from the start of the run.
    :param linear: linear impulse in N·s, body-frame REP-103 (+X forward,
        +Y left, +Z up).
    :param angular: angular impulse in N·m·s about the same axes.
    """

    at_time: float
    linear: Vec3 = (0.0, 0.0, 0.0)
    angular: Vec3 = (0.0, 0.0, 0.0)
    label: str = ""

    def __post_init__(self) -> None:
        if self.at_time < 0.0:
            raise ValueError(f"impulse time must be >= 0, got {self.at_time}")
        for name in ("linear", "angular"):
            vec = getattr(self, name)
            if len(vec) != 3:
                raise ValueError(f"{name} must be a 3-vector, got {vec!r}")

    @property
    def magnitude(self) -> float:
        """Euclidean norm of the linear part, in N·s."""
        return math.sqrt(sum(v * v for v in self.linear))

    @property
    def is_zero(self) -> bool:
        """True if this impulse would not move the body at all.

        A run configured with a zero kick measures nothing, and callers should
        be able to detect that without unpacking axes.
        """
        return not any(self.linear) and not any(self.angular)

    def as_dict(self) -> dict:
        return {
            "at_time": self.at_time,
            "linear": list(self.linear),
            "angular": list(self.angular),
            "label": self.label,
        }


class DisturbanceSchedule:
    """Fires each :class:`Impulse` exactly once, in the step that reaches it.

    The caller pumps :meth:`due` with its own step window and applies whatever
    comes back.  Firing is decided by simulation time rather than step count so
    that the same schedule lands at the same moment regardless of frame rate —
    which is what makes two arms of an A/B comparable.
    """

    def __init__(self, impulses: Iterable[Impulse]) -> None:
        items = tuple(sorted(impulses, key=lambda i: i.at_time))
        if not items:
            raise ValueError(
                "a disturbance schedule needs at least one impulse; an empty "
                "schedule would run an experiment with no manipulated variable"
            )
        self._impulses = items
        self._fired: list[Impulse] = []
        self._next = 0

    @property
    def impulses(self) -> tuple[Impulse, ...]:
        return self._impulses

    @property
    def fired(self) -> tuple[Impulse, ...]:
        """Impulses that have actually been handed to the caller."""
        return tuple(self._fired)

    @property
    def pending(self) -> int:
        """How many impulses have not fired yet."""
        return len(self._impulses) - self._next

    @property
    def all_fired(self) -> bool:
        """True only if every impulse landed.

        Check this before reporting a result.  A run that ends with impulses
        pending did not perform the experiment it claims to have performed.
        """
        return self.pending == 0

    def due(self, t_prev: float, t_now: float) -> list[Impulse]:
        """Return impulses whose time falls at or before ``t_now``.

        Impulses already passed are still returned on the first call that sees
        them — a slow frame that jumps clean over the scheduled time must not
        silently cancel the disturbance.
        """
        if t_now < t_prev:
            raise ValueError(
                f"simulation time went backwards ({t_prev} -> {t_now}); "
                "refusing to guess whether impulses should re-fire"
            )
        out: list[Impulse] = []
        while self._next < len(self._impulses):
            nxt = self._impulses[self._next]
            if nxt.at_time > t_now:
                break
            out.append(nxt)
            self._fired.append(nxt)
            self._next += 1
        return out

    def reset(self) -> None:
        """Re-arm every impulse for the next trial.

        Trials rebuild the scene between runs; a schedule that stayed spent
        would disturb only the first trial and leave the rest of the sample
        undisturbed, which is the vacuous-experiment failure mode again.
        """
        self._fired.clear()
        self._next = 0


@dataclass(frozen=True)
class RecoveryScore:
    """How a body responded to a known disturbance.

    :param baseline_deg: worst tilt seen *before* the kick — the gait's own
        wobble, which the kick's effect should be read against.
    :param peak_deg: worst tilt seen after the kick.
    :param peak_at: simulation time of that worst sample.
    :param settle_time: seconds from the kick until tilt dropped below the
        threshold *and stayed there*, or ``None`` if it never did.
    :param recovered: whether settling was actually demonstrated, hold
        included.
    :param samples_after: how much post-kick evidence the score rests on.
    """

    baseline_deg: float
    peak_deg: float
    peak_at: float
    settle_time: float | None
    recovered: bool
    samples_after: int
    disturbed_at: float
    settled_below_deg: float
    hold_for: float

    @property
    def excursion_deg(self) -> float:
        """How much worse the kick made things than the gait's own baseline."""
        return self.peak_deg - self.baseline_deg

    def as_dict(self) -> dict:
        return {
            "baseline_deg": round(self.baseline_deg, 4),
            "peak_deg": round(self.peak_deg, 4),
            "peak_at": round(self.peak_at, 4),
            "excursion_deg": round(self.excursion_deg, 4),
            "settle_time": (None if self.settle_time is None
                            else round(self.settle_time, 4)),
            "recovered": self.recovered,
            "samples_after": self.samples_after,
            "disturbed_at": round(self.disturbed_at, 4),
            "settled_below_deg": self.settled_below_deg,
            "hold_for": self.hold_for,
        }


def score_recovery(
    samples: Sequence[tuple[float, float]],
    *,
    disturbed_at: float,
    settled_below_deg: float,
    hold_for: float,
) -> RecoveryScore:
    """Score a tilt trace against a disturbance applied at ``disturbed_at``.

    :param samples: ``(sim_time_s, tilt_deg)`` pairs in ascending time order,
        typically body-up-vs-world-up angle from
        :mod:`tritium_lib.geo.body_attitude`.
    :param disturbed_at: simulation time the impulse landed.
    :param settled_below_deg: tilt considered "level again".
    :param hold_for: how long tilt must *stay* below that threshold before
        settling is believed.
    :raises ValueError: if the trace is empty, out of order, or contains no
        samples after the disturbance — scoring the last of those would report
        a peak of zero and read as flawless rejection.
    """
    if not samples:
        raise ValueError("cannot score recovery from an empty trace")
    if settled_below_deg < 0.0:
        raise ValueError(
            f"settled_below_deg must be >= 0, got {settled_below_deg}")
    if hold_for < 0.0:
        raise ValueError(f"hold_for must be >= 0, got {hold_for}")

    times = [float(t) for t, _ in samples]
    if any(b < a for a, b in zip(times, times[1:])):
        raise ValueError(
            "samples must be in ascending time order; an out-of-order trace "
            "would mis-slice the pre/post-disturbance split"
        )

    before = [(t, v) for t, v in samples if t < disturbed_at]
    after = [(float(t), float(v)) for t, v in samples if t >= disturbed_at]
    if not after:
        raise ValueError(
            f"no samples at or after the disturbance at t={disturbed_at}; "
            "the run ended before the kick landed, so there is nothing to "
            "score"
        )

    baseline = max((float(v) for _, v in before), default=0.0)
    # Take the value and its time separately so peak_at is the FIRST worst
    # sample; max() over the pairs would tie-break on time and report the last.
    peak = max(v for _, v in after)
    peak_at = next(t for t, v in after if v == peak)

    settle_time = None
    end_t = after[-1][0]
    for idx, (t, _v) in enumerate(after):
        if all(v < settled_below_deg for _, v in after[idx:]):
            # Below-and-staying-below, but only believable if the trace ran
            # long enough to witness the full hold.
            if end_t - t >= hold_for:
                settle_time = t - disturbed_at
            break

    return RecoveryScore(
        baseline_deg=baseline,
        peak_deg=peak,
        peak_at=peak_at,
        settle_time=settle_time,
        recovered=settle_time is not None,
        samples_after=len(after),
        disturbed_at=float(disturbed_at),
        settled_below_deg=float(settled_below_deg),
        hold_for=float(hold_for),
    )


@dataclass(frozen=True)
class KickVerdict:
    """Whether a commanded impulse actually reached the body.

    :param landed: whether the body gained enough velocity **along the
        commanded direction** to treat the trial as genuinely disturbed.
    :param expected_dv: ``|J| / m`` — the speed change an unopposed push
        would produce.
    :param projected_dv: the measured velocity change projected onto the
        commanded direction.  Signed: negative means the body moved the
        *other* way, which is evidence something other than the push moved it.
    :param fraction: ``projected_dv / expected_dv``.
    :param min_fraction: the threshold this verdict was judged against.
    """

    landed: bool
    expected_dv: float
    projected_dv: float
    fraction: float
    min_fraction: float

    def as_dict(self) -> dict:
        return {
            "landed": self.landed,
            "expected_dv": round(self.expected_dv, 6),
            "projected_dv": round(self.projected_dv, 6),
            "fraction": round(self.fraction, 4),
            "min_fraction": self.min_fraction,
        }


def kick_landed(
    *,
    commanded: Sequence[float],
    measured_dv: Sequence[float],
    body_mass: float,
    min_fraction: float = 0.4,
) -> KickVerdict:
    """Did the push land on the axis it was aimed at?

    :func:`score_recovery` already refuses to score a run in which no impulse
    ever fired.  This closes the next hole down, and it is not hypothetical:
    the first live 3 N-s A/B recorded a lateral ``+Y`` push whose measured
    velocity change was ``[-0.089, 0.0121, 0.1921]``.  The *magnitude* of that
    vector is 0.212 m/s against an expected 0.2 — so a norm check calls it a
    textbook push.  But the commanded axis gained 0.0121 m/s, about 6% of what
    was asked.  The body was **falling**, not shoved.  That trial's 178-degree
    tumble was then charged to the controller as a failure to reject a
    disturbance it never received.

    Hence a **projection onto the commanded direction**, never a norm.  A norm
    check is worse than no check here, because the trials it waves through are
    exactly the ones worth catching.

    ``min_fraction`` defaults to 0.4 rather than something near 1.0 because a
    real push into a real contact is partly absorbed by foot friction — about
    half of ``J/m`` reaches the body, and that is physically correct, not a
    miss.  The threshold separates "mostly delivered" from "barely moved".
    """
    if body_mass <= 0:
        raise ValueError(f"body mass must be positive, got {body_mass!r}")

    j = [float(c) for c in commanded]
    j_mag = math.sqrt(sum(c * c for c in j))
    if j_mag <= 0:
        raise ValueError(
            "commanded impulse is zero, so 'did it land' has no answer — "
            "scoring it either way would silently validate an undisturbed run")

    unit = [c / j_mag for c in j]
    dv = [float(c) for c in measured_dv]
    projected = sum(unit[i] * dv[i] for i in range(3))
    expected = j_mag / float(body_mass)
    fraction = projected / expected

    # Tolerance, not cosmetics: a fraction landing exactly on the threshold
    # is a division result, and 0.08/0.2 evaluates to 0.39999999999999997.
    # A bare >= would reject a push delivered to spec, and the trial it threw
    # away would be indistinguishable from one that never landed.
    return KickVerdict(
        landed=fraction >= min_fraction - 1e-9,
        expected_dv=expected,
        projected_dv=projected,
        fraction=fraction,
        min_fraction=float(min_fraction),
    )
