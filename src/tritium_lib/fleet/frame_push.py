# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Robot-side policy for pushing camera frames to the operator.

Until now a camera reached the Command Center one way: the operator **dialled
in** to an MJPEG or RTSP address the robot advertised (see
:mod:`tritium_lib.fleet.sensor_rig`, which plans exactly those registrations).
That works on a bench and fails everywhere else — a robot on a field radio, a
unit behind CGNAT, a simulator whose renderer binds to ``localhost`` — because
the operator has no inbound route to it.  The fix is the one real fleets use:
the robot **dials out** and pushes its frames.

Pushing is easy; pushing *well* over a link that is sometimes dead is not, and
the four decisions that make the difference all fit in pure logic:

  1. **Decimate.** Never send faster than the operator asked for.  A 60 fps
     renderer must not put 60 fps on a link sized for 10.
  2. **Drop, never queue.** If a send is still in flight, throw the new frame
     away.  A queue converts a slow link into a *lagging* one, and for live
     video a gap is honest where staleness is a lie — the operator cannot tell
     an old frame from a current one.
  3. **Back off.** Consecutive failures widen the retry interval, so an
     operator that is down or restarting is not hammered once per frame.
  4. **Count honestly.** "Sent", "dropped because too fast", "dropped because
     busy" and "refused by the far end" are four different things.  Collapsing
     them is how a rig that has never delivered a single frame reports itself
     green.

No sockets live here, on purpose.  The transport is three lines of ``requests``
or ``urllib`` in the caller; the *decisions* above are what deserve tests, and
none of them need a network to exercise.  That split also keeps this module
importable on a bare Jetson, which the whole library guarantees.

Both North Star halves: FUN — the operator sees the robot's eye view without
opening a port, configuring a tunnel, or knowing where the robot is.
PRODUCTION — inbound reachability is the single most common reason a fielded
camera is dark, and a push path is how every real ground station solves it.

Typical use::

    policy = FramePushPolicy(target_fps=10.0)
    url = operator_url + frame_push_path("isaac_rgb")

    while running:
        now = time.monotonic()
        if not policy.offer(now).send:
            continue
        try:
            requests.post(url, data=jpeg, headers={"Content-Type": "image/jpeg"})
            policy.sent(time.monotonic())
        except Exception:
            policy.failed(time.monotonic())
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote

__all__ = [
    "FRAME_PUSH_PATH",
    "PushDecision",
    "PushStats",
    "FramePushPolicy",
    "frame_push_path",
]

#: The route the Command Center serves for pushed frames.  ``{source_id}`` is
#: the same id :meth:`~tritium_lib.fleet.sensor_rig.RigSensor.feed_source_id`
#: hands out, so a rig that registered a feed can push to it without being told
#: anything new.
FRAME_PUSH_PATH = "/api/camera-feeds/sources/{source_id}/frame"


def frame_push_path(source_id: str) -> str:
    """Path to POST JPEG bytes for *source_id*.

    The id is percent-encoded rather than interpolated raw: an id containing a
    slash would otherwise silently forge a *different*, probably nonexistent
    route, and the resulting 404 would surface as "the operator is refusing my
    frames" long after the real mistake.
    """
    if not source_id or not source_id.strip():
        raise ValueError("source_id must be a non-empty string")
    return FRAME_PUSH_PATH.format(source_id=quote(source_id, safe=""))


@dataclass(frozen=True)
class PushDecision:
    """Whether to send this frame, and — when not — precisely why.

    The reason is not decoration.  ``rate_limited`` means the policy is working
    as designed, ``in_flight`` means the link is slower than the frame rate,
    and ``backoff`` means the far end is refusing; an operator debugging a dark
    feed needs to tell those three apart.
    """

    send: bool
    reason: str  # "ok" | "rate_limited" | "in_flight" | "backoff"


@dataclass(frozen=True)
class PushStats:
    """Honest tally of what happened to the frames offered so far."""

    sent: int
    dropped_rate_limited: int
    dropped_in_flight: int
    dropped_backoff: int
    failed: int
    consecutive_failures: int

    @property
    def offered(self) -> int:
        return (
            self.sent
            + self.dropped_rate_limited
            + self.dropped_in_flight
            + self.dropped_backoff
            + self.failed
        )


class FramePushPolicy:
    """Decides which frames to push, and when to stop trying.

    Time is always passed in rather than read from the clock, so the whole
    policy — including exponential backoff over a simulated multi-hour outage —
    is testable without sleeping.

    The caller drives a strict cycle: :meth:`offer` returns a decision, and
    every ``send=True`` decision must be closed out with exactly one
    :meth:`sent` or :meth:`failed`.
    """

    def __init__(
        self,
        target_fps: float = 10.0,
        *,
        base_backoff_s: float = 0.5,
        max_backoff_s: float = 30.0,
    ) -> None:
        if target_fps <= 0:
            raise ValueError(f"target_fps must be positive, got {target_fps!r}")
        if base_backoff_s < 0:
            raise ValueError("base_backoff_s must not be negative")
        if max_backoff_s < 0:
            raise ValueError("max_backoff_s must not be negative")

        self._interval_s = 1.0 / target_fps
        self._base_backoff_s = base_backoff_s
        self._max_backoff_s = max_backoff_s

        self._in_flight = False
        self._last_sent_at: float | None = None
        self._retry_not_before: float = 0.0

        self._sent = 0
        self._dropped_rate = 0
        self._dropped_in_flight = 0
        self._dropped_backoff = 0
        self._failed = 0
        self._consecutive_failures = 0

    # -- the cycle ----------------------------------------------------------

    def offer(self, now: float) -> PushDecision:
        """Offer a freshly captured frame.  Records the drop when refused.

        Checks run cheapest-consequence first: being early is normal, being
        busy is a link problem, being in backoff is a far-end problem.
        """
        if self._in_flight:
            self._dropped_in_flight += 1
            return PushDecision(send=False, reason="in_flight")

        if now < self._retry_not_before:
            self._dropped_backoff += 1
            return PushDecision(send=False, reason="backoff")

        if self._last_sent_at is not None and (now - self._last_sent_at) < self._interval_s:
            self._dropped_rate += 1
            return PushDecision(send=False, reason="rate_limited")

        self._in_flight = True
        return PushDecision(send=True, reason="ok")

    def sent(self, now: float) -> None:
        """Record a frame the operator accepted.  Clears any backoff."""
        if not self._in_flight:
            raise RuntimeError(
                "sent() without a preceding offer() that returned send=True; "
                "counting an unoffered frame would report throughput never achieved"
            )
        self._in_flight = False
        self._sent += 1
        self._last_sent_at = now
        self._consecutive_failures = 0
        self._retry_not_before = 0.0

    def failed(self, now: float) -> None:
        """Record a push the operator refused or the link dropped.

        Clearing ``_in_flight`` here is load-bearing: without it a single
        dropped connection wedges the pusher for the life of the process.
        """
        if not self._in_flight:
            raise RuntimeError(
                "failed() without a preceding offer() that returned send=True"
            )
        self._in_flight = False
        self._failed += 1
        self._consecutive_failures += 1
        self._retry_not_before = now + self._current_backoff()

    # -- introspection ------------------------------------------------------

    def backoff_remaining(self, now: float) -> float:
        """Seconds until the next attempt is permitted (0.0 when clear)."""
        return max(0.0, self._retry_not_before - now)

    @property
    def stats(self) -> PushStats:
        return PushStats(
            sent=self._sent,
            dropped_rate_limited=self._dropped_rate,
            dropped_in_flight=self._dropped_in_flight,
            dropped_backoff=self._dropped_backoff,
            failed=self._failed,
            consecutive_failures=self._consecutive_failures,
        )

    @property
    def healthy(self) -> bool:
        """True only once a frame has actually landed and none is failing now.

        Deliberately conservative — a pusher that has never delivered anything
        is not "starting up", it is not working.
        """
        return self._sent > 0 and self._consecutive_failures == 0

    # -- internal -----------------------------------------------------------

    def _current_backoff(self) -> float:
        """Exponential in the failure count, capped.

        The cap matters as much as the growth: uncapped doubling means a rig
        that was unreachable overnight sits idle for hours after the operator
        comes back.
        """
        exponent = max(0, self._consecutive_failures - 1)
        # Saturate the exponent before shifting so a long outage cannot
        # overflow into a float infinity on the way to being capped.
        if exponent > 32:
            return self._max_backoff_s
        return min(self._base_backoff_s * (2**exponent), self._max_backoff_s)
