"""The live command surface a driven body listens on.

Every controller in this package so far has *produced* a twist — the waypoint
follower from a route, the teleop shaper from a stick.  Nothing has ever let a
body take a twist from *outside itself, mid-run*.  A simulated body's motion
was decided before it started moving: the Newton gait driver bakes a fixed
twist into generated source at build time, so a run is a recording, not a
robot.  This module is the seam that changes that.

It is deliberately transport-free.  It ingests *bytes* and answers *what to
command right now*, so the same object serves a UDP socket polled from a
physics step callback, a WebSocket in the Command Center, or a test handing it
literals.  Keeping the transport out is what lets the safety rules below be
unit-tested at all — they are the whole point of the module, and they are
exactly the rules that go untested when they live inside a socket loop.

Three properties carry the safety weight:

* **Garbage is not liveness.**  A rejected packet never refreshes the
  watchdog.  If corrupt traffic counted as a heartbeat, a sender emitting
  nothing but noise would pin the body at its last good command forever —
  precisely the runaway the watchdog exists to stop.
* **Old is not new.**  Datagrams reorder.  A late-arriving earlier command
  must not rewind the body, and being handed one is not evidence the sender
  is still alive *now*.
* **A restarted sender is adopted, not locked out.**  Strict monotonicity
  alone bricks a body: the operator's teleop process restarts, its counter
  returns to zero, and a receiver holding a high sequence discards every
  packet for the rest of the run.  A large *backwards* jump is a new session;
  real reordering spans a few packets, never thousands.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from tritium_lib.control.teleop import TeleopWatchdog
from tritium_lib.control.waypoint_follower import TwistCommand

__all__ = ["CommandLimits", "CommandLink", "SESSION_RESET_GAP"]


#: A backwards sequence jump at least this large is read as a restarted
#: sender rather than a reordered datagram.  Network reordering is a
#: small-window phenomenon; a counter that fell by a thousand did not
#: arrive late, it started over.
SESSION_RESET_GAP = 1000


@dataclass(frozen=True)
class CommandLimits:
    """The body's envelope, enforced on arrival.

    A sender is never trusted to respect it — clamping here means one bad
    operator client cannot command a speed the body cannot survive.
    """

    max_linear_mps: float = 1.0
    max_angular_rps: float = 2.0

    def clamp(self, twist: TwistCommand) -> TwistCommand:
        return TwistCommand(
            linear_mps=_clamp(twist.linear_mps, self.max_linear_mps),
            angular_rps=_clamp(twist.angular_rps, self.max_angular_rps),
        )


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


class CommandLink:
    """Decodes inbound command frames into the twist to command this tick.

    Feed it every datagram that arrives with :meth:`ingest`; ask it what to do
    every control tick with :meth:`poll`.  The two run at completely different
    rates on purpose — a physics loop steps far faster than any link delivers,
    so ``poll`` holding the last accepted command is the normal case, not a
    degraded one.
    """

    def __init__(
        self,
        limits: CommandLimits | None = None,
        timeout_s: float | None = 0.5,
    ) -> None:
        self.limits = limits or CommandLimits()
        self._watchdog = TeleopWatchdog(timeout_s=timeout_s)
        self._last_seq: int | None = None
        self.accepted = 0
        self.rejected = 0

    # ------------------------------------------------------------ ingest

    def ingest(self, payload: bytes | str, now_s: float) -> bool:
        """Offer one raw frame. Returns whether it was accepted.

        Never raises on bad input: this sits directly behind a socket, and a
        malformed packet is an expected event on any real link, not an error
        condition worth taking the body down for.
        """
        twist = self._decode(payload)
        if twist is None:
            self.rejected += 1
            return False
        self._watchdog.feed(self.limits.clamp(twist), now_s)
        self.accepted += 1
        return True

    def _decode(self, payload: bytes | str) -> TwistCommand | None:
        try:
            if isinstance(payload, bytes):
                payload = payload.decode("utf-8")
            frame = json.loads(payload)
        except (UnicodeDecodeError, ValueError):
            return None
        if not isinstance(frame, dict) or frame.get("cmd") != "twist":
            return None

        try:
            linear = float(frame["linear_mps"])
            angular = float(frame["angular_rps"])
        except (KeyError, TypeError, ValueError):
            return None
        # NaN through the stride mixer poisons every joint target downstream,
        # and it propagates silently — nothing raises, the body just stops
        # making sense.  Reject it at the boundary.
        if not (math.isfinite(linear) and math.isfinite(angular)):
            return None

        if not self._sequence_ok(frame.get("seq")):
            return None
        return TwistCommand(linear_mps=linear, angular_rps=angular)

    def _sequence_ok(self, seq) -> bool:
        """Ordering gate. A frame with no ``seq`` is always in order.

        Omitting the counter is a legitimate choice for a simple sender — it
        forfeits reorder protection while keeping liveness and clamping,
        rather than being refused outright.
        """
        if seq is None:
            return True
        try:
            seq = int(seq)
        except (TypeError, ValueError):
            return False
        if self._last_seq is not None:
            if seq <= self._last_seq and seq > self._last_seq - SESSION_RESET_GAP:
                return False
        self._last_seq = seq
        return True

    # -------------------------------------------------------------- poll

    def poll(self, now_s: float) -> TwistCommand:
        """The twist to command right now, stopping if the link has gone quiet."""
        return self._watchdog.poll(now_s)

    @property
    def expired(self) -> bool:
        """True until the first frame is accepted, and never again."""
        return self._watchdog.expired
