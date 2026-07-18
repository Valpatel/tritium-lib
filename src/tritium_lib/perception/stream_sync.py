# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Approximate-time pairing for independently-arriving sensor streams.

A depth camera is really two (or three) streams — RGB, depth, sometimes a
second eye — that arrive on separate connections with separate latencies.
The naive pairing is "latest of each", and it is wrong the moment the camera
or the subject moves: the detector runs on one instant and the range is read
from another, so the bounding box samples depth pixels that belong to a
different moment.  The failure is silent and confident — a real range, on the
wrong object, placed on the operator's tactical map.

This was observed live: a `Depth16Source` pulling from Isaac Sim on the RTX
4090 while the camera orbited produced a bbox visibly offset from the same
object in the depth frame.

The fix is the one the robotics world already standardised —
ROS ``message_filters::ApproximateTimeSynchronizer``: buffer a few frames per
stream, emit the combination whose timestamps are closest together, and
**refuse to emit at all** when the best available combination is still
further apart than the caller's skew budget.  Refusing is the important half:
a dropped frame costs one cycle, a misregistered frame costs a wrong contact.

Pure stdlib.  No numpy, no ROS, no camera — this is a timing algorithm, so it
unit-tests with strings and floats and runs anywhere the fleet runs.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

__all__ = ["ApproximateTimeSync", "SyncedFrames"]

#: Frames buffered per stream by default. Enough to absorb a few frames of
#: jitter between channels; small enough that a stalled consumer cannot grow
#: memory. Matches the order of ROS's default message_filters queue size.
DEFAULT_DEPTH = 8

#: Default skew budget in seconds. At 10 fps one frame period is 100 ms, so
#: 50 ms keeps the pair inside half a frame — tight enough that a walking
#: subject (~1.4 m/s) moves under 7 cm between the two channels.
DEFAULT_MAX_SKEW_S = 0.05


@dataclass(frozen=True)
class SyncedFrames:
    """One time-aligned set of frames, plus the skew it was aligned to.

    Attributes:
        values: Stream key -> payload, one entry per declared key.
        stamps: Stream key -> timestamp of the chosen payload.
        skew_s: Full span between the earliest and latest chosen stamp.
            Reported so a caller can degrade rather than trust blindly — a
            pair inside the budget but near it is worth logging.
    """

    values: dict
    stamps: dict
    skew_s: float


class ApproximateTimeSync:
    """Pair N streams by nearest timestamp, refusing pairs beyond a budget.

    Thread-safe: the producing threads call :meth:`push` and the consumer
    calls :meth:`pair`, which is exactly how a multi-channel HTTP camera
    source is shaped.

    Example::

        sync = ApproximateTimeSync(max_skew_s=0.05)
        sync.push("rgb", frame, time.monotonic())      # RGB thread
        sync.push("depth", depth, time.monotonic())    # depth thread
        pair = sync.pair()                             # consumer
        if pair is not None:
            run_detector(pair.values["rgb"], pair.values["depth"])

    Args:
        max_skew_s: Largest acceptable span between the earliest and latest
            frame in a pair. Beyond this, :meth:`pair` returns ``None``.
        keys: Stream names to pair. Defaults to ``("rgb", "depth")``.
        depth: Frames buffered per stream.
    """

    def __init__(
        self,
        max_skew_s: float = DEFAULT_MAX_SKEW_S,
        keys: tuple = ("rgb", "depth"),
        depth: int = DEFAULT_DEPTH,
    ) -> None:
        self.max_skew_s = float(max_skew_s)
        self.keys = tuple(keys)
        self._buf = {k: deque(maxlen=max(1, int(depth))) for k in self.keys}
        self._lock = threading.Lock()

        #: Pairs successfully emitted.
        self.paired = 0
        #: Pair attempts refused because the best combination exceeded budget.
        self.rejected = 0
        #: Skew of the most recent emitted pair, for operator display.
        self.last_skew_s: Optional[float] = None

    # ------------------------------------------------------------ producers

    def push(self, key: str, value: Any, stamp: float) -> None:
        """Record a frame. Unknown keys, ``None`` payloads and non-finite
        timestamps are dropped rather than raised — one misconfigured or
        malformed channel must never kill a feed loop."""
        if key not in self._buf or value is None:
            return
        try:
            ts = float(stamp)
        except (TypeError, ValueError):
            return
        if not math.isfinite(ts):
            return
        with self._lock:
            self._buf[key].append((ts, value))

    # ------------------------------------------------------------ consumer

    def pair(self) -> Optional[SyncedFrames]:
        """Newest time-aligned set within budget, or ``None``.

        **Consuming**, and newest-first rather than smallest-skew-first. Both
        choices exist to prevent the same failure: if one stream stalls, a
        smallest-skew search over a non-destructive buffer keeps re-emitting
        the last good pair — serving a frame that is seconds old as if it
        were current, with a *flattering* skew number attached. Consuming the
        frames an emitted pair used, and preferring the newest qualifying
        pair, means this returns fresh data or nothing at all.

        ``None`` is a normal result, not an error: it means "no aligned frame
        yet", and the caller should simply try again on the next tick.
        """
        with self._lock:
            if any(not self._buf[k] for k in self.keys):
                return None
            snapshot = {k: list(self._buf[k]) for k in self.keys}

            # Anchor on each candidate of the first stream, newest first, and
            # take from every other stream the frame nearest that anchor. The
            # first anchor that lands inside the budget wins. With a bounded
            # buffer this is a handful of comparisons — the exhaustive search
            # ROS approximates is not worth the complexity at this depth.
            anchor_key = self.keys[0]
            best: Optional[SyncedFrames] = None
            for a_ts, a_val in reversed(snapshot[anchor_key]):
                values = {anchor_key: a_val}
                stamps = {anchor_key: a_ts}
                for k in self.keys[1:]:
                    ts, val = min(snapshot[k], key=lambda p: abs(p[0] - a_ts))
                    values[k] = val
                    stamps[k] = ts
                skew = max(stamps.values()) - min(stamps.values())
                if skew <= self.max_skew_s:
                    best = SyncedFrames(values=values, stamps=stamps,
                                        skew_s=skew)
                    break

            if best is None:
                self.rejected += 1
                return None

            # Consume: drop every frame at or before the ones just emitted so
            # the next call cannot go backwards in time.
            for k in self.keys:
                cutoff = best.stamps[k]
                self._buf[k] = deque(
                    (p for p in self._buf[k] if p[0] > cutoff),
                    maxlen=self._buf[k].maxlen,
                )

            self.paired += 1
            self.last_skew_s = best.skew_s
            return best

    # ------------------------------------------------------------- display

    def stats(self) -> dict:
        """Counters for ``to_dict()`` / operator display.

        Read ``reject_rate`` in context: a consumer polling faster than the
        camera publishes will pair a fresh frame against the previous cycle's
        leftover on the other channel, which is correctly refused. At 5 fps
        that alone puts the rate around 0.5 with nothing wrong. What matters
        is ``last_skew_s`` — if that sits near ``max_skew_s``, the channels
        really are drifting apart.
        """
        total = self.paired + self.rejected
        return {
            "paired": self.paired,
            "rejected": self.rejected,
            "reject_rate": round(self.rejected / total, 4) if total else 0.0,
            "last_skew_s": (round(self.last_skew_s, 4)
                            if self.last_skew_s is not None else None),
            "max_skew_s": self.max_skew_s,
        }
