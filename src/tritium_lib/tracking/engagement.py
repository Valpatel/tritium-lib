# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""What a fired round leaves behind for the operator to see.

:mod:`tritium_lib.geo.hitscan` answers *where the ray went and what it
touched*.  That answer dies inside the connector that asked for it: a live
sim can grade a shot perfectly and the operator's map still shows nothing
happened.  This module is the missing half -- one shot becomes a **drawable
record**, and a bounded history of those records becomes an after-action
review.

Two decisions carry the module:

**A miss is still drawn.**  :class:`~tritium_lib.geo.hitscan.ShotResult`
leaves ``impact`` as ``None`` when nothing was hit, which is honest for
grading and useless for rendering -- a tracer that ends nowhere tells the
operator nothing about where the round went.  :meth:`ShotEvent.terminus_of`
therefore falls back to the point at the range gate along the aim, so every
shot has two endpoints and the map can show the near-miss that a bare
hit/miss boolean hides.  The same fallback covers the subtler case of a
geometric hit that the range gate refused: drawing to that impact point would
paint a round reaching a body the verdict says it never reached.

**The log is bounded.**  A turret firing at 5 Hz for an hour is 18,000
records, and the operator only ever looks at the recent ones.  An unbounded
list of them inside a long-lived Command Center is a slow leak, so the ring
drops the oldest rather than growing.

No engine, no simulator, no framework -- standard library only, so this
imports on a bare aarch64 brain alongside the geometry it records.
"""

from __future__ import annotations

import itertools
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from ..geo.hitscan import Muzzle, ShotResult

__all__ = ["EngagementLog", "ShotEvent"]

Vec3 = tuple[float, float, float]

# Monotonic and process-local: a counter, not a random id.  Two shots fired in
# the same millisecond must not collide, which a timestamp-derived id does.
_shot_counter = itertools.count(1)

DEFAULT_MAX_EVENTS = 500


def _muzzle_from_dict(data: dict) -> Muzzle:
    return Muzzle(
        east_m=float(data.get("east_m", 0.0)),
        north_m=float(data.get("north_m", 0.0)),
        up_m=float(data.get("up_m", 0.0)),
        heading_deg=float(data.get("heading_deg", 0.0)),
        elevation_deg=float(data.get("elevation_deg", 0.0)),
    )


@dataclass(frozen=True)
class ShotEvent:
    """One round, in the form the tactical map and an AAR both need.

    Deliberately a superset of what either consumer uses on its own: the map
    wants the two endpoints and the verdict, the review wants the range and
    the miss distance, and splitting them into two records would let the
    picture and the report disagree about the same round.
    """

    shot_id: str
    timestamp: float
    origin: Vec3
    terminus: Vec3
    hit: bool
    shooter_id: str | None = None
    target_id: str | None = None
    range_m: float | None = None
    miss_distance_m: float | None = None
    max_range_m: float = 0.0
    muzzle: Muzzle | None = field(default=None, repr=False)

    # --- construction -----------------------------------------------------

    @staticmethod
    def terminus_of(shot: ShotResult) -> Vec3:
        """The point where the tracer STOPS -- impact, or the range gate.

        ``shot.impact()`` is only populated on a hit that passed the range
        gate, which is exactly the case that needs no fallback.  Everything
        else -- a clean miss, a hit refused by the gate -- resolves to the
        aim vector scaled to ``max_range_m``, so the drawn line always agrees
        with the verdict printed beside it.
        """
        impact = shot.impact()
        if shot.hit and impact is not None:
            return impact
        ox, oy, oz = shot.muzzle.origin()
        ax, ay, az = shot.muzzle.direction()
        reach = float(shot.max_range_m)
        return (ox + ax * reach, oy + ay * reach, oz + az * reach)

    @classmethod
    def from_shot(
        cls,
        shot: ShotResult,
        shooter_id: str | None = None,
        timestamp: float | None = None,
    ) -> ShotEvent:
        """Wrap a resolved shot, adding only who fired it and when."""
        return cls(
            shot_id=f"shot_{next(_shot_counter)}",
            timestamp=time.time() if timestamp is None else float(timestamp),
            origin=shot.muzzle.origin(),
            terminus=cls.terminus_of(shot),
            hit=bool(shot.hit),
            shooter_id=shooter_id,
            target_id=shot.target_id,
            range_m=shot.range_m,
            miss_distance_m=shot.miss_distance_m,
            max_range_m=float(shot.max_range_m),
            muzzle=shot.muzzle,
        )

    @classmethod
    def from_payload(cls, payload: dict) -> ShotEvent:
        """Rebuild an event from a wire dict -- ``ShotResult.to_dict()`` plus
        optional ``shooter_id`` / ``timestamp``.

        A payload with no muzzle is refused rather than defaulted to the
        origin: a tracer drawn from (0, 0, 0) is not a degraded picture, it is
        a wrong one, and it would put a phantom line across the operator's map
        every time a connector sent a malformed frame.
        """
        raw_muzzle = payload.get("muzzle")
        if not isinstance(raw_muzzle, dict):
            raise ValueError("shot payload requires a 'muzzle' object")

        muzzle = _muzzle_from_dict(raw_muzzle)
        max_range = float(payload.get("max_range_m", 0.0))
        if not math.isfinite(max_range) or max_range < 0.0:
            raise ValueError(f"max_range_m must be finite and non-negative, got {max_range!r}")

        impact = payload.get("impact")
        shot = ShotResult(
            hit=bool(payload.get("hit", False)),
            muzzle=muzzle,
            max_range_m=max_range,
            target_id=payload.get("target_id"),
            range_m=payload.get("range_m"),
            impact_east_m=impact[0] if impact else None,
            impact_north_m=impact[1] if impact else None,
            impact_up_m=impact[2] if impact else None,
            miss_distance_m=payload.get("miss_distance_m"),
        )
        return cls.from_shot(
            shot,
            shooter_id=payload.get("shooter_id"),
            timestamp=payload.get("timestamp"),
        )

    # --- serialisation ----------------------------------------------------

    def to_dict(self) -> dict:
        """A JSON-safe record with BOTH endpoints, so a line can be drawn."""
        return {
            "shot_id": self.shot_id,
            "timestamp": self.timestamp,
            "origin": list(self.origin),
            "terminus": list(self.terminus),
            "hit": self.hit,
            "shooter_id": self.shooter_id,
            "target_id": self.target_id,
            "range_m": self.range_m,
            "miss_distance_m": self.miss_distance_m,
            "max_range_m": self.max_range_m,
        }


class EngagementLog:
    """A bounded, thread-safe history of fired rounds.

    Thread-safe because the producers and the consumer are genuinely
    different threads in the deployed shape: a connector ingests shots off a
    socket while the operator's map polls for them.  ``deque`` append is
    atomic under CPython, but ``recent()`` snapshots and ``stats()`` walks are
    not, and an unlocked walk during a burst of fire yields a report that
    counts a round twice or not at all.
    """

    def __init__(self, max_events: int = DEFAULT_MAX_EVENTS) -> None:
        if max_events < 1:
            raise ValueError(f"max_events must be >= 1, got {max_events!r}")
        self._events: deque[ShotEvent] = deque(maxlen=max_events)
        self._lock = threading.Lock()

    def record(self, payload: dict | ShotResult | ShotEvent) -> ShotEvent:
        """Append one round and return the stored event."""
        if isinstance(payload, ShotEvent):
            event = payload
        elif isinstance(payload, ShotResult):
            event = ShotEvent.from_shot(payload)
        else:
            event = ShotEvent.from_payload(payload)

        with self._lock:
            self._events.append(event)
        return event

    def recent(self, limit: int = 50, since: float | None = None) -> list[ShotEvent]:
        """Most-recent-first, optionally only rounds fired after ``since``.

        Newest first because the map draws the freshest tracers on top and a
        poller that asks for 20 wants the LAST 20, not the first 20 of a run
        that started an hour ago.
        """
        with self._lock:
            events = list(self._events)

        events.reverse()
        if since is not None:
            events = [e for e in events if e.timestamp > since]
        return events[: max(0, int(limit))]

    def stats(self) -> dict:
        """Shots, hits, and accuracy -- the AAR's headline numbers."""
        with self._lock:
            events = list(self._events)

        shots = len(events)
        hits = sum(1 for e in events if e.hit)
        return {
            "shots": shots,
            "hits": hits,
            "accuracy": (hits / shots) if shots else 0.0,
        }

    def clear(self) -> None:
        with self._lock:
            self._events.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)
