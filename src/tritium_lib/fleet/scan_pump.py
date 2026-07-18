# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""LiDAR sweep -> Command Center sighting: the forwarding decision.

:mod:`tritium_lib.fleet.sensor_rig` states that LiDAR "streams *sightings*,
not frames", and gives it no camera-feed registration on that basis.  That
sentence described a path nobody had built: the rig's LiDAR came up healthy,
answered ``/status``, and reached the operator's tactical map never.

This module is the decision half of the pump that closes it.  Given a sweep
polled off a ``/scan`` endpoint it says whether that sweep should be POSTed to
``/api/sighting {"source": "lidar"}``, and with what payload.  The caller owns
the socket, exactly as in ``sensor_rig`` — what is worth testing here is
*which sweeps are worth sending*, and that needs no network.

Why the refusals are the substance
----------------------------------
Forwarding every polled sweep is a two-line loop, and it is wrong:

* **A stopped LiDAR keeps answering.** Its ``/scan`` serves the last sweep it
  ever took, forever.  Forwarded on a timer, that refreshes every derived
  track's ``last_seen`` on every poll, so a dead sensor renders on the
  operator's map as a live contact against a static wall.  Refusing an
  unchanged sweep is what makes a dead LiDAR *look* dead.
* **An all-no-return sweep carries no information either way.** An open field
  and an unplugged sensor produce byte-identical arrays of ``range_max``, so
  forwarding one buys no obstacle and grants a dead sensor a heartbeat.
* **A wedged Command Center must not be hammered.** Consecutive POST failures
  trip a breaker rather than retrying a sweep per poll indefinitely.

Both North Star halves: FUN — the rig's LiDAR ring finally paints obstacles on
the tactical map, so a body walking a scene leaves contacts behind it instead
of an empty grid.  PRODUCTION — a scanner that has stopped scanning is the
single most common LiDAR failure in the field, and it is invisible to every
health check that only asks whether the service answers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["ScanDecision", "ScanPump", "REFUSALS"]

#: Every reason a sweep can be refused.  Named so a caller can log or count
#: them without matching on free text.
REFUSALS = ("malformed", "stale", "no_returns", "tripped")

#: A range is a *return* only if it is short of the sensor's own no-return
#: ceiling by more than this.  Float noise across the wire (and the resample
#: clamp in ``lidar_server``) lands beams a hair under ``range_max``.
_NO_RETURN_EPS_M = 1e-6


@dataclass(frozen=True)
class ScanDecision:
    """Whether one polled sweep should reach the Command Center.

    ``payload`` is the ``/api/sighting`` body when ``forward`` is True, and
    ``None`` otherwise — a refused sweep has no body by construction, so a
    caller cannot accidentally POST one.
    """

    forward: bool
    reason: str
    payload: dict[str, Any] | None = None


class ScanPump:
    """Turns polled ``/scan`` sweeps into lidar sighting decisions.

    Stateful on purpose: staleness and the failure run are both properties of
    the *sequence* of sweeps, not of any one of them.  One pump per LiDAR —
    ids are namespaced per ``lidar_id`` downstream, and sharing a pump between
    two sensors would let one sensor's sweep mark the other's as stale.
    """

    def __init__(self, lidar_id: str, sensor_x: float = 0.0,
                 sensor_y: float = 0.0, sensor_yaw_deg: float = 0.0,
                 max_failures: int = 3) -> None:
        self.lidar_id = str(lidar_id)
        self.max_failures = int(max_failures)
        self._x = float(sensor_x)
        self._y = float(sensor_y)
        self._yaw = float(sensor_yaw_deg)
        self._last_ranges: tuple[float, ...] | None = None
        self._failure_run = 0
        self._forwarded = 0
        self._accepted = 0
        self._refusals: dict[str, int] = {r: 0 for r in REFUSALS}

    # ------------------------------------------------------------------ #
    # Pose — per sweep, because a body-mounted LiDAR moves
    # ------------------------------------------------------------------ #

    def set_sensor_pose(self, x: float, y: float, yaw_deg: float) -> None:
        """Update where the sensor is before offering the next sweep.

        A mast-mounted scanner never calls this; one riding a walking Go2
        calls it every sweep, and a pump that baked the pose in at
        construction would pile every obstacle at the body's start position.
        """
        self._x, self._y, self._yaw = float(x), float(y), float(yaw_deg)

    # ------------------------------------------------------------------ #
    # The decision
    # ------------------------------------------------------------------ #

    def offer(self, scan: dict | None) -> ScanDecision:
        """Score one polled sweep.  Never raises on bad input."""
        if self.tripped:
            return self._refuse("tripped")

        ranges = self._clean_ranges(scan)
        if ranges is None:
            return self._refuse("malformed")

        if self._last_ranges is not None and ranges == self._last_ranges:
            # Deliberately NOT recorded as the new baseline: a sensor stuck on
            # one sweep must stay refused, not alternate forward/stale.
            return self._refuse("stale")

        range_max = _as_float(scan.get("range_max"), 0.0)
        if range_max > 0.0 and all(r >= range_max - _NO_RETURN_EPS_M
                                   for r in ranges):
            self._last_ranges = ranges
            return self._refuse("no_returns")

        self._last_ranges = ranges
        self._forwarded += 1
        return ScanDecision(True, "forward", self._payload(scan, ranges))

    def record_result(self, ok: bool) -> None:
        """Report what the POST of the last forwarded sweep did.

        A *run* of failures trips the breaker; an isolated one does not.  An
        intermittently slow Command Center is normal operation, and a pump
        that gave up on the first timeout would need a human to restart it.
        """
        if ok:
            self._accepted += 1
            self._failure_run = 0
        else:
            self._failure_run += 1

    @property
    def tripped(self) -> bool:
        return self._failure_run >= self.max_failures

    def stats(self) -> dict[str, Any]:
        """Counters for the rig's own bring-up report."""
        return {
            "lidar_id": self.lidar_id,
            "forwarded": self._forwarded,
            "accepted": self._accepted,
            "refused": sum(self._refusals.values()),
            "refusals": dict(self._refusals),
            "tripped": self.tripped,
        }

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _refuse(self, reason: str) -> ScanDecision:
        self._refusals[reason] = self._refusals.get(reason, 0) + 1
        return ScanDecision(False, reason, None)

    @staticmethod
    def _clean_ranges(scan: dict | None) -> tuple[float, ...] | None:
        if not isinstance(scan, dict):
            return None
        raw = scan.get("ranges")
        if not raw:
            return None
        try:
            return tuple(float(r) for r in raw)
        except (TypeError, ValueError):
            return None

    def _payload(self, scan: dict,
                 ranges: tuple[float, ...]) -> dict[str, Any]:
        """The ``/api/sighting`` body for a forwarded sweep.

        The scan's own geometry is copied through rather than defaulted: the
        tracker converts polar -> world with exactly these fields, so a
        payload missing ``angle_min`` places a sweep taken over [-pi, pi) as
        though it were taken over [0, 2pi) — every obstacle half a turn out.
        """
        payload: dict[str, Any] = {
            "source": "lidar",
            "lidar_id": self.lidar_id,
            "ranges": list(ranges),
            "sensor_x": self._x,
            "sensor_y": self._y,
            "sensor_yaw_deg": self._yaw,
        }
        for key in ("angle_min", "angle_increment", "range_min", "range_max"):
            if key in scan:
                payload[key] = _as_float(scan[key], 0.0)
        return payload


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
