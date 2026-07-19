# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Sensor-rig bring-up -> Command Center registration.

A robot boots several sensor services at once (RGB, metric depth, the stereo
right eye, a LiDAR ring, the body itself).  Bringing them *up* is one problem
and connectors already solve it; making the **operator** see them is a second
one, and until now it was done by hand, one POST per sensor, from a docstring.

This module is the missing map: given what came up healthy, it says exactly
what the Command Center must be told — and, just as importantly, refuses to
report a rig as healthy when it isn't.

Two pure functions, no network, no Isaac, no framework:

  * :func:`registration_plan` — healthy sensors -> the ``RegistrationCall``
    list that puts them on the operator's wall.
  * :func:`summarize_bringup` — per-sensor outcomes -> a
    :class:`RigBringupReport` whose ``ok`` is trustworthy.

The caller owns the HTTP.  That split is deliberate: the decisions worth
testing are *which* calls and *whether it worked*, and neither needs a socket.

Both North Star halves: FUN — one command turns a sim body into an
instrumented unit whose eyes light up on the tactical map with nothing typed.
PRODUCTION — fleet sensor bring-up is *routinely partial* (one service wedges,
one port is taken), and an operator who is shown green on a half-dead rig is
worse off than one shown nothing.

Design notes that are load-bearing, not style
---------------------------------------------
* **An unready sensor is never registered.** A feed tile that can never render
  is worse than an absent one — it reads as a camera that is merely quiet.
* **LiDAR gets no camera-feed call.** It streams *sightings* (``POST
  /api/sighting {"source": "lidar"}``), not frames; registering it as a feed
  would pin a permanently black tile to the wall.
* **``attach_to`` is omitted, never null**, when the rig does not know the
  mount — a null would overwrite what the camera server advertises about its
  own mount via ``/status``.
* **An empty rig is not ``ok``.** ``all([])`` is ``True``, which is precisely
  how a bring-up in which *every* server died reports success.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

__all__ = [
    "RigSensor",
    "RegistrationCall",
    "RigBringupReport",
    "registration_plan",
    "SOURCES_PATH",
    "summarize_bringup",
    "OUTCOMES",
]

#: The Isaac auto-bind route: it probes the server's own ``/status`` and adopts
#: the mount it advertises, so the rig need not know the mount to bind one.
ISAAC_FEED_PATH = "/api/camera-feeds/isaac"

#: The generic route, used for push registrations.
#:
#: :data:`ISAAC_FEED_PATH` registers a source SC will *dial*, which is exactly
#: wrong when the sensor is on another machine: a kit binds its MJPEG server to
#: the render host's own loopback, so the address the rig would advertise is
#: unreachable by definition.  A push source inverts that — the sensor POSTs
#: frames to the operator — and it is registered here instead.
SOURCES_PATH = "/api/camera-feeds/sources"

#: Roles that carry pixels, and the ``stream`` each maps to on that route.
#: ``depth`` is ``depth16`` (metric uint16-mm), not the colormapped preview —
#: the colormap is for humans, the metric frame is what perception consumes.
_PIXEL_STREAMS: dict[str, str] = {
    "camera": "rgb",
    "depth": "depth16",
    "stereo_right": "right",
}

#: Roles that come up in a rig but carry no pixels — ``lidar`` (streams
#: sightings) and ``body`` (streams pose).  They are absent from
#: ``_PIXEL_STREAMS`` on purpose, and that absence is the whole rejection:
#: a role with no stream gets no feed, which is also the right answer for a
#: role this version has never heard of.
NON_PIXEL_ROLES = frozenset({"lidar", "body"})

#: Every outcome :func:`summarize_bringup` will score.  Anything else raises
#: rather than being silently counted as fine.
OUTCOMES = frozenset({"registered", "already_registered", "failed", "skipped"})

_OK_OUTCOMES = frozenset({"registered", "already_registered"})


@dataclass(frozen=True)
class RigSensor:
    """One sensor service the rig brought up (or tried to).

    ``ready`` is the health-probe verdict, not an intention: the rig sets it
    from the server's own ``/status`` (or TCP accept for the body).
    ``attach_to`` names the tracked target the sensor rides on, when the rig
    knows it; empty means "let the server's advertisement decide".
    """

    role: str
    host: str
    port: int
    ready: bool = False
    attach_to: str = ""
    source_id: str = ""

    def feed_source_id(self) -> str:
        """Stable, collision-free id for this sensor's feed.

        Depth and the right eye live on the *same* host:port as RGB, so the id
        must come from the stream, not the address, or the second registration
        would collide with the first.
        """
        if self.source_id:
            return self.source_id
        stream = _PIXEL_STREAMS.get(self.role, self.role)
        return f"isaac_{stream}"


@dataclass(frozen=True)
class RegistrationCall:
    """One HTTP call the caller should make against the Command Center."""

    method: str
    path: str
    payload: dict
    role: str = ""


@dataclass(frozen=True)
class RigBringupReport:
    """Scored outcome of a whole rig bring-up.

    ``ok`` is the only field callers should branch on, and it is deliberately
    conservative: it is ``True`` only when at least one sensor reached the
    operator and nothing failed.
    """

    registered: int
    already: int
    failed: int
    skipped: int
    detail: str

    @property
    def ok(self) -> bool:
        reached = self.registered + self.already
        return self.failed == 0 and reached > 0

    def __str__(self) -> str:  # pragma: no cover - human sugar
        state = "RIG ONLINE" if self.ok else "RIG DEGRADED"
        return f"{state}: {self.detail}"


def registration_plan(
    sensors: Iterable[RigSensor],
    *,
    detect: bool = True,
    push: bool = False,
) -> list[RegistrationCall]:
    """Healthy pixel sensors -> the registration calls that surface them.

    Order follows the input, so the same rig always yields the same plan and a
    bring-up is reproducible run to run.

    Sensors that are not ``ready``, and roles with no feed surface (LiDAR, the
    body), yield nothing — see the module docstring for why each of those is a
    refusal rather than an omission.

    ``push`` inverts the transport.  By default the plan registers feeds SC
    will dial; with ``push=True`` it registers sources that *accept* frames the
    sensor sends in.  Two consequences are deliberate, not oversights:

    * **No address survives into the payload.**  Carrying host/port would
      invite SC to pull from a loopback address on someone else's machine —
      the precise failure push mode exists to remove.
    * **No mount discovery.**  Discovery works by probing the camera server's
      ``/status``, which is unreachable for the same reason the stream is.  So
      a push registration binds to a body only when the rig was *told* the
      mount via ``attach_to``; it never claims a mount it could not have read.

    The ``source_id`` is identical in both modes on purpose — it is the join
    between the pusher and the source it feeds, and a mode-dependent id would
    make every pushed frame 404.
    """
    calls: list[RegistrationCall] = []
    for sensor in sensors:
        if not sensor.ready:
            continue
        stream = _PIXEL_STREAMS.get(sensor.role)
        if stream is None:
            continue

        payload: dict
        if push:
            payload = {
                "source_id": sensor.feed_source_id(),
                "source_type": "push",
                "detect": detect,
            }
        else:
            payload = {
                "mode": "mjpeg",
                "stream": stream,
                "host": sensor.host,
                "port": sensor.port,
                "source_id": sensor.feed_source_id(),
                "detect": detect,
                "discover": True,
            }
        # Present only when known: an explicit null would clobber the mount the
        # camera server advertises for itself (pull), and under push there is
        # nothing else that could supply one.
        if sensor.attach_to:
            payload["attach_to"] = sensor.attach_to

        calls.append(
            RegistrationCall(
                method="POST",
                path=SOURCES_PATH if push else ISAAC_FEED_PATH,
                payload=payload,
                role=sensor.role,
            )
        )
    return calls


def summarize_bringup(
    outcomes: Sequence[tuple[str, str]],
) -> RigBringupReport:
    """Per-sensor outcomes -> a report whose ``ok`` can be trusted.

    ``outcomes`` is a sequence of ``(source_id, outcome)`` pairs, where the
    outcome is one of :data:`OUTCOMES`.  An unrecognised outcome raises
    ``ValueError`` — scoring a string nobody defined as success is exactly how
    a degraded rig reports green.
    """
    counts = {name: 0 for name in OUTCOMES}
    failed_ids: list[str] = []
    skipped_ids: list[str] = []

    for source_id, outcome in outcomes:
        if outcome not in OUTCOMES:
            raise ValueError(
                f"unknown bring-up outcome {outcome!r} for {source_id!r}; "
                f"expected one of {sorted(OUTCOMES)}"
            )
        counts[outcome] += 1
        if outcome == "failed":
            failed_ids.append(source_id)
        elif outcome == "skipped":
            skipped_ids.append(source_id)

    reached = counts["registered"] + counts["already_registered"]
    parts = [f"{reached} sensor(s) on the map"]
    if failed_ids:
        parts.append("FAILED: " + ", ".join(failed_ids))
    if skipped_ids:
        parts.append("skipped (unhealthy): " + ", ".join(skipped_ids))
    if not outcomes:
        parts = ["no sensors registered — the rig reached the operator with nothing"]

    return RigBringupReport(
        registered=counts["registered"],
        already=counts["already_registered"],
        failed=counts["failed"],
        skipped=counts["skipped"],
        detail="; ".join(parts),
    )
