# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Indoor positioning via WiFi/BLE RSSI fingerprinting.

Provides location fingerprinting for tracking targets inside buildings
where GPS is unavailable. The workflow:

1. **Survey phase** — walk through a building recording RSSI from visible
   access points / BLE beacons at known reference positions. Each recording
   becomes a :class:`Fingerprint` stored in a :class:`FingerprintDB`.

2. **Position phase** — a live device reports current RSSI readings.
   :class:`PositionEstimator` finds the *k* closest reference fingerprints
   (k-nearest-neighbour in RSSI space) and returns a weighted centroid as
   the estimated position.

3. **Zone mapping** — :class:`ZoneMapper` clusters reference positions into
   named zones (lobby, office-A, server-room, ...) so that position estimates
   can be resolved to human-readable locations.

4. **Floor plans** — :class:`FloorPlan` provides a lightweight spatial model
   of rooms, corridors, and doors for path-based reasoning and zone
   containment checks.

Algorithms are pure Python (stdlib ``math`` only, no numpy). RSSI distance
metrics reuse conventions from :mod:`tritium_lib.signals`.

Usage::

    from tritium_lib.indoor import (
        Fingerprint,
        FingerprintDB,
        PositionEstimator,
        FloorPlan,
        ZoneMapper,
    )

    # Survey
    db = FingerprintDB(building_id="hq")
    db.add(Fingerprint(x=1.0, y=2.0, floor=0,
                       rssi={"ap1": -45, "ap2": -67, "ap3": -80}))
    db.add(Fingerprint(x=5.0, y=2.0, floor=0,
                       rssi={"ap1": -70, "ap2": -42, "ap3": -55}))

    # Locate
    estimator = PositionEstimator(db, k=3)
    pos = estimator.estimate({"ap1": -50, "ap2": -60, "ap3": -75})
    print(pos.x, pos.y, pos.confidence)
"""

from .fingerprint import Fingerprint, FingerprintDB
from .estimator import PositionEstimator, PositionResult
from .floorplan import FloorPlan, Room, Door, RoomType
from .zone_mapper import ZoneMapper, Zone

__all__ = [
    # Core fingerprinting
    "Fingerprint",
    "FingerprintDB",
    # Position estimation
    "PositionEstimator",
    "PositionResult",
    # Floor plan
    "FloorPlan",
    "Room",
    "Door",
    "RoomType",
    # Zone mapping
    "ZoneMapper",
    "Zone",
]
