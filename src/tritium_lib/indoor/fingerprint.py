# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""RSSI location fingerprints and fingerprint database.

A :class:`Fingerprint` captures RSSI measurements from multiple access
points (WiFi BSSIDs or BLE beacon MACs) at a known physical position
inside a building. A :class:`FingerprintDB` is the collection of all
reference fingerprints for a building, used by the position estimator.

Pure Python — no numpy required.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------

@dataclass
class Fingerprint:
    """RSSI measurements from multiple APs at a known reference position.

    Attributes:
        x: X-coordinate in local metres (east = positive).
        y: Y-coordinate in local metres (north = positive).
        floor: Floor level (0 = ground floor).
        rssi: Mapping of AP identifier (BSSID / beacon MAC) to RSSI in dBm.
        label: Optional human-readable label for this position.
        fingerprint_id: Unique ID (auto-generated UUID4 if not provided).
        timestamp: Collection time as epoch seconds.
    """
    x: float
    y: float
    floor: int = 0
    rssi: dict[str, float] = field(default_factory=dict)
    label: str = ""
    fingerprint_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)

    # -- Metrics ------------------------------------------------------------

    @property
    def ap_count(self) -> int:
        """Number of access points in this fingerprint."""
        return len(self.rssi)

    @property
    def mean_rssi(self) -> float:
        """Mean RSSI across all APs, or -100 if empty."""
        if not self.rssi:
            return -100.0
        return sum(self.rssi.values()) / len(self.rssi)

    @property
    def strongest_ap(self) -> Optional[str]:
        """AP identifier with the strongest (highest) RSSI, or None."""
        if not self.rssi:
            return None
        return max(self.rssi, key=self.rssi.get)  # type: ignore[arg-type]

    # -- RSSI distance ------------------------------------------------------

    def rssi_distance(self, other: dict[str, float]) -> float:
        """Euclidean distance in RSSI-space between this fingerprint and a
        live observation.

        Only APs present in *both* this fingerprint and ``other`` contribute.
        Missing APs are ignored (not penalised) so that partial scans still
        work. If no APs overlap, returns ``float('inf')``.

        Args:
            other: Live RSSI mapping {ap_id: rssi_dbm}.

        Returns:
            Euclidean distance in dBm-space (lower = more similar).
        """
        common_aps = set(self.rssi.keys()) & set(other.keys())
        if not common_aps:
            return float("inf")
        sq_sum = sum(
            (self.rssi[ap] - other[ap]) ** 2
            for ap in common_aps
        )
        return math.sqrt(sq_sum)

    def weighted_rssi_distance(self, other: dict[str, float],
                                default_rssi: float = -100.0) -> float:
        """RSSI distance that penalises missing APs.

        APs present in one side but not the other use ``default_rssi``
        (a very weak signal) as the stand-in value. This avoids rewarding
        fingerprints that simply have fewer APs.

        Args:
            other: Live RSSI mapping.
            default_rssi: Substitute RSSI for missing APs (default -100 dBm).

        Returns:
            Euclidean distance in dBm-space.
        """
        all_aps = set(self.rssi.keys()) | set(other.keys())
        if not all_aps:
            return float("inf")
        sq_sum = sum(
            (self.rssi.get(ap, default_rssi) - other.get(ap, default_rssi)) ** 2
            for ap in all_aps
        )
        return math.sqrt(sq_sum)

    # -- Serialisation ------------------------------------------------------

    def to_dict(self) -> dict:
        """Export as a JSON-serialisable dict."""
        return {
            "fingerprint_id": self.fingerprint_id,
            "x": self.x,
            "y": self.y,
            "floor": self.floor,
            "rssi": dict(self.rssi),
            "label": self.label,
            "timestamp": self.timestamp,
            "ap_count": self.ap_count,
            "mean_rssi": round(self.mean_rssi, 2),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Fingerprint:
        """Reconstruct from a dict (inverse of ``to_dict``)."""
        return cls(
            x=float(data["x"]),
            y=float(data["y"]),
            floor=int(data.get("floor", 0)),
            rssi={str(k): float(v) for k, v in data.get("rssi", {}).items()},
            label=str(data.get("label", "")),
            fingerprint_id=str(data.get("fingerprint_id", uuid.uuid4().hex[:12])),
            timestamp=float(data.get("timestamp", time.time())),
        )


# ---------------------------------------------------------------------------
# FingerprintDB
# ---------------------------------------------------------------------------

class FingerprintDB:
    """Database of reference fingerprints for a building.

    Stores :class:`Fingerprint` objects and provides lookup, filtering,
    and nearest-neighbour search by RSSI distance.

    Args:
        building_id: Identifier for the building this DB covers.
    """

    def __init__(self, building_id: str = "default") -> None:
        self.building_id = building_id
        self._fingerprints: list[Fingerprint] = []
        self._by_id: dict[str, Fingerprint] = {}

    # -- Mutation -----------------------------------------------------------

    def add(self, fp: Fingerprint) -> None:
        """Add a reference fingerprint to the database.

        If a fingerprint with the same ``fingerprint_id`` already exists,
        it is replaced.
        """
        existing = self._by_id.get(fp.fingerprint_id)
        if existing is not None:
            self._fingerprints.remove(existing)
        self._fingerprints.append(fp)
        self._by_id[fp.fingerprint_id] = fp

    def add_many(self, fingerprints: list[Fingerprint]) -> int:
        """Bulk-add fingerprints. Returns the count added."""
        for fp in fingerprints:
            self.add(fp)
        return len(fingerprints)

    def remove(self, fingerprint_id: str) -> bool:
        """Remove a fingerprint by ID. Returns True if it existed."""
        fp = self._by_id.pop(fingerprint_id, None)
        if fp is not None:
            self._fingerprints.remove(fp)
            return True
        return False

    def clear(self) -> None:
        """Remove all fingerprints."""
        self._fingerprints.clear()
        self._by_id.clear()

    # -- Queries ------------------------------------------------------------

    @property
    def count(self) -> int:
        """Number of fingerprints stored."""
        return len(self._fingerprints)

    def get(self, fingerprint_id: str) -> Optional[Fingerprint]:
        """Look up a fingerprint by ID."""
        return self._by_id.get(fingerprint_id)

    def all(self) -> list[Fingerprint]:
        """Return a copy of all fingerprints."""
        return list(self._fingerprints)

    def filter_by_floor(self, floor: int) -> list[Fingerprint]:
        """Return fingerprints on a specific floor."""
        return [fp for fp in self._fingerprints if fp.floor == floor]

    def get_all_aps(self) -> set[str]:
        """Return the set of all AP identifiers across all fingerprints."""
        aps: set[str] = set()
        for fp in self._fingerprints:
            aps.update(fp.rssi.keys())
        return aps

    def get_floors(self) -> list[int]:
        """Return sorted list of distinct floor levels."""
        return sorted({fp.floor for fp in self._fingerprints})

    # -- Nearest neighbour search -------------------------------------------

    def find_nearest(
        self,
        live_rssi: dict[str, float],
        k: int = 3,
        floor: Optional[int] = None,
        use_weighted: bool = False,
        max_distance: float = float("inf"),
    ) -> list[tuple[Fingerprint, float]]:
        """Find the *k* nearest reference fingerprints to a live observation.

        Args:
            live_rssi: Current RSSI mapping {ap_id: rssi_dbm}.
            k: Number of neighbours to return.
            floor: If set, restrict to fingerprints on this floor.
            use_weighted: If True, use weighted distance (penalise missing APs).
            max_distance: Exclude matches farther than this RSSI distance.

        Returns:
            List of (fingerprint, distance) tuples, sorted by distance
            ascending. Length is min(k, total matches within max_distance).
        """
        candidates = self._fingerprints
        if floor is not None:
            candidates = [fp for fp in candidates if fp.floor == floor]

        scored: list[tuple[Fingerprint, float]] = []
        for fp in candidates:
            if use_weighted:
                dist = fp.weighted_rssi_distance(live_rssi)
            else:
                dist = fp.rssi_distance(live_rssi)
            if dist <= max_distance:
                scored.append((fp, dist))

        scored.sort(key=lambda pair: pair[1])
        return scored[:k]

    # -- Serialisation ------------------------------------------------------

    def to_dict(self) -> dict:
        """Export the database as a JSON-serialisable dict."""
        return {
            "building_id": self.building_id,
            "count": self.count,
            "floors": self.get_floors(),
            "aps": sorted(self.get_all_aps()),
            "fingerprints": [fp.to_dict() for fp in self._fingerprints],
        }

    @classmethod
    def from_dict(cls, data: dict) -> FingerprintDB:
        """Reconstruct from a dict (inverse of ``to_dict``)."""
        db = cls(building_id=data.get("building_id", "default"))
        for fp_data in data.get("fingerprints", []):
            db.add(Fingerprint.from_dict(fp_data))
        return db

    def get_status(self) -> dict:
        """Return a summary status dict."""
        return {
            "building_id": self.building_id,
            "fingerprint_count": self.count,
            "floor_count": len(self.get_floors()),
            "ap_count": len(self.get_all_aps()),
            "floors": self.get_floors(),
        }
