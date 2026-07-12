# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Geofence engine — polygon-based zone monitoring with enter/exit detection.

Defines GeoZone (polygon regions on the tactical map) and GeofenceEngine
which tracks per-target zone membership and detects enter/exit transitions.
Uses ray-casting for point-in-polygon tests.

Events published to EventBus:
    geofence:enter  — target entered a zone
    geofence:exit   — target exited a zone
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field

logger = logging.getLogger("geofence")


@dataclass
class GeoZone:
    """A polygon zone on the tactical map."""

    zone_id: str
    name: str
    polygon: list[tuple[float, float]]  # ordered vertices
    zone_type: str = "monitored"  # "restricted", "monitored", "safe"
    alert_on_enter: bool = True
    alert_on_exit: bool = True
    enabled: bool = True
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "zone_id": self.zone_id,
            "name": self.name,
            "polygon": [list(p) for p in self.polygon],
            "zone_type": self.zone_type,
            "alert_on_enter": self.alert_on_enter,
            "alert_on_exit": self.alert_on_exit,
            "enabled": self.enabled,
            "created_at": self.created_at,
        }


@dataclass
class GeoEvent:
    """A geofence transition event."""

    event_id: str
    event_type: str  # "enter", "exit", "inside"
    target_id: str
    zone_id: str
    zone_name: str
    zone_type: str
    position: tuple[float, float]
    timestamp: float

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "target_id": self.target_id,
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "zone_type": self.zone_type,
            "position": list(self.position),
            "timestamp": self.timestamp,
        }


from tritium_lib.geo import point_in_polygon  # noqa: E402


class GeofenceEngine:
    """Tracks targets against polygon zones and detects enter/exit transitions.

    Thread-safe. Maintains per-target zone membership state so that
    transitions (enter/exit) are detected on each check() call.

    Args:
        event_bus: Optional event bus with a .publish(topic, data) method
            for broadcasting geofence events.
    """

    def __init__(self, event_bus=None) -> None:
        self._event_bus = event_bus
        self._lock = threading.Lock()
        self._zones: dict[str, GeoZone] = {}
        self._target_zones: dict[str, set[str]] = {}
        self._events: list[GeoEvent] = []
        self._max_events = 10000

    # ------------------------------------------------------------------
    # Zone CRUD
    # ------------------------------------------------------------------

    def add_zone(self, zone: GeoZone) -> GeoZone:
        """Add a zone. Returns the zone."""
        with self._lock:
            self._zones[zone.zone_id] = zone
        logger.info(f"Geofence zone added: {zone.name} ({zone.zone_id})")
        return zone

    def remove_zone(self, zone_id: str) -> bool:
        """Remove a zone by ID. Returns True if found and removed."""
        with self._lock:
            if zone_id not in self._zones:
                return False
            del self._zones[zone_id]
            for target_id in list(self._target_zones):
                self._target_zones[target_id].discard(zone_id)
                if not self._target_zones[target_id]:
                    del self._target_zones[target_id]
        logger.info(f"Geofence zone removed: {zone_id}")
        return True

    def get_zone(self, zone_id: str) -> GeoZone | None:
        """Get a zone by ID."""
        with self._lock:
            return self._zones.get(zone_id)

    def list_zones(self) -> list[GeoZone]:
        """Return all zones."""
        with self._lock:
            return list(self._zones.values())

    # ------------------------------------------------------------------
    # Target checking
    # ------------------------------------------------------------------

    def check(
        self, target_id: str, position: tuple[float, float]
    ) -> list[GeoEvent]:
        """Check a target position against all zones."""
        now = time.time()
        events: list[GeoEvent] = []

        with self._lock:
            prev_zones = self._target_zones.get(target_id, set()).copy()
            current_zones: set[str] = set()

            for zone_id, zone in self._zones.items():
                if not zone.enabled:
                    continue

                inside = point_in_polygon(position[0], position[1], zone.polygon)

                if inside:
                    current_zones.add(zone_id)

                    if zone_id not in prev_zones:
                        ev = GeoEvent(
                            event_id=uuid.uuid4().hex[:12],
                            event_type="enter",
                            target_id=target_id,
                            zone_id=zone_id,
                            zone_name=zone.name,
                            zone_type=zone.zone_type,
                            position=position,
                            timestamp=now,
                        )
                        events.append(ev)
                        self._record_event(ev)

                        if zone.alert_on_enter and self._event_bus is not None:
                            self._event_bus.publish("geofence:enter", ev.to_dict())
                    else:
                        events.append(
                            GeoEvent(
                                event_id=uuid.uuid4().hex[:12],
                                event_type="inside",
                                target_id=target_id,
                                zone_id=zone_id,
                                zone_name=zone.name,
                                zone_type=zone.zone_type,
                                position=position,
                                timestamp=now,
                            )
                        )

            exited_zones = prev_zones - current_zones
            for zone_id in exited_zones:
                zone = self._zones.get(zone_id)
                if zone is None:
                    continue
                ev = GeoEvent(
                    event_id=uuid.uuid4().hex[:12],
                    event_type="exit",
                    target_id=target_id,
                    zone_id=zone_id,
                    zone_name=zone.name,
                    zone_type=zone.zone_type,
                    position=position,
                    timestamp=now,
                )
                events.append(ev)
                self._record_event(ev)

                if zone.alert_on_exit and self._event_bus is not None:
                    self._event_bus.publish("geofence:exit", ev.to_dict())

            if current_zones:
                self._target_zones[target_id] = current_zones
            else:
                self._target_zones.pop(target_id, None)

        return events

    def get_target_zones(self, target_id: str) -> set[str]:
        """Get the set of zone IDs a target is currently inside."""
        with self._lock:
            return self._target_zones.get(target_id, set()).copy()

    def get_zone_occupants(self, zone_id: str) -> list[str]:
        """Get target IDs currently inside a given zone."""
        with self._lock:
            return [
                tid for tid, zones in self._target_zones.items()
                if zone_id in zones
            ]

    # ------------------------------------------------------------------
    # Cognition grounding
    # ------------------------------------------------------------------

    def zone_brief(self, occupant_resolver=None, max_zones: int = 6) -> str:
        """Concise operator-facing inventory of zones + live occupancy.

        Surfaces geofence state to the cognition layer (Amy / a Graphling /
        the operator) so a question like "what's in the restricted zone?" or
        "how many targets crossed the perimeter?" can be answered from grounded
        live data instead of a hallucination. This is the read-side companion to
        the ``geofence:enter`` / ``geofence:exit`` events: those are momentary
        alerts, this is the current state of the board.

        Args:
            occupant_resolver: optional ``callable(target_id) -> dict | None``
                returning at least ``{"alliance": str}`` (``"classification"``
                optional). When provided, each occupied zone is broken down by
                alliance, and a hostile inside a ``restricted`` zone is flagged
                as a BREACH — the operationally critical signal.
            max_zones: cap on zones listed (occupied + restricted first) to keep
                the brief short for a small chat model.

        Returns ``""`` when no zones are defined.
        """
        with self._lock:
            zones = [z for z in self._zones.values() if z.enabled]
            if not zones:
                return ""
            occ: dict[str, list[str]] = {z.zone_id: [] for z in zones}
            for tid, zids in self._target_zones.items():
                for zid in zids:
                    if zid in occ:
                        occ[zid].append(tid)

        total_inside = sum(len(v) for v in occ.values())
        occupied = sum(1 for v in occ.values() if v)

        # Order: occupied first, restricted-with-occupants ranked highest,
        # then by occupant count desc, then name for a stable tie-break.
        def _rank(z: GeoZone) -> tuple:
            n = len(occ[z.zone_id])
            restricted = z.zone_type == "restricted"
            return (n == 0, not (restricted and n), -n, z.name)

        zones_sorted = sorted(zones, key=_rank)

        breaches = 0
        zone_lines: list[str] = []
        for z in zones_sorted[:max_zones]:
            ids = occ[z.zone_id]
            ztype = z.zone_type.upper()
            if not ids:
                zone_lines.append(f"{ztype} '{z.name}': empty")
                continue

            if occupant_resolver is not None:
                alliance_counts: dict[str, int] = {}
                for tid in ids:
                    try:
                        info = occupant_resolver(tid) or {}
                    except Exception:
                        info = {}
                    al = info.get("alliance") or "unknown"
                    alliance_counts[al] = alliance_counts.get(al, 0) + 1
                order = ["hostile", "friendly", "neutral", "unknown"]
                ordered = [a for a in order if a in alliance_counts]
                ordered += [a for a in alliance_counts if a not in order]
                breakdown = ", ".join(
                    f"{alliance_counts[a]} {a}" for a in ordered
                )
                hostile_in = alliance_counts.get("hostile", 0)
                if z.zone_type == "restricted" and hostile_in:
                    breaches += 1
                    zone_lines.append(
                        f"{ztype} '{z.name}': {len(ids)} inside "
                        f"({breakdown}) -- BREACH"
                    )
                else:
                    zone_lines.append(
                        f"{ztype} '{z.name}': {len(ids)} inside ({breakdown})"
                    )
            else:
                zone_lines.append(f"{ztype} '{z.name}': {len(ids)} inside")

        header = (
            f"ZONES: {len(zones)} defined, {occupied} occupied, "
            f"{total_inside} target(s) inside"
        )
        if breaches:
            header += f" -- {breaches} BREACH(es)"
        if len(zones) > max_zones:
            zone_lines.append(f"(+{len(zones) - max_zones} more zone(s))")

        return "\n".join([header] + zone_lines)

    # ------------------------------------------------------------------
    # Event log
    # ------------------------------------------------------------------

    def get_events(
        self,
        limit: int = 100,
        zone_id: str | None = None,
        target_id: str | None = None,
        event_type: str | None = None,
    ) -> list[GeoEvent]:
        """Get recent geofence events, optionally filtered."""
        with self._lock:
            filtered = self._events
            if zone_id is not None:
                filtered = [e for e in filtered if e.zone_id == zone_id]
            if target_id is not None:
                filtered = [e for e in filtered if e.target_id == target_id]
            if event_type is not None:
                filtered = [e for e in filtered if e.event_type == event_type]
            return list(reversed(filtered[-limit:]))

    def _record_event(self, event: GeoEvent) -> None:
        """Append event to internal log (caller holds lock)."""
        self._events.append(event)
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]
