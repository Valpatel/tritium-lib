# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Data exchange utilities for Tritium — export and import targets, dossiers,
events, and zones in JSON, CSV, and GeoJSON formats.

DEPRECATED (Gap-fix D D-4, 2026-04-29): no production consumers.
Only the dedicated tests (``test_data_exchange.py``,
``test_end_to_end_pipeline.py``, ``test_full_integration.py``)
import ``TritiumExporter``/``TritiumImporter``.  The dossier export
endpoints under ``/api/dossier/*`` and the heatmap GeoJSON endpoint
serve the live operator path.

TODO: delete this package and its dedicated tests once the
integration tests can be rewritten without the Exporter/Importer stage.


Designed for:
  - Sharing data between Tritium instances (federation)
  - Archiving operational data for post-action review
  - Exporting to analysis tools (CSV for pandas/Excel, GeoJSON for QGIS)
  - Incremental sync (export only records updated since a timestamp)

Usage::

    from tritium_lib.store import TargetStore, DossierStore, EventStore
    from tritium_lib.data_exchange import TritiumExporter, TritiumImporter

    # Export
    exporter = TritiumExporter(target_store=ts, dossier_store=ds, event_store=es)
    json_str = exporter.export_json()                     # full export
    csv_str = exporter.export_targets_csv()               # targets as CSV
    geojson = exporter.export_geojson()                   # positions as GeoJSON
    incremental = exporter.export_json(since=1711000000)  # only recent data

    # Import
    importer = TritiumImporter(target_store=ts, dossier_store=ds, event_store=es)
    result = importer.import_json(json_str)
    result = importer.import_csv(csv_str, section="targets")
"""

from __future__ import annotations

import csv
import io
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from ..store.targets import TargetStore
from ..store.dossiers import DossierStore
from ..store.event_store import EventStore, TacticalEvent


# ---------------------------------------------------------------------------
# Export metadata envelope
# ---------------------------------------------------------------------------

_EXPORT_VERSION = "1.0.0"
_MAGIC = "tritium-data-exchange"

# Safety limits for imported data
_MAX_FIELD_LENGTH = 10_000       # Max length of any single string field
_MAX_JSON_DOC_SIZE = 50_000_000  # 50 MB max for JSON import documents
_MAX_CSV_ROWS = 1_000_000       # Max rows in a CSV import


def _sanitize_str(value: Any, max_len: int = _MAX_FIELD_LENGTH) -> str:
    """Sanitize a string field value by truncating and stripping null bytes."""
    if not isinstance(value, str):
        value = str(value) if value is not None else ""
    # Strip null bytes which can cause issues in databases/filesystems
    value = value.replace("\x00", "")
    if len(value) > max_len:
        value = value[:max_len]
    return value


def _make_header(
    scope: str,
    since: float | None = None,
    filters: dict | None = None,
) -> dict:
    """Build the metadata header for an export document."""
    return {
        "_magic": _MAGIC,
        "_version": _EXPORT_VERSION,
        "_exported_at": time.time(),
        "_export_id": str(uuid.uuid4()),
        "_scope": scope,
        "_since": since,
        "_filters": filters or {},
    }


# ---------------------------------------------------------------------------
# Import result
# ---------------------------------------------------------------------------


@dataclass
class ImportResult:
    """Summary of an import operation."""

    success: bool = True
    targets_imported: int = 0
    targets_skipped: int = 0
    dossiers_imported: int = 0
    dossiers_skipped: int = 0
    events_imported: int = 0
    events_skipped: int = 0
    zones_imported: int = 0
    zones_skipped: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def total_imported(self) -> int:
        return (
            self.targets_imported
            + self.dossiers_imported
            + self.events_imported
            + self.zones_imported
        )

    @property
    def total_skipped(self) -> int:
        return (
            self.targets_skipped
            + self.dossiers_skipped
            + self.events_skipped
            + self.zones_skipped
        )


# ---------------------------------------------------------------------------
# TritiumExporter
# ---------------------------------------------------------------------------


class TritiumExporter:
    """Export targets, dossiers, events, and zones from Tritium stores.

    Supports three output formats:
      - **JSON**: Full fidelity with all metadata and nested structures.
      - **CSV**: Flat tabular rows suitable for pandas, Excel, etc.
      - **GeoJSON**: Standard geographic format for map tools.

    All exports can be scoped with a ``since`` timestamp for incremental
    sync or filtered by source/alliance.

    Parameters
    ----------
    target_store:
        TargetStore instance (or None to skip targets).
    dossier_store:
        DossierStore instance (or None to skip dossiers).
    event_store:
        EventStore instance (or None to skip events).
    zones:
        Optional list of zone dicts to include.  Each dict should have
        at least ``zone_id``, ``name``, and geometry fields.
    """

    def __init__(
        self,
        target_store: TargetStore | None = None,
        dossier_store: DossierStore | None = None,
        event_store: EventStore | None = None,
        zones: list[dict] | None = None,
    ) -> None:
        self._targets = target_store
        self._dossiers = dossier_store
        self._events = event_store
        self._zones = zones or []

    # ------------------------------------------------------------------
    # Internal data collection
    # ------------------------------------------------------------------

    def _collect_targets(
        self,
        since: float | None = None,
        source: str | None = None,
        alliance: str | None = None,
    ) -> list[dict]:
        """Fetch targets from the store with optional filters."""
        if self._targets is None:
            return []
        return self._targets.get_all_targets(
            since=since, source=source, alliance=alliance,
        )

    def _collect_target_history(
        self,
        target_ids: list[str],
    ) -> dict[str, list[dict]]:
        """Fetch position history for a set of targets."""
        if self._targets is None:
            return {}
        result: dict[str, list[dict]] = {}
        for tid in target_ids:
            history = self._targets.get_history(tid, limit=10000)
            if history:
                result[tid] = history
        return result

    def _collect_dossiers(
        self,
        since: float | None = None,
    ) -> list[dict]:
        """Fetch dossiers from the store, optionally filtered by time."""
        if self._dossiers is None:
            return []
        return self._dossiers.get_recent(limit=100000, since=since)

    def _collect_dossier_full(self, dossier_id: str) -> dict | None:
        """Fetch a full dossier with signals and enrichments."""
        if self._dossiers is None:
            return None
        return self._dossiers.get_dossier(dossier_id)

    def _collect_events(
        self,
        since: float | None = None,
        event_type: str | None = None,
        limit: int = 100000,
    ) -> list[dict]:
        """Fetch events from the store."""
        if self._events is None:
            return []
        if event_type:
            events = self._events.query_by_type(
                event_type, start=since, limit=limit,
            )
        else:
            events = self._events.query_time_range(
                start=since, limit=limit,
            )
        return [e.to_dict() for e in events]

    def _collect_zones(self) -> list[dict]:
        """Return zone data."""
        return list(self._zones)

    # ------------------------------------------------------------------
    # JSON export
    # ------------------------------------------------------------------

    def export_json(
        self,
        since: float | None = None,
        source: str | None = None,
        alliance: str | None = None,
        event_type: str | None = None,
        include_history: bool = True,
        include_signals: bool = True,
        indent: int | None = 2,
    ) -> str:
        """Export all data as a JSON string.

        Parameters
        ----------
        since:
            Only include records updated at or after this unix timestamp.
        source:
            Filter targets by source string.
        alliance:
            Filter targets by alliance string.
        event_type:
            Filter events by type.
        include_history:
            If True, includes position history per target.
        include_signals:
            If True, includes signals/enrichments per dossier.
        indent:
            JSON indentation (None for compact).

        Returns a JSON string with envelope metadata and data sections.
        """
        filters = {}
        if source:
            filters["source"] = source
        if alliance:
            filters["alliance"] = alliance
        if event_type:
            filters["event_type"] = event_type

        scope = "incremental" if since else "full"
        header = _make_header(scope, since=since, filters=filters)

        # Targets
        targets = self._collect_targets(since=since, source=source, alliance=alliance)
        history: dict[str, list[dict]] = {}
        if include_history and targets:
            target_ids = [t["target_id"] for t in targets]
            history = self._collect_target_history(target_ids)

        # Dossiers
        dossier_summaries = self._collect_dossiers(since=since)
        dossiers: list[dict] = []
        if include_signals:
            for ds in dossier_summaries:
                full = self._collect_dossier_full(ds["dossier_id"])
                if full:
                    dossiers.append(full)
                else:
                    dossiers.append(ds)
        else:
            dossiers = dossier_summaries

        # Events
        events = self._collect_events(since=since, event_type=event_type)

        # Zones
        zones = self._collect_zones()

        doc = {
            **header,
            "targets": targets,
            "target_history": history,
            "dossiers": dossiers,
            "events": events,
            "zones": zones,
            "_counts": {
                "targets": len(targets),
                "dossiers": len(dossiers),
                "events": len(events),
                "zones": len(zones),
            },
        }
        return json.dumps(doc, indent=indent, default=str)

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    _TARGET_CSV_FIELDS = [
        "target_id", "name", "alliance", "asset_type", "source",
        "first_seen", "last_seen", "position_x", "position_y",
        "position_confidence",
    ]

    _DOSSIER_CSV_FIELDS = [
        "dossier_id", "name", "entity_type", "confidence", "alliance",
        "threat_level", "first_seen", "last_seen",
    ]

    _EVENT_CSV_FIELDS = [
        "event_id", "timestamp", "event_type", "severity", "source",
        "target_id", "operator", "summary", "position_lat", "position_lng",
        "site_id",
    ]

    def export_targets_csv(
        self,
        since: float | None = None,
        source: str | None = None,
        alliance: str | None = None,
    ) -> str:
        """Export targets as a CSV string.

        Metadata (JSON dict) is serialized as a JSON string in the
        ``metadata`` column.
        """
        targets = self._collect_targets(since=since, source=source, alliance=alliance)
        buf = io.StringIO()
        fields = self._TARGET_CSV_FIELDS + ["metadata"]
        writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for t in targets:
            row = {k: t.get(k, "") for k in self._TARGET_CSV_FIELDS}
            meta = t.get("metadata", {})
            row["metadata"] = json.dumps(meta) if isinstance(meta, dict) else str(meta)
            writer.writerow(row)
        return buf.getvalue()

    def export_dossiers_csv(
        self,
        since: float | None = None,
    ) -> str:
        """Export dossiers as a CSV string (summaries only, no signals)."""
        dossiers = self._collect_dossiers(since=since)
        buf = io.StringIO()
        fields = self._DOSSIER_CSV_FIELDS + ["identifiers", "tags"]
        writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for d in dossiers:
            row = {k: d.get(k, "") for k in self._DOSSIER_CSV_FIELDS}
            ids = d.get("identifiers", {})
            row["identifiers"] = json.dumps(ids) if isinstance(ids, dict) else str(ids)
            tags = d.get("tags", [])
            row["tags"] = json.dumps(tags) if isinstance(tags, list) else str(tags)
            writer.writerow(row)
        return buf.getvalue()

    def export_events_csv(
        self,
        since: float | None = None,
        event_type: str | None = None,
    ) -> str:
        """Export events as a CSV string."""
        events = self._collect_events(since=since, event_type=event_type)
        buf = io.StringIO()
        fields = self._EVENT_CSV_FIELDS + ["data"]
        writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for e in events:
            row = {k: e.get(k, "") for k in self._EVENT_CSV_FIELDS}
            data = e.get("data", {})
            row["data"] = json.dumps(data) if isinstance(data, dict) else str(data)
            writer.writerow(row)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # GeoJSON export
    # ------------------------------------------------------------------

    def export_geojson(
        self,
        since: float | None = None,
        source: str | None = None,
        alliance: str | None = None,
        include_trajectories: bool = False,
    ) -> str:
        """Export target positions (and optionally trajectories) as GeoJSON.

        Each target becomes a Point feature.  If ``include_trajectories``
        is True, targets with position history also get a LineString
        feature showing their movement trail.

        Zone polygons (circles approximated as 32-point polygons) are
        included as Polygon features.

        Coordinates use (longitude, latitude) ordering per the GeoJSON
        spec.  Since Tritium stores use local (x, y) coordinates, we
        treat x as longitude and y as latitude for portability.  If
        your installation uses a real geo converter, adapt accordingly.

        Returns a GeoJSON FeatureCollection string.
        """
        features: list[dict] = []

        # Target points
        targets = self._collect_targets(since=since, source=source, alliance=alliance)
        for t in targets:
            px = t.get("position_x")
            py = t.get("position_y")
            if px is None or py is None:
                continue
            feature: dict[str, Any] = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [px, py],
                },
                "properties": {
                    "target_id": t["target_id"],
                    "name": t.get("name", ""),
                    "alliance": t.get("alliance", ""),
                    "asset_type": t.get("asset_type", ""),
                    "source": t.get("source", ""),
                    "first_seen": t.get("first_seen"),
                    "last_seen": t.get("last_seen"),
                    "feature_type": "target_position",
                },
            }
            features.append(feature)

        # Trajectories
        if include_trajectories and self._targets is not None:
            target_ids = [t["target_id"] for t in targets]
            for tid in target_ids:
                traj = self._targets.get_trajectory(tid)
                if len(traj) < 2:
                    continue
                coords = [[p["x"], p["y"]] for p in traj]
                line_feature: dict[str, Any] = {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coords,
                    },
                    "properties": {
                        "target_id": tid,
                        "feature_type": "trajectory",
                        "point_count": len(coords),
                        "start_time": traj[0]["timestamp"],
                        "end_time": traj[-1]["timestamp"],
                    },
                }
                features.append(line_feature)

        # Zones as polygons
        import math
        for z in self._zones:
            vertices = z.get("vertices")
            center_lat = z.get("center_lat")
            center_lng = z.get("center_lng")
            radius = z.get("radius_m")

            if vertices and len(vertices) >= 3:
                # Polygon from vertices — close the ring
                ring = [[v[1], v[0]] for v in vertices]  # (lat,lng) -> [lng,lat]
                if ring[0] != ring[-1]:
                    ring.append(ring[0])
                geom: dict = {"type": "Polygon", "coordinates": [ring]}
            elif (
                center_lat is not None
                and center_lng is not None
                and radius is not None
            ):
                # Approximate circle as 32-point polygon
                n_pts = 32
                ring = []
                for i in range(n_pts + 1):
                    angle = 2 * math.pi * i / n_pts
                    # Rough meter-to-degree conversion
                    dlng = (radius / 111320.0) * math.cos(angle) / max(
                        math.cos(math.radians(center_lat)), 1e-6
                    )
                    dlat = (radius / 110540.0) * math.sin(angle)
                    ring.append([center_lng + dlng, center_lat + dlat])
                geom = {"type": "Polygon", "coordinates": [ring]}
            else:
                continue

            zone_feature: dict[str, Any] = {
                "type": "Feature",
                "geometry": geom,
                "properties": {
                    "zone_id": z.get("zone_id", ""),
                    "name": z.get("name", ""),
                    "zone_type": z.get("zone_type", ""),
                    "feature_type": "zone",
                },
            }
            features.append(zone_feature)

        # Events with positions
        events = self._collect_events(since=since)
        for e in events:
            lat = e.get("position_lat")
            lng = e.get("position_lng")
            if lat is None or lng is None:
                continue
            evt_feature: dict[str, Any] = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lng, lat],
                },
                "properties": {
                    "event_id": e.get("event_id", ""),
                    "event_type": e.get("event_type", ""),
                    "severity": e.get("severity", ""),
                    "timestamp": e.get("timestamp"),
                    "summary": e.get("summary", ""),
                    "feature_type": "event",
                },
            }
            features.append(evt_feature)

        collection: dict[str, Any] = {
            "type": "FeatureCollection",
            "features": features,
        }
        return json.dumps(collection, indent=2, default=str)

    # ------------------------------------------------------------------
    # Export stats
    # ------------------------------------------------------------------

    def get_export_stats(
        self,
        since: float | None = None,
    ) -> dict:
        """Return counts of what would be exported without doing the export."""
        targets = self._collect_targets(since=since)
        dossiers = self._collect_dossiers(since=since)
        events = self._collect_events(since=since)
        zones = self._collect_zones()
        return {
            "targets": len(targets),
            "dossiers": len(dossiers),
            "events": len(events),
            "zones": len(zones),
        }


# ---------------------------------------------------------------------------
# TritiumImporter
# ---------------------------------------------------------------------------


class TritiumImporter:
    """Import targets, dossiers, and events from JSON or CSV into Tritium stores.

    Parameters
    ----------
    target_store:
        TargetStore instance (or None to skip target import).
    dossier_store:
        DossierStore instance (or None to skip dossier import).
    event_store:
        EventStore instance (or None to skip event import).
    """

    def __init__(
        self,
        target_store: TargetStore | None = None,
        dossier_store: DossierStore | None = None,
        event_store: EventStore | None = None,
    ) -> None:
        self._targets = target_store
        self._dossiers = dossier_store
        self._events = event_store

    # ------------------------------------------------------------------
    # JSON import
    # ------------------------------------------------------------------

    def import_json(self, json_str: str) -> ImportResult:
        """Import data from a JSON export string.

        Expects the envelope produced by ``TritiumExporter.export_json()``.
        Records are upserted: existing targets/dossiers are updated,
        new ones are created.

        Returns an ImportResult summarizing what was imported.
        """
        result = ImportResult()

        # Guard against excessively large documents
        if len(json_str) > _MAX_JSON_DOC_SIZE:
            result.success = False
            result.errors.append(
                f"JSON document too large ({len(json_str)} bytes, "
                f"max {_MAX_JSON_DOC_SIZE})"
            )
            return result

        try:
            doc = json.loads(json_str)
        except json.JSONDecodeError as exc:
            result.success = False
            result.errors.append(f"Invalid JSON: {exc}")
            return result

        # Validate envelope
        if doc.get("_magic") != _MAGIC:
            result.warnings.append(
                "Missing or unrecognized _magic field — importing anyway"
            )

        # Import targets
        for t in doc.get("targets", []):
            self._import_target(t, result)

        # Import target history
        for tid, history_list in doc.get("target_history", {}).items():
            self._import_target_history(tid, history_list, result)

        # Import dossiers
        for d in doc.get("dossiers", []):
            self._import_dossier(d, result)

        # Import events
        for e in doc.get("events", []):
            self._import_event(e, result)

        # Zones (stored in-memory only — caller must handle persistence)
        for z in doc.get("zones", []):
            result.zones_imported += 1

        return result

    def _import_target(self, data: dict, result: ImportResult) -> None:
        """Import a single target record."""
        if self._targets is None:
            result.targets_skipped += 1
            return

        target_id = _sanitize_str(data.get("target_id", ""), max_len=256)
        if not target_id:
            result.targets_skipped += 1
            result.warnings.append("Target missing target_id — skipped")
            return

        try:
            meta = data.get("metadata", {})
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except json.JSONDecodeError:
                    meta = {}

            self._targets.record_sighting(
                target_id=target_id,
                name=_sanitize_str(data.get("name", ""), max_len=512),
                alliance=_sanitize_str(data.get("alliance", ""), max_len=64),
                asset_type=_sanitize_str(data.get("asset_type", ""), max_len=128),
                source=_sanitize_str(data.get("source", ""), max_len=256),
                position_x=_safe_float(data.get("position_x")),
                position_y=_safe_float(data.get("position_y")),
                position_confidence=_safe_float(data.get("position_confidence")),
                metadata=meta if isinstance(meta, dict) else {},
                timestamp=_safe_float(data.get("last_seen")) or time.time(),
            )
            result.targets_imported += 1
        except Exception as exc:
            result.targets_skipped += 1
            result.errors.append(f"Target {target_id}: {exc}")

    def _import_target_history(
        self, target_id: str, history: list[dict], result: ImportResult,
    ) -> None:
        """Import position history entries for a target.

        Uses record_sighting to upsert each history point — this ensures
        the target exists and history is appended correctly.
        """
        if self._targets is None:
            return

        for pt in history:
            try:
                x = _safe_float(pt.get("x"))
                y = _safe_float(pt.get("y"))
                if x is None or y is None:
                    continue
                self._targets.record_sighting(
                    target_id=target_id,
                    source=pt.get("source", ""),
                    position_x=x,
                    position_y=y,
                    timestamp=_safe_float(pt.get("timestamp")) or time.time(),
                )
            except Exception:
                pass  # history import is best-effort

    def _import_dossier(self, data: dict, result: ImportResult) -> None:
        """Import a single dossier record."""
        if self._dossiers is None:
            result.dossiers_skipped += 1
            return

        dossier_id = _sanitize_str(data.get("dossier_id", ""), max_len=256)
        if not dossier_id:
            result.dossiers_skipped += 1
            result.warnings.append("Dossier missing dossier_id — skipped")
            return

        try:
            # Check if dossier already exists
            existing = self._dossiers.get_dossier(dossier_id)
            if existing is None:
                # Parse identifiers/tags/notes
                identifiers = data.get("identifiers", {})
                if isinstance(identifiers, str):
                    try:
                        identifiers = json.loads(identifiers)
                    except json.JSONDecodeError:
                        identifiers = {}

                tags = data.get("tags", [])
                if isinstance(tags, str):
                    try:
                        tags = json.loads(tags)
                    except json.JSONDecodeError:
                        tags = []

                notes = data.get("notes", [])
                if isinstance(notes, str):
                    try:
                        notes = json.loads(notes)
                    except json.JSONDecodeError:
                        notes = []

                # Create new dossier — DossierStore generates its own ID,
                # so we write directly to get the exact imported ID.
                ts = _safe_float(data.get("first_seen")) or time.time()
                self._dossiers._conn.execute(
                    """INSERT OR REPLACE INTO dossiers
                       (dossier_id, name, entity_type, confidence, alliance,
                        threat_level, first_seen, last_seen,
                        identifiers, tags, notes)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        dossier_id,
                        _sanitize_str(data.get("name", "Unknown"), max_len=512),
                        _sanitize_str(data.get("entity_type", "unknown"), max_len=64),
                        _safe_float(data.get("confidence")) or 0.0,
                        _sanitize_str(data.get("alliance", "unknown"), max_len=64),
                        _sanitize_str(data.get("threat_level", "none"), max_len=64),
                        ts,
                        _safe_float(data.get("last_seen")) or ts,
                        json.dumps(identifiers if isinstance(identifiers, dict) else {}),
                        json.dumps(tags if isinstance(tags, list) else []),
                        json.dumps(notes if isinstance(notes, list) else []),
                    ),
                )
                self._dossiers._conn.commit()

            # Import signals if present
            for sig in data.get("signals", []):
                try:
                    self._dossiers.add_signal(
                        dossier_id=dossier_id,
                        source=sig.get("source", ""),
                        signal_type=sig.get("signal_type", ""),
                        data=sig.get("data"),
                        position_x=_safe_float(sig.get("position_x")),
                        position_y=_safe_float(sig.get("position_y")),
                        confidence=_safe_float(sig.get("confidence")) or 0.5,
                        timestamp=_safe_float(sig.get("timestamp")),
                    )
                except Exception:
                    pass  # signals are best-effort

            # Import enrichments if present
            for enr in data.get("enrichments", []):
                try:
                    self._dossiers.add_enrichment(
                        dossier_id=dossier_id,
                        provider=enr.get("provider", ""),
                        enrichment_type=enr.get("enrichment_type", ""),
                        data=enr.get("data"),
                        timestamp=_safe_float(enr.get("timestamp")),
                    )
                except Exception:
                    pass  # enrichments are best-effort

            result.dossiers_imported += 1
        except Exception as exc:
            result.dossiers_skipped += 1
            result.errors.append(f"Dossier {dossier_id}: {exc}")

    def _import_event(self, data: dict, result: ImportResult) -> None:
        """Import a single event record."""
        if self._events is None:
            result.events_skipped += 1
            return

        event_type = _sanitize_str(data.get("event_type", ""), max_len=128)
        if not event_type:
            result.events_skipped += 1
            result.warnings.append("Event missing event_type — skipped")
            return

        try:
            self._events.record(
                event_type=event_type,
                severity=_sanitize_str(data.get("severity", "info"), max_len=32),
                source=_sanitize_str(data.get("source", ""), max_len=256),
                target_id=_sanitize_str(data.get("target_id", ""), max_len=256),
                operator=_sanitize_str(data.get("operator", ""), max_len=256),
                summary=_sanitize_str(data.get("summary", "")),
                data=data.get("data"),
                position_lat=_safe_float(data.get("position_lat")),
                position_lng=_safe_float(data.get("position_lng")),
                site_id=_sanitize_str(data.get("site_id", ""), max_len=128),
                timestamp=_safe_float(data.get("timestamp")),
                event_id=_sanitize_str(data.get("event_id", ""), max_len=256) or None,
            )
            result.events_imported += 1
        except Exception as exc:
            result.events_skipped += 1
            result.errors.append(f"Event: {exc}")

    # ------------------------------------------------------------------
    # CSV import
    # ------------------------------------------------------------------

    def import_csv(self, csv_str: str, section: str) -> ImportResult:
        """Import data from a CSV string.

        Parameters
        ----------
        csv_str:
            CSV text (with header row).
        section:
            Which data type the CSV contains.  One of:
            ``"targets"``, ``"dossiers"``, ``"events"``.

        Returns an ImportResult summarizing what was imported.
        """
        result = ImportResult()
        reader = csv.DictReader(io.StringIO(csv_str))

        handlers = {
            "targets": self._import_target,
            "dossiers": self._import_dossier,
            "events": self._import_event,
        }

        handler = handlers.get(section)
        if handler is None:
            result.success = False
            result.errors.append(
                f"Unknown section '{section}'. Must be one of: {list(handlers.keys())}"
            )
            return result

        row_count = 0
        for row in reader:
            row_count += 1
            if row_count > _MAX_CSV_ROWS:
                result.warnings.append(
                    f"CSV import truncated at {_MAX_CSV_ROWS} rows"
                )
                break
            handler(dict(row), result)

        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(value: Any) -> float | None:
    """Convert a value to float, returning None if not possible."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


__all__ = [
    "TritiumExporter",
    "TritiumImporter",
    "ImportResult",
]
