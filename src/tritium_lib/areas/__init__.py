# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""tritium_lib.areas — named geographic area management.

Manage named geographic regions with polygon boundaries, hierarchical
containment (campus > building > floor > room > zone), classification,
and activity statistics.  Uses :mod:`tritium_lib.geo` for all polygon
math (point-in-polygon, area computation, coordinate transforms).

Key classes:

* :class:`Area` — a named region with a polygon boundary and metadata.
* :class:`AreaHierarchy` — nested parent/child relationships between areas.
* :class:`AreaManager` — CRUD, spatial queries, import/export (GeoJSON).
* :class:`AreaClassifier` — rule-based classification of areas.
* :class:`AreaStats` — per-area activity statistics.

Usage::

    from tritium_lib.areas import Area, AreaManager

    mgr = AreaManager()
    campus = mgr.create(Area(
        area_id="campus_main",
        name="Main Campus",
        polygon=[(40.0, -74.0), (40.001, -74.0),
                 (40.001, -73.999), (40.0, -73.999)],
        area_type="campus",
    ))

    # Point-in-area query
    areas = mgr.areas_containing(40.0005, -74.0005)

    # Hierarchical containment
    mgr.hierarchy.set_parent("building_a", "campus_main")
    ancestors = mgr.hierarchy.ancestors("building_a")
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from tritium_lib.geo import (
    compute_area_latlng,
    haversine_distance,
    point_in_polygon_latlng,
)

logger = logging.getLogger("tritium.areas")


# ---------------------------------------------------------------------------
# Area classification
# ---------------------------------------------------------------------------

class AreaType(str, Enum):
    """Well-known area classifications."""

    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    INDUSTRIAL = "industrial"
    MILITARY = "military"
    CAMPUS = "campus"
    BUILDING = "building"
    FLOOR = "floor"
    ROOM = "room"
    ZONE = "zone"
    PARK = "park"
    PARKING = "parking"
    ROAD = "road"
    WATER = "water"
    RESTRICTED = "restricted"
    OTHER = "other"


# ---------------------------------------------------------------------------
# Area
# ---------------------------------------------------------------------------

@dataclass
class Area:
    """A named geographic region with a polygon boundary.

    Coordinates are in WGS84 (lat, lng).  The polygon is a list of
    ``(lat, lng)`` tuples forming a closed ring (the last vertex is
    implicitly connected to the first).

    Attributes:
        area_id: Unique identifier.  Auto-generated if empty.
        name: Human-readable name.
        polygon: Ordered list of (lat, lng) vertices.
        area_type: Classification string (see :class:`AreaType`).
        properties: Arbitrary key/value metadata.
        tags: Freeform string tags for filtering.
        created_at: Unix epoch creation time.
        updated_at: Unix epoch last-update time.
    """

    area_id: str = ""
    name: str = ""
    polygon: list[tuple[float, float]] = field(default_factory=list)
    area_type: str = "other"
    properties: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.area_id:
            self.area_id = f"area_{uuid.uuid4().hex[:12]}"

    # -- Geometry helpers ---------------------------------------------------

    @property
    def centroid(self) -> tuple[float, float]:
        """Compute the centroid of the polygon as (lat, lng).

        Returns (0, 0) if the polygon has fewer than 3 vertices.
        """
        n = len(self.polygon)
        if n == 0:
            return (0.0, 0.0)
        lat = sum(p[0] for p in self.polygon) / n
        lng = sum(p[1] for p in self.polygon) / n
        return (lat, lng)

    @property
    def area_sq_meters(self) -> float:
        """Approximate area of the polygon in square meters."""
        return compute_area_latlng(self.polygon)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Axis-aligned bounding box (min_lat, min_lng, max_lat, max_lng)."""
        if not self.polygon:
            return (0.0, 0.0, 0.0, 0.0)
        lats = [p[0] for p in self.polygon]
        lngs = [p[1] for p in self.polygon]
        return (min(lats), min(lngs), max(lats), max(lngs))

    def contains_point(self, lat: float, lng: float) -> bool:
        """Test whether a (lat, lng) point lies inside this area."""
        return point_in_polygon_latlng(lat, lng, self.polygon)

    def overlaps(self, other: Area) -> bool:
        """Test whether this area overlaps with *other*.

        Uses a fast bounding-box pre-check followed by vertex-in-polygon
        tests in both directions.  This is an approximation — it catches
        most practical overlaps but can miss edge-only intersections where
        no vertex of either polygon lies inside the other.
        """
        # Fast bbox rejection
        a = self.bbox
        b = other.bbox
        if a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]:
            return False
        # Check if any vertex of self is inside other
        for v in self.polygon:
            if other.contains_point(v[0], v[1]):
                return True
        # Check if any vertex of other is inside self
        for v in other.polygon:
            if self.contains_point(v[0], v[1]):
                return True
        return False

    @property
    def perimeter_meters(self) -> float:
        """Approximate perimeter of the polygon in meters."""
        n = len(self.polygon)
        if n < 2:
            return 0.0
        total = 0.0
        for i in range(n):
            j = (i + 1) % n
            total += haversine_distance(
                self.polygon[i][0], self.polygon[i][1],
                self.polygon[j][0], self.polygon[j][1],
            )
        return total

    # -- Serialization ------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "area_id": self.area_id,
            "name": self.name,
            "polygon": [list(p) for p in self.polygon],
            "area_type": self.area_type,
            "properties": dict(self.properties),
            "tags": list(self.tags),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Area:
        """Deserialize from a plain dict."""
        return cls(
            area_id=data.get("area_id", ""),
            name=data.get("name", ""),
            polygon=[tuple(p) for p in data.get("polygon", [])],
            area_type=data.get("area_type", "other"),
            properties=data.get("properties", {}),
            tags=data.get("tags", []),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
        )

    def to_geojson_feature(self) -> dict:
        """Export as a GeoJSON Feature (Polygon geometry).

        GeoJSON uses ``[longitude, latitude]`` ordering.
        """
        # Close the ring for GeoJSON
        coords = [[p[1], p[0]] for p in self.polygon]
        if coords and coords[0] != coords[-1]:
            coords.append(coords[0])
        return {
            "type": "Feature",
            "id": self.area_id,
            "properties": {
                "name": self.name,
                "area_type": self.area_type,
                "tags": self.tags,
                **self.properties,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords],
            },
        }

    @classmethod
    def from_geojson_feature(cls, feature: dict) -> Area:
        """Import from a GeoJSON Feature with Polygon geometry.

        Converts ``[lng, lat]`` coordinate ordering to ``(lat, lng)``.
        """
        props = feature.get("properties", {})
        geom = feature.get("geometry", {})
        coords = geom.get("coordinates", [[]])[0]
        # Remove closing duplicate if present
        polygon = [(c[1], c[0]) for c in coords]
        if len(polygon) > 1 and polygon[0] == polygon[-1]:
            polygon = polygon[:-1]
        return cls(
            area_id=feature.get("id", ""),
            name=props.get("name", ""),
            polygon=polygon,
            area_type=props.get("area_type", "other"),
            tags=props.get("tags", []),
            properties={
                k: v for k, v in props.items()
                if k not in ("name", "area_type", "tags")
            },
        )


# ---------------------------------------------------------------------------
# AreaHierarchy — nested parent/child containment
# ---------------------------------------------------------------------------

class AreaHierarchy:
    """Manage parent/child relationships between areas.

    Supports arbitrary depth: campus > building > floor > room > zone.
    Thread-safe.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # area_id -> parent_area_id
        self._parent: dict[str, str] = {}
        # area_id -> set of child area_ids
        self._children: dict[str, set[str]] = {}

    def set_parent(self, child_id: str, parent_id: str) -> None:
        """Set *parent_id* as the parent of *child_id*.

        Raises ValueError if this would create a cycle.
        """
        if child_id == parent_id:
            raise ValueError("An area cannot be its own parent")
        with self._lock:
            # Cycle detection: walk up from parent_id — if we ever
            # reach child_id, setting this link would create a cycle.
            current = parent_id
            visited: set[str] = set()
            while current is not None:
                if current == child_id:
                    raise ValueError(
                        f"Setting {parent_id} as parent of {child_id} "
                        f"would create a cycle"
                    )
                if current in visited:
                    break  # existing cycle in data — don't loop forever
                visited.add(current)
                current = self._parent.get(current)

            # Remove old parent link
            old_parent = self._parent.get(child_id)
            if old_parent is not None and old_parent in self._children:
                self._children[old_parent].discard(child_id)

            self._parent[child_id] = parent_id
            self._children.setdefault(parent_id, set()).add(child_id)

    def remove(self, area_id: str) -> None:
        """Remove an area from the hierarchy entirely.

        Children of the removed area become parentless (roots).
        """
        with self._lock:
            # Remove as child of its parent
            parent = self._parent.pop(area_id, None)
            if parent is not None and parent in self._children:
                self._children[parent].discard(area_id)

            # Orphan children
            children = self._children.pop(area_id, set())
            for cid in children:
                self._parent.pop(cid, None)

    def parent(self, area_id: str) -> str | None:
        """Return the parent area ID, or None if root."""
        with self._lock:
            return self._parent.get(area_id)

    def children(self, area_id: str) -> list[str]:
        """Return the direct child area IDs."""
        with self._lock:
            return list(self._children.get(area_id, set()))

    def ancestors(self, area_id: str) -> list[str]:
        """Return all ancestor area IDs from immediate parent to root.

        The first element is the direct parent; the last is the root.
        """
        result: list[str] = []
        with self._lock:
            current = self._parent.get(area_id)
            visited: set[str] = set()
            while current is not None and current not in visited:
                result.append(current)
                visited.add(current)
                current = self._parent.get(current)
        return result

    def descendants(self, area_id: str) -> list[str]:
        """Return all descendant area IDs (breadth-first)."""
        result: list[str] = []
        with self._lock:
            queue = list(self._children.get(area_id, set()))
            visited: set[str] = set()
            while queue:
                cid = queue.pop(0)
                if cid in visited:
                    continue
                visited.add(cid)
                result.append(cid)
                queue.extend(self._children.get(cid, set()))
        return result

    def roots(self) -> list[str]:
        """Return area IDs that have children but no parent (top-level)."""
        with self._lock:
            all_parents = set(self._children.keys())
            all_children = set(self._parent.keys())
            return sorted(all_parents - all_children)

    def depth(self, area_id: str) -> int:
        """Return the depth of an area (0 for root, 1 for direct child, etc.)."""
        return len(self.ancestors(area_id))

    def to_dict(self) -> dict:
        """Serialize the hierarchy to a dict."""
        with self._lock:
            return {
                "parent_map": dict(self._parent),
                "children_map": {
                    k: sorted(v) for k, v in self._children.items()
                },
            }

    @classmethod
    def from_dict(cls, data: dict) -> AreaHierarchy:
        """Restore hierarchy from a serialized dict."""
        h = cls()
        parent_map = data.get("parent_map", {})
        for child_id, parent_id in parent_map.items():
            h.set_parent(child_id, parent_id)
        return h


# ---------------------------------------------------------------------------
# AreaStats — per-area activity statistics
# ---------------------------------------------------------------------------

@dataclass
class AreaStats:
    """Activity statistics for an area.

    Attributes:
        area_id: The area these stats belong to.
        target_count: Number of targets currently inside.
        total_entries: Cumulative entries since tracking started.
        total_exits: Cumulative exits since tracking started.
        avg_dwell_seconds: Rolling average dwell time in seconds.
        last_entry_at: Timestamp of most recent entry (0 if none).
        last_exit_at: Timestamp of most recent exit (0 if none).
        peak_occupancy: Highest simultaneous target count observed.
        active_target_ids: Set of target IDs currently inside.
    """

    area_id: str = ""
    target_count: int = 0
    total_entries: int = 0
    total_exits: int = 0
    avg_dwell_seconds: float = 0.0
    last_entry_at: float = 0.0
    last_exit_at: float = 0.0
    peak_occupancy: int = 0
    active_target_ids: set[str] = field(default_factory=set)

    # Internal: entry timestamps per target for dwell calculation
    _entry_times: dict[str, float] = field(
        default_factory=dict, repr=False
    )
    _total_dwell: float = field(default=0.0, repr=False)
    _dwell_count: int = field(default=0, repr=False)

    def record_entry(self, target_id: str, timestamp: float | None = None) -> None:
        """Record a target entering this area."""
        ts = timestamp or time.time()
        self.active_target_ids.add(target_id)
        self._entry_times[target_id] = ts
        self.target_count = len(self.active_target_ids)
        self.total_entries += 1
        self.last_entry_at = ts
        if self.target_count > self.peak_occupancy:
            self.peak_occupancy = self.target_count

    def record_exit(self, target_id: str, timestamp: float | None = None) -> None:
        """Record a target exiting this area."""
        ts = timestamp or time.time()
        self.active_target_ids.discard(target_id)
        self.target_count = len(self.active_target_ids)
        self.total_exits += 1
        self.last_exit_at = ts

        entry_ts = self._entry_times.pop(target_id, None)
        if entry_ts is not None:
            dwell = ts - entry_ts
            self._total_dwell += dwell
            self._dwell_count += 1
            self.avg_dwell_seconds = self._total_dwell / self._dwell_count

    def to_dict(self) -> dict:
        """Serialize to a plain dict (excludes internal bookkeeping)."""
        return {
            "area_id": self.area_id,
            "target_count": self.target_count,
            "total_entries": self.total_entries,
            "total_exits": self.total_exits,
            "avg_dwell_seconds": round(self.avg_dwell_seconds, 2),
            "last_entry_at": self.last_entry_at,
            "last_exit_at": self.last_exit_at,
            "peak_occupancy": self.peak_occupancy,
            "active_target_ids": sorted(self.active_target_ids),
        }


# ---------------------------------------------------------------------------
# AreaClassifier
# ---------------------------------------------------------------------------

class AreaClassifier:
    """Rule-based classifier that assigns an :class:`AreaType` to an area.

    Uses heuristics based on the area's size, name, tags, and properties.
    Additional rules can be registered at runtime.
    """

    def __init__(self) -> None:
        self._rules: list[tuple[str, Any]] = []
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register built-in classification rules."""

        def _name_keywords(area: Area) -> str | None:
            name_lower = area.name.lower()
            kw_map = {
                "residential": ["house", "home", "apartment", "residence",
                                "dwelling", "condo"],
                "commercial": ["shop", "store", "mall", "market", "office",
                               "business", "retail"],
                "industrial": ["factory", "warehouse", "plant", "depot",
                               "manufacturing", "industrial"],
                "military": ["base", "barracks", "armory", "fort",
                             "military", "camp"],
                "campus": ["campus", "university", "school", "college"],
                "park": ["park", "garden", "playground", "recreation"],
                "parking": ["parking", "garage", "lot"],
                "water": ["lake", "river", "pond", "reservoir", "ocean"],
                "road": ["road", "highway", "street", "avenue", "boulevard"],
            }
            for area_type, keywords in kw_map.items():
                for kw in keywords:
                    if kw in name_lower:
                        return area_type
            return None

        self._rules.append(("name_keywords", _name_keywords))

        def _tag_match(area: Area) -> str | None:
            tags_lower = {t.lower() for t in area.tags}
            for at in AreaType:
                if at.value in tags_lower:
                    return at.value
            return None

        self._rules.append(("tag_match", _tag_match))

        def _size_heuristic(area: Area) -> str | None:
            if len(area.polygon) < 3:
                return None  # can't classify without a real polygon
            sq_m = area.area_sq_meters
            if sq_m < 20:
                return "room"
            if sq_m < 200:
                return "zone"
            if sq_m < 5000:
                return "building"
            if sq_m < 50000:
                return "campus"
            return None

        self._rules.append(("size_heuristic", _size_heuristic))

    def add_rule(self, name: str, rule_fn: Any) -> None:
        """Register a custom classification rule.

        *rule_fn* takes an :class:`Area` and returns a type string or None.
        Rules are evaluated in registration order; first non-None wins.
        """
        self._rules.append((name, rule_fn))

    def classify(self, area: Area) -> str:
        """Classify an area by running rules in order.

        Returns the first non-None result, or ``"other"`` if no rule matches.
        """
        for _name, rule_fn in self._rules:
            result = rule_fn(area)
            if result is not None:
                return result
        return "other"


# ---------------------------------------------------------------------------
# AreaManager — main CRUD + spatial query interface
# ---------------------------------------------------------------------------

class AreaManager:
    """Manage a collection of named areas with spatial queries.

    Thread-safe.  Provides:

    * CRUD operations for areas.
    * Point-in-area containment queries.
    * Overlap detection between areas.
    * GeoJSON import/export for the entire collection.
    * Integrated :class:`AreaHierarchy` and per-area :class:`AreaStats`.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._areas: dict[str, Area] = {}
        self.hierarchy = AreaHierarchy()
        self._stats: dict[str, AreaStats] = {}
        self._classifier = AreaClassifier()

    # -- CRUD ---------------------------------------------------------------

    def create(self, area: Area) -> Area:
        """Add an area.  Returns the area (with generated ID if needed)."""
        with self._lock:
            self._areas[area.area_id] = area
            self._stats[area.area_id] = AreaStats(area_id=area.area_id)
        logger.info("Area created: %s (%s)", area.name, area.area_id)
        return area

    def get(self, area_id: str) -> Area | None:
        """Look up an area by ID."""
        with self._lock:
            return self._areas.get(area_id)

    def update(self, area: Area) -> Area | None:
        """Replace an area's data.  Returns the updated area, or None if not found."""
        with self._lock:
            if area.area_id not in self._areas:
                return None
            area.updated_at = time.time()
            self._areas[area.area_id] = area
        logger.info("Area updated: %s (%s)", area.name, area.area_id)
        return area

    def delete(self, area_id: str) -> bool:
        """Remove an area by ID.  Returns True if found and removed."""
        with self._lock:
            if area_id not in self._areas:
                return False
            del self._areas[area_id]
            self._stats.pop(area_id, None)
        self.hierarchy.remove(area_id)
        logger.info("Area deleted: %s", area_id)
        return True

    def list_areas(
        self,
        area_type: str | None = None,
        tag: str | None = None,
    ) -> list[Area]:
        """Return all areas, optionally filtered by type or tag."""
        with self._lock:
            areas = list(self._areas.values())
        if area_type is not None:
            areas = [a for a in areas if a.area_type == area_type]
        if tag is not None:
            areas = [a for a in areas if tag in a.tags]
        return areas

    def count(self) -> int:
        """Return the number of managed areas."""
        with self._lock:
            return len(self._areas)

    # -- Spatial queries ----------------------------------------------------

    def areas_containing(self, lat: float, lng: float) -> list[Area]:
        """Return all areas whose polygon contains the point (lat, lng)."""
        with self._lock:
            candidates = list(self._areas.values())
        return [a for a in candidates if a.contains_point(lat, lng)]

    def areas_near(
        self, lat: float, lng: float, radius_m: float
    ) -> list[tuple[Area, float]]:
        """Return areas whose centroid is within *radius_m* meters.

        Returns a list of (area, distance_meters) sorted by distance.
        """
        with self._lock:
            candidates = list(self._areas.values())
        results: list[tuple[Area, float]] = []
        for area in candidates:
            clat, clng = area.centroid
            dist = haversine_distance(lat, lng, clat, clng)
            if dist <= radius_m:
                results.append((area, dist))
        results.sort(key=lambda x: x[1])
        return results

    def find_overlaps(self) -> list[tuple[str, str]]:
        """Find all pairs of areas that overlap.

        Returns a list of (area_id_a, area_id_b) pairs.
        """
        with self._lock:
            areas = list(self._areas.values())
        overlaps: list[tuple[str, str]] = []
        for i, a in enumerate(areas):
            for b in areas[i + 1:]:
                if a.overlaps(b):
                    overlaps.append((a.area_id, b.area_id))
        return overlaps

    # -- Classification -----------------------------------------------------

    def classify(self, area_id: str) -> str:
        """Classify an area using the built-in classifier.

        Returns the classification string, or ``"other"`` if not found.
        """
        area = self.get(area_id)
        if area is None:
            return "other"
        return self._classifier.classify(area)

    def auto_classify_all(self) -> dict[str, str]:
        """Run the classifier on all areas and update their area_type.

        Returns a mapping of area_id to assigned type.
        """
        results: dict[str, str] = {}
        with self._lock:
            for area in self._areas.values():
                classified = self._classifier.classify(area)
                area.area_type = classified
                area.updated_at = time.time()
                results[area.area_id] = classified
        return results

    # -- Statistics ---------------------------------------------------------

    def get_stats(self, area_id: str) -> AreaStats | None:
        """Return the stats tracker for an area."""
        with self._lock:
            return self._stats.get(area_id)

    def record_target_position(
        self, target_id: str, lat: float, lng: float, timestamp: float | None = None
    ) -> list[str]:
        """Update stats based on a target's current position.

        Checks all areas for containment and records entries/exits.
        Returns the list of area IDs the target is currently inside.
        """
        ts = timestamp or time.time()
        with self._lock:
            areas = list(self._areas.values())
            stats = dict(self._stats)

        inside_ids: list[str] = []
        for area in areas:
            aid = area.area_id
            st = stats.get(aid)
            if st is None:
                continue
            was_inside = target_id in st.active_target_ids
            is_inside = area.contains_point(lat, lng)

            if is_inside and not was_inside:
                st.record_entry(target_id, ts)
            elif not is_inside and was_inside:
                st.record_exit(target_id, ts)

            if is_inside:
                inside_ids.append(aid)

        return inside_ids

    # -- GeoJSON import/export ---------------------------------------------

    def to_geojson(self) -> dict:
        """Export all areas as a GeoJSON FeatureCollection."""
        with self._lock:
            features = [a.to_geojson_feature() for a in self._areas.values()]
        return {
            "type": "FeatureCollection",
            "features": features,
        }

    def from_geojson(self, geojson: dict) -> list[Area]:
        """Import areas from a GeoJSON FeatureCollection.

        Existing areas with the same ID are replaced.  Returns the
        list of imported areas.
        """
        features = geojson.get("features", [])
        imported: list[Area] = []
        for f in features:
            if f.get("geometry", {}).get("type") != "Polygon":
                continue
            area = Area.from_geojson_feature(f)
            self.create(area)
            imported.append(area)
        return imported

    def export_json(self) -> str:
        """Export all areas and hierarchy as a JSON string."""
        data = {
            "areas": [a.to_dict() for a in self.list_areas()],
            "hierarchy": self.hierarchy.to_dict(),
        }
        return json.dumps(data, indent=2)

    def import_json(self, json_str: str) -> int:
        """Import areas and hierarchy from a JSON string.

        Returns the number of areas imported.
        """
        data = json.loads(json_str)
        count = 0
        for ad in data.get("areas", []):
            area = Area.from_dict(ad)
            self.create(area)
            count += 1
        h_data = data.get("hierarchy")
        if h_data:
            self.hierarchy = AreaHierarchy.from_dict(h_data)
        return count


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "Area",
    "AreaType",
    "AreaHierarchy",
    "AreaManager",
    "AreaClassifier",
    "AreaStats",
]
