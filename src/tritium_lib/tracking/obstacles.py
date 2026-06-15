# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Building obstacle detection from OpenStreetMap Overpass API.

Pulls building footprints and stores them as polygons in local (x, z)
coordinates for collision detection. Uses ray-casting for
point-in-polygon (no Shapely dependency).

Coordinate convention:
    +X = East, +Y = North (same as geo.py — Y in comments here means
    the second coordinate, labeled "z" in the 3D layout convention).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Optional

try:
    import httpx
    _HAS_HTTPX = True
except ImportError:
    httpx = None  # type: ignore[assignment]
    _HAS_HTTPX = False

from tritium_lib.geo import point_in_polygon as _point_in_polygon

logger = logging.getLogger(__name__)

_OVERPASS_URL = "https://overpass-api.de/api/interpreter"
_USER_AGENT = "TRITIUM/0.1.0"
_DEFAULT_CACHE_DIR = "~/.cache/tritium"

# Meters per degree latitude (constant)
_METERS_PER_DEG_LAT = 111_320.0


def _latlng_to_local(
    lat: float, lng: float, ref_lat: float, ref_lng: float
) -> tuple[float, float]:
    """Convert lat/lng to local (x, y) meters relative to reference point."""
    y = (lat - ref_lat) * _METERS_PER_DEG_LAT
    meters_per_deg_lng = _METERS_PER_DEG_LAT * math.cos(math.radians(ref_lat))
    x = (lng - ref_lng) * meters_per_deg_lng
    return (x, y)


def _fetch_buildings(
    lat: float, lng: float, radius_m: float
) -> list[dict]:
    """Fetch building footprints from Overpass API (synchronous).

    Returns a list of OSM way elements with geometry.
    Requires httpx to be installed.
    """
    if not _HAS_HTTPX:
        logger.warning("httpx not installed — cannot fetch buildings from Overpass API")
        return []

    query = f'[out:json];way["building"](around:{radius_m},{lat},{lng});out geom;'
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            _OVERPASS_URL,
            data={"data": query},
            headers={"User-Agent": _USER_AGENT},
        )
        resp.raise_for_status()
    data = resp.json()
    return data.get("elements", [])


def _segments_intersect(
    ax: float, ay: float, bx: float, by: float,
    cx: float, cy: float, dx: float, dy: float,
) -> bool:
    """Check if line segment AB intersects line segment CD.

    Uses the cross-product orientation test.
    """
    def cross(ox: float, oy: float, px: float, py: float, qx: float, qy: float) -> float:
        return (px - ox) * (qy - oy) - (py - oy) * (qx - ox)

    d1 = cross(cx, cy, dx, dy, ax, ay)
    d2 = cross(cx, cy, dx, dy, bx, by)
    d3 = cross(ax, ay, bx, by, cx, cy)
    d4 = cross(ax, ay, bx, by, dx, dy)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    # Collinear cases (not needed for building detection — skip for simplicity)
    return False


def _dist_to_polygon_edge(
    x: float, y: float, poly: list[tuple[float, float]]
) -> float:
    """Shortest distance from point (x, y) to any edge of *poly*."""
    best = float("inf")
    n = len(poly)
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        dx, dy = bx - ax, by - ay
        seg2 = dx * dx + dy * dy
        if seg2 <= 1e-12:
            d = math.hypot(x - ax, y - ay)
        else:
            t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / seg2))
            d = math.hypot(x - (ax + t * dx), y - (ay + t * dy))
        if d < best:
            best = d
    return best


class BuildingObstacles:
    """Building footprints for collision/obstacle detection.

    Stores building polygons in local (x, y) coordinates.
    Uses AABB bounding boxes for fast pre-filtering (~95% rejection
    in 4 float comparisons before full ray-cast).

    Usage:
        obs = BuildingObstacles()
        obs.load(lat, lng, radius_m=300)
        if obs.point_in_building(x, y):
            ...
    """

    # Uniform-grid broad-phase cell size (meters). Buildings are bucketed into
    # every grid cell their AABB overlaps; queries only test buildings in the
    # cells they touch. ~30 m ≈ a few city footprints per cell — small enough to
    # reject the vast majority of buildings, large enough to keep the bucket map
    # bounded. Narrow-phase math (ray-cast, segment-intersect) is UNCHANGED, so
    # the index only changes WHICH buildings are tested, never the answer.
    _GRID_CELL_M: float = 30.0

    def __init__(self) -> None:
        self.polygons: list[list[tuple[float, float]]] = []
        # Per-building roof heights (meters), parallel to self.polygons
        self._heights: list[float] = []
        # AABB bounding boxes: (min_x, min_y, max_x, max_y) per polygon
        self._aabbs: list[tuple[float, float, float, float]] = []
        # Uniform-grid spatial index (broad phase). Maps a cell key (ci, cj) to
        # the list of building indices whose AABB overlaps that cell. Built once
        # in _compute_aabbs(); empty => fall back to the linear scan (so any code
        # path that mutates polygons without rebuilding still works correctly).
        self._grid: dict[tuple[int, int], list[int]] = {}
        # Grid origin (min corner of the bucketed extent) in local meters.
        self._grid_origin_x: float = 0.0
        self._grid_origin_y: float = 0.0
        # Unit-radius standoff (m). When > 0, point_in_building also flags
        # points within this distance of a building EDGE — so paths keep a
        # clean margin off walls instead of riding them (units are points
        # in the planner/checker but have width in reality). Honored by both
        # planning (path_crosses_building) and per-tick enforcement, with no
        # call-site changes. Default 0.0 = exact footprint (unchanged).
        self.clearance: float = 0.0

    def load(
        self,
        lat: float,
        lng: float,
        radius_m: float = 300,
        cache_dir: str = _DEFAULT_CACHE_DIR,
    ) -> None:
        """Load building footprints for the area around (lat, lng).

        Tries cache first, then Overpass API. On failure, polygons is empty.
        """
        cache_path = self._cache_path(lat, lng, radius_m, cache_dir)

        # Try loading from cache
        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    self.polygons = json.load(f)
                # Convert inner lists back to tuples
                self.polygons = [
                    [(pt[0], pt[1]) for pt in poly]
                    for poly in self.polygons
                ]
                # Cache doesn't store heights — default all to 8m
                self._heights = [8.0] * len(self.polygons)
                self._compute_aabbs()
                logger.info(f"Building obstacles loaded from cache: {len(self.polygons)} buildings")
                return
            except Exception as e:
                logger.warning(f"Cache load failed: {e}")

        # Fetch from Overpass API
        try:
            elements = _fetch_buildings(lat, lng, radius_m)
        except Exception as e:
            logger.warning(f"Overpass building fetch failed: {e}")
            self.polygons = []
            return

        if not elements:
            self.polygons = []
            return

        # Convert building footprints to local polygons
        self._build_polygons(elements, lat, lng)
        self._compute_aabbs()

        # Save to cache
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            # Store as list of lists for JSON serialization
            serializable = [
                [[pt[0], pt[1]] for pt in poly]
                for poly in self.polygons
            ]
            with open(cache_path, "w") as f:
                json.dump(serializable, f)
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")

    def point_in_building(self, x: float, y: float) -> bool:
        """Check if the point (x, y) in local meters is inside any building.

        Broad phase: the uniform-grid index narrows the candidate set to the
        buildings in the cell(s) covering the point (when ``clearance`` > 0 the
        point's clearance-expanded box, so wall-margin hits are never missed).
        Narrow phase (AABB ± clearance pre-filter, then ray-cast / edge-distance)
        is UNCHANGED, so the result is byte-identical to the full linear scan.
        """
        c = self.clearance
        if self._grid:
            seen: set[int] = set()
            for i in self._candidate_cells_point(x, y, c):
                if i in seen:
                    continue
                seen.add(i)
                min_x, min_y, max_x, max_y = self._aabbs[i]
                if x < min_x - c or x > max_x + c or y < min_y - c or y > max_y + c:
                    continue
                if _point_in_polygon(x, y, self.polygons[i]):
                    return True
                if c > 0.0 and _dist_to_polygon_edge(x, y, self.polygons[i]) <= c:
                    return True
            return False
        if self._aabbs:
            for i, (min_x, min_y, max_x, max_y) in enumerate(self._aabbs):
                if x < min_x - c or x > max_x + c or y < min_y - c or y > max_y + c:
                    continue
                if _point_in_polygon(x, y, self.polygons[i]):
                    return True
                if c > 0.0 and _dist_to_polygon_edge(x, y, self.polygons[i]) <= c:
                    return True
            return False
        # Fallback: no AABBs computed (shouldn't happen after load)
        for poly in self.polygons:
            if _point_in_polygon(x, y, poly):
                return True
            if c > 0.0 and _dist_to_polygon_edge(x, y, poly) <= c:
                return True
        return False

    def building_height_at(self, x: float, y: float) -> float | None:
        """Return the roof height of the building containing (x, y), or None.

        Uses AABB pre-filter then ray-casting, same as point_in_building().
        Returns the height from ``_heights`` for the first containing polygon.
        """
        if not self._heights:
            return None
        if self._aabbs:
            for i, (min_x, min_y, max_x, max_y) in enumerate(self._aabbs):
                if x < min_x or x > max_x or y < min_y or y > max_y:
                    continue
                if _point_in_polygon(x, y, self.polygons[i]):
                    return self._heights[i]
            return None
        for i, poly in enumerate(self.polygons):
            if _point_in_polygon(x, y, poly):
                return self._heights[i]
        return None

    def path_crosses_building(
        self, waypoints: list[tuple[float, float]]
    ) -> bool:
        """Check if any segment of the path crosses a building polygon.

        Tests both: (a) segment-edge intersection, and
        (b) midpoint inside a building (catches paths entirely inside).
        """
        if len(waypoints) < 2:
            return False

        grid = self._grid
        # AABB pre-filter is only usable when boxes line up 1:1 with polygons.
        # If polygons were mutated without _compute_aabbs(), fall back to the
        # original unfiltered linear scan so the answer stays identical.
        have_aabbs = len(self._aabbs) == len(self.polygons)

        for i in range(len(waypoints) - 1):
            ax, ay = waypoints[i]
            bx, by = waypoints[i + 1]

            # Check midpoint of the segment
            mx = (ax + bx) / 2
            my = (ay + by) / 2
            if self.point_in_building(mx, my):
                return True

            if grid and have_aabbs:
                # Broad phase: only the buildings whose AABB can meet this
                # segment. A building whose AABB is disjoint from the segment's
                # bbox cannot be crossed (its edges all lie inside its AABB), so
                # it is rejected without any _segments_intersect call. This
                # pre-filter is exact — it never skips a building the linear
                # scan would have hit.
                seg_min_x = ax if ax < bx else bx
                seg_max_x = ax if ax > bx else bx
                seg_min_y = ay if ay < by else by
                seg_max_y = ay if ay > by else by
                seen: set[int] = set()
                for bi in self._candidate_cells_segment(ax, ay, bx, by):
                    if bi in seen:
                        continue
                    seen.add(bi)
                    min_x, min_y, max_x, max_y = self._aabbs[bi]
                    if (max_x < seg_min_x or min_x > seg_max_x
                            or max_y < seg_min_y or min_y > seg_max_y):
                        continue
                    poly = self.polygons[bi]
                    n = len(poly)
                    for j in range(n):
                        cx, cy = poly[j]
                        dx, dy = poly[(j + 1) % n]
                        if _segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
                            return True
            else:
                # Linear fallback (no index): check segment against all edges.
                for poly in self.polygons:
                    n = len(poly)
                    for j in range(n):
                        cx, cy = poly[j]
                        dx, dy = poly[(j + 1) % n]
                        if _segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
                            return True

        return False

    def load_from_overture(self, building_dicts: list[dict]) -> None:
        """Load building polygons from pre-fetched data.

        Accepts a list of dicts with "polygon" key containing
        [(x, y), ...] tuples in local meters coordinates, and
        optional "height" key (default 8m).

        This allows the overlay API to pass pre-computed local-coordinate
        polygons without re-fetching from Overpass.
        """
        self.polygons = []
        self._heights = []
        for bldg in building_dicts:
            poly = bldg.get("polygon", [])
            if len(poly) < 3:
                continue
            # Ensure tuples
            self.polygons.append([(pt[0], pt[1]) for pt in poly])
            self._heights.append(bldg.get("height", 8.0))
        self._compute_aabbs()
        logger.info(f"Building obstacles: loaded {len(self.polygons)} buildings from overture data")

    def add_polygons(self, footprints, heights=None) -> int:
        """Merge additional building footprints into this obstacle set.

        Unlike ``load_from_overture`` (which REPLACES), this EXTENDS the
        existing polygons — so a network-independent source (a layout file,
        derived terrain footprints) can contribute buildings whether or not
        Overpass succeeded. Footprints with fewer than 3 vertices are
        ignored. Returns the number of polygons actually added.

        Args:
            footprints: iterable of [(x, y), ...] polygons in local meters.
            heights: optional parallel iterable of roof heights (default 8m).
        """
        added = 0
        heights = list(heights) if heights is not None else None
        for i, poly in enumerate(footprints):
            pts = [(float(p[0]), float(p[1])) for p in poly]
            if len(pts) < 3:
                continue
            self.polygons.append(pts)
            h = heights[i] if (heights is not None and i < len(heights)) else 8.0
            self._heights.append(h)
            added += 1
        if added:
            self._compute_aabbs()
            logger.info(f"Building obstacles: merged {added} footprint(s) "
                        f"(now {len(self.polygons)} total)")
        return added

    def to_dicts(self, default_height: float = 8.0) -> list[dict]:
        """Export building polygons as a list of dicts for the frontend.

        Returns: [{"polygon": [[x, y], ...], "height": <h>}, ...]
        Uses per-building heights from ``_heights`` when available,
        otherwise falls back to *default_height*.
        """
        result = []
        for i, poly in enumerate(self.polygons):
            h = self._heights[i] if i < len(self._heights) else default_height
            result.append({"polygon": [list(pt) for pt in poly], "height": h})
        return result

    def _compute_aabbs(self) -> None:
        """Compute axis-aligned bounding boxes for all polygons and (re)build
        the uniform-grid broad-phase index.

        The grid is built ONCE here (on load / set), never per query. Each
        building is bucketed into every grid cell its AABB overlaps, so a
        building straddling a cell boundary (or spanning many cells) is found
        from any of those cells. Empty obstacles => empty grid (queries fall
        back to the trivially-empty linear scan).
        """
        self._aabbs = []
        for poly in self.polygons:
            if not poly:
                self._aabbs.append((0.0, 0.0, 0.0, 0.0))
                continue
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            self._aabbs.append((min(xs), min(ys), max(xs), max(ys)))
        self._build_grid()

    def _build_grid(self) -> None:
        """Bucket every building index into the grid cells its AABB overlaps."""
        self._grid = {}
        if not self._aabbs:
            self._grid_origin_x = 0.0
            self._grid_origin_y = 0.0
            return
        # Grid origin = min corner of the whole extent (keeps cell indices small
        # and positive regardless of where the local-coordinate map sits).
        self._grid_origin_x = min(b[0] for b in self._aabbs)
        self._grid_origin_y = min(b[1] for b in self._aabbs)
        cell = self._GRID_CELL_M
        ox, oy = self._grid_origin_x, self._grid_origin_y
        grid = self._grid
        for idx, (min_x, min_y, max_x, max_y) in enumerate(self._aabbs):
            ci0 = int(math.floor((min_x - ox) / cell))
            ci1 = int(math.floor((max_x - ox) / cell))
            cj0 = int(math.floor((min_y - oy) / cell))
            cj1 = int(math.floor((max_y - oy) / cell))
            for ci in range(ci0, ci1 + 1):
                for cj in range(cj0, cj1 + 1):
                    grid.setdefault((ci, cj), []).append(idx)

    def _cell_of(self, x: float, y: float) -> tuple[int, int]:
        """Grid cell key containing the local-meters point (x, y)."""
        cell = self._GRID_CELL_M
        ci = int(math.floor((x - self._grid_origin_x) / cell))
        cj = int(math.floor((y - self._grid_origin_y) / cell))
        return (ci, cj)

    def _candidate_cells_point(self, x: float, y: float, c: float):
        """Yield building indices in the cell(s) covering the point.

        When ``c`` (clearance) > 0 we cover the point's clearance-expanded box
        so a building whose footprint is up to ``c`` meters away — and therefore
        a wall-margin hit — is never missed by the broad phase.
        """
        grid = self._grid
        cell = self._GRID_CELL_M
        ox, oy = self._grid_origin_x, self._grid_origin_y
        if c <= 0.0:
            bucket = grid.get(self._cell_of(x, y))
            if bucket:
                yield from bucket
            return
        ci0 = int(math.floor((x - c - ox) / cell))
        ci1 = int(math.floor((x + c - ox) / cell))
        cj0 = int(math.floor((y - c - oy) / cell))
        cj1 = int(math.floor((y + c - oy) / cell))
        for ci in range(ci0, ci1 + 1):
            for cj in range(cj0, cj1 + 1):
                bucket = grid.get((ci, cj))
                if bucket:
                    yield from bucket

    def _candidate_cells_segment(self, ax: float, ay: float, bx: float, by: float):
        """Yield building indices in the cells the segment AB passes through.

        Covers every cell overlapping the segment's bounding box — a superset of
        the exact swept cells, which keeps the answer identical (no false skips)
        while still rejecting the bulk of a hundreds-of-building map. The
        per-building AABB-vs-segment-bbox check downstream trims the rest.
        """
        grid = self._grid
        cell = self._GRID_CELL_M
        ox, oy = self._grid_origin_x, self._grid_origin_y
        smin_x = ax if ax < bx else bx
        smax_x = ax if ax > bx else bx
        smin_y = ay if ay < by else by
        smax_y = ay if ay > by else by
        ci0 = int(math.floor((smin_x - ox) / cell))
        ci1 = int(math.floor((smax_x - ox) / cell))
        cj0 = int(math.floor((smin_y - oy) / cell))
        cj1 = int(math.floor((smax_y - oy) / cell))
        for ci in range(ci0, ci1 + 1):
            for cj in range(cj0, cj1 + 1):
                bucket = grid.get((ci, cj))
                if bucket:
                    yield from bucket

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_polygons(
        self, elements: list[dict], ref_lat: float, ref_lng: float
    ) -> None:
        """Convert Overpass way elements to local-coordinate polygons."""
        self.polygons = []
        self._heights = []
        for el in elements:
            if el.get("type") != "way":
                continue
            geometry = el.get("geometry", [])
            if len(geometry) < 3:
                continue

            poly = []
            for pt in geometry:
                local = _latlng_to_local(pt["lat"], pt["lon"], ref_lat, ref_lng)
                poly.append(local)

            self.polygons.append(poly)

            # Extract height from OSM tags
            tags = el.get("tags", {})
            height = tags.get("height")
            if height is not None:
                try:
                    self._heights.append(float(height))
                    continue
                except (ValueError, TypeError):
                    pass
            levels = tags.get("building:levels")
            if levels is not None:
                try:
                    self._heights.append(float(levels) * 3.0)
                    continue
                except (ValueError, TypeError):
                    pass
            self._heights.append(8.0)

        logger.info(f"Building obstacles: {len(self.polygons)} buildings loaded")

    @staticmethod
    def _cache_path(
        lat: float, lng: float, radius_m: float, cache_dir: str
    ) -> Path:
        """Return the cache file path for these parameters."""
        key = f"buildings_{lat:.6f}_{lng:.6f}_{radius_m:.0f}"
        h = hashlib.sha256(key.encode()).hexdigest()[:16]
        base = Path(cache_dir).expanduser()
        return base / f"{h}.json"
