# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""tritium_lib.geoint — Geospatial Intelligence (GEOINT).

Combines target tracking data with terrain and building analysis to answer
tactical questions:

* **Line of sight** — can two points see each other considering buildings?
* **Cover analysis** — where can a person hide from observation?
* **Approach routes** — what is the most concealed path between two points?
* **Observation points** — where should sensors go to watch an area?
* **Surveillance coverage** — what fraction of an area can sensors observe?

All geometry operates in 2D local meters (x = East, y = North), consistent
with :mod:`tritium_lib.geo` and :mod:`tritium_lib.tracking.obstacles`.
Lat/lng conversion is available via the geo module when a reference point
is initialized.

Key classes:

* :class:`LineOfSight` — ray vs building-polygon intersection.
* :class:`CoverAnalysis` — find concealed positions behind buildings.
* :class:`ApproachRoute` — find concealed routes through an area.
* :class:`ObservationPoint` — score and rank observation positions.
* :class:`SurveillanceCoverage` — compute observable fraction of an area.
* :class:`GeointAnalyzer` — facade combining all of the above.

Usage::

    from tritium_lib.geoint import GeointAnalyzer

    analyzer = GeointAnalyzer()
    analyzer.load_buildings([
        {"polygon": [(0, 0), (10, 0), (10, 10), (0, 10)], "height": 8.0},
    ])

    # Check line of sight
    visible = analyzer.line_of_sight.check((20, 5), (−5, 5))

    # Find cover positions
    cover = analyzer.cover_analysis.find_cover(
        observer=(50, 50), search_radius=100,
    )
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any

from tritium_lib.geo import point_in_polygon as _pip_geo


# ---------------------------------------------------------------------------
# Geometry helpers (self-contained, no external deps)
# ---------------------------------------------------------------------------

def _segments_intersect(
    ax: float, ay: float, bx: float, by: float,
    cx: float, cy: float, dx: float, dy: float,
) -> bool:
    """Check if line segment AB intersects line segment CD.

    Uses the cross-product orientation test.  Does not count collinear
    overlaps (good enough for building-edge intersection).
    """
    def cross(ox: float, oy: float, px: float, py: float,
              qx: float, qy: float) -> float:
        return (px - ox) * (qy - oy) - (py - oy) * (qx - ox)

    d1 = cross(cx, cy, dx, dy, ax, ay)
    d2 = cross(cx, cy, dx, dy, bx, by)
    d3 = cross(ax, ay, bx, by, cx, cy)
    d4 = cross(ax, ay, bx, by, dx, dy)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


_point_in_polygon = _pip_geo


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance between two 2D points."""
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _polygon_centroid(poly: list[tuple[float, float]]) -> tuple[float, float]:
    """Centroid of a polygon (average of vertices)."""
    n = len(poly)
    if n == 0:
        return (0.0, 0.0)
    cx = sum(p[0] for p in poly) / n
    cy = sum(p[1] for p in poly) / n
    return (cx, cy)


def _polygon_edges(
    poly: list[tuple[float, float]],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Return edges of a polygon as pairs of vertices."""
    n = len(poly)
    return [(poly[i], poly[(i + 1) % n]) for i in range(n)]


# ---------------------------------------------------------------------------
# Building representation
# ---------------------------------------------------------------------------

@dataclass
class Building:
    """A building footprint in local meters.

    Attributes:
        polygon: List of (x, y) vertices forming the footprint.
        height: Roof height in meters (default 8 m).
        aabb: Axis-aligned bounding box (min_x, min_y, max_x, max_y).
    """

    polygon: list[tuple[float, float]] = field(default_factory=list)
    height: float = 8.0
    aabb: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if self.polygon and self.aabb == (0.0, 0.0, 0.0, 0.0):
            self._compute_aabb()

    def _compute_aabb(self) -> None:
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        self.aabb = (min(xs), min(ys), max(xs), max(ys))

    @property
    def centroid(self) -> tuple[float, float]:
        return _polygon_centroid(self.polygon)

    def contains(self, x: float, y: float) -> bool:
        """Check if point is inside this building footprint."""
        mn_x, mn_y, mx_x, mx_y = self.aabb
        if x < mn_x or x > mx_x or y < mn_y or y > mx_y:
            return False
        return _point_in_polygon(x, y, self.polygon)


# ---------------------------------------------------------------------------
# LineOfSight
# ---------------------------------------------------------------------------

class LineOfSight:
    """2D line-of-sight checks against building footprints.

    Tests whether a straight line between two points is blocked by any
    building polygon edge.  Pure 2D — does not consider elevation.
    """

    def __init__(self) -> None:
        self._buildings: list[Building] = []

    def set_buildings(self, buildings: list[Building]) -> None:
        """Replace the building set."""
        self._buildings = list(buildings)

    def check(
        self,
        origin: tuple[float, float],
        target: tuple[float, float],
    ) -> bool:
        """Return True if *origin* can see *target* (no building blocks the ray).

        Both points are in local meters (x, y).
        """
        ax, ay = origin
        bx, by = target
        for bldg in self._buildings:
            # Quick AABB rejection: if the line segment's own bounding box
            # does not overlap the building AABB, skip.
            mn_x, mn_y, mx_x, mx_y = bldg.aabb
            seg_mn_x = min(ax, bx)
            seg_mx_x = max(ax, bx)
            seg_mn_y = min(ay, by)
            seg_mx_y = max(ay, by)
            if seg_mx_x < mn_x or seg_mn_x > mx_x or seg_mx_y < mn_y or seg_mn_y > mx_y:
                continue
            # Test segment against each building edge
            for (cx, cy), (dx, dy) in _polygon_edges(bldg.polygon):
                if _segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
                    return False
        return True

    def blocking_buildings(
        self,
        origin: tuple[float, float],
        target: tuple[float, float],
    ) -> list[int]:
        """Return indices of buildings that block the line of sight."""
        ax, ay = origin
        bx, by = target
        blockers: list[int] = []
        for idx, bldg in enumerate(self._buildings):
            mn_x, mn_y, mx_x, mx_y = bldg.aabb
            seg_mn_x = min(ax, bx)
            seg_mx_x = max(ax, bx)
            seg_mn_y = min(ay, by)
            seg_mx_y = max(ay, by)
            if seg_mx_x < mn_x or seg_mn_x > mx_x or seg_mx_y < mn_y or seg_mn_y > mx_y:
                continue
            for (cx, cy), (dx, dy) in _polygon_edges(bldg.polygon):
                if _segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
                    blockers.append(idx)
                    break
        return blockers

    def visible_targets(
        self,
        observer: tuple[float, float],
        targets: list[tuple[float, float]],
    ) -> list[tuple[int, tuple[float, float]]]:
        """Return indices and positions of targets visible from *observer*."""
        result = []
        for i, t in enumerate(targets):
            if self.check(observer, t):
                result.append((i, t))
        return result


# ---------------------------------------------------------------------------
# CoverAnalysis
# ---------------------------------------------------------------------------

@dataclass
class CoverPosition:
    """A position that provides concealment from an observer.

    Attributes:
        position: (x, y) in local meters.
        cover_score: 0.0 (no cover) to 1.0 (fully concealed).
        building_index: Index of the building providing cover.
        distance_to_building: Distance from position to the covering building centroid.
    """
    position: tuple[float, float] = (0.0, 0.0)
    cover_score: float = 0.0
    building_index: int = -1
    distance_to_building: float = 0.0


class CoverAnalysis:
    """Find positions concealed from one or more observers by buildings.

    Samples candidate positions on the far side of each building relative
    to the observer, then verifies with line-of-sight checks.
    """

    def __init__(self, los: LineOfSight, buildings: list[Building]) -> None:
        self._los = los
        self._buildings = buildings

    def find_cover(
        self,
        observer: tuple[float, float],
        search_radius: float = 200.0,
        sample_spacing: float = 5.0,
        min_cover_score: float = 0.3,
    ) -> list[CoverPosition]:
        """Find concealed positions within *search_radius* of the observer.

        For each building within range, sample candidate positions on the
        far side (away from the observer) and keep those hidden from the
        observer's line of sight.

        Args:
            observer: (x, y) observer position in local meters.
            search_radius: Maximum distance from observer to consider.
            sample_spacing: Spacing between candidate samples in meters.
            min_cover_score: Minimum cover score to include in results.

        Returns:
            List of :class:`CoverPosition` sorted by descending cover_score.
        """
        results: list[CoverPosition] = []

        for idx, bldg in enumerate(self._buildings):
            bc = bldg.centroid
            dist_to_bldg = _distance(observer, bc)
            if dist_to_bldg > search_radius or dist_to_bldg < 1e-6:
                continue

            # Direction from observer to building centroid
            dx = bc[0] - observer[0]
            dy = bc[1] - observer[1]
            norm = math.hypot(dx, dy)
            ux, uy = dx / norm, dy / norm

            # Sample candidates on the far side of the building
            # Offset from centroid in the direction away from observer
            mn_x, mn_y, mx_x, mx_y = bldg.aabb
            bldg_size = max(mx_x - mn_x, mx_y - mn_y)
            offset_start = bldg_size / 2 + 1.0
            offset_end = bldg_size / 2 + sample_spacing * 3

            for offset in _frange(offset_start, offset_end, sample_spacing):
                # Sample along the away-from-observer direction
                cx = bc[0] + ux * offset
                cy = bc[1] + uy * offset

                cand_dist = _distance(observer, (cx, cy))
                if cand_dist > search_radius:
                    continue
                # Must not be inside a building
                if any(b.contains(cx, cy) for b in self._buildings):
                    continue

                if not self._los.check(observer, (cx, cy)):
                    # Observer cannot see this point — good cover
                    score = min(1.0, (1.0 - cand_dist / search_radius) * 0.5 + 0.5)
                    if score >= min_cover_score:
                        results.append(CoverPosition(
                            position=(cx, cy),
                            cover_score=score,
                            building_index=idx,
                            distance_to_building=_distance((cx, cy), bc),
                        ))
                else:
                    # Also sample perpendicular offsets
                    for perp_sign in (-1, 1):
                        px = cx + perp_sign * uy * sample_spacing
                        py = cy - perp_sign * ux * sample_spacing
                        pd = _distance(observer, (px, py))
                        if pd > search_radius:
                            continue
                        if any(b.contains(px, py) for b in self._buildings):
                            continue
                        if not self._los.check(observer, (px, py)):
                            score = min(1.0, (1.0 - pd / search_radius) * 0.5 + 0.5)
                            if score >= min_cover_score:
                                results.append(CoverPosition(
                                    position=(px, py),
                                    cover_score=score,
                                    building_index=idx,
                                    distance_to_building=_distance((px, py), bc),
                                ))

        # Deduplicate positions that are very close together
        results = self._deduplicate(results, min_distance=sample_spacing * 0.5)
        results.sort(key=lambda c: c.cover_score, reverse=True)
        return results

    def is_concealed(
        self,
        position: tuple[float, float],
        observer: tuple[float, float],
    ) -> bool:
        """Check if *position* is concealed from *observer* by buildings."""
        return not self._los.check(observer, position)

    def concealment_from_multiple(
        self,
        position: tuple[float, float],
        observers: list[tuple[float, float]],
    ) -> float:
        """Fraction of observers that cannot see *position* (0.0 to 1.0)."""
        if not observers:
            return 0.0
        hidden_count = sum(
            1 for obs in observers if not self._los.check(obs, position)
        )
        return hidden_count / len(observers)

    @staticmethod
    def _deduplicate(
        positions: list[CoverPosition], min_distance: float
    ) -> list[CoverPosition]:
        """Remove duplicate cover positions that are too close together."""
        kept: list[CoverPosition] = []
        for cp in positions:
            too_close = False
            for existing in kept:
                if _distance(cp.position, existing.position) < min_distance:
                    too_close = True
                    break
            if not too_close:
                kept.append(cp)
        return kept


# ---------------------------------------------------------------------------
# ApproachRoute
# ---------------------------------------------------------------------------

@dataclass
class RouteWaypoint:
    """A waypoint along a concealed approach route.

    Attributes:
        position: (x, y) in local meters.
        concealment: 0.0 (fully exposed) to 1.0 (fully hidden).
        distance_from_start: Cumulative distance along the route.
    """
    position: tuple[float, float] = (0.0, 0.0)
    concealment: float = 0.0
    distance_from_start: float = 0.0


@dataclass
class Route:
    """A complete approach route with concealment scoring.

    Attributes:
        waypoints: Ordered list of :class:`RouteWaypoint`.
        total_distance: Total route length in meters.
        avg_concealment: Average concealment score along the route.
        min_concealment: Worst-case concealment (most exposed segment).
        exposed_distance: Total distance spent in exposed segments.
    """
    waypoints: list[RouteWaypoint] = field(default_factory=list)
    total_distance: float = 0.0
    avg_concealment: float = 0.0
    min_concealment: float = 1.0
    exposed_distance: float = 0.0


class ApproachRoute:
    """Find concealed routes between two points using A* on a grid.

    Discretizes the area into a grid.  Each cell is scored for concealment
    (blocked from the observer by buildings).  A* finds the path that
    maximizes concealment while minimizing distance.
    """

    def __init__(self, los: LineOfSight, buildings: list[Building]) -> None:
        self._los = los
        self._buildings = buildings

    def find_route(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
        observer: tuple[float, float],
        grid_spacing: float = 5.0,
        concealment_weight: float = 2.0,
    ) -> Route:
        """Find the most concealed route from *start* to *goal*.

        Uses A* search on a grid.  The cost function balances distance
        against exposure to the observer.

        Args:
            start: Origin position (x, y) in local meters.
            goal: Destination position (x, y).
            observer: Observer position to hide from.
            grid_spacing: Grid cell size in meters.
            concealment_weight: How much to penalize exposed cells
                (higher = more concealed route, possibly longer).

        Returns:
            A :class:`Route` with waypoints from start to goal.
        """
        # Determine grid bounds
        all_x = [start[0], goal[0], observer[0]]
        all_y = [start[1], goal[1], observer[1]]
        for bldg in self._buildings:
            for p in bldg.polygon:
                all_x.append(p[0])
                all_y.append(p[1])

        margin = grid_spacing * 3
        min_x = min(all_x) - margin
        max_x = max(all_x) + margin
        min_y = min(all_y) - margin
        max_y = max(all_y) + margin

        # Snap start and goal to grid
        def to_grid(p: tuple[float, float]) -> tuple[int, int]:
            gx = round((p[0] - min_x) / grid_spacing)
            gy = round((p[1] - min_y) / grid_spacing)
            return (gx, gy)

        def to_world(g: tuple[int, int]) -> tuple[float, float]:
            return (min_x + g[0] * grid_spacing, min_y + g[1] * grid_spacing)

        cols = int((max_x - min_x) / grid_spacing) + 1
        rows = int((max_y - min_y) / grid_spacing) + 1

        # Safety cap to prevent excessive memory
        max_cells = 10000
        if cols * rows > max_cells:
            # Increase grid spacing to fit
            scale = math.sqrt((cols * rows) / max_cells)
            grid_spacing *= scale
            cols = int((max_x - min_x) / grid_spacing) + 1
            rows = int((max_y - min_y) / grid_spacing) + 1

        start_g = to_grid(start)
        goal_g = to_grid(goal)

        # Precompute concealment scores
        concealment_cache: dict[tuple[int, int], float] = {}

        def get_concealment(g: tuple[int, int]) -> float:
            if g in concealment_cache:
                return concealment_cache[g]
            wp = to_world(g)
            # Inside a building is impassable
            for b in self._buildings:
                if b.contains(wp[0], wp[1]):
                    concealment_cache[g] = -1.0
                    return -1.0
            # Concealment: 1.0 if observer can't see, 0.0 if observer can see
            hidden = 0.0 if self._los.check(observer, wp) else 1.0
            concealment_cache[g] = hidden
            return hidden

        # A* search
        # Cost = distance + concealment_weight * exposure_penalty
        def heuristic(g: tuple[int, int]) -> float:
            wp = to_world(g)
            gp = to_world(goal_g)
            return _distance(wp, gp)

        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]

        open_set: list[tuple[float, tuple[int, int]]] = [(0.0, start_g)]
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score: dict[tuple[int, int], float] = {start_g: 0.0}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal_g:
                break

            for dnx, dny in neighbors:
                nb = (current[0] + dnx, current[1] + dny)
                if nb[0] < 0 or nb[0] >= cols or nb[1] < 0 or nb[1] >= rows:
                    continue

                conc = get_concealment(nb)
                if conc < 0:
                    continue  # inside building, impassable

                step_dist = grid_spacing * (1.414 if dnx != 0 and dny != 0 else 1.0)
                exposure_penalty = (1.0 - conc) * concealment_weight * step_dist
                tentative = g_score[current] + step_dist + exposure_penalty

                if tentative < g_score.get(nb, float("inf")):
                    came_from[nb] = current
                    g_score[nb] = tentative
                    f = tentative + heuristic(nb)
                    heapq.heappush(open_set, (f, nb))

        # Reconstruct path
        path_grid: list[tuple[int, int]] = []
        node = goal_g
        while node in came_from:
            path_grid.append(node)
            node = came_from[node]
        path_grid.append(start_g)
        path_grid.reverse()

        # Build Route
        waypoints: list[RouteWaypoint] = []
        cumulative_dist = 0.0
        total_concealment = 0.0
        min_conc = 1.0
        exposed_dist = 0.0

        for i, g in enumerate(path_grid):
            wp = to_world(g)
            conc = get_concealment(g)
            if conc < 0:
                conc = 0.0  # shouldn't happen, but safety
            if i > 0:
                step = _distance(to_world(path_grid[i - 1]), wp)
                cumulative_dist += step
                if conc < 0.5:
                    exposed_dist += step
            total_concealment += conc
            if conc < min_conc:
                min_conc = conc
            waypoints.append(RouteWaypoint(
                position=wp,
                concealment=conc,
                distance_from_start=cumulative_dist,
            ))

        n = len(waypoints)
        avg_conc = total_concealment / n if n > 0 else 0.0

        return Route(
            waypoints=waypoints,
            total_distance=cumulative_dist,
            avg_concealment=avg_conc,
            min_concealment=min_conc,
            exposed_distance=exposed_dist,
        )


# ---------------------------------------------------------------------------
# ObservationPoint
# ---------------------------------------------------------------------------

@dataclass
class ObservationScore:
    """Scoring for a candidate observation position.

    Attributes:
        position: (x, y) in local meters.
        visible_fraction: Fraction of the area of interest visible [0, 1].
        visible_count: Number of sample points visible.
        total_samples: Total sample points tested.
        avg_distance: Average distance to visible sample points.
    """
    position: tuple[float, float] = (0.0, 0.0)
    visible_fraction: float = 0.0
    visible_count: int = 0
    total_samples: int = 0
    avg_distance: float = 0.0


class ObservationPoint:
    """Find the best positions for observing a target area.

    Evaluates candidate positions by how much of the area of interest
    they can see (unblocked by buildings).
    """

    def __init__(self, los: LineOfSight, buildings: list[Building]) -> None:
        self._los = los
        self._buildings = buildings

    def score_position(
        self,
        position: tuple[float, float],
        area_center: tuple[float, float],
        area_radius: float,
        sample_spacing: float = 5.0,
    ) -> ObservationScore:
        """Score how well *position* can observe the circular area.

        Samples a grid of points inside the area and counts how many
        are visible from *position*.

        Args:
            position: Candidate observer position (x, y).
            area_center: Center of the area of interest.
            area_radius: Radius of the area of interest in meters.
            sample_spacing: Distance between sample points.

        Returns:
            :class:`ObservationScore` for this position.
        """
        samples = self._sample_area(area_center, area_radius, sample_spacing)
        visible_count = 0
        total_dist = 0.0

        for s in samples:
            if self._los.check(position, s):
                visible_count += 1
                total_dist += _distance(position, s)

        n = len(samples)
        fraction = visible_count / n if n > 0 else 0.0
        avg_d = total_dist / visible_count if visible_count > 0 else 0.0

        return ObservationScore(
            position=position,
            visible_fraction=fraction,
            visible_count=visible_count,
            total_samples=n,
            avg_distance=avg_d,
        )

    def find_best(
        self,
        area_center: tuple[float, float],
        area_radius: float,
        search_radius: float = 100.0,
        candidate_spacing: float = 10.0,
        sample_spacing: float = 5.0,
        top_n: int = 5,
    ) -> list[ObservationScore]:
        """Find the top-N observation points around the area.

        Generates candidate positions in a ring around the area of interest,
        scores each, and returns the best.

        Args:
            area_center: Center of the area to observe.
            area_radius: Radius of the area in meters.
            search_radius: How far from area_center to search for OPs.
            candidate_spacing: Spacing between candidate positions.
            sample_spacing: Spacing for area sampling.
            top_n: Number of best positions to return.

        Returns:
            List of :class:`ObservationScore` sorted by descending
            visible_fraction.
        """
        candidates: list[tuple[float, float]] = []

        # Generate candidate positions in a ring
        inner = area_radius + 5.0
        outer = search_radius
        r = inner
        while r <= outer:
            circumference = 2 * math.pi * r
            n_points = max(4, int(circumference / candidate_spacing))
            for i in range(n_points):
                angle = 2 * math.pi * i / n_points
                cx = area_center[0] + r * math.cos(angle)
                cy = area_center[1] + r * math.sin(angle)
                # Skip if inside a building
                if any(b.contains(cx, cy) for b in self._buildings):
                    continue
                candidates.append((cx, cy))
            r += candidate_spacing

        # Score all candidates
        scores = [
            self.score_position(c, area_center, area_radius, sample_spacing)
            for c in candidates
        ]

        scores.sort(key=lambda s: s.visible_fraction, reverse=True)
        return scores[:top_n]

    @staticmethod
    def _sample_area(
        center: tuple[float, float], radius: float, spacing: float,
    ) -> list[tuple[float, float]]:
        """Generate grid sample points within a circular area."""
        samples: list[tuple[float, float]] = []
        x = center[0] - radius
        while x <= center[0] + radius:
            y = center[1] - radius
            while y <= center[1] + radius:
                if _distance(center, (x, y)) <= radius:
                    samples.append((x, y))
                y += spacing
            x += spacing
        return samples


# ---------------------------------------------------------------------------
# SurveillanceCoverage
# ---------------------------------------------------------------------------

@dataclass
class CoverageResult:
    """Result of a surveillance coverage analysis.

    Attributes:
        coverage_fraction: Fraction of the area observable [0, 1].
        covered_points: Number of sample points visible by at least one sensor.
        total_points: Total sample points in the area.
        blind_spots: Sample points not visible by any sensor.
        sensor_contributions: Per-sensor count of unique visible points.
    """
    coverage_fraction: float = 0.0
    covered_points: int = 0
    total_points: int = 0
    blind_spots: list[tuple[float, float]] = field(default_factory=list)
    sensor_contributions: dict[int, int] = field(default_factory=dict)


class SurveillanceCoverage:
    """Compute what fraction of an area is observable by a set of sensors.

    Sensors are modeled as omnidirectional observers at fixed positions.
    Coverage is limited only by building occlusion and max sensor range.
    """

    def __init__(self, los: LineOfSight, buildings: list[Building]) -> None:
        self._los = los
        self._buildings = buildings

    def compute(
        self,
        sensor_positions: list[tuple[float, float]],
        area_center: tuple[float, float],
        area_radius: float,
        sample_spacing: float = 5.0,
        max_sensor_range: float = 0.0,
    ) -> CoverageResult:
        """Compute surveillance coverage of a circular area.

        Args:
            sensor_positions: List of (x, y) sensor locations.
            area_center: Center of the area of interest.
            area_radius: Radius of the area in meters.
            sample_spacing: Spacing between sample points.
            max_sensor_range: Maximum detection range (0 = unlimited).

        Returns:
            :class:`CoverageResult` with coverage statistics.
        """
        samples = ObservationPoint._sample_area(
            area_center, area_radius, sample_spacing
        )
        # Filter out sample points inside buildings
        samples = [
            s for s in samples
            if not any(b.contains(s[0], s[1]) for b in self._buildings)
        ]

        covered: set[int] = set()
        contributions: dict[int, int] = {i: 0 for i in range(len(sensor_positions))}

        for si, sensor in enumerate(sensor_positions):
            for pi, pt in enumerate(samples):
                if max_sensor_range > 0 and _distance(sensor, pt) > max_sensor_range:
                    continue
                if self._los.check(sensor, pt):
                    if pi not in covered:
                        contributions[si] = contributions.get(si, 0) + 1
                    covered.add(pi)

        total = len(samples)
        covered_count = len(covered)
        blind = [samples[i] for i in range(total) if i not in covered]

        return CoverageResult(
            coverage_fraction=covered_count / total if total > 0 else 0.0,
            covered_points=covered_count,
            total_points=total,
            blind_spots=blind,
            sensor_contributions=contributions,
        )

    def find_blind_spots(
        self,
        sensor_positions: list[tuple[float, float]],
        area_center: tuple[float, float],
        area_radius: float,
        sample_spacing: float = 5.0,
        max_sensor_range: float = 0.0,
    ) -> list[tuple[float, float]]:
        """Return points in the area that no sensor can observe."""
        result = self.compute(
            sensor_positions, area_center, area_radius,
            sample_spacing, max_sensor_range,
        )
        return result.blind_spots


# ---------------------------------------------------------------------------
# GeointAnalyzer — facade
# ---------------------------------------------------------------------------

class GeointAnalyzer:
    """Facade combining all GEOINT analysis capabilities.

    Manages a shared building set and provides access to:
    :class:`LineOfSight`, :class:`CoverAnalysis`, :class:`ApproachRoute`,
    :class:`ObservationPoint`, and :class:`SurveillanceCoverage`.

    Usage::

        analyzer = GeointAnalyzer()
        analyzer.load_buildings([
            {"polygon": [(0, 0), (10, 0), (10, 10), (0, 10)], "height": 8},
        ])
        los = analyzer.line_of_sight.check((20, 5), (-5, 5))
    """

    def __init__(self) -> None:
        self._buildings: list[Building] = []
        self._los = LineOfSight()
        self._cover: CoverAnalysis | None = None
        self._approach: ApproachRoute | None = None
        self._observation: ObservationPoint | None = None
        self._surveillance: SurveillanceCoverage | None = None

    def load_buildings(self, building_dicts: list[dict]) -> int:
        """Load buildings from a list of dicts.

        Each dict should have:
        - "polygon": list of [x, y] or (x, y) pairs in local meters.
        - "height" (optional): building height in meters (default 8).

        Returns the number of buildings loaded.
        """
        self._buildings = []
        for bd in building_dicts:
            poly_raw = bd.get("polygon", [])
            if len(poly_raw) < 3:
                continue
            poly = [(float(p[0]), float(p[1])) for p in poly_raw]
            height = float(bd.get("height", 8.0))
            self._buildings.append(Building(polygon=poly, height=height))

        self._los.set_buildings(self._buildings)
        self._cover = CoverAnalysis(self._los, self._buildings)
        self._approach = ApproachRoute(self._los, self._buildings)
        self._observation = ObservationPoint(self._los, self._buildings)
        self._surveillance = SurveillanceCoverage(self._los, self._buildings)

        return len(self._buildings)

    @property
    def buildings(self) -> list[Building]:
        """The loaded buildings."""
        return self._buildings

    @property
    def line_of_sight(self) -> LineOfSight:
        """Line-of-sight checker."""
        return self._los

    @property
    def cover_analysis(self) -> CoverAnalysis:
        """Cover analysis (requires buildings loaded)."""
        if self._cover is None:
            self._cover = CoverAnalysis(self._los, self._buildings)
        return self._cover

    @property
    def approach_route(self) -> ApproachRoute:
        """Approach route planner (requires buildings loaded)."""
        if self._approach is None:
            self._approach = ApproachRoute(self._los, self._buildings)
        return self._approach

    @property
    def observation_point(self) -> ObservationPoint:
        """Observation point scorer (requires buildings loaded)."""
        if self._observation is None:
            self._observation = ObservationPoint(self._los, self._buildings)
        return self._observation

    @property
    def surveillance_coverage(self) -> SurveillanceCoverage:
        """Surveillance coverage analyzer (requires buildings loaded)."""
        if self._surveillance is None:
            self._surveillance = SurveillanceCoverage(self._los, self._buildings)
        return self._surveillance

    def analyze_area(
        self,
        area_center: tuple[float, float],
        area_radius: float,
        sensor_positions: list[tuple[float, float]],
        sample_spacing: float = 5.0,
    ) -> dict[str, Any]:
        """Run a comprehensive GEOINT analysis on an area.

        Returns a dict with coverage, observation scores, and blind spots.
        """
        coverage = self.surveillance_coverage.compute(
            sensor_positions, area_center, area_radius, sample_spacing,
        )
        obs_scores = []
        for sp in sensor_positions:
            score = self.observation_point.score_position(
                sp, area_center, area_radius, sample_spacing,
            )
            obs_scores.append({
                "position": score.position,
                "visible_fraction": score.visible_fraction,
            })

        return {
            "coverage_fraction": coverage.coverage_fraction,
            "covered_points": coverage.covered_points,
            "total_points": coverage.total_points,
            "blind_spot_count": len(coverage.blind_spots),
            "sensor_scores": obs_scores,
        }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _frange(start: float, stop: float, step: float):
    """Float range generator."""
    val = start
    while val <= stop:
        yield val
        val += step


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "GeointAnalyzer",
    "LineOfSight",
    "CoverAnalysis",
    "CoverPosition",
    "ApproachRoute",
    "Route",
    "RouteWaypoint",
    "ObservationPoint",
    "ObservationScore",
    "SurveillanceCoverage",
    "CoverageResult",
    "Building",
]
