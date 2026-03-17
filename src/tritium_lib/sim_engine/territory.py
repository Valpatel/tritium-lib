# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Territory control and influence map system for tactical simulation.

Provides three main classes:

    InfluenceMap     — Grid-based per-faction influence (0-1 per cell) with
                       radial falloff, decay, frontline detection, and
                       contested zone analysis.
    TerritoryControl — Strategic control points with capture/contest logic,
                       progress tracking, and faction territory summaries.
    StrategicValue   — Rate positions by tactical value (elevation, road
                       proximity, building density, crossroads).

All spatial coordinates use Vec2 = tuple[float, float] from the steering
module.  Grid cells are indexed by integer (x, y) pairs where x is the
column and y is the row.  World positions are converted to grid cells by
dividing by cell_size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from tritium_lib.sim_engine.ai.steering import Vec2, distance

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_CELL_SIZE = 10.0
_DEFAULT_DECAY_RATE = 0.01
_CONTESTED_THRESHOLD = 0.15  # Factions within this delta are "contested"
_MIN_INFLUENCE = 0.001  # Below this, treat as zero
_CAPTURE_RANGE = 30.0  # World-space distance to contest a control point
_CAPTURE_RATE = 0.1  # Progress per second per unit in range
_UNIT_INFLUENCE_STRENGTH = 0.3
_UNIT_INFLUENCE_RADIUS = 50.0


# ---------------------------------------------------------------------------
# InfluenceMap
# ---------------------------------------------------------------------------

class InfluenceMap:
    """Grid-based per-faction influence tracker.

    Each cell stores a float in [0, 1] per faction representing that
    faction's influence over the area.  Influence radiates from unit
    positions with configurable falloff, decays over time, and can be
    queried for controllers, contested zones, and frontlines.

    Parameters
    ----------
    width : int
        Number of columns in the grid.
    height : int
        Number of rows in the grid.
    cell_size : float
        World-space size of each cell in meters.  Default 10.0.
    """

    __slots__ = ("width", "height", "cell_size", "_grid")

    def __init__(self, width: int, height: int, cell_size: float = _DEFAULT_CELL_SIZE) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive integers")
        if cell_size <= 0:
            raise ValueError("cell_size must be positive")
        self.width = width
        self.height = height
        self.cell_size = cell_size
        # _grid[faction] -> list[list[float]] (row-major: _grid[f][y][x])
        self._grid: dict[str, list[list[float]]] = {}

    # -- Internal helpers ---------------------------------------------------

    def _ensure_faction(self, faction: str) -> None:
        """Create a zero grid for *faction* if it doesn't exist yet."""
        if faction not in self._grid:
            self._grid[faction] = [[0.0] * self.width for _ in range(self.height)]

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def _world_to_cell(self, pos: Vec2) -> tuple[int, int]:
        return (int(math.floor(pos[0] / self.cell_size)),
                int(math.floor(pos[1] / self.cell_size)))

    def _cell_to_world(self, x: int, y: int) -> Vec2:
        return ((x + 0.5) * self.cell_size, (y + 0.5) * self.cell_size)

    @property
    def factions(self) -> list[str]:
        """Return all factions that have any influence on the map."""
        return list(self._grid.keys())

    # -- Core operations ----------------------------------------------------

    def add_influence(
        self,
        faction: str,
        position: Vec2,
        strength: float,
        radius: float,
    ) -> None:
        """Add radial influence for *faction* centered at *position*.

        Influence falls off linearly from *strength* at center to 0 at
        *radius*.  Values are clamped to [0, 1].

        Parameters
        ----------
        faction : str
            Faction identifier.
        position : Vec2
            World-space center of influence.
        strength : float
            Peak influence value at center (clamped to [0, 1]).
        radius : float
            World-space radius of influence effect.
        """
        if radius <= 0 or strength <= 0:
            return
        strength = min(strength, 1.0)
        self._ensure_faction(faction)
        grid = self._grid[faction]

        cx, cy = self._world_to_cell(position)
        cell_radius = int(math.ceil(radius / self.cell_size))

        for dy in range(-cell_radius, cell_radius + 1):
            gy = cy + dy
            if gy < 0 or gy >= self.height:
                continue
            for dx in range(-cell_radius, cell_radius + 1):
                gx = cx + dx
                if gx < 0 or gx >= self.width:
                    continue
                cell_center = self._cell_to_world(gx, gy)
                dist = distance(position, cell_center)
                if dist >= radius:
                    continue
                falloff = 1.0 - (dist / radius)
                value = strength * falloff
                grid[gy][gx] = min(grid[gy][gx] + value, 1.0)

    def decay(self, rate: float = _DEFAULT_DECAY_RATE) -> None:
        """Decay all influence values by *rate* per tick.

        Each cell's value is reduced by *rate*, clamped to 0.
        """
        for faction_grid in self._grid.values():
            for y in range(self.height):
                row = faction_grid[y]
                for x in range(self.width):
                    if row[x] > 0:
                        row[x] = max(0.0, row[x] - rate)

    def get_controller(self, position: Vec2) -> str | None:
        """Return the faction with highest influence at *position*, or None.

        Returns None if no faction has influence above _MIN_INFLUENCE.
        """
        cx, cy = self._world_to_cell(position)
        if not self._in_bounds(cx, cy):
            return None

        best_faction: str | None = None
        best_value = _MIN_INFLUENCE
        for faction, grid in self._grid.items():
            val = grid[cy][cx]
            if val > best_value:
                best_value = val
                best_faction = faction
        return best_faction

    def get_influence(self, position: Vec2, faction: str) -> float:
        """Return the influence value for *faction* at *position*.

        Returns 0.0 for unknown factions or out-of-bounds positions.
        """
        cx, cy = self._world_to_cell(position)
        if not self._in_bounds(cx, cy):
            return 0.0
        if faction not in self._grid:
            return 0.0
        return self._grid[faction][cy][cx]

    def get_contested_zones(self) -> list[dict[str, Any]]:
        """Return areas where two or more factions have near-equal influence.

        Each returned dict contains:
            - ``position``: world-space center (Vec2)
            - ``cell``: grid indices (x, y)
            - ``factions``: dict of faction -> influence at that cell
            - ``spread``: difference between top two faction values

        Only cells where at least two factions exceed _MIN_INFLUENCE and
        the spread is within _CONTESTED_THRESHOLD are included.
        """
        zones: list[dict[str, Any]] = []
        faction_list = list(self._grid.keys())
        if len(faction_list) < 2:
            return zones

        for y in range(self.height):
            for x in range(self.width):
                values: list[tuple[float, str]] = []
                for f in faction_list:
                    v = self._grid[f][y][x]
                    if v > _MIN_INFLUENCE:
                        values.append((v, f))
                if len(values) < 2:
                    continue
                values.sort(reverse=True)
                spread = values[0][0] - values[1][0]
                if spread <= _CONTESTED_THRESHOLD:
                    factions_dict = {f: v for v, f in values}
                    zones.append({
                        "position": self._cell_to_world(x, y),
                        "cell": (x, y),
                        "factions": factions_dict,
                        "spread": spread,
                    })
        return zones

    def get_frontline(self, faction_a: str, faction_b: str) -> list[Vec2]:
        """Return world-space positions of cells on the border between two factions.

        A cell is on the frontline if it is controlled by *faction_a* and
        at least one neighbor is controlled by *faction_b*, or vice versa.
        """
        if faction_a not in self._grid or faction_b not in self._grid:
            return []

        grid_a = self._grid[faction_a]
        grid_b = self._grid[faction_b]
        frontline: list[Vec2] = []

        for y in range(self.height):
            for x in range(self.width):
                a_val = grid_a[y][x]
                b_val = grid_b[y][x]
                # Determine which faction controls this cell
                if a_val <= _MIN_INFLUENCE and b_val <= _MIN_INFLUENCE:
                    continue
                my_faction = "a" if a_val >= b_val else "b"

                # Check 4-connected neighbors for different controller
                is_border = False
                for nx, ny in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                    if not self._in_bounds(nx, ny):
                        continue
                    na_val = grid_a[ny][nx]
                    nb_val = grid_b[ny][nx]
                    if na_val <= _MIN_INFLUENCE and nb_val <= _MIN_INFLUENCE:
                        continue
                    neighbor_faction = "a" if na_val >= nb_val else "b"
                    if neighbor_faction != my_faction:
                        is_border = True
                        break

                if is_border:
                    frontline.append(self._cell_to_world(x, y))

        return frontline

    def tick(
        self,
        dt: float,
        unit_positions: dict[str, tuple[Vec2, str]],
    ) -> None:
        """Advance the influence map by *dt* seconds.

        Units project influence based on their position and faction.
        Existing influence decays.

        Parameters
        ----------
        dt : float
            Time step in seconds.
        unit_positions : dict
            Mapping of unit_id -> (position, faction).
        """
        # Decay first
        self.decay(rate=_DEFAULT_DECAY_RATE * dt)

        # Units project influence
        for _uid, (pos, faction) in unit_positions.items():
            scaled_strength = _UNIT_INFLUENCE_STRENGTH * dt
            self.add_influence(faction, pos, scaled_strength, _UNIT_INFLUENCE_RADIUS)

    def clear(self) -> None:
        """Reset all influence to zero."""
        self._grid.clear()

    def clear_faction(self, faction: str) -> None:
        """Remove all influence for a specific faction."""
        if faction in self._grid:
            del self._grid[faction]

    def total_influence(self, faction: str) -> float:
        """Sum of all influence values for *faction*."""
        if faction not in self._grid:
            return 0.0
        total = 0.0
        for row in self._grid[faction]:
            for val in row:
                total += val
        return total

    def controlled_cell_count(self, faction: str) -> int:
        """Number of cells where *faction* has the highest influence."""
        count = 0
        for y in range(self.height):
            for x in range(self.width):
                best_f = None
                best_v = _MIN_INFLUENCE
                for f, grid in self._grid.items():
                    v = grid[y][x]
                    if v > best_v:
                        best_v = v
                        best_f = f
                if best_f == faction:
                    count += 1
        return count

    def to_three_js(self) -> dict[str, Any]:
        """Export influence data for Three.js rendering.

        Returns a dict with:
            - ``width``, ``height``, ``cell_size``: grid dimensions
            - ``heatmaps``: dict of faction -> flat list of influence values
                            (row-major, length = width * height)
            - ``frontlines``: dict of ``"factionA_vs_factionB"`` -> list of Vec2
        """
        heatmaps: dict[str, list[float]] = {}
        for faction, grid in self._grid.items():
            flat: list[float] = []
            for row in grid:
                flat.extend(row)
            heatmaps[faction] = flat

        # Compute frontlines for all faction pairs
        frontlines: dict[str, list[Vec2]] = {}
        faction_list = list(self._grid.keys())
        for i in range(len(faction_list)):
            for j in range(i + 1, len(faction_list)):
                fa, fb = faction_list[i], faction_list[j]
                fl = self.get_frontline(fa, fb)
                if fl:
                    frontlines[f"{fa}_vs_{fb}"] = fl

        return {
            "width": self.width,
            "height": self.height,
            "cell_size": self.cell_size,
            "heatmaps": heatmaps,
            "frontlines": frontlines,
        }


# ---------------------------------------------------------------------------
# CaptureState — for control point progress tracking
# ---------------------------------------------------------------------------

class CaptureState(Enum):
    """State of a control point."""
    NEUTRAL = "neutral"
    CAPTURING = "capturing"
    CONTESTED = "contested"
    CAPTURED = "captured"


# ---------------------------------------------------------------------------
# TerritoryControl
# ---------------------------------------------------------------------------

@dataclass
class ControlPoint:
    """A strategic control point on the map.

    Attributes
    ----------
    point_id : str
        Unique identifier.
    position : Vec2
        World-space location.
    name : str
        Human-readable name.
    capture_radius : float
        World-space radius within which units can capture.
    capture_time : float
        Seconds required for a single unit to capture (more units = faster).
    value : float
        Strategic value multiplier (1.0 = normal).
    owner : str or None
        Faction that currently controls this point.
    """
    point_id: str
    position: Vec2
    name: str = ""
    capture_radius: float = _CAPTURE_RANGE
    capture_time: float = 10.0
    value: float = 1.0
    owner: str | None = None


class TerritoryControl:
    """Strategic control point manager with capture/contest logic.

    Parameters
    ----------
    influence_map : InfluenceMap or None
        If provided, capturing a control point also projects influence.
    """

    def __init__(self, influence_map: InfluenceMap | None = None) -> None:
        self.control_points: list[ControlPoint] = []
        self.capture_progress: dict[str, dict[str, float]] = {}
        # Maps point_id -> {faction: progress_0_to_1}
        self._influence_map = influence_map

    def add_control_point(self, point: ControlPoint) -> None:
        """Register a control point."""
        self.control_points.append(point)
        self.capture_progress[point.point_id] = {}

    def remove_control_point(self, point_id: str) -> None:
        """Remove a control point by ID."""
        self.control_points = [p for p in self.control_points if p.point_id != point_id]
        self.capture_progress.pop(point_id, None)

    def get_point(self, point_id: str) -> ControlPoint | None:
        """Look up a control point by ID."""
        for p in self.control_points:
            if p.point_id == point_id:
                return p
        return None

    def _units_near_point(
        self,
        point: ControlPoint,
        unit_positions: dict[str, tuple[Vec2, str]],
    ) -> dict[str, int]:
        """Count units per faction within capture radius of *point*."""
        counts: dict[str, int] = {}
        for _uid, (pos, faction) in unit_positions.items():
            if distance(pos, point.position) <= point.capture_radius:
                counts[faction] = counts.get(faction, 0) + 1
        return counts

    def tick(
        self,
        dt: float,
        unit_positions: dict[str, tuple[Vec2, str]],
    ) -> list[dict[str, Any]]:
        """Advance capture logic by *dt* seconds.

        Returns a list of event dicts for state changes (captures, contests).

        Parameters
        ----------
        dt : float
            Time step in seconds.
        unit_positions : dict
            Mapping of unit_id -> (position, faction).
        """
        events: list[dict[str, Any]] = []

        for point in self.control_points:
            faction_counts = self._units_near_point(point, unit_positions)
            progress = self.capture_progress.setdefault(point.point_id, {})

            if not faction_counts:
                # No units nearby — progress decays toward 0
                for faction in list(progress.keys()):
                    progress[faction] = max(0.0, progress[faction] - _CAPTURE_RATE * dt * 0.5)
                    if progress[faction] <= 0:
                        del progress[faction]
                continue

            # Determine if contested (multiple factions present)
            active_factions = [f for f, c in faction_counts.items() if c > 0]

            if len(active_factions) > 1:
                # Contested — no progress for anyone, emit event
                events.append({
                    "type": "contested",
                    "point_id": point.point_id,
                    "point_name": point.name,
                    "factions": dict(faction_counts),
                })
                continue

            # Single faction capturing
            faction = active_factions[0]
            unit_count = faction_counts[faction]
            rate = _CAPTURE_RATE * dt * math.sqrt(unit_count)
            # More units = faster, but sqrt diminishing returns

            # If another faction had progress, reduce it first
            for other_f in list(progress.keys()):
                if other_f != faction:
                    progress[other_f] = max(0.0, progress[other_f] - rate)
                    if progress[other_f] <= 0:
                        del progress[other_f]

            # Advance capturing faction
            old_progress = progress.get(faction, 0.0)
            new_progress = min(1.0, old_progress + rate / max(point.capture_time, 0.1))
            progress[faction] = new_progress

            if new_progress >= 1.0 and point.owner != faction:
                old_owner = point.owner
                point.owner = faction
                events.append({
                    "type": "captured",
                    "point_id": point.point_id,
                    "point_name": point.name,
                    "faction": faction,
                    "previous_owner": old_owner,
                })
                # Project bonus influence if linked to an influence map
                if self._influence_map is not None:
                    self._influence_map.add_influence(
                        faction, point.position,
                        0.8 * point.value, point.capture_radius * 2,
                    )

        return events

    def get_state(self, point_id: str) -> CaptureState:
        """Current capture state of a control point."""
        point = self.get_point(point_id)
        if point is None:
            return CaptureState.NEUTRAL

        progress = self.capture_progress.get(point_id, {})
        if not progress:
            if point.owner:
                return CaptureState.CAPTURED
            return CaptureState.NEUTRAL

        active = {f: v for f, v in progress.items() if v > _MIN_INFLUENCE}
        if len(active) > 1:
            return CaptureState.CONTESTED
        if len(active) == 1:
            faction = next(iter(active))
            if active[faction] >= 1.0 and point.owner == faction:
                return CaptureState.CAPTURED
            return CaptureState.CAPTURING
        if point.owner:
            return CaptureState.CAPTURED
        return CaptureState.NEUTRAL

    def get_territory_summary(self, faction: str) -> dict[str, Any]:
        """Return a summary of territory owned by *faction*.

        Returns
        -------
        dict with keys:
            - ``controlled_points``: list of ControlPoint dicts owned
            - ``total_value``: sum of values of owned points
            - ``capturing``: list of points being captured by faction
            - ``contested``: list of points where faction is contesting
            - ``total_points``: total number of control points
        """
        controlled: list[dict[str, Any]] = []
        capturing: list[dict[str, Any]] = []
        contested: list[dict[str, Any]] = []
        total_value = 0.0

        for point in self.control_points:
            point_info = {
                "point_id": point.point_id,
                "name": point.name,
                "position": point.position,
                "value": point.value,
            }
            if point.owner == faction:
                controlled.append(point_info)
                total_value += point.value
            else:
                progress = self.capture_progress.get(point.point_id, {})
                if faction in progress and progress[faction] > _MIN_INFLUENCE:
                    # Check if contested
                    others = [f for f in progress if f != faction and progress[f] > _MIN_INFLUENCE]
                    if others:
                        contested.append({**point_info, "progress": progress[faction]})
                    else:
                        capturing.append({**point_info, "progress": progress[faction]})

        return {
            "controlled_points": controlled,
            "total_value": total_value,
            "capturing": capturing,
            "contested": contested,
            "total_points": len(self.control_points),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize state for API/frontend consumption."""
        points_list = []
        for p in self.control_points:
            state = self.get_state(p.point_id)
            progress = self.capture_progress.get(p.point_id, {})
            points_list.append({
                "point_id": p.point_id,
                "name": p.name,
                "position": p.position,
                "owner": p.owner,
                "value": p.value,
                "capture_radius": p.capture_radius,
                "state": state.value,
                "progress": dict(progress),
            })
        return {"control_points": points_list}


# ---------------------------------------------------------------------------
# StrategicValue
# ---------------------------------------------------------------------------

class StrategicValue:
    """Rate positions by tactical value.

    Considers elevation (hilltops), road proximity, building density,
    and crossroad intersections to produce a 0-1 strategic value score.

    Parameters
    ----------
    cell_size : float
        World-space size per evaluation cell.
    width : int
        Grid width in cells.
    height : int
        Grid height in cells.
    """

    __slots__ = ("cell_size", "width", "height",
                 "_elevation", "_roads", "_buildings", "_cache")

    def __init__(self, width: int, height: int, cell_size: float = _DEFAULT_CELL_SIZE) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive")
        if cell_size <= 0:
            raise ValueError("cell_size must be positive")
        self.width = width
        self.height = height
        self.cell_size = cell_size
        # Optional data layers
        self._elevation: list[list[float]] = [[0.0] * width for _ in range(height)]
        self._roads: set[tuple[int, int]] = set()  # Cells containing roads
        self._buildings: set[tuple[int, int]] = set()  # Cells containing buildings
        self._cache: dict[tuple[int, int], float] = {}

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def _world_to_cell(self, pos: Vec2) -> tuple[int, int]:
        return (int(math.floor(pos[0] / self.cell_size)),
                int(math.floor(pos[1] / self.cell_size)))

    def _cell_to_world(self, x: int, y: int) -> Vec2:
        return ((x + 0.5) * self.cell_size, (y + 0.5) * self.cell_size)

    def set_elevation(self, x: int, y: int, elevation: float) -> None:
        """Set terrain elevation at grid cell (x, y)."""
        if self._in_bounds(x, y):
            self._elevation[y][x] = elevation
            self._cache.pop((x, y), None)

    def set_elevation_from_heightmap(self, heightmap: Any) -> None:
        """Import elevation from a HeightMap (from terrain module).

        Resamples the heightmap to match this grid's resolution.
        """
        for y in range(self.height):
            for x in range(self.width):
                world_pos = self._cell_to_world(x, y)
                try:
                    self._elevation[y][x] = heightmap.get_elevation_world(world_pos)
                except (AttributeError, IndexError):
                    pass
        self._cache.clear()

    def add_road(self, position: Vec2) -> None:
        """Mark a cell as containing a road."""
        cx, cy = self._world_to_cell(position)
        if self._in_bounds(cx, cy):
            self._roads.add((cx, cy))
            self._cache.pop((cx, cy), None)

    def add_road_segment(self, start: Vec2, end: Vec2) -> None:
        """Mark all cells along a line segment as road cells."""
        sx, sy = self._world_to_cell(start)
        ex, ey = self._world_to_cell(end)
        # Simple DDA line rasterization
        dx = ex - sx
        dy = ey - sy
        steps = max(abs(dx), abs(dy), 1)
        for i in range(steps + 1):
            t = i / steps
            cx = int(round(sx + dx * t))
            cy = int(round(sy + dy * t))
            if self._in_bounds(cx, cy):
                self._roads.add((cx, cy))
                self._cache.pop((cx, cy), None)

    def add_building(self, position: Vec2) -> None:
        """Mark a cell as containing a building."""
        cx, cy = self._world_to_cell(position)
        if self._in_bounds(cx, cy):
            self._buildings.add((cx, cy))
            self._cache.pop((cx, cy), None)

    def _elevation_score(self, x: int, y: int) -> float:
        """Score based on relative elevation — hilltops score higher.

        Compares this cell's elevation to its neighbors.  Being higher
        than all neighbors yields max score.
        """
        if not self._in_bounds(x, y):
            return 0.0
        my_elev = self._elevation[y][x]
        higher_count = 0
        neighbor_count = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if self._in_bounds(nx, ny):
                    neighbor_count += 1
                    if my_elev >= self._elevation[ny][nx]:
                        higher_count += 1
        if neighbor_count == 0:
            return 0.0
        return higher_count / neighbor_count

    def _road_score(self, x: int, y: int) -> float:
        """Score based on road proximity and crossroads.

        Being on a road = 0.5.  Being at an intersection (3+ road neighbors) = 1.0.
        Adjacent to road = 0.25.
        """
        if (x, y) in self._roads:
            # Count road neighbors for crossroad detection
            road_neighbors = 0
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    if (x + dx, y + dy) in self._roads:
                        road_neighbors += 1
            if road_neighbors >= 3:
                return 1.0  # Crossroads
            return 0.5  # On a road
        # Check adjacency
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                if (x + dx, y + dy) in self._roads:
                    return 0.25
        return 0.0

    def _building_score(self, x: int, y: int) -> float:
        """Score based on building density in the neighborhood.

        More nearby buildings = higher value (urban areas are strategic).
        """
        count = 0
        radius = 2
        total = 0
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if self._in_bounds(nx, ny):
                    total += 1
                    if (nx, ny) in self._buildings:
                        count += 1
        if total == 0:
            return 0.0
        return min(count / max(total * 0.3, 1), 1.0)

    def rate_cell(self, x: int, y: int) -> float:
        """Compute strategic value of grid cell (x, y) in [0, 1].

        Weighted combination of elevation, road, and building scores.
        """
        if not self._in_bounds(x, y):
            return 0.0
        if (x, y) in self._cache:
            return self._cache[(x, y)]

        elev = self._elevation_score(x, y)
        road = self._road_score(x, y)
        bldg = self._building_score(x, y)

        # Weighted combination: elevation 40%, roads 35%, buildings 25%
        score = 0.40 * elev + 0.35 * road + 0.25 * bldg
        score = min(max(score, 0.0), 1.0)
        self._cache[(x, y)] = score
        return score

    def rate_position(self, position: Vec2) -> float:
        """Compute strategic value of a world-space position in [0, 1]."""
        cx, cy = self._world_to_cell(position)
        return self.rate_cell(cx, cy)

    def find_high_value_positions(
        self,
        min_value: float = 0.5,
        max_results: int = 20,
    ) -> list[dict[str, Any]]:
        """Find the highest strategic-value cells on the map.

        Returns a sorted list (highest first) of dicts with ``position``,
        ``cell``, ``value``, and component scores.
        """
        candidates: list[tuple[float, int, int]] = []
        for y in range(self.height):
            for x in range(self.width):
                val = self.rate_cell(x, y)
                if val >= min_value:
                    candidates.append((val, x, y))

        candidates.sort(reverse=True)
        results: list[dict[str, Any]] = []
        for val, x, y in candidates[:max_results]:
            results.append({
                "position": self._cell_to_world(x, y),
                "cell": (x, y),
                "value": val,
                "elevation_score": self._elevation_score(x, y),
                "road_score": self._road_score(x, y),
                "building_score": self._building_score(x, y),
            })
        return results

    def to_grid(self) -> list[list[float]]:
        """Return the full strategic value grid as a 2D list (row-major)."""
        return [[self.rate_cell(x, y) for x in range(self.width)]
                for y in range(self.height)]
