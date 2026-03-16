# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Terrain height maps, line-of-sight, cover analysis, and movement cost.

Provides four main classes:

    HeightMap     — 2D elevation grid with procedural generation
    LineOfSight   — Bresenham ray-march visibility checks
    CoverMap      — Obstacle-aware cover evaluation
    MovementCost  — Slope-based movement penalties

All spatial coordinates use Vec2 = tuple[float, float] from the steering
module. Grid cells are indexed by integer (x, y) pairs where x is the
column and y is the row. World positions are converted to grid cells by
dividing by cell_size.
"""

from __future__ import annotations

import math
import random
from typing import Optional

# Vec2 type alias — same definition as in ai.steering
Vec2 = tuple[float, float]

# ---------------------------------------------------------------------------
# Optional numpy acceleration
# ---------------------------------------------------------------------------
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False


# ---------------------------------------------------------------------------
# Value noise (simple Perlin-like, no external deps)
# ---------------------------------------------------------------------------

def _hash_2d(ix: int, iy: int, seed: int) -> float:
    """Deterministic pseudo-random float in [0, 1) for grid point (ix, iy)."""
    # Simple integer hash
    h = seed
    h ^= ix * 374761393
    h ^= iy * 668265263
    h = (h * 1274126177) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 1911520717) & 0xFFFFFFFF
    h ^= h >> 16
    return (h & 0x7FFFFFFF) / 0x7FFFFFFF


def _smoothstep(t: float) -> float:
    """Hermite smoothstep for interpolation."""
    return t * t * (3.0 - 2.0 * t)


def _value_noise_2d(x: float, y: float, seed: int) -> float:
    """Single-octave value noise returning a float in roughly [0, 1]."""
    ix = int(math.floor(x))
    iy = int(math.floor(y))
    fx = x - ix
    fy = y - iy
    sx = _smoothstep(fx)
    sy = _smoothstep(fy)
    n00 = _hash_2d(ix, iy, seed)
    n10 = _hash_2d(ix + 1, iy, seed)
    n01 = _hash_2d(ix, iy + 1, seed)
    n11 = _hash_2d(ix + 1, iy + 1, seed)
    nx0 = n00 + sx * (n10 - n00)
    nx1 = n01 + sx * (n11 - n01)
    return nx0 + sy * (nx1 - nx0)


def _fbm(x: float, y: float, octaves: int, seed: int) -> float:
    """Fractal Brownian motion — layered value noise."""
    value = 0.0
    amplitude = 1.0
    frequency = 1.0
    total_amp = 0.0
    for i in range(octaves):
        value += amplitude * _value_noise_2d(x * frequency, y * frequency, seed + i * 31)
        total_amp += amplitude
        amplitude *= 0.5
        frequency *= 2.0
    return value / total_amp if total_amp > 0 else 0.0


# ===========================================================================
# HeightMap
# ===========================================================================

class HeightMap:
    """2D grid of elevation values.

    Parameters
    ----------
    width : int
        Number of columns.
    height : int
        Number of rows.
    cell_size : float
        World-space size of each cell (meters). Default 1.0.
    """

    __slots__ = ("width", "height", "cell_size", "_data")

    def __init__(self, width: int, height: int, cell_size: float = 1.0) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive integers")
        if cell_size <= 0:
            raise ValueError("cell_size must be positive")
        self.width = width
        self.height = height
        self.cell_size = cell_size
        if _HAS_NUMPY:
            self._data = np.zeros((height, width), dtype=np.float64)
        else:
            self._data: list[list[float]] = [[0.0] * width for _ in range(height)]

    # -- Element access -----------------------------------------------------

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def set_elevation(self, x: int, y: int, elevation: float) -> None:
        """Set elevation at grid cell (x, y)."""
        if not self._in_bounds(x, y):
            raise IndexError(f"Cell ({x}, {y}) out of bounds ({self.width}x{self.height})")
        self._data[y][x] = elevation

    def get_elevation(self, x: int, y: int) -> float:
        """Get elevation at grid cell (x, y). Returns 0.0 for out-of-bounds."""
        if not self._in_bounds(x, y):
            return 0.0
        return float(self._data[y][x])

    def world_to_cell(self, pos: Vec2) -> tuple[int, int]:
        """Convert world position to grid cell indices."""
        return (int(math.floor(pos[0] / self.cell_size)),
                int(math.floor(pos[1] / self.cell_size)))

    def cell_to_world(self, x: int, y: int) -> Vec2:
        """Convert grid cell to world-space center position."""
        return ((x + 0.5) * self.cell_size, (y + 0.5) * self.cell_size)

    def get_elevation_world(self, pos: Vec2) -> float:
        """Get elevation at a world-space position (nearest cell)."""
        cx, cy = self.world_to_cell(pos)
        return self.get_elevation(cx, cy)

    # -- Constructors -------------------------------------------------------

    @classmethod
    def from_noise(
        cls,
        width: int,
        height: int,
        cell_size: float = 1.0,
        octaves: int = 4,
        seed: Optional[int] = None,
        scale: float = 10.0,
        amplitude: float = 20.0,
    ) -> "HeightMap":
        """Generate procedural terrain using value noise.

        Parameters
        ----------
        scale : float
            Spatial frequency scale (higher = more zoomed out). Default 10.
        amplitude : float
            Maximum elevation produced. Default 20.
        """
        if seed is None:
            seed = random.randint(0, 2**31)
        hm = cls(width, height, cell_size)
        for y in range(height):
            for x in range(width):
                nx = x / max(scale, 0.001)
                ny = y / max(scale, 0.001)
                hm._data[y][x] = _fbm(nx, ny, octaves, seed) * amplitude
        return hm

    @classmethod
    def from_array(cls, data: list[list[float]], cell_size: float = 1.0) -> "HeightMap":
        """Create from a 2D nested list (rows x cols). data[y][x]."""
        if not data or not data[0]:
            raise ValueError("Data must be non-empty 2D array")
        height = len(data)
        width = len(data[0])
        hm = cls(width, height, cell_size)
        for y in range(height):
            if len(data[y]) != width:
                raise ValueError("All rows must have the same width")
            for x in range(width):
                hm._data[y][x] = float(data[y][x])
        return hm

    # -- Analysis -----------------------------------------------------------

    def slope_at(self, x: int, y: int) -> float:
        """Slope angle in radians at cell (x, y) using central differences.

        Returns 0.0 for out-of-bounds cells.
        """
        if not self._in_bounds(x, y):
            return 0.0
        # Central difference with clamping at edges
        xl = max(x - 1, 0)
        xr = min(x + 1, self.width - 1)
        yl = max(y - 1, 0)
        yr = min(y + 1, self.height - 1)
        dzdx = (self.get_elevation(xr, y) - self.get_elevation(xl, y)) / (
            (xr - xl) * self.cell_size if xr != xl else self.cell_size
        )
        dzdy = (self.get_elevation(x, yr) - self.get_elevation(x, yl)) / (
            (yr - yl) * self.cell_size if yr != yl else self.cell_size
        )
        gradient_magnitude = math.hypot(dzdx, dzdy)
        return math.atan(gradient_magnitude)

    def normal_at(self, x: int, y: int) -> tuple[float, float, float]:
        """Surface normal vector at cell (x, y) as (nx, ny, nz).

        Returns (0, 0, 1) for flat terrain or out-of-bounds.
        """
        if not self._in_bounds(x, y):
            return (0.0, 0.0, 1.0)
        xl = max(x - 1, 0)
        xr = min(x + 1, self.width - 1)
        yl = max(y - 1, 0)
        yr = min(y + 1, self.height - 1)
        dx_dist = (xr - xl) * self.cell_size if xr != xl else self.cell_size
        dy_dist = (yr - yl) * self.cell_size if yr != yl else self.cell_size
        dzdx = (self.get_elevation(xr, y) - self.get_elevation(xl, y)) / dx_dist
        dzdy = (self.get_elevation(x, yr) - self.get_elevation(x, yl)) / dy_dist
        # Normal = (-dz/dx, -dz/dy, 1) normalized
        nx, ny, nz = -dzdx, -dzdy, 1.0
        length = math.sqrt(nx * nx + ny * ny + nz * nz)
        if length < 1e-12:
            return (0.0, 0.0, 1.0)
        return (nx / length, ny / length, nz / length)


# ===========================================================================
# LineOfSight
# ===========================================================================

class LineOfSight:
    """Bresenham ray-march line-of-sight checks over a HeightMap.

    Parameters
    ----------
    heightmap : HeightMap
        The terrain to check against.
    observer_height : float
        Default eye height above terrain (meters). Default 1.8.
    """

    __slots__ = ("heightmap", "observer_height")

    def __init__(self, heightmap: HeightMap, observer_height: float = 1.8) -> None:
        self.heightmap = heightmap
        self.observer_height = observer_height

    def can_see(
        self,
        from_pos: Vec2,
        to_pos: Vec2,
        from_height: Optional[float] = None,
        to_height: Optional[float] = None,
    ) -> bool:
        """Check if *from_pos* can see *to_pos* over terrain.

        Heights are above-terrain offsets. If None, uses observer_height.
        Uses Bresenham-style ray march along grid cells.
        """
        hm = self.heightmap
        x0, y0 = hm.world_to_cell(from_pos)
        x1, y1 = hm.world_to_cell(to_pos)

        fh = from_height if from_height is not None else self.observer_height
        th = to_height if to_height is not None else self.observer_height

        # Observer and target absolute elevations
        obs_z = hm.get_elevation(x0, y0) + fh
        tgt_z = hm.get_elevation(x1, y1) + th

        # Bresenham line cells
        cells = _bresenham(x0, y0, x1, y1)
        n = len(cells)
        if n <= 2:
            return True  # Adjacent or same cell — always visible

        # Check intermediate cells (skip first and last)
        for i in range(1, n - 1):
            cx, cy = cells[i]
            t = i / (n - 1)  # interpolation factor
            ray_z = obs_z + t * (tgt_z - obs_z)
            terrain_z = hm.get_elevation(cx, cy)
            if terrain_z >= ray_z:
                return False
        return True

    def visibility_map(self, from_pos: Vec2, radius: float) -> set[tuple[int, int]]:
        """Return all grid cells visible from *from_pos* within *radius* (world units)."""
        hm = self.heightmap
        cx, cy = hm.world_to_cell(from_pos)
        cell_radius = max(1, int(math.ceil(radius / hm.cell_size)))
        visible: set[tuple[int, int]] = set()

        for dy in range(-cell_radius, cell_radius + 1):
            for dx in range(-cell_radius, cell_radius + 1):
                tx, ty = cx + dx, cy + dy
                if not hm._in_bounds(tx, ty):
                    continue
                # Distance check in world space
                wp = hm.cell_to_world(tx, ty)
                dist = math.hypot(wp[0] - from_pos[0], wp[1] - from_pos[1])
                if dist > radius:
                    continue
                if self.can_see(from_pos, wp):
                    visible.add((tx, ty))
        return visible

    def find_defilade(
        self,
        from_pos: Vec2,
        threat_pos: Vec2,
        search_radius: float,
    ) -> list[Vec2]:
        """Find positions near *from_pos* that are hidden from *threat_pos*.

        Returns world-space positions sorted by distance from *from_pos* (nearest first).
        """
        hm = self.heightmap
        cx, cy = hm.world_to_cell(from_pos)
        cell_radius = max(1, int(math.ceil(search_radius / hm.cell_size)))
        hidden: list[tuple[float, Vec2]] = []

        for dy in range(-cell_radius, cell_radius + 1):
            for dx in range(-cell_radius, cell_radius + 1):
                tx, ty = cx + dx, cy + dy
                if not hm._in_bounds(tx, ty):
                    continue
                wp = hm.cell_to_world(tx, ty)
                dist = math.hypot(wp[0] - from_pos[0], wp[1] - from_pos[1])
                if dist > search_radius:
                    continue
                if not self.can_see(threat_pos, wp):
                    hidden.append((dist, wp))

        hidden.sort(key=lambda t: t[0])
        return [pos for _, pos in hidden]


def _bresenham(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Bresenham's line algorithm returning all cells along the ray."""
    cells: list[tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        cells.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return cells


# ===========================================================================
# CoverMap
# ===========================================================================

# 8 cardinal + intercardinal directions (unit vectors)
_EIGHT_DIRS: list[Vec2] = [
    (1.0, 0.0), (0.707, 0.707), (0.0, 1.0), (-0.707, 0.707),
    (-1.0, 0.0), (-0.707, -0.707), (0.0, -1.0), (0.707, -0.707),
]


class CoverMap:
    """Obstacle-based cover evaluation.

    Parameters
    ----------
    heightmap : HeightMap
        Terrain elevation data.
    obstacles : list of (center, radius)
        Each obstacle is a circular area providing hard cover.
    """

    __slots__ = ("heightmap", "obstacles")

    def __init__(
        self,
        heightmap: HeightMap,
        obstacles: Optional[list[tuple[Vec2, float]]] = None,
    ) -> None:
        self.heightmap = heightmap
        self.obstacles = obstacles or []

    def cover_value(self, pos: Vec2, threat_dir: Vec2) -> float:
        """Cover quality at *pos* against threats coming from *threat_dir*.

        Returns 0.0 (fully exposed) to 1.0 (full cover).
        Considers both terrain slope (being behind a ridge) and obstacles.
        """
        cover = 0.0

        # Terrain cover: check if there's higher ground between pos and threat direction
        hm = self.heightmap
        cx, cy = hm.world_to_cell(pos)
        my_elev = hm.get_elevation(cx, cy)

        # Normalize threat direction
        td_len = math.hypot(threat_dir[0], threat_dir[1])
        if td_len < 1e-9:
            return 0.0
        td_nx = threat_dir[0] / td_len
        td_ny = threat_dir[1] / td_len

        # Sample terrain in the threat direction (a few cells toward the threat)
        max_terrain_cover = 0.0
        for dist_cells in range(1, 4):
            sx = int(round(cx + td_nx * dist_cells))
            sy = int(round(cy + td_ny * dist_cells))
            if hm._in_bounds(sx, sy):
                elev_diff = hm.get_elevation(sx, sy) - my_elev
                if elev_diff > 0:
                    # Higher terrain in threat direction = cover
                    max_terrain_cover = max(max_terrain_cover, min(elev_diff / 3.0, 0.5))

        cover += max_terrain_cover

        # Obstacle cover: check if any obstacle is between pos and threat direction
        for obs_center, obs_radius in self.obstacles:
            # Vector from pos toward the threat
            to_obs_x = obs_center[0] - pos[0]
            to_obs_y = obs_center[1] - pos[1]
            obs_dist = math.hypot(to_obs_x, to_obs_y)
            if obs_dist < 1e-9:
                # Inside obstacle center
                cover = min(cover + 0.8, 1.0)
                continue

            # Project obstacle onto threat direction
            dot = (to_obs_x * td_nx + to_obs_y * td_ny)
            if dot <= 0:
                # Obstacle is behind us relative to threat — no cover
                continue

            # Perpendicular distance from threat line to obstacle center
            perp_dist = abs(to_obs_x * td_ny - to_obs_y * td_nx)

            if perp_dist < obs_radius:
                # Obstacle blocks the threat line
                # Cover quality depends on proximity and obstacle size
                proximity_factor = max(0.0, 1.0 - obs_dist / 10.0)
                blocking_factor = 1.0 - (perp_dist / obs_radius)
                obs_cover = min(blocking_factor * (0.5 + 0.5 * proximity_factor), 1.0)
                cover = min(cover + obs_cover, 1.0)

        return min(cover, 1.0)

    def generate_cover_grid(self, cell_size: float) -> dict[tuple[int, int], list[float]]:
        """Precompute cover values for each cell from 8 directions.

        Returns dict mapping (cell_x, cell_y) -> [cover_from_dir0, ..., cover_from_dir7].
        The 8 directions are E, NE, N, NW, W, SW, S, SE.
        """
        hm = self.heightmap
        grid_w = max(1, int(math.ceil(hm.width * hm.cell_size / cell_size)))
        grid_h = max(1, int(math.ceil(hm.height * hm.cell_size / cell_size)))
        result: dict[tuple[int, int], list[float]] = {}

        for gy in range(grid_h):
            for gx in range(grid_w):
                wx = (gx + 0.5) * cell_size
                wy = (gy + 0.5) * cell_size
                pos: Vec2 = (wx, wy)
                covers = [self.cover_value(pos, d) for d in _EIGHT_DIRS]
                result[(gx, gy)] = covers

        return result


# ===========================================================================
# MovementCost
# ===========================================================================

class MovementCost:
    """Slope-based movement cost over a HeightMap.

    Parameters
    ----------
    heightmap : HeightMap
        Terrain elevation data.
    base_cost : float
        Base movement cost per cell on flat terrain. Default 1.0.
    """

    __slots__ = ("heightmap", "base_cost")

    def __init__(self, heightmap: HeightMap, base_cost: float = 1.0) -> None:
        self.heightmap = heightmap
        self.base_cost = base_cost

    def cost(self, from_pos: Vec2, to_pos: Vec2) -> float:
        """Movement cost from *from_pos* to *to_pos*.

        Factors in distance and slope. Uphill costs more, downhill slightly less.
        """
        hm = self.heightmap
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        horiz_dist = math.hypot(dx, dy)
        if horiz_dist < 1e-9:
            return 0.0

        from_elev = hm.get_elevation_world(from_pos)
        to_elev = hm.get_elevation_world(to_pos)
        elev_diff = to_elev - from_elev

        # Slope factor: tan(angle) = elev_diff / horiz_dist
        slope = elev_diff / horiz_dist
        if slope > 0:
            # Uphill: exponentially harder
            slope_factor = 1.0 + 2.0 * slope * slope + slope
        else:
            # Downhill: slightly easier, but steep downhill is harder
            abs_slope = abs(slope)
            if abs_slope < 0.5:
                slope_factor = max(0.8, 1.0 - 0.2 * abs_slope)
            else:
                slope_factor = 1.0 + abs_slope  # Very steep downhill is also hard

        return self.base_cost * horiz_dist * slope_factor

    def max_speed_modifier(self, pos: Vec2) -> float:
        """Speed modifier at *pos*: 1.0 on flat terrain, approaches 0.0 on steep slopes.

        Returns a value in [0.0, 1.0].
        """
        hm = self.heightmap
        cx, cy = hm.world_to_cell(pos)
        slope = hm.slope_at(cx, cy)
        # Gentle linear falloff: 0 slope = 1.0, 45 deg (pi/4) = 0.2, 90 deg = 0.0
        modifier = max(0.0, 1.0 - slope / (math.pi / 2.0))
        return modifier
