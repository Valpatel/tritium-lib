# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Procedural tactical map generator.

Creates rich, playable maps with buildings, roads, rivers, terrain features,
spawn points, and objectives.  Every generated map is deterministic for a
given seed so that scenarios are reproducible.

Usage::

    from tritium_lib.sim_engine.mapgen import MapGenerator, MAP_PRESETS

    gen = MapGenerator(500, 500, seed=42)
    m = gen.generate_terrain("hilly")
    gen.add_city((250, 250), radius=80, density=0.6)
    gen.add_river((0, 100), (500, 400))
    gen.place_spawn_points(["blue", "red"])
    gen.place_objectives(3)
    result = gen.result()
"""

from __future__ import annotations

import enum
import math
import random
import uuid
from dataclasses import dataclass, field
from typing import Optional

from tritium_lib.sim_engine.terrain import (
    HeightMap,
    Vec2,
    _fbm,
    _value_noise_2d,
)


# ---------------------------------------------------------------------------
# TerrainType enum
# ---------------------------------------------------------------------------

class TerrainType(enum.Enum):
    """Terrain surface types that affect movement, cover, and visuals."""
    GRASS = "grass"
    DIRT = "dirt"
    SAND = "sand"
    ROCK = "rock"
    WATER = "water"
    SWAMP = "swamp"
    FOREST = "forest"
    URBAN = "urban"
    ROAD = "road"
    BRIDGE = "bridge"


# ---------------------------------------------------------------------------
# MapFeature
# ---------------------------------------------------------------------------

@dataclass
class MapFeature:
    """A discrete feature placed on the map (building, wall, tower, etc.)."""
    feature_id: str
    feature_type: str  # building, road, river, hill, forest, bridge, wall, tower
    position: Vec2
    size: tuple[float, float]
    rotation: float = 0.0
    properties: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# GeneratedMap
# ---------------------------------------------------------------------------

@dataclass
class GeneratedMap:
    """Complete output of the map generation process."""
    width: float
    height: float
    heightmap: list[list[float]]
    terrain_types: list[list[str]]
    features: list[MapFeature]
    spawn_points: dict[str, list[Vec2]]
    objectives: list[dict]
    roads: list[list[Vec2]]
    rivers: list[list[Vec2]]
    seed: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uid() -> str:
    return uuid.uuid4().hex[:12]


def _dist(a: Vec2, b: Vec2) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _meander(start: Vec2, end: Vec2, rng: random.Random,
             amplitude: float = 20.0, segments: int = 30,
             seed: int = 0) -> list[Vec2]:
    """Generate a meandering path from *start* to *end* using noise offsets."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return [start, end]
    # Perpendicular direction
    px, py = -dy / length, dx / length
    points: list[Vec2] = []
    for i in range(segments + 1):
        t = i / segments
        bx = start[0] + dx * t
        by = start[1] + dy * t
        # Noise offset perpendicular to the line
        noise_val = _value_noise_2d(t * 5.0, seed * 0.1 + 0.5, seed) * 2.0 - 1.0
        offset = noise_val * amplitude
        # Taper offset to zero at endpoints
        taper = math.sin(t * math.pi)
        points.append((bx + px * offset * taper, by + py * offset * taper))
    return points


def _point_in_bounds(p: Vec2, w: float, h: float, margin: float = 0.0) -> bool:
    return margin <= p[0] <= w - margin and margin <= p[1] <= h - margin


# ---------------------------------------------------------------------------
# MapGenerator
# ---------------------------------------------------------------------------

class MapGenerator:
    """Procedural tactical map generator.

    Parameters
    ----------
    width : float
        Map width in world units (meters).
    height : float
        Map height in world units (meters).
    cell_size : float
        Size of each terrain grid cell.  Determines heightmap resolution.
    seed : int or None
        Random seed for reproducible generation.  If None a random seed is
        chosen.
    """

    def __init__(
        self,
        width: float = 500.0,
        height: float = 500.0,
        cell_size: float = 5.0,
        seed: Optional[int] = None,
    ) -> None:
        self.width = float(width)
        self.height = float(height)
        self.cell_size = float(cell_size)
        self.seed: int = seed if seed is not None else random.randint(0, 2**31)
        self._rng = random.Random(self.seed)

        self._cols = max(1, int(math.ceil(self.width / self.cell_size)))
        self._rows = max(1, int(math.ceil(self.height / self.cell_size)))

        # Heightmap (row-major: [row][col])
        self._hm: list[list[float]] = [
            [0.0] * self._cols for _ in range(self._rows)
        ]
        # Terrain type per cell (row-major)
        self._terrain: list[list[str]] = [
            [TerrainType.GRASS.value] * self._cols for _ in range(self._rows)
        ]

        self._features: list[MapFeature] = []
        self._spawn_points: dict[str, list[Vec2]] = {}
        self._objectives: list[dict] = []
        self._roads: list[list[Vec2]] = []
        self._rivers: list[list[Vec2]] = []
        self._generated = False

    # -- Internal helpers ---------------------------------------------------

    def _world_to_cell(self, pos: Vec2) -> tuple[int, int]:
        cx = int(math.floor(pos[0] / self.cell_size))
        cy = int(math.floor(pos[1] / self.cell_size))
        return (_clamp_int(cx, 0, self._cols - 1), _clamp_int(cy, 0, self._rows - 1))

    def _in_bounds_cell(self, cx: int, cy: int) -> bool:
        return 0 <= cx < self._cols and 0 <= cy < self._rows

    def _paint_circle(self, center: Vec2, radius: float, terrain: str) -> None:
        """Set terrain type for all cells within *radius* of *center*."""
        cx0, cy0 = self._world_to_cell(
            (center[0] - radius, center[1] - radius)
        )
        cx1, cy1 = self._world_to_cell(
            (center[0] + radius, center[1] + radius)
        )
        for row in range(cy0, cy1 + 1):
            for col in range(cx0, cx1 + 1):
                if not self._in_bounds_cell(col, row):
                    continue
                wx = (col + 0.5) * self.cell_size
                wy = (row + 0.5) * self.cell_size
                if _dist((wx, wy), center) <= radius:
                    self._terrain[row][col] = terrain

    def _paint_road_cells(self, polyline: list[Vec2], width: float) -> None:
        """Mark cells along a polyline as ROAD terrain."""
        half_w = width / 2.0
        for i in range(len(polyline) - 1):
            a, b = polyline[i], polyline[i + 1]
            seg_len = _dist(a, b)
            if seg_len < 0.1:
                continue
            steps = max(1, int(seg_len / (self.cell_size * 0.5)))
            for s in range(steps + 1):
                t = s / steps
                px = _lerp(a[0], b[0], t)
                py = _lerp(a[1], b[1], t)
                # Paint cells within half_w of this point
                cr = max(1, int(math.ceil(half_w / self.cell_size)))
                ccx, ccy = self._world_to_cell((px, py))
                for dr in range(-cr, cr + 1):
                    for dc in range(-cr, cr + 1):
                        r2, c2 = ccy + dr, ccx + dc
                        if not self._in_bounds_cell(c2, r2):
                            continue
                        wx = (c2 + 0.5) * self.cell_size
                        wy = (r2 + 0.5) * self.cell_size
                        if _dist((wx, wy), (px, py)) <= half_w:
                            self._terrain[r2][c2] = TerrainType.ROAD.value

    def _paint_river_cells(self, polyline: list[Vec2], width: float) -> None:
        """Mark cells along a river polyline as WATER terrain."""
        half_w = width / 2.0
        for i in range(len(polyline) - 1):
            a, b = polyline[i], polyline[i + 1]
            seg_len = _dist(a, b)
            if seg_len < 0.1:
                continue
            steps = max(1, int(seg_len / (self.cell_size * 0.5)))
            for s in range(steps + 1):
                t = s / steps
                px = _lerp(a[0], b[0], t)
                py = _lerp(a[1], b[1], t)
                cr = max(1, int(math.ceil(half_w / self.cell_size)))
                ccx, ccy = self._world_to_cell((px, py))
                for dr in range(-cr, cr + 1):
                    for dc in range(-cr, cr + 1):
                        r2, c2 = ccy + dr, ccx + dc
                        if not self._in_bounds_cell(c2, r2):
                            continue
                        wx = (c2 + 0.5) * self.cell_size
                        wy = (r2 + 0.5) * self.cell_size
                        if _dist((wx, wy), (px, py)) <= half_w:
                            self._terrain[r2][c2] = TerrainType.WATER.value

    # -- Public API ---------------------------------------------------------

    def generate_terrain(self, style: str = "mixed") -> "MapGenerator":
        """Generate base terrain heightmap.

        Parameters
        ----------
        style : str
            One of ``"flat"``, ``"hilly"``, ``"mountainous"``, ``"coastal"``,
            ``"island"``, ``"valley"``, ``"mixed"``.

        Returns
        -------
        MapGenerator
            Self, for chaining.
        """
        s = self.seed
        for row in range(self._rows):
            for col in range(self._cols):
                nx = col / max(self._cols, 1) * 10.0
                ny = row / max(self._rows, 1) * 10.0

                if style == "flat":
                    h = _fbm(nx, ny, 2, s) * 2.0
                elif style == "hilly":
                    h = _fbm(nx, ny, 4, s) * 25.0
                elif style == "mountainous":
                    h = _fbm(nx, ny, 6, s) * 80.0
                elif style == "coastal":
                    # Left side is land, right side drops to water
                    land_factor = 1.0 - (col / self._cols)
                    h = _fbm(nx, ny, 4, s) * 30.0 * land_factor - 5.0
                elif style == "island":
                    # Distance from center → falloff
                    cx = col / self._cols - 0.5
                    cy = row / self._rows - 0.5
                    d = math.sqrt(cx * cx + cy * cy) * 2.0
                    falloff = max(0.0, 1.0 - d * 1.4)
                    h = _fbm(nx, ny, 4, s) * 30.0 * falloff - 3.0
                elif style == "valley":
                    # High at edges, low in center
                    cy = abs(row / self._rows - 0.5) * 2.0
                    h = _fbm(nx, ny, 4, s) * 20.0 + cy * cy * 40.0
                else:  # "mixed"
                    h = _fbm(nx, ny, 4, s) * 20.0

                self._hm[row][col] = h

        # Assign base terrain types from elevation
        for row in range(self._rows):
            for col in range(self._cols):
                elev = self._hm[row][col]
                if elev < -2.0:
                    self._terrain[row][col] = TerrainType.WATER.value
                elif elev < 0.0:
                    self._terrain[row][col] = TerrainType.SAND.value
                elif elev < 15.0:
                    self._terrain[row][col] = TerrainType.GRASS.value
                elif elev < 35.0:
                    self._terrain[row][col] = TerrainType.DIRT.value
                else:
                    self._terrain[row][col] = TerrainType.ROCK.value

        self._generated = True
        return self

    def add_city(
        self,
        center: Vec2,
        radius: float,
        density: float = 0.5,
    ) -> "MapGenerator":
        """Generate a city with a grid of roads and buildings.

        Parameters
        ----------
        center : Vec2
            World-space center of the city.
        radius : float
            Approximate radius of the urban area.
        density : float
            Building density 0.0-1.0.  Higher values produce more buildings.
        """
        density = _clamp(density, 0.0, 1.0)
        self._paint_circle(center, radius, TerrainType.URBAN.value)

        # Generate grid roads
        block_size = 30.0 + (1.0 - density) * 30.0  # 30-60m blocks
        road_width = 6.0
        x_start = center[0] - radius
        x_end = center[0] + radius
        y_start = center[1] - radius
        y_end = center[1] + radius

        # Horizontal roads
        y = y_start
        while y <= y_end:
            road_pts: list[Vec2] = [(x_start, y), (x_end, y)]
            self._roads.append(road_pts)
            self._paint_road_cells(road_pts, road_width)
            y += block_size

        # Vertical roads
        x = x_start
        while x <= x_end:
            road_pts = [(x, y_start), (x, y_end)]
            self._roads.append(road_pts)
            self._paint_road_cells(road_pts, road_width)
            x += block_size

        # Optional diagonal road through center
        if density > 0.3 and radius > 40:
            diag = [
                (center[0] - radius * 0.7, center[1] - radius * 0.7),
                (center[0] + radius * 0.7, center[1] + radius * 0.7),
            ]
            self._roads.append(diag)
            self._paint_road_cells(diag, road_width)

        # Place buildings in blocks
        building_types = [
            ("small_house", (6, 6), (10, 10)),
            ("medium_office", (12, 12), (20, 15)),
            ("large_warehouse", (20, 15), (30, 20)),
        ]
        x = x_start + road_width
        while x < x_end - road_width:
            y = y_start + road_width
            while y < y_end - road_width:
                if _dist((x, y), center) > radius:
                    y += block_size
                    continue
                if self._rng.random() > density:
                    y += block_size
                    continue

                # Pick building type weighted toward small
                r = self._rng.random()
                if r < 0.5:
                    bt = building_types[0]
                elif r < 0.8:
                    bt = building_types[1]
                else:
                    bt = building_types[2]

                bname, (min_w, min_h), (max_w, max_h) = bt
                bw = self._rng.uniform(min_w, max_w)
                bh = self._rng.uniform(min_h, max_h)
                rot = self._rng.choice([0.0, 90.0])
                height = self._rng.uniform(3.0, 12.0)
                mat = self._rng.choice(["concrete", "brick", "steel", "wood"])

                feat = MapFeature(
                    feature_id=f"bldg_{_uid()}",
                    feature_type="building",
                    position=(x + bw / 2, y + bh / 2),
                    size=(bw, bh),
                    rotation=rot,
                    properties={"material": mat, "height": height,
                                "building_class": bname},
                )
                self._features.append(feat)
                y += max(bh + 4, block_size * 0.4)
            x += block_size
        return self

    def add_village(
        self,
        center: Vec2,
        radius: float,
    ) -> "MapGenerator":
        """Generate a small village with scattered buildings and dirt roads.

        Parameters
        ----------
        center : Vec2
            World-space center.
        radius : float
            Approximate radius.
        """
        # A couple of dirt roads through the village
        for _ in range(self._rng.randint(2, 4)):
            angle = self._rng.uniform(0, math.pi * 2)
            start = (
                center[0] + math.cos(angle) * radius,
                center[1] + math.sin(angle) * radius,
            )
            end = (
                center[0] - math.cos(angle) * radius,
                center[1] - math.sin(angle) * radius,
            )
            pts = _meander(start, end, self._rng, amplitude=8.0,
                           segments=15, seed=self._rng.randint(0, 10000))
            self._roads.append(pts)
            self._paint_road_cells(pts, 4.0)
            # Paint surrounding dirt
            for p in pts:
                self._paint_circle(p, 8.0, TerrainType.DIRT.value)

        # Scatter buildings
        count = self._rng.randint(5, 15)
        for _ in range(count):
            angle = self._rng.uniform(0, math.pi * 2)
            dist = self._rng.uniform(0, radius * 0.85)
            bx = center[0] + math.cos(angle) * dist
            by = center[1] + math.sin(angle) * dist
            bw = self._rng.uniform(5, 12)
            bh = self._rng.uniform(5, 10)
            feat = MapFeature(
                feature_id=f"vbldg_{_uid()}",
                feature_type="building",
                position=(bx, by),
                size=(bw, bh),
                rotation=self._rng.uniform(0, 360),
                properties={
                    "material": self._rng.choice(["wood", "stone", "thatch"]),
                    "height": self._rng.uniform(2.5, 5.0),
                    "building_class": "cottage",
                },
            )
            self._features.append(feat)

        # Add some farm fields
        for _ in range(self._rng.randint(1, 3)):
            angle = self._rng.uniform(0, math.pi * 2)
            dist = self._rng.uniform(radius * 0.5, radius)
            fx = center[0] + math.cos(angle) * dist
            fy = center[1] + math.sin(angle) * dist
            self._paint_circle((fx, fy), self._rng.uniform(15, 30),
                               TerrainType.DIRT.value)

        return self

    def add_river(
        self,
        start: Vec2,
        end: Vec2,
        width: float = 10.0,
    ) -> "MapGenerator":
        """Add a meandering river.

        Bridges are automatically added where the river crosses existing roads.

        Parameters
        ----------
        start, end : Vec2
            Endpoints of the river.
        width : float
            River width in world units.
        """
        pts = _meander(start, end, self._rng, amplitude=width * 2.5,
                       segments=40, seed=self._rng.randint(0, 100000))
        self._rivers.append(pts)
        self._paint_river_cells(pts, width)

        # Depress heightmap along river
        for p in pts:
            cx, cy = self._world_to_cell(p)
            cr = max(1, int(math.ceil(width / self.cell_size)))
            for dr in range(-cr, cr + 1):
                for dc in range(-cr, cr + 1):
                    r2, c2 = cy + dr, cx + dc
                    if self._in_bounds_cell(c2, r2):
                        d = _dist(
                            ((c2 + 0.5) * self.cell_size,
                             (r2 + 0.5) * self.cell_size),
                            p,
                        )
                        if d <= width / 2:
                            self._hm[r2][c2] = min(self._hm[r2][c2], -3.0)

        # Auto-bridge at road crossings
        self._add_bridges_at_crossings(pts, width)
        return self

    def _add_bridges_at_crossings(
        self, river_pts: list[Vec2], river_width: float
    ) -> None:
        """Place bridge features where existing roads cross the river."""
        for road in self._roads:
            for ri in range(len(river_pts) - 1):
                rp = river_pts[ri]
                for rdi in range(len(road) - 1):
                    rdp = road[rdi]
                    if _dist(rp, rdp) < river_width * 2:
                        # Mark bridge
                        mid = (
                            (rp[0] + rdp[0]) / 2,
                            (rp[1] + rdp[1]) / 2,
                        )
                        feat = MapFeature(
                            feature_id=f"bridge_{_uid()}",
                            feature_type="bridge",
                            position=mid,
                            size=(river_width + 4, 8.0),
                            properties={"material": "concrete"},
                        )
                        self._features.append(feat)
                        # Paint bridge cells
                        bcx, bcy = self._world_to_cell(mid)
                        br = max(1, int(math.ceil((river_width + 4) /
                                                  self.cell_size / 2)))
                        for dr in range(-br, br + 1):
                            for dc in range(-br, br + 1):
                                r2, c2 = bcy + dr, bcx + dc
                                if self._in_bounds_cell(c2, r2):
                                    self._terrain[r2][c2] = TerrainType.BRIDGE.value
                        # Only one bridge per road segment
                        break

    def add_forest(
        self,
        center: Vec2,
        radius: float,
        density: float = 0.7,
    ) -> "MapGenerator":
        """Add a forested area.

        Trees are placed as features for LOS blocking and cover.

        Parameters
        ----------
        center : Vec2
            Center of the forest.
        radius : float
            Approximate radius.
        density : float
            Tree density 0.0-1.0.
        """
        density = _clamp(density, 0.0, 1.0)
        self._paint_circle(center, radius, TerrainType.FOREST.value)

        # Place individual tree clusters as features
        area = math.pi * radius * radius
        tree_count = int(area / 100.0 * density)
        for _ in range(tree_count):
            angle = self._rng.uniform(0, math.pi * 2)
            dist = self._rng.uniform(0, radius)
            tx = center[0] + math.cos(angle) * dist
            ty = center[1] + math.sin(angle) * dist
            if not _point_in_bounds((tx, ty), self.width, self.height):
                continue
            feat = MapFeature(
                feature_id=f"tree_{_uid()}",
                feature_type="forest",
                position=(tx, ty),
                size=(self._rng.uniform(2, 5), self._rng.uniform(2, 5)),
                rotation=self._rng.uniform(0, 360),
                properties={"tree_type": self._rng.choice(
                    ["oak", "pine", "birch", "maple"])},
            )
            self._features.append(feat)
        return self

    def add_road(
        self,
        start: Vec2,
        end: Vec2,
        width: float = 6.0,
    ) -> "MapGenerator":
        """Add a road (straight or gentle curve).

        Parameters
        ----------
        start, end : Vec2
            Road endpoints.
        width : float
            Road width.
        """
        # Gentle curve via small meander
        pts = _meander(start, end, self._rng, amplitude=width * 0.5,
                       segments=20, seed=self._rng.randint(0, 100000))
        self._roads.append(pts)
        self._paint_road_cells(pts, width)
        return self

    def place_spawn_points(
        self,
        factions: list[str],
        min_distance: float = 100.0,
    ) -> "MapGenerator":
        """Place spawn points for each faction, spread far apart near cover.

        Parameters
        ----------
        factions : list[str]
            Faction/alliance names (e.g. ["blue", "red"]).
        min_distance : float
            Minimum distance between faction spawn clusters.
        """
        margin = min(self.width, self.height) * 0.1
        placed: list[Vec2] = []

        for faction in factions:
            best: Optional[Vec2] = None
            best_min_dist = -1.0
            for _ in range(200):
                x = self._rng.uniform(margin, self.width - margin)
                y = self._rng.uniform(margin, self.height - margin)
                cand = (x, y)
                # Check terrain — avoid water
                cx, cy = self._world_to_cell(cand)
                if self._terrain[cy][cx] == TerrainType.WATER.value:
                    continue
                # Min dist from all previously placed spawns
                if placed:
                    md = min(_dist(cand, p) for p in placed)
                else:
                    md = self.width + self.height
                if md > best_min_dist:
                    best = cand
                    best_min_dist = md

            if best is None:
                best = (margin, margin)

            # Create a small cluster of spawn positions
            spawn_list: list[Vec2] = [best]
            for _ in range(3):
                sx = best[0] + self._rng.uniform(-15, 15)
                sy = best[1] + self._rng.uniform(-15, 15)
                sx = _clamp(sx, 0, self.width)
                sy = _clamp(sy, 0, self.height)
                spawn_list.append((sx, sy))
            self._spawn_points[faction] = spawn_list
            placed.extend(spawn_list)
        return self

    def place_objectives(self, count: int = 3) -> "MapGenerator":
        """Place strategic objectives (capture points, defend points).

        Objectives favour interesting locations: hilltops, crossroads,
        near buildings.

        Parameters
        ----------
        count : int
            Number of objectives to place.
        """
        placed: list[Vec2] = []
        for idx in range(count):
            best: Optional[Vec2] = None
            best_score = -1.0
            for _ in range(300):
                x = self._rng.uniform(20, self.width - 20)
                y = self._rng.uniform(20, self.height - 20)
                cand = (x, y)
                cx, cy = self._world_to_cell(cand)
                if self._terrain[cy][cx] == TerrainType.WATER.value:
                    continue

                score = 0.0
                # Prefer high ground
                score += self._hm[cy][cx] * 0.5
                # Prefer near buildings
                for f in self._features:
                    if f.feature_type == "building":
                        d = _dist(cand, f.position)
                        if d < 50:
                            score += 10.0 / (1.0 + d)
                # Prefer near roads
                for road in self._roads:
                    for rp in road:
                        d = _dist(cand, rp)
                        if d < 30:
                            score += 5.0 / (1.0 + d)
                            break
                # Spread objectives apart
                if placed:
                    md = min(_dist(cand, p) for p in placed)
                    score += md * 0.1

                if score > best_score:
                    best = cand
                    best_score = score

            if best is None:
                best = (self.width / 2, self.height / 2)

            obj_type = "capture_point" if idx % 2 == 0 else "defend_point"
            self._objectives.append({
                "id": f"obj_{_uid()}",
                "type": obj_type,
                "position": best,
                "radius": 15.0,
                "name": f"Objective {chr(65 + idx)}",
            })
            placed.append(best)
        return self

    # -- Output methods -----------------------------------------------------

    def result(self) -> GeneratedMap:
        """Return the completed GeneratedMap."""
        return GeneratedMap(
            width=self.width,
            height=self.height,
            heightmap=[row[:] for row in self._hm],
            terrain_types=[row[:] for row in self._terrain],
            features=list(self._features),
            spawn_points={k: list(v) for k, v in self._spawn_points.items()},
            objectives=list(self._objectives),
            roads=[list(r) for r in self._roads],
            rivers=[list(r) for r in self._rivers],
            seed=self.seed,
        )

    def to_three_js(self) -> dict:
        """Export full map data for Three.js rendering.

        Returns a JSON-serialisable dictionary with all map data needed
        by a 3D renderer.
        """
        features_out = []
        for f in self._features:
            features_out.append({
                "id": f.feature_id,
                "type": f.feature_type,
                "position": {"x": f.position[0], "y": f.position[1]},
                "size": {"w": f.size[0], "h": f.size[1]},
                "rotation": f.rotation,
                "properties": f.properties,
            })

        spawn_out: dict[str, list[dict]] = {}
        for faction, pts in self._spawn_points.items():
            spawn_out[faction] = [{"x": p[0], "y": p[1]} for p in pts]

        obj_out = []
        for obj in self._objectives:
            pos = obj["position"]
            obj_out.append({
                "id": obj["id"],
                "type": obj["type"],
                "position": {"x": pos[0], "y": pos[1]},
                "radius": obj["radius"],
                "name": obj["name"],
            })

        roads_out = []
        for road in self._roads:
            roads_out.append([{"x": p[0], "y": p[1]} for p in road])

        rivers_out = []
        for river in self._rivers:
            rivers_out.append([{"x": p[0], "y": p[1]} for p in river])

        return {
            "width": self.width,
            "height": self.height,
            "cell_size": self.cell_size,
            "seed": self.seed,
            "heightmap": self._hm,
            "terrain_types": self._terrain,
            "features": features_out,
            "spawn_points": spawn_out,
            "objectives": obj_out,
            "roads": roads_out,
            "rivers": rivers_out,
        }

    def to_heightmap(self) -> HeightMap:
        """Convert the internal heightmap to a sim_engine HeightMap object."""
        hm = HeightMap(self._cols, self._rows, self.cell_size)
        for row in range(self._rows):
            for col in range(self._cols):
                hm.set_elevation(col, row, self._hm[row][col])
        return hm


# ---------------------------------------------------------------------------
# Helper used by _world_to_cell
# ---------------------------------------------------------------------------

def _clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# MAP_PRESETS
# ---------------------------------------------------------------------------

def _build_city_block(seed: int) -> GeneratedMap:
    gen = MapGenerator(200, 200, cell_size=5.0, seed=seed)
    gen.generate_terrain("flat")
    gen.add_city((100, 100), radius=80, density=0.7)
    gen.place_spawn_points(["blue", "red"], min_distance=120)
    gen.place_objectives(2)
    return gen.result()


def _build_village(seed: int) -> GeneratedMap:
    gen = MapGenerator(300, 300, cell_size=5.0, seed=seed)
    gen.generate_terrain("hilly")
    gen.add_village((150, 150), radius=60)
    gen.add_forest((60, 60), radius=40)
    gen.add_forest((240, 240), radius=35)
    gen.place_spawn_points(["blue", "red"], min_distance=150)
    gen.place_objectives(2)
    return gen.result()


def _build_coastal_base(seed: int) -> GeneratedMap:
    gen = MapGenerator(500, 300, cell_size=5.0, seed=seed)
    gen.generate_terrain("coastal")
    gen.add_city((120, 150), radius=50, density=0.5)
    gen.add_road((0, 150), (500, 150))
    gen.place_spawn_points(["blue", "red"], min_distance=200)
    gen.place_objectives(3)
    return gen.result()


def _build_mountain_pass(seed: int) -> GeneratedMap:
    gen = MapGenerator(400, 200, cell_size=5.0, seed=seed)
    gen.generate_terrain("valley")
    gen.add_road((0, 100), (400, 100))
    gen.add_village((200, 100), radius=30)
    gen.place_spawn_points(["blue", "red"], min_distance=200)
    gen.place_objectives(2)
    return gen.result()


def _build_island(seed: int) -> GeneratedMap:
    gen = MapGenerator(300, 300, cell_size=5.0, seed=seed)
    gen.generate_terrain("island")
    gen.add_village((150, 150), radius=40)
    gen.add_forest((100, 100), radius=30)
    gen.place_spawn_points(["blue", "red"], min_distance=100)
    gen.place_objectives(2)
    return gen.result()


def _build_desert_town(seed: int) -> GeneratedMap:
    gen = MapGenerator(400, 400, cell_size=5.0, seed=seed)
    gen.generate_terrain("flat")
    # Override base terrain to sand
    for row in range(gen._rows):
        for col in range(gen._cols):
            gen._terrain[row][col] = TerrainType.SAND.value
    gen.add_city((200, 200), radius=60, density=0.4)
    gen.add_road((0, 200), (400, 200))
    gen.add_road((200, 0), (200, 400))
    gen.place_spawn_points(["blue", "red"], min_distance=200)
    gen.place_objectives(3)
    return gen.result()


def _build_forest_camp(seed: int) -> GeneratedMap:
    gen = MapGenerator(300, 300, cell_size=5.0, seed=seed)
    gen.generate_terrain("hilly")
    gen.add_forest((150, 150), radius=120, density=0.6)
    # Clearings
    gen._paint_circle((150, 150), 25, TerrainType.GRASS.value)
    gen._paint_circle((80, 220), 20, TerrainType.GRASS.value)
    gen.add_road((0, 150), (300, 150))
    gen.add_village((150, 150), radius=20)
    gen.place_spawn_points(["blue", "red"], min_distance=150)
    gen.place_objectives(3)
    return gen.result()


MAP_PRESETS: dict[str, callable] = {
    "city_block": _build_city_block,
    "village": _build_village,
    "coastal_base": _build_coastal_base,
    "mountain_pass": _build_mountain_pass,
    "island": _build_island,
    "desert_town": _build_desert_town,
    "forest_camp": _build_forest_camp,
}


def generate_preset(name: str, seed: int = 42) -> GeneratedMap:
    """Generate a map from a named preset.

    Parameters
    ----------
    name : str
        Preset name (one of MAP_PRESETS keys).
    seed : int
        Random seed.

    Returns
    -------
    GeneratedMap

    Raises
    ------
    KeyError
        If *name* is not a known preset.
    """
    if name not in MAP_PRESETS:
        raise KeyError(f"Unknown preset {name!r}. Available: {sorted(MAP_PRESETS)}")
    return MAP_PRESETS[name](seed)
