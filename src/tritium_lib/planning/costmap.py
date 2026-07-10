# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Layer-driven costmap generation.

A :class:`Costmap` is a regular grid of per-cell traversal costs in local
meters.  Cheap cells (roads) are preferred by the planner; ``LETHAL``
(``inf``) cells are impassable.  :class:`CostmapBuilder` stamps input
layers — obstacle polygons, road lines, and an elevation DEM — into a
single cost grid with a **deterministic application order** that does not
depend on the order the layers were added.

Grid convention (shared with telemetry and the DEM):
    - ``origin_x``/``origin_y`` is the south-west corner of the grid.
    - ``grid[row][col]``; ``row 0`` is the southernmost row, ``col 0`` the
      westernmost column.
    - ``grid_to_world`` returns **cell centers**.

Cost model (applied per cell, fixed order regardless of call order):
    1. ``base = base_cost + slope_weight * slope``
    2. ``* road_discount`` if the cell is a road cell
    3. ``* cost_zone_multiplier`` if the cell falls in one or more soft-cost
       zones (the MAX multiplier over covering zones — zones never compound
       and never make a cell lethal)
    4. ``slope > max_slope`` -> ``LETHAL``
    5. obstacle / water cell -> ``LETHAL``
    6. optional inflation: non-lethal cells within ``obstacle_inflation_m``
       of a lethal cell are raised to at least ``inflation_cost``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from tritium_lib.geo import point_in_polygon

from .layers import (
    LINE_TYPES,
    POLYGON_TYPES,
    LocalElevationGrid,
    iter_features,
)

__all__ = [
    "CostmapWeights",
    "Costmap",
    "CostmapBuilder",
    "MTFCC_WIDTHS_M",
    "builder_from_terrain_map",
    "costmap_from_terrain_map",
]


# ---------------------------------------------------------------------------
# TIGER/Line MTFCC -> stamp width (meters)
# ---------------------------------------------------------------------------
#
# Consumer-side road-width table.  The GIS lane ships TIGER road features
# tagged with an MTFCC class code (``properties.kind``) but does NOT define a
# physical width — width is a planning concern, so the mapping lives here.  A
# feature's own ``properties.width_m`` always wins over this table; an MTFCC
# code absent from the table falls back to ``weights.road_width_m``.
MTFCC_WIDTHS_M: dict[str, float] = {
    "S1100": 18.0,   # primary road (interstate / highway)
    "S1200": 12.0,   # secondary road
    "S1400": 8.0,    # local neighborhood street
    "S1630": 8.0,    # ramp
    "S1640": 8.0,    # service drive
    "S1710": 3.0,    # walkway / pedestrian trail
    "S1720": 3.0,    # stairway
    "S1730": 4.0,    # alley
}


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

@dataclass
class CostmapWeights:
    """Tunable weights for costmap generation.

    Attributes:
        base_cost: Baseline traversal cost of an open cell.
        road_discount: Multiplier applied to road cells (< 1 -> preferred).
        road_width_m: Default road stamp width when a road feature lacks a
            ``properties.width_m``.
        slope_weight: Added cost per unit of slope (``slope_weight * slope``).
        max_slope: Slope above this (rise/run) is lethal (~0.7 ≈ 35 deg).
        obstacle_inflation_m: Inflate non-lethal cells within this radius of
            a lethal cell (0 disables inflation).
        inflation_cost: Floor cost applied to inflated cells.
    """

    base_cost: float = 1.0
    road_discount: float = 0.5
    road_width_m: float = 8.0
    slope_weight: float = 5.0
    max_slope: float = 0.7
    obstacle_inflation_m: float = 0.0
    inflation_cost: float = 3.0


# ---------------------------------------------------------------------------
# Costmap
# ---------------------------------------------------------------------------

@dataclass
class Costmap:
    """A grid of per-cell traversal costs in local meters.

    Attributes:
        origin_x: South-west corner X (meters).
        origin_y: South-west corner Y (meters).
        resolution: Cell size (meters).
        width: Number of columns.
        height: Number of rows.
        grid: ``grid[row][col]`` float costs; ``row 0`` = south.
    """

    LETHAL: float = float("inf")

    origin_x: float = 0.0
    origin_y: float = 0.0
    resolution: float = 5.0
    width: int = 0
    height: int = 0
    grid: list[list[float]] = field(default_factory=list)

    # -- Coordinate conversion ---------------------------------------------

    def world_to_grid(self, x: float, y: float) -> tuple[int, int] | None:
        """World ``(x, y)`` -> ``(col, row)``.  ``None`` if out of bounds."""
        col = int(math.floor((x - self.origin_x) / self.resolution))
        row = int(math.floor((y - self.origin_y) / self.resolution))
        if col < 0 or col >= self.width or row < 0 or row >= self.height:
            return None
        return (col, row)

    def grid_to_world(self, col: int, row: int) -> tuple[float, float]:
        """``(col, row)`` -> world ``(x, y)`` at the **cell center**."""
        x = self.origin_x + (col + 0.5) * self.resolution
        y = self.origin_y + (row + 0.5) * self.resolution
        return (x, y)

    # -- Cost queries -------------------------------------------------------

    def in_bounds(self, col: int, row: int) -> bool:
        """True if ``(col, row)`` indexes a real cell."""
        return 0 <= col < self.width and 0 <= row < self.height

    def cost_at(self, col: int, row: int) -> float:
        """Cost of cell ``(col, row)``.  Out-of-bounds cells are ``LETHAL``."""
        if not self.in_bounds(col, row):
            return self.LETHAL
        return self.grid[row][col]

    def is_lethal(self, col: int, row: int) -> bool:
        """True if the cell is impassable (or out of bounds)."""
        return self.cost_at(col, row) == self.LETHAL

    def min_traversable_cost(self) -> float:
        """Minimum cost over non-lethal cells (floored at ``1e-6``)."""
        best = math.inf
        for row in self.grid:
            for c in row:
                if c != self.LETHAL and c < best:
                    best = c
        if best == math.inf:
            return 1e-6
        return max(best, 1e-6)

    def bounds(self) -> tuple[float, float, float, float]:
        """``(min_x, min_y, max_x, max_y)`` of the grid extent."""
        return (
            self.origin_x,
            self.origin_y,
            self.origin_x + self.width * self.resolution,
            self.origin_y + self.height * self.resolution,
        )

    # -- Telemetry ----------------------------------------------------------

    def to_telemetry(self, max_cells: int = 40000) -> dict:
        """Serialize to a JSON-friendly dict for frontend rendering.

        Returns ``{"grid", "cell_size", "bounds", "max_cost"}`` where
        ``grid`` is row-major (``row 0`` = south) with ``LETHAL`` encoded
        as ``-1.0``.  If ``width * height > max_cells`` the grid is
        downsampled by an integer stride: lethal wins within a block, else
        the block takes the max cost.  ``cell_size`` scales with the stride.
        """
        min_x, min_y, max_x, max_y = self.bounds()
        max_cost = 0.0
        for row in self.grid:
            for c in row:
                if c != self.LETHAL and c > max_cost:
                    max_cost = c

        total = self.width * self.height
        stride = 1
        if total > max_cells and total > 0:
            stride = int(math.ceil(math.sqrt(total / max_cells)))
            while (
                math.ceil(self.width / stride) * math.ceil(self.height / stride)
                > max_cells
            ):
                stride += 1

        if stride == 1:
            out = [
                [(-1.0 if c == self.LETHAL else c) for c in row]
                for row in self.grid
            ]
        else:
            out = []
            for row0 in range(0, self.height, stride):
                out_row: list[float] = []
                for col0 in range(0, self.width, stride):
                    block_lethal = False
                    block_max = 0.0
                    has_val = False
                    for r in range(row0, min(row0 + stride, self.height)):
                        for c in range(col0, min(col0 + stride, self.width)):
                            v = self.grid[r][c]
                            if v == self.LETHAL:
                                block_lethal = True
                            else:
                                has_val = True
                                if v > block_max:
                                    block_max = v
                    if block_lethal:
                        out_row.append(-1.0)
                    else:
                        out_row.append(block_max if has_val else 0.0)
                out.append(out_row)

        return {
            "grid": out,
            "cell_size": self.resolution * stride,
            "bounds": [min_x, min_y, max_x, max_y],
            "max_cost": max_cost,
        }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class CostmapBuilder:
    """Accumulate layers, then :meth:`build` a :class:`Costmap`.

    Layers may be added in any order; :meth:`build` applies them in a fixed
    deterministic order so the result is independent of call order.
    """

    def __init__(
        self,
        bounds: tuple[float, float, float, float],
        resolution: float = 5.0,
        weights: CostmapWeights | None = None,
    ) -> None:
        min_x, min_y, max_x, max_y = bounds
        self.origin_x = float(min_x)
        self.origin_y = float(min_y)
        self.resolution = float(resolution)
        self.width = max(1, int(math.ceil((max_x - min_x) / resolution)))
        self.height = max(1, int(math.ceil((max_y - min_y) / resolution)))
        self.weights = weights or CostmapWeights()

        # Layer accumulators keyed by (row, col).
        self._road_cells: set[tuple[int, int]] = set()
        # obstacle cell -> kind tag (last writer wins for introspection).
        self._obstacle_cells: dict[tuple[int, int], str] = {}
        # soft-cost zone cell -> MAX multiplier over covering zones.
        self._cost_zone_cells: dict[tuple[int, int], float] = {}
        self._dem: LocalElevationGrid | None = None

    # -- Cell helpers -------------------------------------------------------

    def _cell_center(self, col: int, row: int) -> tuple[float, float]:
        x = self.origin_x + (col + 0.5) * self.resolution
        y = self.origin_y + (row + 0.5) * self.resolution
        return (x, y)

    def _grid_range_for_bbox(
        self, min_x: float, min_y: float, max_x: float, max_y: float
    ) -> tuple[int, int, int, int]:
        """Inclusive ``(col_lo, col_hi, row_lo, row_hi)`` covering a bbox."""
        col_lo = int(math.floor((min_x - self.origin_x) / self.resolution))
        col_hi = int(math.floor((max_x - self.origin_x) / self.resolution))
        row_lo = int(math.floor((min_y - self.origin_y) / self.resolution))
        row_hi = int(math.floor((max_y - self.origin_y) / self.resolution))
        col_lo = max(0, col_lo)
        row_lo = max(0, row_lo)
        col_hi = min(self.width - 1, col_hi)
        row_hi = min(self.height - 1, row_hi)
        return (col_lo, col_hi, row_lo, row_hi)

    # -- Layer stamping -----------------------------------------------------

    def _stamp_polygon(self, ring: list[tuple[float, float]], mark) -> None:
        """Mark every cell whose center is inside the polygon ``ring``."""
        if len(ring) < 3:
            return
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        col_lo, col_hi, row_lo, row_hi = self._grid_range_for_bbox(
            min(xs), min(ys), max(xs), max(ys)
        )
        for row in range(row_lo, row_hi + 1):
            for col in range(col_lo, col_hi + 1):
                cx, cy = self._cell_center(col, row)
                if point_in_polygon(cx, cy, ring):
                    mark(col, row)

    def add_obstacles(
        self,
        feature_collection: dict,
        kind: str = "obstacle",
        to_local=None,
    ) -> "CostmapBuilder":
        """Stamp Polygon/MultiPolygon features as lethal cells.

        Use for buildings, water, and flood masks.  ``kind`` is a tag
        recorded per cell for introspection; the lethal semantics are the
        same regardless of tag.  Non-polygon features are ignored.
        """
        for gtype, seqs, _props in iter_features(feature_collection, to_local):
            if gtype not in POLYGON_TYPES:
                continue
            for ring in seqs:
                self._stamp_polygon(
                    ring, lambda c, r: self._obstacle_cells.__setitem__((r, c), kind)
                )
        return self

    def add_roads(
        self,
        feature_collection: dict,
        to_local=None,
    ) -> "CostmapBuilder":
        """Stamp road features as road (discounted) cells.

        LineString/MultiLineString features are stamped with a half-width of
        ``(properties.width_m or road_width_m) / 2``: every cell whose center
        lies within that half-width of any segment becomes a road cell.
        Polygon road features mark all covered cells.
        """
        for gtype, seqs, props in iter_features(feature_collection, to_local):
            if gtype in LINE_TYPES:
                width_m = props.get("width_m")
                if width_m is None or width_m <= 0:
                    width_m = self.weights.road_width_m
                half = float(width_m) / 2.0
                for line in seqs:
                    self._stamp_line(line, half)
            elif gtype in POLYGON_TYPES:
                for ring in seqs:
                    self._stamp_polygon(
                        ring, lambda c, r: self._road_cells.add((r, c))
                    )
        return self

    def _stamp_line(self, line: list[tuple[float, float]], half: float) -> None:
        """Mark cells within ``half`` meters of any segment of ``line``."""
        if len(line) < 2:
            if line:
                # Degenerate single point — stamp its cell if in bounds.
                col_lo, col_hi, row_lo, row_hi = self._grid_range_for_bbox(
                    line[0][0] - half, line[0][1] - half,
                    line[0][0] + half, line[0][1] + half,
                )
                for row in range(row_lo, row_hi + 1):
                    for col in range(col_lo, col_hi + 1):
                        cx, cy = self._cell_center(col, row)
                        if math.hypot(cx - line[0][0], cy - line[0][1]) <= half:
                            self._road_cells.add((row, col))
            return
        for (x0, y0), (x1, y1) in zip(line, line[1:]):
            seg_min_x = min(x0, x1) - half
            seg_max_x = max(x0, x1) + half
            seg_min_y = min(y0, y1) - half
            seg_max_y = max(y0, y1) + half
            col_lo, col_hi, row_lo, row_hi = self._grid_range_for_bbox(
                seg_min_x, seg_min_y, seg_max_x, seg_max_y
            )
            for row in range(row_lo, row_hi + 1):
                for col in range(col_lo, col_hi + 1):
                    cx, cy = self._cell_center(col, row)
                    if _point_segment_distance(cx, cy, x0, y0, x1, y1) <= half:
                        self._road_cells.add((row, col))

    def add_dem(self, elevation: LocalElevationGrid) -> "CostmapBuilder":
        """Attach a DEM used for per-cell slope cost at :meth:`build`.

        ``elevation`` is a :class:`~tritium_lib.planning.layers.LocalElevationGrid`
        (local meters, ``row 0`` = south).  To feed a GIS-lane WGS-84 wire
        grid, convert it first with
        :func:`~tritium_lib.planning.layers.local_grid_from_gis`.
        """
        self._dem = elevation
        return self

    # -- Semantic GIS ingestion --------------------------------------------

    def _mark_zone(self, row: int, col: int, multiplier: float) -> None:
        """Record ``multiplier`` for a soft-cost cell, keeping the MAX."""
        key = (row, col)
        cur = self._cost_zone_cells.get(key)
        if cur is None or multiplier > cur:
            self._cost_zone_cells[key] = multiplier

    def add_cost_zones(
        self,
        feature_collection: dict,
        multiplier: float,
        to_local=None,
    ) -> "CostmapBuilder":
        """Stamp Polygon/MultiPolygon features as soft-cost zones.

        Every cell whose center falls inside a zone has its final cost
        MULTIPLIED by ``multiplier`` (``> 1`` = avoid, ``< 1`` = prefer).
        Stacking rule: when several zones cover one cell the **MAX**
        multiplier is kept — zones never compound.  The multiplier is applied
        in :meth:`build` AFTER the road discount and BEFORE any lethal
        override, so a cost zone can raise a cell's cost but never make it
        lethal.  Non-polygon features are ignored.
        """
        mult = float(multiplier)
        for gtype, seqs, _props in iter_features(feature_collection, to_local):
            if gtype not in POLYGON_TYPES:
                continue
            for ring in seqs:
                self._stamp_polygon(
                    ring, lambda c, r, m=mult: self._mark_zone(r, c, m)
                )
        return self

    def add_gis_features(
        self,
        feature_collection: dict,
        to_local=None,
    ) -> dict:
        """Ingest a mixed GIS ``FeatureCollection`` routed by ``properties.source``.

        Dispatches each feature by its ``source`` tag (set by the GIS lane):

        - ``"tiger"`` -> road stamp.  Width is ``properties.width_m`` when
          present, else :data:`MTFCC_WIDTHS_M` keyed by ``properties.kind``
          (the MTFCC code), else ``weights.road_width_m``.
        - ``"fema"`` -> flood.  A truthy ``properties.sfha`` (Special Flood
          Hazard Area) stamps a **lethal** obstacle tagged ``"flood"``.
          ``sfha`` falsey (e.g. zone X minimal hazard) is traversable and
          IGNORED.  Rasterization is driven by ``sfha``/``kind`` only — never
          from style properties.
        - ``"noaa"`` -> weather alert.  ``properties.severity`` of ``"Severe"``
          or ``"Extreme"`` stamps a soft-cost zone with multiplier ``2.0``;
          other severities are IGNORED.  ``expires`` is ignored here — expiry
          filtering is the fetch layer's job.
        - unknown / missing ``source`` -> IGNORED gracefully.

        Returns a summary ``{"roads", "flood", "zones", "ignored"}`` counting
        features routed to each effect (useful for logging).  Unlike the
        other ``add_*`` methods this returns the summary dict, not ``self``.
        """
        summary = {"roads": 0, "flood": 0, "zones": 0, "ignored": 0}
        for gtype, seqs, props in iter_features(feature_collection, to_local):
            source = props.get("source")
            if source == "tiger":
                width_m = props.get("width_m")
                if width_m is None or width_m <= 0:
                    width_m = MTFCC_WIDTHS_M.get(
                        props.get("kind"), self.weights.road_width_m
                    )
                if gtype in LINE_TYPES:
                    half = float(width_m) / 2.0
                    for line in seqs:
                        self._stamp_line(line, half)
                    summary["roads"] += 1
                elif gtype in POLYGON_TYPES:
                    for ring in seqs:
                        self._stamp_polygon(
                            ring, lambda c, r: self._road_cells.add((r, c))
                        )
                    summary["roads"] += 1
                else:
                    summary["ignored"] += 1
            elif source == "fema":
                if props.get("sfha") and gtype in POLYGON_TYPES:
                    for ring in seqs:
                        self._stamp_polygon(
                            ring,
                            lambda c, r: self._obstacle_cells.__setitem__(
                                (r, c), "flood"
                            ),
                        )
                    summary["flood"] += 1
                else:
                    summary["ignored"] += 1
            elif source == "noaa":
                if props.get("severity") in ("Severe", "Extreme") and (
                    gtype in POLYGON_TYPES
                ):
                    for ring in seqs:
                        self._stamp_polygon(
                            ring, lambda c, r, m=2.0: self._mark_zone(r, c, m)
                        )
                    summary["zones"] += 1
                else:
                    summary["ignored"] += 1
            else:
                summary["ignored"] += 1
        return summary

    # -- Build --------------------------------------------------------------

    def build(self) -> Costmap:
        """Apply all layers in fixed order and return a :class:`Costmap`."""
        w = self.weights
        lethal = float("inf")
        grid: list[list[float]] = [
            [0.0] * self.width for _ in range(self.height)
        ]

        for row in range(self.height):
            for col in range(self.width):
                cx, cy = self._cell_center(col, row)
                slope = self._dem.slope_at(cx, cy) if self._dem is not None else 0.0

                cost = w.base_cost + w.slope_weight * slope
                if (row, col) in self._road_cells:
                    cost *= w.road_discount

                # Soft-cost zones: MAX multiplier over covering zones, applied
                # after the road discount but before any lethal override so a
                # zone can raise cost yet never make a cell impassable.
                zone_mult = self._cost_zone_cells.get((row, col))
                if zone_mult is not None:
                    cost *= zone_mult

                if slope > w.max_slope:
                    cost = lethal
                if (row, col) in self._obstacle_cells:
                    cost = lethal

                grid[row][col] = cost

        costmap = Costmap(
            origin_x=self.origin_x,
            origin_y=self.origin_y,
            resolution=self.resolution,
            width=self.width,
            height=self.height,
            grid=grid,
        )

        if w.obstacle_inflation_m > 0:
            _inflate(costmap, w.obstacle_inflation_m, w.inflation_cost)

        return costmap


# ---------------------------------------------------------------------------
# Inflation
# ---------------------------------------------------------------------------

def _inflate(costmap: Costmap, radius_m: float, inflation_cost: float) -> None:
    """Raise non-lethal cells within ``radius_m`` of a lethal cell.

    Implemented as a bounded disk stamp expanding outward from every lethal
    cell: each non-lethal cell whose center is within ``radius_m`` of a
    lethal cell's center is raised to at least ``inflation_cost``.
    """
    res = costmap.resolution
    r_cells = int(math.ceil(radius_m / res))
    lethal = costmap.LETHAL
    lethal_cells = [
        (row, col)
        for row in range(costmap.height)
        for col in range(costmap.width)
        if costmap.grid[row][col] == lethal
    ]
    for lr, lc in lethal_cells:
        for dr in range(-r_cells, r_cells + 1):
            nr = lr + dr
            if nr < 0 or nr >= costmap.height:
                continue
            for dc in range(-r_cells, r_cells + 1):
                nc = lc + dc
                if nc < 0 or nc >= costmap.width:
                    continue
                if costmap.grid[nr][nc] == lethal:
                    continue
                dist = math.hypot(dr * res, dc * res)
                if dist <= radius_m:
                    if costmap.grid[nr][nc] < inflation_cost:
                        costmap.grid[nr][nc] = inflation_cost


# ---------------------------------------------------------------------------
# TerrainMap adapter
# ---------------------------------------------------------------------------

def builder_from_terrain_map(
    terrain_map, weights: CostmapWeights | None = None
) -> CostmapBuilder:
    """Seed a :class:`CostmapBuilder` from a sim ``TerrainMap``.

    Unlike :func:`costmap_from_terrain_map` — which returns a *finished*
    :class:`Costmap` and so cannot be layered on — this returns the builder
    with its terrain baseline stamped in, ready to be enriched with GIS
    layers (DEM slope, TIGER roads, FEMA flood, NOAA weather zones) via
    :meth:`~CostmapBuilder.add_dem`, :meth:`~CostmapBuilder.add_roads`,
    :meth:`~CostmapBuilder.add_obstacles`,
    :meth:`~CostmapBuilder.add_cost_zones`, or
    :meth:`~CostmapBuilder.add_gis_features` before :meth:`~CostmapBuilder.build`.

    Duck-typed against the TerrainMap API (``grid_size``, ``resolution``,
    ``get_terrain_at(col, row)``, ``_grid_to_world(col, row)``) so tests can
    pass a fake.  The grid frame is identical to
    :func:`costmap_from_terrain_map`: the south-west corner is cell
    ``(0, 0)``'s center minus half a cell, and the grid is ``grid_size`` cells
    square at the terrain resolution.

    Terrain seeding (the same mapping table as
    :func:`costmap_from_terrain_map`, expressed as builder layers):

        - ``building`` / ``water`` -> lethal obstacle cell (tagged with the
          terrain kind for introspection)
        - ``road`` -> road (discounted) cell
        - everything else (open, yard, ...) -> untouched open cell

    Args:
        terrain_map: A sim ``TerrainMap`` (or duck-typed fake) exposing
            ``grid_size``, ``resolution``, ``get_terrain_at(col, row)`` and
            ``_grid_to_world(col, row)``.
        weights: Cost weights for the builder.  Defaults to
            :class:`CostmapWeights`.

    Returns:
        A :class:`CostmapBuilder` pre-seeded with the terrain obstacle and
        road cells, ready for further ``add_*`` enrichment and ``build()``.
    """
    w = weights or CostmapWeights()
    gs = int(terrain_map.grid_size)
    res = float(terrain_map.resolution)

    # Derive the SW corner from cell (0, 0)'s center — identical frame to
    # costmap_from_terrain_map.
    c0x, c0y = terrain_map._grid_to_world(0, 0)
    origin_x = c0x - res / 2.0
    origin_y = c0y - res / 2.0

    builder = CostmapBuilder(
        bounds=(origin_x, origin_y, origin_x + gs * res, origin_y + gs * res),
        resolution=res,
        weights=w,
    )
    # The ctor derives width/height with math.ceil, which can drift by a cell
    # on a non-integer span due to float rounding.  Pin the dimensions to
    # grid_size so the frame is bit-identical to costmap_from_terrain_map.
    builder.width = gs
    builder.height = gs

    for row in range(gs):
        for col in range(gs):
            terrain = terrain_map.get_terrain_at(col, row)
            if terrain in ("building", "water"):
                builder._obstacle_cells[(row, col)] = terrain
            elif terrain == "road":
                builder._road_cells.add((row, col))

    return builder


def costmap_from_terrain_map(terrain_map, weights: CostmapWeights | None = None) -> Costmap:
    """Adapt an existing sim ``TerrainMap`` into a :class:`Costmap`.

    Duck-typed against the TerrainMap API (``grid_size``, ``resolution``,
    ``get_terrain_at(col, row)``, ``_grid_to_world(col, row)``) so tests can
    pass a fake.  Terrain mapping:

        - ``building`` / ``water`` -> ``LETHAL``
        - ``road`` -> ``base_cost * road_discount``
        - everything else (open, yard, ...) -> ``base_cost``

    This is the terrain-only convenience path.  It is exactly
    :func:`builder_from_terrain_map` followed by
    :meth:`~CostmapBuilder.build` with no enrichment layers added: with the
    default weights every open cell is ``base_cost``, every road cell is
    ``base_cost * road_discount``, every building/water cell is ``LETHAL``,
    there is no DEM (slope 0) and no inflation (``obstacle_inflation_m`` 0.0).
    Use :func:`builder_from_terrain_map` directly when you want to enrich the
    terrain baseline with GIS layers before building.
    """
    return builder_from_terrain_map(terrain_map, weights).build()


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _point_segment_distance(
    px: float, py: float, x0: float, y0: float, x1: float, y1: float
) -> float:
    """Shortest distance from point ``(px, py)`` to segment ``(x0,y0)-(x1,y1)``."""
    dx = x1 - x0
    dy = y1 - y0
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq <= 1e-12:
        return math.hypot(px - x0, py - y0)
    t = ((px - x0) * dx + (py - y0) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    cx = x0 + t * dx
    cy = y0 + t * dy
    return math.hypot(px - cx, py - cy)
