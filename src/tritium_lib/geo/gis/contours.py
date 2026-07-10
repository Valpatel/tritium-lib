# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Iso-elevation contour lines from an :class:`ElevationGrid` (marching squares).

Pure stdlib.  Turns a sampled terrain raster into a GeoJSON ``FeatureCollection``
of ``LineString`` contours the tactical map and costmap lane can render directly.

The algorithm is the standard 16-case *marching squares* run over the grid's
cell-corner lattice: the grid samples **are** the lattice nodes, crossings are
linearly interpolated along cell edges, and the two ambiguous saddle cases
(5 and 10) are resolved by the sign of the cell-centre average.  Cells that
touch a NoData sample are skipped entirely, so contours never cross a hole.

Row convention (load-bearing, do not flip): ``ElevationGrid`` row 0 is the
NORTH edge.  We read ``grid.cell_lon`` / ``grid.cell_lat`` for node coordinates,
so the row-0=north convention is honoured transparently — output coordinates are
WGS-84 ``[lon, lat]``.

Each emitted feature's ``properties`` follow the vector contract:

    {"source": "usgs", "kind": "contour",
     "elevation_m": <level rounded 1 dp>, "level_index": <i>}
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .models import ElevationGrid

__all__ = ["auto_levels", "contour_lines"]

#: Endpoint-matching tolerance when joining segments into polylines (degrees).
_JOIN_TOL_DIGITS = 9

# Marching-squares segment table, keyed by the 4-corner bitmask
#   bit 1 = TL(a), bit 2 = TR(b), bit 4 = BR(c), bit 8 = BL(d)   (>= level).
# Values are the unambiguous single-segment cases as (edge, edge) pairs where
# edge is one of 'T'(op) 'R'(ight) 'B'(ottom) 'L'(eft).  Cases 0/15 (no line)
# and the saddles 5/10 are handled separately.
_SEGMENT_TABLE = {
    1: [("L", "T")],
    2: [("T", "R")],
    3: [("L", "R")],
    4: [("R", "B")],
    6: [("T", "B")],
    7: [("L", "B")],
    8: [("L", "B")],
    9: [("T", "B")],
    11: [("R", "B")],
    12: [("L", "R")],
    13: [("T", "R")],
    14: [("L", "T")],
}


def auto_levels(grid: "ElevationGrid", n: int = 8) -> list:
    """Return ``n`` evenly spaced contour levels strictly inside ``(min, max)``.

    The levels sit at fractions ``i / (n + 1)`` of the value range for
    ``i = 1..n`` — every level is strictly greater than the grid minimum and
    strictly less than the maximum (so a level never coincides with the flat
    outer boundary and always has terrain on both sides).  Returns ``[]`` when
    the grid has fewer than two distinct present values (nothing to contour).
    """
    mn, mx = grid.min_max()
    if mn is None or mx is None or mx <= mn or n < 1:
        return []
    distinct = {v for v in grid.values if v is not None}
    if len(distinct) < 2:
        return []
    span = mx - mn
    return [mn + span * (i / (n + 1)) for i in range(1, n + 1)]


def contour_lines(grid: "ElevationGrid", levels: list) -> dict:
    """Trace each level in ``levels`` as GeoJSON ``LineString`` contour features.

    Marching squares over the cell-corner lattice with linear edge
    interpolation; saddle cases resolved by the cell-centre average; cells that
    touch a NoData corner are skipped.  Segments are joined into polylines where
    endpoints coincide (tolerance ``1e-9`` deg).  Returns a
    ``FeatureCollection`` dict; each feature carries the contour ``properties``
    contract documented in the module docstring.
    """
    features: list = []
    if not levels or grid.ncols < 2 or grid.nrows < 2:
        return {"type": "FeatureCollection", "features": features}

    for level_index, level in enumerate(levels):
        segments = _level_segments(grid, level)
        for poly in _join_segments(segments):
            if len(poly) < 2:
                continue
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[lon, lat] for lon, lat in poly],
                    },
                    "properties": {
                        "source": "usgs",
                        "kind": "contour",
                        "elevation_m": round(level, 1),
                        "level_index": level_index,
                    },
                }
            )
    return {"type": "FeatureCollection", "features": features}


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _level_segments(grid: "ElevationGrid", level: float) -> list:
    """All marching-squares line segments for a single iso-level.

    Returns a list of ``((lon0, lat0), (lon1, lat1))`` segment endpoints.
    """
    segments: list = []
    for iy in range(grid.nrows - 1):
        lat_n = grid.cell_lat(iy)
        lat_s = grid.cell_lat(iy + 1)
        for ix in range(grid.ncols - 1):
            a = grid.value_at(ix, iy)          # TL
            b = grid.value_at(ix + 1, iy)      # TR
            c = grid.value_at(ix + 1, iy + 1)  # BR
            d = grid.value_at(ix, iy + 1)      # BL
            # Skip any cell touching a NoData sample — never cross a hole.
            if a is None or b is None or c is None or d is None:
                continue

            case = (
                (1 if a >= level else 0)
                | (2 if b >= level else 0)
                | (4 if c >= level else 0)
                | (8 if d >= level else 0)
            )
            if case == 0 or case == 15:
                continue

            seg_edges = _cell_segments(case, a, b, c, d, level)
            if not seg_edges:
                continue

            lon_w = grid.cell_lon(ix)
            lon_e = grid.cell_lon(ix + 1)

            def edge_point(edge: str) -> tuple:
                # Linear interpolation along the crossed cell edge.  Every edge
                # in the case table separates a >= corner from a < corner, so
                # the denominator is always non-zero.
                if edge == "T":
                    t = (level - a) / (b - a)
                    return (lon_w + t * (lon_e - lon_w), lat_n)
                if edge == "R":
                    t = (level - b) / (c - b)
                    return (lon_e, lat_n + t * (lat_s - lat_n))
                if edge == "B":
                    t = (level - d) / (c - d)
                    return (lon_w + t * (lon_e - lon_w), lat_s)
                # "L"
                t = (level - a) / (d - a)
                return (lon_w, lat_n + t * (lat_s - lat_n))

            for e0, e1 in seg_edges:
                segments.append((edge_point(e0), edge_point(e1)))
    return segments


def _cell_segments(case: int, a: float, b: float, c: float, d: float, level: float) -> list:
    """Edge pairs for one cell.  Saddles (5, 10) resolved by centre average."""
    table_entry = _SEGMENT_TABLE.get(case)
    if table_entry is not None:
        return table_entry

    center = (a + b + c + d) / 4.0
    if case == 5:
        # TL & BR above.  Centre above => the two below corners (TR, BL) are the
        # isolated pockets; centre below => the above corners are separate.
        if center >= level:
            return [("T", "R"), ("L", "B")]
        return [("L", "T"), ("R", "B")]
    if case == 10:
        # TR & BL above (mirror of case 5).
        if center >= level:
            return [("L", "T"), ("R", "B")]
        return [("T", "R"), ("L", "B")]
    return []


def _join_segments(segments: list) -> list:
    """Chain unordered segments into polylines by matching shared endpoints.

    Endpoints are matched after rounding to :data:`_JOIN_TOL_DIGITS` decimal
    places (the ``1e-9`` deg tolerance) — crossings shared by adjacent cells are
    computed from identical corner values and coordinates, so they coincide
    exactly.  Open chains are emitted first (walked from odd-degree endpoints),
    then any remaining closed loops.
    """
    if not segments:
        return []

    def qkey(pt: tuple) -> tuple:
        return (round(pt[0], _JOIN_TOL_DIGITS), round(pt[1], _JOIN_TOL_DIGITS))

    edges: list = []              # (key_a, key_b)
    point_of: dict = {}           # key -> representative (lon, lat)
    for p0, p1 in segments:
        ka, kb = qkey(p0), qkey(p1)
        if ka == kb:
            continue              # drop degenerate zero-length segment
        point_of[ka] = p0
        point_of[kb] = p1
        edges.append((ka, kb))

    adj: dict = defaultdict(list)  # key -> list of edge indices
    for i, (ka, kb) in enumerate(edges):
        adj[ka].append(i)
        adj[kb].append(i)
    used = [False] * len(edges)

    def build_chain(start):
        keys = [start]
        cur = start
        while True:
            nxt = next((i for i in adj[cur] if not used[i]), None)
            if nxt is None:
                break
            used[nxt] = True
            ka, kb = edges[nxt]
            cur = kb if ka == cur else ka
            keys.append(cur)
        return keys

    polylines: list = []

    # 1) Open chains — start at odd-degree endpoints so nothing is left dangling.
    endpoints = [k for k, es in adj.items() if len(es) % 2 == 1]
    for start in endpoints:
        if all(used[i] for i in adj[start]):
            continue
        keys = build_chain(start)
        if len(keys) >= 2:
            polylines.append([point_of[k] for k in keys])

    # 2) Remaining closed loops.
    for i in range(len(edges)):
        if used[i]:
            continue
        keys = build_chain(edges[i][0])
        if len(keys) >= 2:
            polylines.append([point_of[k] for k in keys])

    return polylines
