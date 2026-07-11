# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Iso-cost contour curves over a :class:`Costmap` (marching squares).

Reuses the GIS marching-squares engine —
:func:`tritium_lib.geo.gis.contours.contour_lines` /
:func:`~tritium_lib.geo.gis.contours.auto_levels` — by adapting the costmap
into a real :class:`~tritium_lib.geo.gis.models.ElevationGrid` whose lon/lat
slots carry **local meters**.  Output coordinates are therefore ``[x, y]``
local meters, directly renderable on the tactical map next to the costmap
telemetry layer.

Adapter conventions (load-bearing):

    - ``Costmap`` row 0 is the SOUTH row; ``ElevationGrid`` row 0 is the NORTH
      row — the adapter flips rows (``values[iy * ncols + ix] =
      grid[height - 1 - iy][ix]``).
    - Lattice nodes are **cell centers**: ``west = origin_x + 0.5 * res`` ..
      ``east = origin_x + (width - 0.5) * res`` (same for south/north), so
      ``cell_lon`` / ``cell_lat`` land exactly on costmap cell centers.
    - ``LETHAL`` (``inf``) cells become ``None`` (NoData) — marching squares
      skips cells touching NoData, so contours never cross a lethal region.

Each emitted feature's ``properties`` follow the iso-cost contract:

    {"source": "costmap", "kind": "iso_cost",
     "cost": <level rounded 3 dp>, "level_index": <i>}

The returned ``FeatureCollection`` carries a foreign member ``"levels"``: the
list of cost levels actually contoured.  Degenerate inputs (grid smaller than
2x2, all-lethal, uniform cost, no usable levels) return an empty collection —
never raise.
"""

from __future__ import annotations

import math

from tritium_lib.geo.gis.contours import auto_levels, contour_lines
from tritium_lib.geo.gis.models import ElevationGrid

from .costmap import Costmap

__all__ = ["iso_cost_contours"]


def _empty_collection() -> dict:
    """A fresh empty result — degenerate inputs all funnel here."""
    return {"type": "FeatureCollection", "features": [], "levels": []}


def _as_elevation_grid(costmap: Costmap) -> ElevationGrid:
    """Adapt a :class:`Costmap` into an :class:`ElevationGrid` for contouring.

    Flips rows (costmap row 0 = south -> grid row 0 = north), converts
    ``LETHAL``/non-finite costs to ``None`` (NoData), and places the lattice
    bounds on the outermost **cell centers** so node coordinates come out as
    local-meter cell centers.
    """
    res = costmap.resolution
    w, h = costmap.width, costmap.height
    values: list = []
    for iy in range(h):
        row = costmap.grid[h - 1 - iy]
        for ix in range(w):
            v = row[ix]
            values.append(None if not math.isfinite(v) else float(v))
    return ElevationGrid(
        west=costmap.origin_x + 0.5 * res,
        south=costmap.origin_y + 0.5 * res,
        east=costmap.origin_x + (w - 0.5) * res,
        north=costmap.origin_y + (h - 0.5) * res,
        ncols=w,
        nrows=h,
        values=values,
        source="costmap",
        resolution_m=res,
    )


def iso_cost_contours(
    costmap: Costmap, levels: list | None = None, n: int = 6
) -> dict:
    """Trace iso-cost contour polylines over a :class:`Costmap`.

    Args:
        costmap: The cost grid to contour (local meters, row 0 = south).
        levels: Explicit cost levels to trace, used as-is (``level_index`` is
            the position in this list).  ``None`` picks ``n`` evenly spaced
            levels strictly inside the (min, max) cost range via
            :func:`~tritium_lib.geo.gis.contours.auto_levels`.
        n: Number of auto levels when ``levels`` is ``None``.

    Returns:
        A GeoJSON ``FeatureCollection`` dict of ``LineString`` features with
        ``[x, y]`` local-meter coordinates, each carrying the iso-cost
        ``properties`` contract (``source``/``kind``/``cost``/``level_index``),
        plus a foreign member ``"levels"`` listing the cost levels actually
        used.  Degenerate inputs (grid smaller than 2x2, all-lethal, uniform
        cost, empty level list) return an empty collection — never raise.
    """
    if costmap.width < 2 or costmap.height < 2:
        return _empty_collection()

    grid = _as_elevation_grid(costmap)
    mn, mx = grid.min_max()
    if mn is None or mx is None or mx <= mn:
        # All-lethal (nothing present) or uniform cost — nothing to contour.
        return _empty_collection()

    if levels is None:
        levels = auto_levels(grid, n)
    levels = [float(level) for level in levels]
    if not levels:
        return _empty_collection()

    collection = contour_lines(grid, levels)
    features: list = []
    for feature in collection.get("features", []):
        level_index = feature["properties"]["level_index"]
        feature["properties"] = {
            "source": "costmap",
            "kind": "iso_cost",
            "cost": round(levels[level_index], 3),
            "level_index": level_index,
        }
        features.append(feature)
    return {"type": "FeatureCollection", "features": features, "levels": levels}
