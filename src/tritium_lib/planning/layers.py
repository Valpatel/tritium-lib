# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Layer inputs for costmap generation — DEM grids + GeoJSON geometry.

This module defines the *input contract* between the parallel GIS lane
(which fetches USGS elevation and OSM/vector features) and the costmap
builder in :mod:`tritium_lib.planning.costmap`.

Two kinds of layer input are supported:

1. :class:`ElevationGrid` — a Digital Elevation Model (DEM).  This is the
   canonical DEM convention for the whole system.  A GIS pipeline that
   downloads USGS 3DEP tiles rasterizes them into this exact structure so
   the costmap builder can read slope without knowing where the data came
   from.

2. GeoJSON ``FeatureCollection`` dicts — polygons (buildings, water,
   flood masks) and lines (roads).  Coordinates are assumed to be **local
   meters** ``[x, y]`` unless a ``to_local(c0, c1) -> (x, y)`` callable is
   supplied, in which case the raw coordinates are treated as WGS-84
   ``[lng, lat]`` and projected.

Coordinate frame everywhere: local meters, +X = East, +Y = North — the
same frame as :mod:`tritium_lib.geo`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterator

__all__ = [
    "ElevationGrid",
    "wgs84_to_local",
    "iter_features",
    "iter_polygons",
    "iter_lines",
    "POLYGON_TYPES",
    "LINE_TYPES",
]

# GeoJSON geometry types handled by the layer helpers.
POLYGON_TYPES = frozenset({"Polygon", "MultiPolygon"})
LINE_TYPES = frozenset({"LineString", "MultiLineString"})

_EPS = 1e-9


# ---------------------------------------------------------------------------
# ElevationGrid — the canonical DEM convention
# ---------------------------------------------------------------------------

@dataclass
class ElevationGrid:
    """A Digital Elevation Model sampled on a regular grid of nodes.

    The value ``data[row][col]`` is the elevation (meters) sampled at the
    world position::

        x = origin_x + col * resolution
        y = origin_y + row * resolution

    Conventions:
        - ``origin_x``/``origin_y`` is the **south-west corner** node
          (local meters).
        - ``row 0`` is the **southernmost** row; ``col 0`` is the
          **westernmost** column.  Row-major nested lists.
        - Samples are treated as node values at cell corners, so bilinear
          interpolation blends the four surrounding nodes — the standard
          DEM convention (matches USGS 3DEP raster sampling).

    Attributes:
        origin_x: World X (meters) of the south-west node.
        origin_y: World Y (meters) of the south-west node.
        resolution: Node spacing in meters.
        data: Row-major elevations, ``data[row][col]`` in meters.
    """

    origin_x: float
    origin_y: float
    resolution: float
    data: list[list[float]] = field(default_factory=list)

    @property
    def height(self) -> int:
        """Number of rows (samples along +Y / north)."""
        return len(self.data)

    @property
    def width(self) -> int:
        """Number of columns (samples along +X / east)."""
        return len(self.data[0]) if self.data else 0

    # -- Interpolation ------------------------------------------------------

    def elevation_at(self, x: float, y: float) -> float | None:
        """Bilinearly interpolated elevation at world ``(x, y)``.

        Returns ``None`` if the point lies outside the sampled region.
        """
        w = self.width
        h = self.height
        if w == 0 or h == 0:
            return None
        res = self.resolution

        fx = (x - self.origin_x) / res
        fy = (y - self.origin_y) / res

        # Fractional node coordinates must lie within the node lattice.
        if fx < -_EPS or fy < -_EPS or fx > (w - 1) + _EPS or fy > (h - 1) + _EPS:
            return None

        col0 = int(math.floor(fx))
        row0 = int(math.floor(fy))
        # Clamp so col0+1 / row0+1 stay valid; tx/ty absorb the edge.
        if col0 >= w - 1:
            col0 = max(0, w - 2)
        if row0 >= h - 1:
            row0 = max(0, h - 2)
        col1 = min(col0 + 1, w - 1)
        row1 = min(row0 + 1, h - 1)

        tx = fx - col0
        ty = fy - row0

        v00 = self.data[row0][col0]
        v01 = self.data[row0][col1]
        v10 = self.data[row1][col0]
        v11 = self.data[row1][col1]

        top = v00 + (v01 - v00) * tx
        bot = v10 + (v11 - v10) * tx
        return top + (bot - top) * ty

    def slope_at(self, x: float, y: float) -> float:
        """Gradient magnitude (rise/run, unitless) at world ``(x, y)``.

        Uses central differences over one ``resolution`` step in each axis.
        Returns ``0.0`` if any required sample is out of bounds.
        """
        res = self.resolution
        e_xp = self.elevation_at(x + res, y)
        e_xm = self.elevation_at(x - res, y)
        e_yp = self.elevation_at(x, y + res)
        e_ym = self.elevation_at(x, y - res)
        if None in (e_xp, e_xm, e_yp, e_ym):
            return 0.0
        dzdx = (e_xp - e_xm) / (2.0 * res)  # type: ignore[operator]
        dzdy = (e_yp - e_ym) / (2.0 * res)  # type: ignore[operator]
        return math.hypot(dzdx, dzdy)

    # -- Construction -------------------------------------------------------

    @classmethod
    def from_callable(
        cls,
        bounds: tuple[float, float, float, float],
        resolution: float,
        fn: Callable[[float, float], float],
    ) -> "ElevationGrid":
        """Build a synthetic DEM by sampling ``fn(x, y) -> elevation``.

        Args:
            bounds: ``(min_x, min_y, max_x, max_y)`` in local meters.
            resolution: Node spacing in meters.
            fn: Elevation function evaluated at each node position.

        The node lattice spans ``min`` to ``max`` inclusive (at least a
        2x2 lattice), so ``elevation_at`` covers the full bounds.
        """
        min_x, min_y, max_x, max_y = bounds
        cols = max(2, int(math.floor((max_x - min_x) / resolution + _EPS)) + 1)
        rows = max(2, int(math.floor((max_y - min_y) / resolution + _EPS)) + 1)
        data = [
            [fn(min_x + col * resolution, min_y + row * resolution)
             for col in range(cols)]
            for row in range(rows)
        ]
        return cls(origin_x=min_x, origin_y=min_y, resolution=resolution, data=data)


# ---------------------------------------------------------------------------
# WGS-84 -> local projection factory
# ---------------------------------------------------------------------------

def wgs84_to_local() -> Callable[[float, float], tuple[float, float]]:
    """Return a ``to_local(lng, lat) -> (x, y)`` projector.

    Uses the process-wide :mod:`tritium_lib.geo` reference singleton.  The
    returned callable takes GeoJSON coordinate order ``(lng, lat)`` and
    returns local meters ``(x, y)``.

    Raises:
        RuntimeError: If the geo reference singleton has not been
            initialised (call :func:`tritium_lib.geo.init_reference` first).
    """
    from tritium_lib import geo

    ref = geo.get_reference()
    if not ref.initialized:
        raise RuntimeError(
            "tritium_lib.geo reference is uninitialised — call "
            "geo.init_reference(lat, lng) before projecting WGS-84 layers"
        )

    def to_local(lng: float, lat: float) -> tuple[float, float]:
        x, y, _ = geo.latlng_to_local(lat, lng)
        return (x, y)

    return to_local


# ---------------------------------------------------------------------------
# GeoJSON iteration
# ---------------------------------------------------------------------------

def _convert_ring(
    coords: list,
    to_local: Callable[[float, float], tuple[float, float]] | None,
) -> list[tuple[float, float]]:
    """Convert a GeoJSON coordinate sequence to local-meter ``(x, y)`` pairs."""
    if to_local is None:
        return [(float(c[0]), float(c[1])) for c in coords]
    out: list[tuple[float, float]] = []
    for c in coords:
        x, y = to_local(c[0], c[1])
        out.append((float(x), float(y)))
    return out


def _extract_sequences(
    gtype: str,
    coords: list,
    to_local: Callable[[float, float], tuple[float, float]] | None,
) -> list[list[tuple[float, float]]] | None:
    """Return the list of rings (polygons) or lines for one geometry.

    Polygon exterior rings only — holes are ignored for now.  Returns
    ``None`` for unsupported geometry types.
    """
    try:
        if gtype == "Polygon":
            if not coords:
                return []
            return [_convert_ring(coords[0], to_local)]
        if gtype == "MultiPolygon":
            return [
                _convert_ring(poly[0], to_local)
                for poly in coords
                if poly
            ]
        if gtype == "LineString":
            return [_convert_ring(coords, to_local)]
        if gtype == "MultiLineString":
            return [_convert_ring(line, to_local) for line in coords if line]
    except (TypeError, IndexError, ValueError):
        return None
    return None


def iter_features(
    feature_collection: dict,
    to_local: Callable[[float, float], tuple[float, float]] | None = None,
) -> Iterator[tuple[str, list[list[tuple[float, float]]], dict]]:
    """Iterate a GeoJSON ``FeatureCollection``.

    Yields ``(geometry_type, sequences, properties)`` for each feature,
    where ``sequences`` is a list of coordinate sequences already in local
    meters:

        - Polygon / MultiPolygon -> one exterior ring per polygon
        - LineString / MultiLineString -> one point list per line

    Unsupported geometry types (Point, GeometryCollection, ...) are skipped
    gracefully.  ``geometry_type`` is the raw GeoJSON type string; callers
    test membership against :data:`POLYGON_TYPES` / :data:`LINE_TYPES`.
    """
    if not isinstance(feature_collection, dict):
        return
    features = feature_collection.get("features")
    if not features:
        # Allow a bare geometry or single Feature too.
        if feature_collection.get("type") == "Feature":
            features = [feature_collection]
        else:
            return
    for feat in features:
        if not isinstance(feat, dict):
            continue
        geom = feat.get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates")
        props = feat.get("properties") or {}
        if gtype is None or coords is None:
            continue
        seqs = _extract_sequences(gtype, coords, to_local)
        if not seqs:
            continue
        yield gtype, seqs, props


def iter_polygons(
    feature_collection: dict,
    to_local: Callable[[float, float], tuple[float, float]] | None = None,
) -> Iterator[tuple[list[tuple[float, float]], dict]]:
    """Yield ``(exterior_ring, properties)`` for every polygon feature."""
    for gtype, seqs, props in iter_features(feature_collection, to_local):
        if gtype in POLYGON_TYPES:
            for ring in seqs:
                yield ring, props


def iter_lines(
    feature_collection: dict,
    to_local: Callable[[float, float], tuple[float, float]] | None = None,
) -> Iterator[tuple[list[tuple[float, float]], dict]]:
    """Yield ``(line, properties)`` for every line feature."""
    for gtype, seqs, props in iter_features(feature_collection, to_local):
        if gtype in LINE_TYPES:
            for line in seqs:
                yield line, props
