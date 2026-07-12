# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Layer inputs for costmap generation — DEM grids + GeoJSON geometry.

This module defines the *input contract* between the parallel GIS lane
(which fetches USGS elevation and OSM/vector features) and the costmap
builder in :mod:`tritium_lib.planning.costmap`.

Two kinds of layer input are supported:

1. :class:`LocalElevationGrid` — a Digital Elevation Model (DEM) sampled in
   **local meters** with ``row 0`` = south.  This is planning's own raster
   convention.  A GIS pipeline that downloads elevation tiles rasterizes
   them into this structure so the costmap builder can read slope without
   knowing where the data came from.

   .. note::
      This class was called ``ElevationGrid`` before the GIS-lane merge.
      It was renamed to :class:`LocalElevationGrid` because the parallel
      GIS lane owns the name ``ElevationGrid`` for its **WGS-84 wire model**
      (``tritium_lib.geo.gis.models.ElevationGrid``) which uses the OPPOSITE
      row convention (``row 0`` = *north*, flat row-major ``values`` over a
      lat/lng bbox).  Planning must not export ``ElevationGrid`` at all.
      Use :func:`local_grid_from_gis` to convert the WGS-84 wire grid into a
      local-meter :class:`LocalElevationGrid`.

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
    "LocalElevationGrid",
    "local_grid_from_gis",
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
# LocalElevationGrid — planning's local-meter DEM convention
# ---------------------------------------------------------------------------

@dataclass
class LocalElevationGrid:
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
    ) -> "LocalElevationGrid":
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
# GIS wire-DEM adapter (WGS-84 row-0-north -> local-meter row-0-south)
# ---------------------------------------------------------------------------

def _gis_field(gis_grid, key: str):
    """Read ``key`` from a GIS grid that may be an object or a plain dict."""
    if isinstance(gis_grid, dict):
        return gis_grid.get(key)
    return getattr(gis_grid, key, None)


def local_grid_from_gis(
    gis_grid,
    *,
    to_local: Callable[[float, float], tuple[float, float]] | None = None,
    resolution: float | None = None,
    nodata_fill: float | None = None,
) -> LocalElevationGrid:
    """Adapt a GIS-lane WGS-84 elevation grid into a :class:`LocalElevationGrid`.

    The GIS lane (``tritium_lib.geo.gis.models.ElevationGrid``) delivers a
    Digital Elevation Model as a **WGS-84 wire model** with the OPPOSITE row
    convention to planning's local raster:

        - bbox fields ``west, south, east, north`` (degrees),
        - ``ncols, nrows`` grid dimensions,
        - ``values`` — a **flat row-major** list of length ``ncols*nrows``
          with **``row 0`` = NORTH edge**, values increasing eastward within
          a row.  ``None`` marks NoData.
        - inclusive-edge sampling: column 0 sits exactly on ``west``, column
          ``ncols-1`` on ``east``; row 0 on ``north``, row ``nrows-1`` on
          ``south``.

    ``gis_grid`` is duck-typed: pass either an object exposing those attrs or
    a plain dict carrying those keys (the ``GET /api/gis/elevation/grid``
    payload).  Extra keys (``source``, ``resolution_m``, ``fixture``, …) are
    tolerated and ignored.

    Args:
        gis_grid: The WGS-84 wire grid (object or dict).
        to_local: ``to_local(lng, lat) -> (x, y)`` projector.  Defaults to
            :func:`wgs84_to_local` (needs an initialised ``tritium_lib.geo``
            reference).
        resolution: Output node spacing in meters.  Defaults to the mean
            source cell spacing in meters (floored at ``1e-6``).
        nodata_fill: Fill value when *all* four source neighbours of an
            output node are NoData.  Defaults to the mean of every non-None
            source value (or ``0.0`` if the grid is entirely NoData — which
            cannot happen because an all-None grid raises first).

    Returns:
        A :class:`LocalElevationGrid` in local meters with ``row 0`` = south.

    Raises:
        ValueError: Empty/degenerate grid (``ncols`` or ``nrows`` < 2, a
            ``values`` length mismatch, an all-NoData grid, or a bbox that
            projects to a zero-area local extent).

    Assumption:
        The tritium geo projection is **equirectangular over an AO-sized
        bbox**, so the mapping from local ``(x, y)`` to fractional source
        index is AFFINE.  The four bbox corners are projected and their
        min/max define the axis-aligned local extent; each output node maps
        back to a fractional ``(iy_frac, ix_frac)`` source index (with a row
        FLIP, since the source ``row 0`` = north but the output ``row 0`` =
        south) and is sampled by BILINEAR interpolation.
    """
    west = _gis_field(gis_grid, "west")
    south = _gis_field(gis_grid, "south")
    east = _gis_field(gis_grid, "east")
    north = _gis_field(gis_grid, "north")
    ncols = _gis_field(gis_grid, "ncols")
    nrows = _gis_field(gis_grid, "nrows")
    values = _gis_field(gis_grid, "values")

    if None in (west, south, east, north, ncols, nrows) or values is None:
        raise ValueError(
            "local_grid_from_gis: gis_grid is missing required fields "
            "(west/south/east/north/ncols/nrows/values)"
        )
    ncols = int(ncols)
    nrows = int(nrows)
    if ncols < 2 or nrows < 2:
        raise ValueError(
            f"local_grid_from_gis: degenerate grid {ncols}x{nrows} — need "
            "ncols >= 2 and nrows >= 2"
        )
    if len(values) != ncols * nrows:
        raise ValueError(
            f"local_grid_from_gis: values length {len(values)} != "
            f"ncols*nrows ({ncols * nrows})"
        )

    non_none = [v for v in values if v is not None]
    if not non_none:
        raise ValueError("local_grid_from_gis: grid is entirely NoData")
    if nodata_fill is None:
        nodata_fill = sum(non_none) / len(non_none)

    if to_local is None:
        to_local = wgs84_to_local()

    # Project the four bbox corners; the local extent is their min/max.
    corners = [
        to_local(west, north),
        to_local(east, north),
        to_local(west, south),
        to_local(east, south),
    ]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    x_west, x_east = min(xs), max(xs)
    y_south, y_north = min(ys), max(ys)
    span_x = x_east - x_west
    span_y = y_north - y_south
    if span_x <= _EPS or span_y <= _EPS:
        raise ValueError(
            "local_grid_from_gis: bbox projects to a zero-area local extent"
        )

    if resolution is None:
        x_spacing = span_x / (ncols - 1)
        y_spacing = span_y / (nrows - 1)
        resolution = max((x_spacing + y_spacing) / 2.0, 1e-6)
    resolution = float(resolution)

    out_cols = max(2, int(math.floor(span_x / resolution + _EPS)) + 1)
    out_rows = max(2, int(math.floor(span_y / resolution + _EPS)) + 1)

    def _sample(iy_frac: float, ix_frac: float) -> float:
        # Clamp fractional index into the valid lattice.
        if ix_frac < 0.0:
            ix_frac = 0.0
        elif ix_frac > ncols - 1:
            ix_frac = float(ncols - 1)
        if iy_frac < 0.0:
            iy_frac = 0.0
        elif iy_frac > nrows - 1:
            iy_frac = float(nrows - 1)

        ic0 = int(math.floor(ix_frac))
        ir0 = int(math.floor(iy_frac))
        if ic0 > ncols - 2:
            ic0 = ncols - 2
        if ir0 > nrows - 2:
            ir0 = nrows - 2
        ic1 = ic0 + 1
        ir1 = ir0 + 1
        tx = ix_frac - ic0
        ty = iy_frac - ir0

        v00 = values[ir0 * ncols + ic0]
        v01 = values[ir0 * ncols + ic1]
        v10 = values[ir1 * ncols + ic0]
        v11 = values[ir1 * ncols + ic1]

        if None not in (v00, v01, v10, v11):
            top = v00 + (v01 - v00) * tx
            bot = v10 + (v11 - v10) * tx
            return top + (bot - top) * ty

        # NoData: nearest non-None of the four neighbours by fractional
        # distance in index space; all-None -> nodata_fill.
        neighbours = [
            (v00, math.hypot(ty, tx)),
            (v01, math.hypot(ty, 1.0 - tx)),
            (v10, math.hypot(1.0 - ty, tx)),
            (v11, math.hypot(1.0 - ty, 1.0 - tx)),
        ]
        candidates = sorted(
            ((d, v) for (v, d) in neighbours if v is not None),
            key=lambda t: t[0],
        )
        if candidates:
            return float(candidates[0][1])
        return float(nodata_fill)

    data: list[list[float]] = []
    for out_row in range(out_rows):
        y = y_south + out_row * resolution
        iy_frac = (y_north - y) / span_y * (nrows - 1)  # row FLIP
        grid_row: list[float] = []
        for out_col in range(out_cols):
            x = x_west + out_col * resolution
            ix_frac = (x - x_west) / span_x * (ncols - 1)
            grid_row.append(_sample(iy_frac, ix_frac))
        data.append(grid_row)

    return LocalElevationGrid(
        origin_x=x_west,
        origin_y=y_south,
        resolution=resolution,
        data=data,
    )


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
